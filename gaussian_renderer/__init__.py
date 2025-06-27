#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os.path
import time

import torch
import torch.nn as nn
import torch.nn.functional as nnf
from einops import repeat

import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.encodings import STE_binary, STE_multistep


def generate_neural_gaussians(viewpoint_camera, pc : GaussianModel, visible_mask=None, is_training=False, step=0):
    ## view frustum filtering for acceleration

    time_sub = 0

    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)

    anchor = pc.get_anchor[visible_mask]
    #
    feat = pc._anchor_feat[visible_mask]
    grid_offsets = pc._offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]
    binary_grid_masks = pc.get_mask[visible_mask]
    mask_anchor = pc.get_mask_anchor[visible_mask]
    mask_anchor_bool = mask_anchor.to(torch.bool)
    mask_anchor_rate = (mask_anchor.sum() / mask_anchor.numel()).detach()

    bit_per_param = None
    bit_per_feat_param = None
    bit_per_scaling_param = None
    bit_per_offsets_param = None
    Q_feat = 1
    Q_scaling = 0.001
    Q_offsets = 0.2
    if is_training:
        if step > 3000 and step <= 10000:
            # quantization
            feat = feat + torch.empty_like(feat).uniform_(-0.5, 0.5) * Q_feat
            grid_scaling = grid_scaling + torch.empty_like(grid_scaling).uniform_(-0.5, 0.5) * Q_scaling
            grid_offsets = grid_offsets + torch.empty_like(grid_offsets).uniform_(-0.5, 0.5) * Q_offsets

        if step == 10000:
            pc.update_anchor_bound()

        if step > 10000:
            feat_context = pc.calc_interp_feat(anchor)
            feat_context = pc.get_grid_mlp(feat_context)
            mean, scale, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
                torch.split(feat_context, split_size_or_sections=[pc.feat_dim, pc.feat_dim, 6, 6, 3*pc.n_offsets, 3*pc.n_offsets, 1, 1, 1], dim=-1)

            Q_feat = Q_feat * (1 + torch.tanh(Q_feat_adj))
            Q_scaling = Q_scaling * (1 + torch.tanh(Q_scaling_adj))
            Q_offsets = Q_offsets * (1 + torch.tanh(Q_offsets_adj))
            feat = feat + torch.empty_like(feat).uniform_(-0.5, 0.5) * Q_feat
            grid_scaling = grid_scaling + torch.empty_like(grid_scaling).uniform_(-0.5, 0.5) * Q_scaling
            grid_offsets = grid_offsets + torch.empty_like(grid_offsets).uniform_(-0.5, 0.5) * Q_offsets.unsqueeze(1)

            choose_idx = torch.rand_like(anchor[:, 0]) <= 0.05
            choose_idx = choose_idx & mask_anchor_bool
            feat_chosen = feat[choose_idx]
            grid_scaling_chosen = grid_scaling[choose_idx]
            grid_offsets_chosen = grid_offsets[choose_idx].view(-1, 3*pc.n_offsets)
            mean = mean[choose_idx]
            scale = scale[choose_idx]
            mean_scaling = mean_scaling[choose_idx]
            scale_scaling = scale_scaling[choose_idx]
            mean_offsets = mean_offsets[choose_idx]
            scale_offsets = scale_offsets[choose_idx]
            Q_feat = Q_feat[choose_idx]
            Q_scaling = Q_scaling[choose_idx]
            Q_offsets = Q_offsets[choose_idx]
            binary_grid_masks_chosen = binary_grid_masks[choose_idx].repeat(1, 1, 3).view(-1, 3*pc.n_offsets)
            bit_feat = pc.entropy_gaussian.forward(feat_chosen, mean, scale, Q_feat, pc._anchor_feat.mean())
            bit_scaling = pc.entropy_gaussian.forward(grid_scaling_chosen, mean_scaling, scale_scaling, Q_scaling, pc.get_scaling.mean())
            bit_offsets = pc.entropy_gaussian.forward(grid_offsets_chosen, mean_offsets, scale_offsets, Q_offsets, pc._offset.mean())
            bit_offsets = bit_offsets * binary_grid_masks_chosen
            bit_per_feat_param = torch.sum(bit_feat) / bit_feat.numel() * mask_anchor_rate
            bit_per_scaling_param = torch.sum(bit_scaling) / bit_scaling.numel() * mask_anchor_rate
            bit_per_offsets_param = torch.sum(bit_offsets) / bit_offsets.numel() * mask_anchor_rate
            bit_per_param = (torch.sum(bit_feat) + torch.sum(bit_scaling) + torch.sum(bit_offsets)) / \
                            (bit_feat.numel() + bit_scaling.numel() + bit_offsets.numel()) * mask_anchor_rate

    elif not pc.decoded_version:
        torch.cuda.synchronize(); t1 = time.time()
        feat_context = pc.calc_interp_feat(anchor)
        mean, scale, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
            torch.split(pc.get_grid_mlp(feat_context), split_size_or_sections=[pc.feat_dim, pc.feat_dim, 6, 6, 3*pc.n_offsets, 3*pc.n_offsets, 1, 1, 1], dim=-1)

        Q_feat = Q_feat * (1 + torch.tanh(Q_feat_adj))
        Q_scaling = Q_scaling * (1 + torch.tanh(Q_scaling_adj))
        Q_offsets = Q_offsets * (1 + torch.tanh(Q_offsets_adj))  # [N_visible_anchor, 1]
        feat = (STE_multistep.apply(feat, Q_feat, pc._anchor_feat.mean())).detach()
        grid_scaling = (STE_multistep.apply(grid_scaling, Q_scaling, pc.get_scaling.mean())).detach()
        grid_offsets = (STE_multistep.apply(grid_offsets, Q_offsets.unsqueeze(1), pc._offset.mean())).detach()
        torch.cuda.synchronize(); time_sub = time.time() - t1

    else:
        pass

    ob_view = anchor - viewpoint_camera.camera_center
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    ob_view = ob_view / ob_dist

    ## view-adaptive feature
    if pc.use_feat_bank:
        cat_view = torch.cat([ob_view, ob_dist], dim=1)  # [3+1]

        bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1)  # [N_visible_anchor, 1, 3]

        feat = feat.unsqueeze(dim=-1)  # feat: [N_visible_anchor, 32]
        feat = \
            feat[:, ::4, :1].repeat([1, 4, 1])*bank_weight[:, :, :1] + \
            feat[:, ::2, :1].repeat([1, 2, 1])*bank_weight[:, :, 1:2] + \
            feat[:, ::1, :1]*bank_weight[:, :, 2:]
        feat = feat.squeeze(dim=-1)  # [N_visible_anchor, 32]

    cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1)  # [N_visible_anchor, 32+3+1]

    neural_opacity = pc.get_opacity_mlp(cat_local_view)  # [N_visible_anchor, K]
    neural_opacity = neural_opacity.reshape([-1, 1])  # [N_visible_anchor*K, 1]
    neural_opacity = neural_opacity * binary_grid_masks.view(-1, 1)
    mask = (neural_opacity > 0.0)
    mask = mask.view(-1)  # [N_visible_anchor*K]

    # select opacity
    opacity = neural_opacity[mask]  # [N_opacity_pos_gaussian, 1]

    # get offset's color
    color = pc.get_color_mlp(cat_local_view)  # [N_visible_anchor, K*3]
    color = color.reshape([anchor.shape[0] * pc.n_offsets, 3])  # [N_visible_anchor*K, 3]

    # get offset's cov
    scale_rot = pc.get_cov_mlp(cat_local_view)  # [N_visible_anchor, K*7]
    scale_rot = scale_rot.reshape([anchor.shape[0] * pc.n_offsets, 7])  # [N_visible_anchor*K, 7]

    offsets = grid_offsets.view([-1, 3])  # [N_visible_anchor*K, 3]

    # combine for parallel masking
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)  # [N_visible_anchor, 6+3]
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)  # [N_visible_anchor*K, 6+3]
    concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets],
                                 dim=-1)  # [N_visible_anchor*K, (6+3)+3+7+3]
    masked = concatenated_all[mask]  # [N_opacity_pos_gaussian, (6+3)+3+7+3]
    scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6, 3, 3, 7, 3], dim=-1)

    # post-process cov
    scaling = scaling_repeat[:, 3:] * torch.sigmoid(
        scale_rot[:, :3])
    rot = pc.rotation_activation(scale_rot[:, 3:7])  # [N_opacity_pos_gaussian, 4]

    offsets = offsets * scaling_repeat[:, :3]  # [N_opacity_pos_gaussian, 3]
    xyz = repeat_anchor + offsets  # [N_opacity_pos_gaussian, 3]

    if is_training:
        return xyz, color, opacity, scaling, rot, neural_opacity, mask, bit_per_param, bit_per_feat_param, bit_per_scaling_param, bit_per_offsets_param
    else:
        return xyz, color, opacity, scaling, rot, time_sub, mask


def render(viewpoint_camera, backward_viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, visible_mask=None, retain_grad=False, step=0):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    is_training = pc.get_color_mlp.training
    
    if is_training:
        xyz, color, opacity, scaling, rot, neural_opacity, mask, bit_per_param, bit_per_feat_param, bit_per_scaling_param, bit_per_offsets_param = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training, step=step)
        xyz_bwd, color_bwd, opacity_bwd, scaling_bwd, rot_bwd, neural_opacity_bwd, mask_bwd, bit_per_param_bwd, bit_per_feat_param_bwd, bit_per_scaling_param_bwd, bit_per_offsets_param_bwd = generate_neural_gaussians(backward_viewpoint_camera, pc, visible_mask, is_training=is_training, step=step)    
    else:
        xyz, color, opacity, scaling, rot, time_sub, mask = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training, step=step)
        xyz_bwd, color_bwd, opacity_bwd, scaling_bwd, rot_bwd, time_sub_bwd, mask_bwd = generate_neural_gaussians(backward_viewpoint_camera, pc, visible_mask, is_training=is_training, step=step)

    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points_copy = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=False, device="cuda") + 0
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass
    screenspace_points_bwd = torch.zeros_like(xyz_bwd, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points_bwd_copy = torch.zeros_like(xyz_bwd, dtype=pc.get_anchor.dtype, requires_grad=False, device="cuda") + 0
    if retain_grad:
        try:
            screenspace_points_bwd.retain_grad()
        except:
            pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii = rasterizer(
        means3D = xyz,
        means2D = screenspace_points,
        shs = None,
        colors_precomp = color,
        opacities = opacity,
        scales = scaling,
        rotations = rot,
        cov3D_precomp = None)
    
    #Set up backward rasterization configuration
    tanfovx_bwd = math.tan(backward_viewpoint_camera.FoVx * 0.5)
    tanfovy_bwd = math.tan(backward_viewpoint_camera.FoVy * 0.5)

    raster_settings_bwd = GaussianRasterizationSettings(
        image_height=int(backward_viewpoint_camera.image_height),
        image_width=int(backward_viewpoint_camera.image_width),
        tanfovx=tanfovx_bwd,
        tanfovy=tanfovy_bwd,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=backward_viewpoint_camera.world_view_transform,
        projmatrix=backward_viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=backward_viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )
    
    rasterizer_bwd = GaussianRasterizer(raster_settings=raster_settings_bwd)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image_bwd, radii_bwd = rasterizer_bwd(
        means3D = xyz_bwd,
        means2D = screenspace_points_bwd,
        shs = None,
        colors_precomp = color_bwd,
        opacities = opacity_bwd,
        scales = scaling_bwd,
        rotations = rot_bwd,
        cov3D_precomp = None)
    
    #img_fwd = rendered_image.permute(1, 2, 0).contiguous()
    #img_bwd = rendered_image_bwd.permute(1, 2, 0).contiguous()

    #rgb_fwd = img_fwd[..., :3]
    #rgb_bwd = img_bwd[..., :3]

    #alpha_fwd = opacitytorch.ones_like(rgb_fwd[..., :1])
    #alpha_bwd = opacity_bwdtorch.ones_like(rgb_bwd[..., :1])

    #rgb_blend = (rgb_fwd * alpha_fwd  + rgb_bwd * alpha_bwd) / (alpha_fwd + alpha_bwd + 1e-5)
    #alpha_blend = torch.clamp(alpha_fwd + alpha_bwd, max=1.0)

    #blended = torch.cat([rgb_blend, alpha_blend], dim=-1)
    #blended = blended.permute(2,0,1).contiguous

    #assert rendered_image.shape == rendered_image_bwd.shape
    #assert rendered_image.device == rendered_image_bwd.device
    
    rendered_image = rendered_image.contiguous()
    rendered_image_bwd = rendered_image_bwd.contiguous()

    blended_image = 0.5 * (rendered_image + rendered_image_bwd)
    blended_mask = mask | mask_bwd
    if is_training:
        blended_neural_opacity = 0.5 * (neural_opacity + neural_opacity_bwd)
        if bit_per_param is not None and bit_per_param_bwd is not None:
            bit_per_param = 0.5 * (bit_per_param + bit_per_param_bwd)
        if bit_per_feat_param is not None and bit_per_feat_param_bwd is not None:
            bit_per_feat_param = 0.5 * (bit_per_feat_param + bit_per_feat_param_bwd)
        if bit_per_scaling_param is not None and bit_per_scaling_param_bwd is not None:
            bit_per_scaling_param = 0.5 * (bit_per_scaling_param + bit_per_scaling_param_bwd)
        if bit_per_offsets_param is not None and bit_per_offsets_param_bwd is not None:
            bit_per_offsets_param = 0.5 * (bit_per_offsets_param + bit_per_offsets_param_bwd)
    else:
        blended_time_sub = 0.5 * (time_sub + time_sub_bwd)

    def blend_parameters(mask_fwd, mask_bwd, merged_mask, param_fwd, param_bwd, size):    
        is_vector = param_fwd.ndim == 1
        shape = (size,) if is_vector else (size, param_fwd.shape[1])
        full_param_fwd = torch.zeros(shape, device=param_fwd.device, dtype=param_fwd.dtype)
        full_param_bwd = torch.zeros_like(full_param_fwd)
        
        full_param_fwd[mask_fwd] = param_fwd
        full_param_bwd[mask_bwd] = param_bwd

        visibility_count = mask_fwd.int() + mask_bwd.int()
        if not is_vector:
            visibility_count = visibility_count.unsqueeze(-1)
        visibility_count = visibility_count.clamp(min=1)

        combined_param = (full_param_fwd + full_param_bwd) / visibility_count
        return combined_param[merged_mask]

    blended_screenspace_points = blend_parameters(mask, mask_bwd, blended_mask, screenspace_points_copy, screenspace_points_bwd_copy, len(mask))
    blended_radii = blend_parameters(mask, mask_bwd, blended_mask, radii, radii_bwd, len(mask))
    blended_scaling = blend_parameters(mask, mask_bwd, blended_mask, scaling, scaling_bwd, len(mask))

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    if is_training:
        return {"render": blended_image,
                "viewspace_points": blended_screenspace_points,
                "visibility_filter" : blended_radii > 0,
                "radii": blended_radii,
                "selection_mask": blended_mask,
                "neural_opacity": blended_neural_opacity,
                "scaling": blended_scaling,
                "bit_per_param": bit_per_param,
                "bit_per_feat_param": bit_per_feat_param,
                "bit_per_scaling_param": bit_per_scaling_param,
                "bit_per_offsets_param": bit_per_offsets_param,
                "fwd_viewspace_points": screenspace_points,
                "bwd_viewspace_points": screenspace_points_bwd,
                "fwd_neural_opacity": neural_opacity,
                "bwd_neural_opacity": neural_opacity_bwd,
                "fwd_selection_mask": mask,
                "bwd_selection_mask": mask_bwd,
                "fwd_visibility_filter": radii > 0,
                "bwd_visibility_filter": radii_bwd > 0,
                }
    else:
        return {"render": blended_image,
                "viewspace_points": blended_screenspace_points,
                "visibility_filter" : blended_radii > 0,
                "radii": blended_radii,
                "time_sub": blended_time_sub,
                }


def prefilter_voxel(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0,
                    override_color=None):
    """
    Render the scene. 

    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_anchor, dtype=pc.get_anchor.dtype, requires_grad=True,
                                          device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_anchor

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:  # False
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:  # into here
        scales = pc.get_scaling  # requires_grad = True
        rotations = pc.get_rotation  # requires_grad = True

    radii_pure = rasterizer.visible_filter(
        means3D=means3D,
        scales=scales[:, :3],
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,  # None
    )

    return radii_pure > 0
