import os
import train

for lmbda in [0.004]:  # Optionally, you can try: 0.003, 0.002, 0.001, 0.0005
    for cuda, scene in enumerate(['ShakeNDry']):
        one_cmd = f'CUDA_VISIBLE_DEVICES={0} python train.py -s data/videos/{scene} --eval --lod 0 --voxel_size 0.001 --update_init_factor 4 --iterations 30_000 -m outputs/videos/{scene}/{lmbda} --lmbda {lmbda}'
        #os.system(one_cmd)
        train.main([
"-s", "data/videos/ShakeNDry",
"--eval",
"--lod", "0",
"--voxel_size", "0.001",
"--update_init_factor", "4",
"--iterations", "30000",
"-m", "outputs/videos/ShakeNDry/0.003",
"--lmbda", "0.003"
])