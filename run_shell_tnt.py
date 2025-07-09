import os
import train
for lmbda in [0.004]:  # Optionally, you can try: 0.003, 0.002, 0.001, 0.0005
    for cuda, scene in enumerate(['truck']):
        one_cmd = f'CUDA_VISIBLE_DEVICES={0} python train.py -s data/tandt/{scene} --eval --lod 0 --voxel_size 0.01 --update_init_factor 16 --iterations 30_000 -m outputs/tandt/{scene}/{lmbda} --lmbda {lmbda}'
        #os.system(one_cmd)
        train.main([
"-s", "data/tandt/truck",
"--eval",
"--lod", "0",
"--voxel_size", "0.001",
"--update_init_factor", "4",
"--iterations", "30000",
"-m", "outputs/tandt/truck/0.004",
"--lmbda", "0.004"
])