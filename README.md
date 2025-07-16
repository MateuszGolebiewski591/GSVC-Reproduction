## Reproduction Goals
Our goal was to reproduce the model featured in https://openreview.net/forum?id=JbRM5QKRDd&utm_source=chatgpt.com 

We begin by adapting the HAC series code found here: https://github.com/YihangChen-ee/HAC

## HAC adaptation
Due to previous dependencies no longer working with one another, we began by updating the environment. 
 - Removed the pytorch channel from environment.yml
 - Changed python version in environment.yml from 3.7.13 to 3.10
 - removed torchaudio dependency from environment.yml
 - removed pip installs from environment.yml and moved them into a newly created requirements.txt
 - moved pytorch-scatter dependency into requirements.txt

The original HAC codebase can be found at https://github.com/YihangChen-ee/HAC

## Commit history:
Base commit after adaptation - 7e891b17141534d797afe5e7c5c0c0e15f1000a7 

Generation of random gaussians for previously colmap dependent data - 6a386d49a919d25050f32b3e35ed4a49262a2f59

Support for any video represented as a sequence of PNGs implemented - 164f3c46cfde89ba75fbbac58b8b65208e6d1438

Implemented double pass render - 8b048586534ef794c118ff8385136a71e3fd81eb

Added a mask that defines which gaussians lie within the toast-like sliding window - 01a00d1dff2d3c77bd21d677d1304866c3311e18

Fully working toast-like sliding window - c387f05a5351a2d5c44c3f987831377cab9decc8

Adding low level of pruning as well as trainable opacities to the base method - b6b600f3044ceaef96deb61d5840eda8b2f26299 

Adding low level of pruning as well as trainable opacities to TSW - 096c4f2df7c12612c6341a94797022fdf5cc8428

Harsher pruning and different functions for scaling and opacity for base method - a5538ccd4136a6d107f83a0da1b757027c4ff842

Harsher pruning and different functions for scaling and opacity for TSW - 5b226a6077b365a75ed4883aa9619fd86f2defd7

## Installation

We tested our code locally with Ubuntu 24.04.2, cuda 11.6, gcc 9.5.0 on a single rtx 3090 graphics card
1. Ensure compatible driver and compiler

Download compatible cuda driver
```
wget https://developer.download.nvidia.com/compute/cuda/11.6.0/local_installers/cuda_11.6.0_510.39.01_linux.run
sudo sh cuda_11.6.0_510.39.01_linux.run --override
export PATH=/usr/local/cuda-11.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64:$LD_LIBRARY_PATH
```
Accept the agreement and install. Once installed, the run file can be removed.
```
rm cuda_11.6.0_510.39.01_linux.run
```
Next, install compatible compilers
```
sudo apt install gcc-9 g++-9
export CC=/usr/bin/gcc-9
export CXX=/usr/bin/g++-9
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 90
sudo update-alternatives --config gcc
sudo update-alternatives --config g++
```
Verify successfull installations with 
```
nvcc --version
gcc --version
```

2. Unzip files
```
cd submodules
unzip diff-gaussian-rasterization.zip
unzip gridencoder.zip
unzip simple-knn.zip
unzip arithmetic.zip
cd ..
```

3. Install environment
```
conda env create --file environment.yml
conda activate HAC_env
```

4. Install dependencies
```
pip install -r requirements.txt
```

## Data

First, create a ```data``` folder inside the project path with 
```
mkdir data
```

All data should be structured as follows:

```
data/
├── dataset_name
│   ├── scene1/
│   │   ├── images
│   │   |   ├── IMG_0.jpg
│   │   |   ├── IMG_1.jpg
│   │   |   ├── ...
│   │   |    
│   ├── scene2/
│   │   ├── images
│   │   |   ├── IMG_0.jpg
│   │   |   ├── IMG_1.jpg
│   │   |   ├── ...
│   │   |
...
```

 - For instance: `./data/HEVC-CTC/BasketballDrive_1920x1080_50/`
 - For instance: `./data/JVET-CTC/B1BQTerrace/`
 - For instance: `./data/MCL-JCV/videoSRC01/`
 - For instance: `./data/UVG/Beauty/`

### Custom Data

For custom data, you should ensure the image sequence is named in order and placed in an images directory as shown above. No additional 

## Training

To train scenes, we provide the following training script: 
 ```run_shell_video.py```
 which can be ran with 
 ```
 python run_shell_xxx.py
 ```
Inside run_shell_python, there is a set of parameters that is passed into train.py's main method. Update these parameters with the location of your dataset and the parameters with which you wish the model to train on them.

The code will automatically run the entire process of: **training, encoding, decoding, testing**.
 - The training log will be recorded in `output.log` found in the output directory corresponding to your dataset. Results of **detailed fidelity, detailed size, detailed time** will all be recorded
 - Encoded bitstreams will be stored in `./bitstreams` of the corresponding output directory.
 - Evaluated output images will be saved in `./test/ours_30000/renders` of the corresponding output directory.
 - Optionally, you can change `lmbda` in within `run_shell_video.py` to test variable bitrates.
 - **After training, the original model `point_cloud.ply` is losslessly compressed as `./bitstreams`. You should refer to `./bitstreams` to get the final model size, not `point_cloud.ply`. You may be deleted `point_cloud.ply` as it's not necessary.**