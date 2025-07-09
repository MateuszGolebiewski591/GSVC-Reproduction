## Reproduction Goals
Our goal is to reproduce the model featured in https://openreview.net/forum?id=JbRM5QKRDd&utm_source=chatgpt.com

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

First, create a ```data/``` folder inside the project path by 
```
mkdir data
```

The data should be structured as follows:

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


### Public Data (We follow suggestions from [Scaffold-GS](https://github.com/city-super/Scaffold-GS))

 - The **BungeeNeRF** dataset is available in [Google Drive](https://drive.google.com/file/d/1nBLcf9Jrr6sdxKa1Hbd47IArQQ_X8lww/view?usp=sharing)/[百度网盘[提取码:4whv]](https://pan.baidu.com/s/1AUYUJojhhICSKO2JrmOnCA). 
 - The **MipNeRF360** scenes are provided by the paper author [here](https://jonbarron.info/mipnerf360/). And we test on its entire 9 scenes ```bicycle, bonsai, counter, garden, kitchen, room, stump, flowers, treehill```. 
 - The SfM datasets for **Tanks&Temples** and **Deep Blending** are hosted by 3D-Gaussian-Splatting [here](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip). Download and uncompress them into the ```data/``` folder.

### Custom Data

For custom data, you should ensure the image sequence is named in order and placed in an images directory as shown above. 

## Training

To train scenes, we provide the following training script: 
 - Modify for your dataset: ```run_shell_video.py```

 run with 
 ```
 python run_shell_xxx.py
 ```

The code will automatically run the entire process of: **training, encoding, decoding, testing**.
 - Training log will be recorded in `output.log` of the output directory. Results of **detailed fidelity, detailed size, detailed time** will all be recorded
 - Encoded bitstreams will be stored in `./bitstreams` of the output directory.
 - Evaluated output images will be saved in `./test/ours_30000/renders` of the output directory.
 - Optionally, you can change `lmbda` in these `run_shell_xxx.py` scripts to try variable bitrate.
 - **After training, the original model `point_cloud.ply` is losslessly compressed as `./bitstreams`. You should refer to `./bitstreams` to get the final model size, but not `point_cloud.ply`. You can even delete `point_cloud.ply` if you like :).**