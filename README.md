<p align="center">

  <h1 align="center">Learning a More Continuous Zero Level Set in Unsigned Distance Fields through Level Set Projection</h1>
  <p align="center">
    <a href="https://junshengzhou.github.io/"><strong>Junsheng Zhou*</strong></a>
    ·
    <a href="https://mabaorui.github.io/"><strong>Baorui Ma*</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=wynhSuQAAAAJ&hl=zh-CN&oi=sra"><strong>Shujuan Li</strong></a>
    ·
    <a href="https://yushen-liu.github.io/"><strong>Yu-Shen Liu</strong></a>
    ·
    <a href="https://h312h.github.io/"><strong>Zhizhong Han</strong></a>

  </p>
  <p align="center"><strong>(* Equal Contribution)</strong></p>
  <h2 align="center">ICCV 2023</h2>
  <div align="center"></div>
</p>

We release the code of the paper <a href="https://arxiv.org/abs/2308.11441">Learning a More Continuous Zero Level Set in Unsigned Distance Fields through Level Set Projection</a> in this repository.


## Reconstruction Results
### ShapeNetCars
<p align="center">
  <img src="figs/cars.png" width="780" />
</p>

### 3DScenes
<p align="center">
  <img src="figs/scenes.png" width="780" />
</p>

### KITTI
<p align="center">
  <img src="figs/kitti.png" width="760" />
</p>

## Point Upsampling Results
<p align="center">
  <img src="figs/pugan.png" width="780" />
</p>

## Point Normal Estimation Results
<p align="center">
  <img src="figs/pcpnet.png" width="780" />
</p>



## Installation
Our code is implemented in Python 3.8, PyTorch 1.11.0 and CUDA 11.3.
- Install python Dependencies
```bash
conda create -n levelsetudf python=3.8
conda activate levelsetudf
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install tqdm pyhocon==0.3.57 trimesh PyMCubes scipy point_cloud_utils==0.29.7
```
- Compile C++ extensions
```
cd extensions/chamfer_dist
python setup.py install
```

## Quick Start

For a quick start, you can train our LevelSetUDF to reconstruct surfaces from a single point cloud as:
```
python run.py --gpu 0 --conf confs/object.conf --dataname demo_car --dir demo_car
```
- We provide the data for a demo car in the `./data` folder for a quick start on LevelSetUDF.

You can find the outputs in the `./outs` folder:

```
│outs/
├──demo_car/
│  ├── mesh
│  ├── densepoints
│  ├── normal
```
- The reconstructed meshes are saved in the `mesh` folder
- The upsampled dense point clouds are saved in the `densepoints` folder
- The estimated normals for the point cloud are saved in the `normal` folder

## Use Your Own Data
We also provide the instructions for training your own data in the following.

### Data
First, you should put your own data to the `./data/input` folder. The datasets is organised as follows:
```
│data/
│── input
│   ├── (dataname).ply/xyz/npy
```
We support the point cloud data format of `.ply`, `.xyz` and `.npy`

### Run
To train your own data, simply run:
```
python run.py --gpu 0 --conf confs/object.conf --dataname (dataname) --dir (dataname)
```

### Notice
- For achieving better performances on point clouds of different complexity, the weights for the losses should be adjusted. For example, we provide two configs in the `./conf` folder, i.e., `object.conf` and `scene.conf`. If you are reconstructing large scale scenes, the `scene.conf` is recomended, otherwise, the `object.conf` should work fine for object-level reconstructions.

- In different datasets or your own data, because of the variation in point cloud density, this hyperparameter [scale](https://github.com/junshengzhou/LevelSetUDF/blob/44cd4e72b895f51bd2d06689392e25b31fed017a/models/dataset.py#L52) has a very strong influence on the final result, which controls the distance between the query points and the point cloud. So if you want to get better results, you should adjust this parameter. We give `0.25 * np.sqrt(POINT_NUM_GT / 20000)` here as a reference value, and this value can be used for most object-level reconstructions. 

## Related works
Please also check out the following works that inspire us a lot:
* [Junsheng Zhou et al. - Learning consistency-aware unsigned distance functions progressively from raw point clouds. (NeurIPS2022)](https://junshengzhou.github.io/CAP-UDF/)
* [Baorui Ma et al. - Neural-Pull: Learning Signed Distance Functions from Point Clouds by Learning to Pull Space onto Surfaces (ICML2021)](https://github.com/mabaorui/NeuralPull-Pytorch)
* [Baorui Ma et al. - Surface Reconstruction from Point Clouds by Learning Predictive Context Priors (CVPR2022)](https://mabaorui.github.io/PredictableContextPrior_page/)
* [Baorui Ma et al. - Reconstructing Surfaces for Sparse Point Clouds with On-Surface Priors (CVPR2022)](https://mabaorui.github.io/-OnSurfacePrior_project_page/)

## Citation
If you find our code or paper useful, please consider citing

    @inproceedings{zhou2023levelset,
    title={Learning a More Continuous Zero Level Set in Unsigned Distance Fields through Level Set Projection},
    author={Zhou, Junsheng and Ma, Baorui and Li, Shujuan and Liu, Yu-Shen and Han, Zhizhong},
    booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
    year={2023}
    }
