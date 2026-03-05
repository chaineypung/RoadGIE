<div align="center">
<img src="/assets/logo.png" alt="drawing" width="100%" height="5%"/>
    <h4 align="center"> </h4>

<p align="center">
  <h1 align="center">RoadGIE: Towards A Global-Scale Aerial Benchmark for Generalizable Interactive Road Extraction (CVPR 2026)
</h1>

<img src="/assets/dataset.png" alt="drawing" width="100%" height="30%"/>
Composition of the proposed dataset. We visualize the collection workflow and geographical distribution of the WorldRoadSeg-360K dataset. Furthermore, representative examples of road segmentation across typical terrain types are displayed on both sides of the figure.
</div>

## Abstract
> *Accurate road segmentation from aerial imagery is fundamental to many geospatial applications. However, existing road segmentation datasets frequently exhibit imbalanced scene diversity, insufficient semantic granularity, and inadequate structural continuity, restricting their generalization capabilities across varied environments. To address these challenges, we introduce WorldRoadSeg-360K, the largest and most diverse road segmentation dataset to date, comprising 366,947 high-resolution images collected from 38 countries and 223 cities across various terrains and continents. WorldRoadSeg-360K provides a comprehensive benchmark for evaluating road segmentation models and highlights the challenges that current methods face in handling diverse and structurally complex scenes. Automated approaches often struggle to preserve road connectivity, while current interactive methods lack efficient, topology-sensitive tools for real-world road editing. To this end, we present RoadGIE, establishing a novel interactive paradigm for road extraction in remote sensing. Unlike prior point- or box-based prompting strategies, RoadGIE supports connectivity-aware prompts, including clicks and scribbles, which inherently align with the topology of road networks. To improve structural consistency and mitigate performance degradation during iterative interactions, RoadGIE integrates an expert-guided prompting strategy and adapts the skeleton-based recall loss for interactive scenarios. RoadGIE achieves state-of-the-art performance in both segmentation accuracy and topological consistency on WorldRoadSeg-360K and other benchmarks, while maintaining efficient operation with only 3.7 million parameters and real-time processing capabilities.*

## Overview

**RoadGIE** supports connectivity-aware prompts, including clicks and scribbles, which inherently align with the topology of road networks.

<p align="center">
  <img src="/assets/1.gif" width="24%" />
  <img src="/assets/2.gif" width="24%" />
  <img src="/assets/3.gif" width="24%" />
  <img src="/assets/4.gif" width="24%" />
</p>

<img src="/assets/data.png" alt="drawing" width="100%" height="30%"/>
</div>

## Dependencies and Installation
```python
# 1. download your environment
https://pan.baidu.com/s/1ViI0hy21-bO42-cvStk89g code:8888 
# 2. make a directory and unzip your environment
cd RoadGIE
mkdir -p env
tar -xzf RoadGIE.tar.gz -C env
# 3. activate environment
source env/bin/activate
```

## Quick Inference
```python
# 1. run the demo
python demo/demo.py
# 2.open the gradio demo
http://192.168.4.12:7860/
```

## Training
```python
python RoadGIE/roadgie/experiment/unet.py -config train_unet.yaml 
```

<img src="/assets/table.png" alt="drawing" width="80%" height="30%"/>
</div>

## To Do

- [x] Release demo 
- [x] Release model code
- [x] Release model weights
- [x] Release training code
- [ ] Release WorldRoadSeg-360K dataset

## Acknowledgements

* Our training code builds on the [`ScribblePrompt`](https://github.com/halleewong/ScribblePrompt) library. Thanks to [@halleewong](https://github.com/halleewong) for sharing this code!

### Citation

If you use this in your research, please cite this project.

```bibtex
@article{peng2026,
	title={RoadGIE: Towards A Global-Scale Aerial Benchmark for Generalizable Interactive Road Extraction},
	author={Chenxu Peng, Chenxu Wang, Yimian Dai, Yongxiang Liu, Ming-Ming Cheng, Xiang Li},
	journal={CVPR},
	year={2026}
}
```
