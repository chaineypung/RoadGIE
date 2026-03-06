<div align="center">
<img src="/assets/logo.png" alt="drawing" width="100%" height="300"/>
    <h4 align="center"> </h4>

<p align="center">
  <h1 align="center">RoadGIE: Towards A Global-Scale Aerial Benchmark for Generalizable Interactive Road Extraction (CVPR 2026)
</h1>
  <p align="center">
      <a href='https://github.com/chaineypung' style='text-decoration: none' >Chenxu Peng</a><sup>1</sup>&emsp;
      <a href='https://github.com/facias914' style='text-decoration: none' >Chenxu Wang</a><sup>1</sup>&emsp;
      <a href='https://yimian.grokcv.ai/' style='text-decoration: none' >Yimian Dai</a><sup>1,2</sup>&emsp;
      <a href='https://scholar.google.com/citations?user=a9tTHSEAAAAJ&hl=zh-CN&oi=ao' style='text-decoration: none' >Yongxiang Liu</a><sup>3</sup>&emsp;
      <a href='https://mmcheng.net/cmm/' style='text-decoration: none' >Ming-Ming Cheng</a><sup>1,2</sup>&emsp;
      <a href='https://implus.github.io/' style='text-decoration: none' >Xiang Li</a><sup>1,2*</sup>&emsp;
        <p align="center">
        $^{1}$ VCIP, CS, Nankai University, $^{2}$ NKIARI, Shenzhen Futian, $^{3}$ National University of Defense Technology
        <p align='center'>
    <p align='center'>
    </p>
   </p>
</p>

<img src="/assets/dataset.png" alt="drawing" width="100%" height="30%"/>
</div>

## 📰Abstract
> *Accurate road segmentation from aerial imagery is fundamental to many geospatial applications. However, existing road segmentation datasets frequently exhibit imbalanced scene diversity, insufficient semantic granularity, and inadequate structural continuity, restricting their generalization capabilities across varied environments. To address these challenges, we introduce WorldRoadSeg-360K, the largest and most diverse road segmentation dataset to date, comprising 366,947 high-resolution images collected from 38 countries and 223 cities across various terrains and continents. WorldRoadSeg-360K provides a comprehensive benchmark for evaluating road segmentation models and highlights the challenges that current methods face in handling diverse and structurally complex scenes. Automated approaches often struggle to preserve road connectivity, while current interactive methods lack efficient, topology-sensitive tools for real-world road editing. To this end, we present RoadGIE, establishing a novel interactive paradigm for road extraction in remote sensing. Unlike prior point- or box-based prompting strategies, RoadGIE supports connectivity-aware prompts, including clicks and scribbles, which inherently align with the topology of road networks. To improve structural consistency and mitigate performance degradation during iterative interactions, RoadGIE integrates an expert-guided prompting strategy and adapts the skeleton-based recall loss for interactive scenarios. RoadGIE achieves state-of-the-art performance in both segmentation accuracy and topological consistency on WorldRoadSeg-360K and other benchmarks, while maintaining efficient operation with only 3.7 million parameters and real-time processing capabilities.*

## 📰 News
- **[2026.03.05]** Released SOTA segmentation model weights pretrained on WorldRoadSeg-360K
- **[2026.02.25]** Released model code and weights
- **[2026.02.21]** Released demo
- **[2026.02.21]** RoadGIE has been accepted to CVPR 2026!

## 🚩Overview

**RoadGIE** supports connectivity-aware prompts, including clicks and scribbles, which inherently align with the topology of road networks.

<p align="center">
  <img src="/assets/1.gif" width="24%" />
  <img src="/assets/2.gif" width="24%" />
  <img src="/assets/3.gif" width="24%" />
  <img src="/assets/4.gif" width="24%" />
</p>

<img src="/assets/data.png" alt="drawing" width="100%" height="30%"/>
</div>

## Installation
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

## Inference
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

## SOTA segmentation model weights pretrained on WorldRoadSeg-360K

| Encoder | Decoder | Dice | Model | Weight |
| :--- | :--- | :---: | :---: | :---: |
| SegNeXt_base | DaFormer33_ASPP | 0.740 | [SegNeXt_base](https://pan.baidu.com/s/184NVdsza8EmLm8D-rG3Uew?pwd=zr3s) | [SegNeXt_base](https://pan.baidu.com/s/1FgSC3mflgx0VT83B7o5S2w?pwd=62ua) |
| tf_efficientnetv2_l_in21ft1k | SiUnet | 0.726 | [tf_efficientnetv2_l](https://pan.baidu.com/s/184NVdsza8EmLm8D-rG3Uew?pwd=zr3s) | [tf_efficientnetv2_l](https://pan.baidu.com/s/1FgSC3mflgx0VT83B7o5S2w?pwd=62ua) |
| tf_efficientnet_b7_ns | SiUnet | 0.719 | [tf_efficientnet_b7_ns](https://pan.baidu.com/s/184NVdsza8EmLm8D-rG3Uew?pwd=zr3s) | [tf_efficientnet_b7_ns](https://pan.baidu.com/s/1FgSC3mflgx0VT83B7o5S2w?pwd=62ua) |
| convnext_large_384_in22ft1k | SiUnet | 0.740 | [convnext_large_384](https://pan.baidu.com/s/184NVdsza8EmLm8D-rG3Uew?pwd=zr3s) | [convnext_large_384](https://pan.baidu.com/s/1FgSC3mflgx0VT83B7o5S2w?pwd=62ua) |
| SwinT_small | UperNet | 0.737 | [SwinT_small](https://pan.baidu.com/s/184NVdsza8EmLm8D-rG3Uew?pwd=zr3s) | [SwinT_small](https://pan.baidu.com/s/1FgSC3mflgx0VT83B7o5S2w?pwd=62ua) |
| PvT_v2_b4 | DaFormer33_ASPP | 0.738 | [PvT_v2_b4](https://pan.baidu.com/s/184NVdsza8EmLm8D-rG3Uew?pwd=zr3s) | [PvT_v2_b4](https://pan.baidu.com/s/1FgSC3mflgx0VT83B7o5S2w?pwd=62ua) |
| Dual_ViT_b | DaFormer33_ASPP | 0.742 | [Dual_ViT_b](https://pan.baidu.com/s/184NVdsza8EmLm8D-rG3Uew?pwd=zr3s) | [Dual_ViT_b](https://pan.baidu.com/s/1FgSC3mflgx0VT83B7o5S2w?pwd=62ua) |
| SMT_base | DaFormer33_ASPP | 0.742 | [SMT_base](https://pan.baidu.com/s/184NVdsza8EmLm8D-rG3Uew?pwd=zr3s) | [SMT_base](https://pan.baidu.com/s/1FgSC3mflgx0VT83B7o5S2w?pwd=62ua) |
| Uniformer_base | DaFormer33_ASPP | 0.740 | [Uniformer_base](https://pan.baidu.com/s/184NVdsza8EmLm8D-rG3Uew?pwd=zr3s) | [Uniformer_base](https://pan.baidu.com/s/1FgSC3mflgx0VT83B7o5S2w?pwd=62ua) |
| WaveViT | DaFormer33_ASPP | 0.738 | [WaveViT](https://pan.baidu.com/s/184NVdsza8EmLm8D-rG3Uew?pwd=zr3s) | [WaveViT](https://pan.baidu.com/s/1FgSC3mflgx0VT83B7o5S2w?pwd=62ua) |
| MiT_b2 | SegFormer | 0.735 | [MiT_b2](https://pan.baidu.com/s/184NVdsza8EmLm8D-rG3Uew?pwd=zr3s) | [MiT_b2](https://pan.baidu.com/s/1FgSC3mflgx0VT83B7o5S2w?pwd=62ua) |
| iFormer_large | DaFormer33_ASPP | 0.739 | [iFormer_large](https://pan.baidu.com/s/184NVdsza8EmLm8D-rG3Uew?pwd=zr3s) | [iFormer_large](https://pan.baidu.com/s/1FgSC3mflgx0VT83B7o5S2w?pwd=62ua) |
| HorNet_small_gf | DaFormer33_ASPP | 0.742 | [HorNet_small_gf](https://pan.baidu.com/s/184NVdsza8EmLm8D-rG3Uew?pwd=zr3s) | [HorNet_small_gf](https://pan.baidu.com/s/1FgSC3mflgx0VT83B7o5S2w?pwd=62ua) |
| CoaT_4level_small | DaFormer33_ASPP | 0.736 | [CoaT_4level_small](https://pan.baidu.com/s/184NVdsza8EmLm8D-rG3Uew?pwd=zr3s) | [CoaT_4level_small](https://pan.baidu.com/s/1FgSC3mflgx0VT83B7o5S2w?pwd=62ua) |
| CoaT_4level_lite_medium | DaFormer33_ASPP | 0.739 | [CoaT_4level_lite](https://pan.baidu.com/s/184NVdsza8EmLm8D-rG3Uew?pwd=zr3s) | [CoaT_4level_lite](https://pan.baidu.com/s/1FgSC3mflgx0VT83B7o5S2w?pwd=62ua) |
| CoaT_5level_parallel_small | DaFormer33_ASPP | 0.741 | [CoaT_5level_parallel](https://pan.baidu.com/s/184NVdsza8EmLm8D-rG3Uew?pwd=zr3s) | [CoaT_5level_parallel](https://pan.baidu.com/s/1FgSC3mflgx0VT83B7o5S2w?pwd=62ua) |

## Results

Comparison of different models using different datasets.

| Method | Baseline dataset (Dice↑) | Baseline dataset (APLS↑) | WorldRoadSeg-360K (Dice↑) | WorldRoadSeg-360K (APLS↑) |
| :--- | :---: | :---: | :---: | :---: |
| EISeg | 0.701 | 0.511 | 0.706 | 0.515 |
| ScribbleSeg-B0 | 0.766 | 0.560 | 0.785 | 0.578 |
| ScribbleSeg-B3 | 0.761 | 0.556 | 0.788 | 0.580 |
| SAM (ViT-b) | 0.719 | 0.522 | 0.737 | 0.539 |
| SAM (ViT-h) | 0.738 | 0.539 | 0.756 | 0.553 |
| PRISM-2D | 0.622 | 0.463 | 0.643 | 0.481 |
| PRISM-2D-Lite | 0.656 | 0.489 | 0.669 | 0.496 |
| ScribblePrompt | 0.791 | 0.584 | 0.809 | 0.592 |
| **RoadGIE** | **0.807** | **0.593** | **0.835** | **0.620** |

<br>

Performance of models pretrained on different datasets and evaluated on the same test set. Best results are highlighted in bold.

| Pretrained dataset | Dice↑ | Recall↑ | clDice↑ | APLS↑ | β₀↓ | β₁↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline dataset | 0.807 | 0.897 | 0.869 | 0.593 | 8.150 | 3.061 |
| WorldRoadSeg-360K | **0.835** | **0.934** | **0.905** | **0.620** | **5.823** | **2.752** |

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
