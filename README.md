<div align="center">
  <img src="utils/logo_white_bg.jpg" alt="Touchstone Benchmark" width="300">
</div>


<h1 align="center" style="font-size: 60px; margin-bottom: 4px">Touchstone Benchmark</h1>
<p align="center">
    <a href='https://www.cs.jhu.edu/~zongwei/advert/TutorialBenchmarkV1.pdf'><img src='https://img.shields.io/badge/Participate-Touchstone 1.0-blue'></a>
    <a href='https://www.cs.jhu.edu/~zongwei/advert/Call4Benchmark.pdf'><img src='https://img.shields.io/badge/Participate-Touchstone 2.0-red'></a>
    <br/>
    <a href="https://github.com/MrGiovanni/Touchstone"><img src="https://img.shields.io/github/stars/MrGiovanni/Touchstone?style=social" /></a>
    <a href="https://twitter.com/bodymaps317"><img src="https://img.shields.io/twitter/follow/BodyMaps" alt="Follow on Twitter" /></a>
</p>

We present Touchstone, a large-scale  medical segmentation benchmark based on annotated **5,195** CT volumes from **76** hospitals for training, and **6,933** CT volumes from **8** additional hospitals for testing. We invite AI inventors to train their models on AbdomenAtlas, and we independently evaluate their algorithms. We have already collaborated with **14** influential research teams, and we remain accepting new submissions.

> [!NOTE]
> Training set
> - Touchstone 1.0: [AbdomenAtlas1.0Mini](https://huggingface.co/datasets/AbdomenAtlas/AbdomenAtlas1.0Mini) (*N*=5,195)
> - Touchstone 2.0: [AbdomenAtlas1.1Mini](https://huggingface.co/datasets/AbdomenAtlas/_AbdomenAtlas1.1Mini) (*N*=9,262)
> 
> Test set 
> - Proprietary [JHH dataset](https://www.sciencedirect.com/science/article/pii/S2211568419301391) (*N*=5,172)
> - Public [TotalSegmentator V2](https://github.com/wasserth/TotalSegmentator) dataset (*N*=1,228)
> - Public [DAP Atlas](https://github.com/alexanderjaus/AtlasDataset) dataset (*N*=533)

# Touchstone 1.0 Leaderboard 

<b>Touchstone Benchmark: Are We on the Right Way for Evaluating AI Algorithms for Medical Segmentation?</b> <br/>
[Pedro R. A. S. Bassi](https://scholar.google.com.hk/citations?user=NftgL6gAAAAJ)<sup>1</sup>, [Wenxuan Li](https://scholar.google.com/citations?hl=en&user=tpNZM2YAAAAJ)<sup>1</sup>, [Yucheng Tang](https://scholar.google.com.hk/citations?hl=en&user=0xheliUAAAAJ)<sup>2</sup>, [Fabian Isensee](https://scholar.google.com.hk/citations?hl=en&user=PjerEe4AAAAJ)<sup>3</sup>, ..., [Alan Yuille](https://www.cs.jhu.edu/~ayuille/)<sup>1</sup>, [Zongwei Zhou](https://www.zongweiz.com/)<sup>1</sup> <br/>
<sup>1</sup>Johns Hopkins University, <sup>2</sup>NVIDIA, <sup>3</sup>DKFZ <br/>
[project](https://www.zongweiz.com/dataset) | [paper](https://www.cs.jhu.edu/~alanlab/Pubs24/bassi2024touchstone.pdf) | [code](https://github.com/MrGiovanni/Touchstone) <br/>

| rank | model  | organization | average DSC | paper | github
|:---|:---|:---|:---:|:---:|:---:|
| 🏆 | MedNeXt | DKFZ | 89.2 | [![arXiv](https://img.shields.io/badge/arXiv-2303.09975-b31b1b.svg)](https://arxiv.org/pdf/2303.09975) | [![GitHub stars](https://img.shields.io/github/stars/MIC-DKFZ/MedNeXt.svg?logo=github&label=Stars)](https://github.com/MIC-DKFZ/MedNeXt) |
| 🏆 | STU-Net-B | Shanghai AI Lab | 89.0 | [![arXiv](https://img.shields.io/badge/arXiv-2304.06716-b31b1b.svg)](https://arxiv.org/pdf/2304.06716) | [![GitHub stars](https://img.shields.io/github/stars/uni-medical/STU-Net.svg?logo=github&label=Stars)](https://github.com/uni-medical/STU-Net) |
| 🏆 | nnU-Net ResEncL | DKFZ | 88.8 | [![arXiv](https://img.shields.io/badge/arXiv-1809.10486-b31b1b.svg)](https://arxiv.org/abs/1809.10486) | [![GitHub stars](https://img.shields.io/github/stars/MIC-DKFZ/nnUNet.svg?logo=github&label=Stars)](https://github.com/MIC-DKFZ/nnUNet) |
| 🏆 | UniSeg | NPU | 88.8 | [![arXiv](https://img.shields.io/badge/arXiv-2304.03493-b31b1b.svg)](https://arxiv.org/abs/2304.03493) | [![GitHub stars](https://img.shields.io/github/stars/yeerwen/UniSeg.svg?logo=github&label=Stars)](https://github.com/yeerwen/UniSeg) |
| 🏆 | Diff-UNet | HKUST | 88.5 | [![arXiv](https://img.shields.io/badge/arXiv-2303.10326-b31b1b.svg)](https://arxiv.org/abs/2303.10326) | [![GitHub stars](https://img.shields.io/github/stars/ge-xing/Diff-UNet.svg?logo=github&label=Stars)](https://github.com/ge-xing/Diff-UNet) |
| 🏆 | NexToU | HIT | 87.8 | [![arXiv](https://img.shields.io/badge/arXiv-2305.15911-b31b1b.svg)](https://arxiv.org/abs/2305.15911) | [![GitHub stars](https://img.shields.io/github/stars/PengchengShi1220/NexToU.svg?logo=github&label=Stars)](https://github.com/PengchengShi1220/NexToU) |
| 7 | SegVol | BAAI | 87.1 | [![arXiv](https://img.shields.io/badge/arXiv-2311.13385-b31b1b.svg)](https://arxiv.org/abs/2311.13385) | [![GitHub stars](https://img.shields.io/github/stars/BAAI-DCAI/SegVol.svg?logo=github&label=Stars)](https://github.com/BAAI-DCAI/SegVol) |
| 8 | U-Net & CLIP | CityU | 87.1 | [![arXiv](https://img.shields.io/badge/arXiv-2301.00785-b31b1b.svg)](https://arxiv.org/abs/2301.00785) | [![GitHub stars](https://img.shields.io/github/stars/ljwztc/CLIP-Driven-Universal-Model.svg?logo=github&label=Stars)](https://github.com/ljwztc/CLIP-Driven-Universal-Model) |
| 9 | Swin UNETR & CLIP | CityU | 86.7 | [![arXiv](https://img.shields.io/badge/arXiv-2301.00785-b31b1b.svg)](https://arxiv.org/abs/2301.00785) | [![GitHub stars](https://img.shields.io/github/stars/ljwztc/CLIP-Driven-Universal-Model.svg?logo=github&label=Stars)](https://github.com/ljwztc/CLIP-Driven-Universal-Model) |
| 10 | Swin UNETR | NVIDIA | 80.1 | [![arXiv](https://img.shields.io/badge/arXiv-2211.11537-b31b1b.svg)](https://arxiv.org/abs/2211.11537) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)](https://github.com/Project-MONAI/tutorials) |
| 11 | UNesT | NVIDIA | 79.1 | [![arXiv](https://img.shields.io/badge/arXiv-2303.10745-b31b1b.svg)](https://arxiv.org/abs/2303.10745) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)](https://github.com/Project-MONAI/tutorials) |
| 12 | SAM-Adapter | Duke | 73.4 | [![arXiv](https://img.shields.io/badge/arXiv-2404.09957-b31b1b.svg)](https://arxiv.org/abs/2404.09957) | [![GitHub stars](https://img.shields.io/github/stars/mazurowski-lab/finetune-SAM.svg?logo=github&label=Stars)](https://github.com/mazurowski-lab/finetune-SAM) |
| 13 | UNETR | NVIDIA| 64.4 | [![arXiv](https://img.shields.io/badge/arXiv-2111.04004-b31b1b.svg)](https://arxiv.org/abs/2111.04004) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)](https://github.com/Project-MONAI/tutorials) |


<details>
<summary style="margin-left: 25px;">Aorta - NexToU 🏆 </summary>
<div style="margin-left: 25px;">

| rank | model  | organization | DSC | paper | github |
|:---|:---|:---|:---:|:---:|:---:|
| 🏆 | NexToU | HIT | 86.4 | [![arXiv](https://img.shields.io/badge/arXiv-2305.15911-b31b1b.svg)](https://arxiv.org/pdf/2305.15911) | [![GitHub stars](https://img.shields.io/github/stars/PengchengShi1220/NexToU.svg?logo=github&label=Stars)](https://github.com/PengchengShi1220/NexToU) |
| 2 | MedNeXt | DKFZ | 83.1 | [![arXiv](https://img.shields.io/badge/arXiv-2303.09975-b31b1b.svg)](https://arxiv.org/pdf/2303.09975) | [![GitHub stars](https://img.shields.io/github/stars/MIC-DKFZ/MedNeXt.svg?logo=github&label=Stars)](https://github.com/MIC-DKFZ/MedNeXt) |
| 3 | UniSeg | NPU | 82.3 | [![arXiv](https://img.shields.io/badge/arXiv-2304.03493-b31b1b.svg)](https://arxiv.org/abs/2304.03493) | [![GitHub stars](https://img.shields.io/github/stars/yeerwen/UniSeg.svg?logo=github&label=Stars)](https://github.com/yeerwen/UniSeg) |
| 4 | STU-Net-B | Shanghai AI Lab | 82.1 | [![arXiv](https://img.shields.io/badge/arXiv-2304.06716-b31b1b.svg)](https://arxiv.org/pdf/2304.06716) | [![GitHub stars](https://img.shields.io/github/stars/uni-medical/STU-Net.svg?logo=github&label=Stars)](https://github.com/uni-medical/STU-Net) |
| 5 | nnU-Net ResEncL | DKFZ | 81.4 | [![arXiv](https://img.shields.io/badge/arXiv-1809.10486-b31b1b.svg)](https://arxiv.org/abs/1809.10486) | [![GitHub stars](https://img.shields.io/github/stars/MIC-DKFZ/nnUNet.svg?logo=github&label=Stars)](https://github.com/MIC-DKFZ/nnUNet) |
| 6 | Diff-UNet | HKUST | 81.2 | [![arXiv](https://img.shields.io/badge/arXiv-2303.10326-b31b1b.svg)](https://arxiv.org/abs/2303.10326) | [![GitHub stars](https://img.shields.io/github/stars/ge-xing/Diff-UNet.svg?logo=github&label=Stars)](https://github.com/ge-xing/Diff-UNet) |
| 7| Swin UNETR | NVIDIA | 81.1 | [![arXiv](https://img.shields.io/badge/arXiv-2211.11537-b31b1b.svg)](https://arxiv.org/abs/2211.11537) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)](https://github.com/Project-MONAI/tutorials) |
| 8 | SegVol | BAAI | 80.2 | [![arXiv](https://img.shields.io/badge/arXiv-2311.13385-b31b1b.svg)](https://arxiv.org/abs/2311.13385) | [![GitHub stars](https://img.shields.io/github/stars/BAAI-DCAI/SegVol.svg?logo=github&label=Stars)](https://github.com/BAAI-DCAI/SegVol) |
| 9 | UNesT | NVIDIA | 78.6 | [![arXiv](https://img.shields.io/badge/arXiv-2303.10745-b31b1b.svg)](https://arxiv.org/abs/2303.10745) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)](https://github.com/Project-MONAI/tutorials) |
| 10 | Swin UNETR & CLIP | CityU | 78.1 | [![arXiv](https://img.shields.io/badge/arXiv-2301.00785-b31b1b.svg)](https://arxiv.org/abs/2301.00785) | [![GitHub stars](https://img.shields.io/github/stars/ljwztc/CLIP-Driven-Universal-Model.svg?logo=github&label=Stars)](https://github.com/ljwztc/CLIP-Driven-Universal-Model) |
| 11 | U-Net & CLIP | CityU | 77.1 | [![arXiv](https://img.shields.io/badge/arXiv-2301.00785-b31b1b.svg)](https://arxiv.org/abs/2301.00785) | [![GitHub stars](https://img.shields.io/github/stars/ljwztc/CLIP-Driven-Universal-Model.svg?logo=github&label=Stars)](https://github.com/ljwztc/CLIP-Driven-Universal-Model) |
| 12 | SAM-Adapter | Duke | 62.8 | [![arXiv](https://img.shields.io/badge/arXiv-2404.09957-b31b1b.svg)](https://arxiv.org/abs/2404.09957) | [![GitHub stars](https://img.shields.io/github/stars/mazurowski-lab/finetune-SAM.svg?logo=github&label=Stars)](https://github.com/mazurowski-lab/finetune-SAM) |
| 13 | UNETR | NVIDIA | 52.1 | [![arXiv](https://img.shields.io/badge/arXiv-2111.04004-b31b1b.svg)](https://arxiv.org/abs/2111.04004) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)](https://github.com/Project-MONAI/tutorials) |


</div>
</details>

<details>
<summary style="margin-left: 25px;">Gallbladder - STU-Net-B & MedNeXt 🏆 </summary>
<div style="margin-left: 25px;">

| rank | model  | organization | DSC | paper | github |
|:---|:---|:---|:---:|:---:|:---:|
| 🏆 | STU-Net-B | Shanghai AI Lab | 85.5 | [![arXiv](https://img.shields.io/badge/arXiv-2304.06716-b31b1b.svg)](https://arxiv.org/pdf/2304.06716) | [![GitHub stars](https://img.shields.io/github/stars/uni-medical/STU-Net.svg?logo=github&label=Stars)](https://github.com/uni-medical/STU-Net) |
| 🏆 | MedNeXt | DKFZ | 85.3 | [![arXiv](https://img.shields.io/badge/arXiv-2303.09975-b31b1b.svg)](https://arxiv.org/pdf/2303.09975) | [![GitHub stars](https://img.shields.io/github/stars/MIC-DKFZ/MedNeXt.svg?logo=github&label=Stars)](https://github.com/MIC-DKFZ/MedNeXt) |
| 3 | nnU-Net ResEncL | DKFZ | 84.9 | [![arXiv](https://img.shields.io/badge/arXiv-1809.10486-b31b1b.svg)](https://arxiv.org/abs/1809.10486) | [![GitHub stars](https://img.shields.io/github/stars/MIC-DKFZ/nnUNet.svg?logo=github&label=Stars)](https://github.com/MIC-DKFZ/nnUNet) |
| 4 | UniSeg | NPU | 84.7 | [![arXiv](https://img.shields.io/badge/arXiv-2304.03493-b31b1b.svg)](https://arxiv.org/abs/2304.03493) | [![GitHub stars](https://img.shields.io/github/stars/yeerwen/UniSeg.svg?logo=github&label=Stars)](https://github.com/yeerwen/UniSeg) |
| 5 | Diff-UNet | HKUST | 83.8 | [![arXiv](https://img.shields.io/badge/arXiv-2303.10326-b31b1b.svg)](https://arxiv.org/abs/2303.10326) | [![GitHub stars](https://img.shields.io/github/stars/ge-xing/Diff-UNet.svg?logo=github&label=Stars)](https://github.com/ge-xing/Diff-UNet) |
| 6 | NexToU | HIT | 82.3 | [![arXiv](https://img.shields.io/badge/arXiv-2305.15911-b31b1b.svg)](https://arxiv.org/abs/2305.15911) | [![GitHub stars](https://img.shields.io/github/stars/PengchengShi1220/NexToU.svg?logo=github&label=Stars)](https://github.com/PengchengShi1220/NexToU) |
| 7 | U-Net & CLIP | CityU | 82.1 | [![arXiv](https://img.shields.io/badge/arXiv-2301.00785-b31b1b.svg)](https://arxiv.org/abs/2301.00785) | [![GitHub stars](https://img.shields.io/github/stars/ljwztc/CLIP-Driven-Universal-Model.svg?logo=github&label=Stars)](https://github.com/ljwztc/CLIP-Driven-Universal-Model) |
| 8 | Swin UNETR & CLIP | CityU | 80.2 | [![arXiv](https://img.shields.io/badge/arXiv-2301.00785-b31b1b.svg)](https://arxiv.org/abs/2301.00785) | [![GitHub stars](https://img.shields.io/github/stars/ljwztc/CLIP-Driven-Universal-Model.svg?logo=github&label=Stars)](https://github.com/ljwztc/CLIP-Driven-Universal-Model) |
| 9 | SegVol | BAAI | 79.3 | [![arXiv](https://img.shields.io/badge/arXiv-2311.13385-b31b1b.svg)](https://arxiv.org/abs/2311.13385) | [![GitHub stars](https://img.shields.io/github/stars/BAAI-DCAI/SegVol.svg?logo=github&label=Stars)](https://github.com/BAAI-DCAI/SegVol) |
| 10 | Swin UNETR | NVIDIA | 69.2 | [![arXiv](https://img.shields.io/badge/arXiv-2211.11537-b31b1b.svg)](https://arxiv.org/abs/2211.11537) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)](https://github.com/Project-MONAI/tutorials) |
| 11 | UNesT | NVIDIA | 62.1 | [![arXiv](https://img.shields.io/badge/arXiv-2303.10745-b31b1b.svg)](https://arxiv.org/abs/2303.10745) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)](https://github.com/Project-MONAI/tutorials) |
| 12 | SAM-Adapter | Duke | 49.4 | [![arXiv](https://img.shields.io/badge/arXiv-2404.09957-b31b1b.svg)](https://arxiv.org/abs/2404.09957) | [![GitHub stars](https://img.shields.io/github/stars/mazurowski-lab/finetune-SAM.svg?logo=github&label=Stars)](https://github.com/mazurowski-lab/finetune-SAM) |
| 13 | UNETR | NVIDIA | 43.8 | [![arXiv](https://img.shields.io/badge/arXiv-2111.04004-b31b1b.svg)](https://arxiv.org/abs/2111.04004) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)](https://github.com/Project-MONAI/tutorials) |


</div>
</details>

<details>
<summary style="margin-left: 25px;">KidneyL - Diff-UNet 🏆 </summary>
<div style="margin-left: 25px;">

| rank | model  | organization | DSC | paper | github |
|:---|:---|:---|:---:|:---:|:---:|
| 🏆 | Diff-UNet | HKUST | 91.9 | [![arXiv](https://img.shields.io/badge/arXiv-2303.10326-b31b1b.svg)](https://arxiv.org/abs/2303.10326) | [![GitHub stars](https://img.shields.io/github/stars/ge-xing/Diff-UNet.svg?logo=github&label=Stars)](https://github.com/ge-xing/Diff-UNet) |
| 2 | nnU-Net ResEncL | DKFZ | 91.9 | [![arXiv](https://img.shields.io/badge/arXiv-1809.10486-b31b1b.svg)](https://arxiv.org/abs/1809.10486) | [![GitHub stars](https://img.shields.io/github/stars/MIC-DKFZ/nnUNet.svg?logo=github&label=Stars)](https://github.com/MIC-DKFZ/nnUNet) |
| 3 | STU-Net-B | Shanghai AI Lab | 91.9 | [![arXiv](https://img.shields.io/badge/arXiv-2304.06716-b31b1b.svg)](https://arxiv.org/pdf/2304.06716) | [![GitHub stars](https://img.shields.io/github/stars/uni-medical/STU-Net.svg?logo=github&label=Stars)](https://github.com/uni-medical/STU-Net) |
| 4 | MedNeXt | DKFZ | 91.8 | [![arXiv](https://img.shields.io/badge/arXiv-2303.09975-b31b1b.svg)](https://arxiv.org/pdf/2303.09975) | [![GitHub stars](https://img.shields.io/github/stars/MIC-DKFZ/MedNeXt.svg?logo=github&label=Stars)](https://github.com/MIC-DKFZ/MedNeXt) |
| 5 | SegVol | BAAI | 91.8 | [![arXiv](https://img.shields.io/badge/arXiv-2311.13385-b31b1b.svg)](https://arxiv.org/abs/2311.13385) | [![GitHub stars](https://img.shields.io/github/stars/BAAI-DCAI/SegVol.svg?logo=github&label=Stars)](https://github.com/BAAI-DCAI/SegVol) |
| 6 | UniSeg | NPU | 91.5 | [![arXiv](https://img.shields.io/badge/arXiv-2304.03493-b31b1b.svg)](https://arxiv.org/abs/2304.03493) | [![GitHub stars](https://img.shields.io/github/stars/yeerwen/UniSeg.svg?logo=github&label=Stars)](https://github.com/yeerwen/UniSeg) |
| 7 | U-Net & CLIP | CityU | 91.1 | [![arXiv](https://img.shields.io/badge/arXiv-2301.00785-b31b1b.svg)](https://arxiv.org/abs/2301.00785) | [![GitHub stars](https://img.shields.io/github/stars/ljwztc/CLIP-Driven-Universal-Model.svg?logo=github&label=Stars)](https://github.com/ljwztc/CLIP-Driven-Universal-Model) |
| 8 | Swin UNETR & CLIP | CityU | 91.0 | [![arXiv](https://img.shields.io/badge/arXiv-2301.00785-b31b1b.svg)](https://arxiv.org/abs/2301.00785) | [![GitHub stars](https://img.shields.io/github/stars/ljwztc/CLIP-Driven-Universal-Model.svg?logo=github&label=Stars)](https://github.com/ljwztc/CLIP-Driven-Universal-Model) |
| 9 | NexToU | HIT | 89.6 | [![arXiv](https://img.shields.io/badge/arXiv-2305.15911-b31b1b.svg)](https://arxiv.org/abs/2305.15911) | [![GitHub stars](https://img.shields.io/github/stars/PengchengShi1220/NexToU.svg?logo=github&label=Stars)](https://github.com/PengchengShi1220/NexToU) |
| 10 | SAM-Adapter | Duke | 87.3 | [![arXiv](https://img.shields.io/badge/arXiv-2404.09957-b31b1b.svg)](https://arxiv.org/abs/2404.09957) | [![GitHub stars](https://img.shields.io/github/stars/mazurowski-lab/finetune-SAM.svg?logo=github&label=Stars)](https://github.com/mazurowski-lab/finetune-SAM) |
| 11 | Swin UNETR | NVIDIA | 85.5 | [![arXiv](https://img.shields.io/badge/arXiv-2211.11537-b31b1b.svg)](https://arxiv.org/abs/2211.11537) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)](https://github.com/Project-MONAI/tutorials) |
| 12 | UNesT | NVIDIA | 85.4 | [![arXiv](https://img.shields.io/badge/arXiv-2303.10745-b31b1b.svg)](https://arxiv.org/abs/2303.10745) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)](https://github.com/Project-MONAI/tutorials) |
| 13 | UNETR | NVIDIA | 63.7 | [![arXiv](https://img.shields.io/badge/arXiv-2111.04004-b31b1b.svg)](https://arxiv.org/abs/2111.04004) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)](https://github.com/Project-MONAI/tutorials) |

                                                                                                                                  
</div>
</details>

<details>
<summary style="margin-left: 25px;">KidneyR - Diff-UNet 🏆 </summary>
<div style="margin-left: 25px;">

| rank | model  | organization | DSC | paper | github |
|:---|:---|:---|:---:|:---:|:---:|
| 🏆 | Diff-UNet | HKUST | 92.8 | [![arXiv](https://img.shields.io/badge/arXiv-2303.10326-b31b1b.svg)](https://arxiv.org/abs/2303.10326) | [![GitHub stars](https://img.shields.io/github/stars/ge-xing/Diff-UNet.svg?logo=github&label=Stars)](https://github.com/ge-xing/Diff-UNet) |
| 2 | MedNeXt | DKFZ | 92.6 | [![arXiv](https://img.shields.io/badge/arXiv-2303.09975-b31b1b.svg)](https://arxiv.org/pdf/2303.09975) | [![GitHub stars](https://img.shields.io/github/stars/MIC-DKFZ/MedNeXt.svg?logo=github&label=Stars)](https://github.com/MIC-DKFZ/MedNeXt) |
| 3 | nnU-Net ResEncL | DKFZ | 92.6 | [![arXiv](https://img.shields.io/badge/arXiv-1809.10486-b31b1b.svg)](https://arxiv.org/abs/1809.10486) | [![GitHub stars](https://img.shields.io/github/stars/MIC-DKFZ/nnUNet.svg?logo=github&label=Stars)](https://github.com/MIC-DKFZ/nnUNet) |
| 4 | STU-Net-B | Shanghai AI Lab | 92.5 | [![arXiv](https://img.shields.io/badge/arXiv-2304.06716-b31b1b.svg)](https://arxiv.org/pdf/2304.06716) | [![GitHub stars](https://img.shields.io/github/stars/uni-medical/STU-Net.svg?logo=github&label=Stars)](https://github.com/uni-medical/STU-Net) |
| 5 | SegVol | BAAI | 92.5 | [![arXiv](https://img.shields.io/badge/arXiv-2311.13385-b31b1b.svg)](https://arxiv.org/abs/2311.13385) | [![GitHub stars](https://img.shields.io/github/stars/BAAI-DCAI/SegVol.svg?logo=github&label=Stars)](https://github.com/BAAI-DCAI/SegVol) |
| 6 | UniSeg | NPU | 92.2 | [![arXiv](https://img.shields.io/badge/arXiv-2304.03493-b31b1b.svg)](https://arxiv.org/abs/2304.03493) | [![GitHub stars](https://img.shields.io/github/stars/yeerwen/UniSeg.svg?logo=github&label=Stars)](https://github.com/yeerwen/UniSeg) |
| 7 | U-Net & CLIP | CityU | 91.9 | [![arXiv](https://img.shields.io/badge/arXiv-2301.00785-b31b1b.svg)](https://arxiv.org/abs/2301.00785) | [![GitHub stars](https://img.shields.io/github/stars/ljwztc/CLIP-Driven-Universal-Model.svg?logo=github&label=Stars)](https://github.com/ljwztc/CLIP-Driven-Universal-Model) |
| 8 | Swin UNETR & CLIP | CityU | 91.7 | [![arXiv](https://img.shields.io/badge/arXiv-2301.00785-b31b1b.svg)](https://arxiv.org/abs/2301.00785) | [![GitHub stars](https://img.shields.io/github/stars/ljwztc/CLIP-Driven-Universal-Model.svg?logo=github&label=Stars)](https://github.com/ljwztc/CLIP-Driven-Universal-Model) |
| 9 | SAM-Adapter | Duke | 90.4 | [![arXiv](https://img.shields.io/badge/arXiv-2404.09957-b31b1b.svg)](https://arxiv.org/abs/2404.09957) | [![GitHub stars](https://img.shields.io/github/stars/mazurowski-lab/finetune-SAM.svg?logo=github&label=Stars)](https://github.com/mazurowski-lab/finetune-SAM) |
| 10 | NexToU | HIT | 90.1 | [![arXiv](https://img.shields.io/badge/arXiv-2305.15911-b31b1b.svg)](https://arxiv.org/abs/2305.15911) | [![GitHub stars](https://img.shields.io/github/stars/PengchengShi1220/NexToU.svg?logo=github&label=Stars)](https://github.com/PengchengShi1220/NexToU) |
| 11 | UNesT | NVIDIA | 83.6 | [![arXiv](https://img.shields.io/badge/arXiv-2303.10745-b31b1b.svg)](https://arxiv.org/abs/2303.10745) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)](https://github.com/Project-MONAI/tutorials) |
| 12 | Swin UNETR | NVIDIA | 81.7 | [![arXiv](https://img.shields.io/badge/arXiv-2211.11537-b31b1b.svg)](https://arxiv.org/abs/2211.11537) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)](https://github.com/Project-MONAI/tutorials) |
| 13 | UNETR | NVIDIA | 69.6 | [![arXiv](https://img.shields.io/badge/arXiv-2111.04004-b31b1b.svg)](https://arxiv.org/abs/2111.04004) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)](https://github.com/Project-MONAI/tutorials) |

</div>
</details>

<details>
<summary style="margin-left: 25px;">Liver - MedNeXt 🏆 </summary>
<div style="margin-left: 25px;">

| rank | model  | organization | DSC | paper | github |
|:---|:---|:---|:---:|:---:|:---:|
| 🏆 | MedNeXt | DKFZ | 96.3 | [![arXiv](https://img.shields.io/badge/arXiv-2303.09975-b31b1b.svg)](https://arxiv.org/pdf/2303.09975) | [![GitHub stars](https://img.shields.io/github/stars/MIC-DKFZ/MedNeXt.svg?logo=github&label=Stars)](https://github.com/MIC-DKFZ/MedNeXt) |
| 2 | nnU-Net ResEncL | DKFZ | 96.3 | [![arXiv](https://img.shields.io/badge/arXiv-1809.10486-b31b1b.svg)](https://arxiv.org/abs/1809.10486) | [![GitHub stars](https://img.shields.io/github/stars/MIC-DKFZ/nnUNet.svg?logo=github&label=Stars)](https://github.com/MIC-DKFZ/nnUNet) |
| 3 | Diff-UNet | HKUST | 96.2 | [![arXiv](https://img.shields.io/badge/arXiv-2303.10326-b31b1b.svg)](https://arxiv.org/abs/2303.10326) | [![GitHub stars](https://img.shields.io/github/stars/ge-xing/Diff-UNet.svg?logo=github&label=Stars)](https://github.com/ge-xing/Diff-UNet) |
| 4 | STU-Net-B | Shanghai AI Lab | 96.2 | [![arXiv](https://img.shields.io/badge/arXiv-2304.06716-b31b1b.svg)](https://arxiv.org/pdf/2304.06716) | [![GitHub stars](https://img.shields.io/github/stars/uni-medical/STU-Net.svg?logo=github&label=Stars)](https://github.com/uni-medical/STU-Net) |
| 5 | UniSeg | NPU | 96.1 | [![arXiv](https://img.shields.io/badge/arXiv-2304.03493-b31b1b.svg)](https://arxiv.org/abs/2304.03493) | [![GitHub stars](https://img.shields.io/github/stars/yeerwen/UniSeg.svg?logo=github&label=Stars)](https://github.com/yeerwen/UniSeg) |
| 6 | U-Net & CLIP | CityU | 96.0 | [![arXiv](https://img.shields.io/badge/arXiv-2301.00785-b31b1b.svg)](https://arxiv.org/abs/2301.00785) | [![GitHub stars](https://img.shields.io/github/stars/ljwztc/CLIP-Driven-Universal-Model.svg?logo=github&label=Stars)](https://github.com/ljwztc/CLIP-Driven-Universal-Model) |
| 7 | SegVol | BAAI | 96.0 | [![arXiv](https://img.shields.io/badge/arXiv-2311.13385-b31b1b.svg)](https://arxiv.org/abs/2311.13385) | [![GitHub stars](https://img.shields.io/github/stars/BAAI-DCAI/SegVol.svg?logo=github&label=Stars)](https://github.com/BAAI-DCAI/SegVol) |
| 8 | Swin UNETR & CLIP | CityU | 95.8 | [![arXiv](https://img.shields.io/badge/arXiv-2301.00785-b31b1b.svg)](https://arxiv.org/abs/2301.00785) | [![GitHub stars](https://img.shields.io/github/stars/ljwztc/CLIP-Driven-Universal-Model.svg?logo=github&label=Stars)](https://github.com/ljwztc/CLIP-Driven-Universal-Model) |
| 9 | NexToU | HIT | 95.7 | [![arXiv](https://img.shields.io/badge/arXiv-2305.15911-b31b1b.svg)](https://arxiv.org/abs/2305.15911) | [![GitHub stars](https://img.shields.io/github/stars/PengchengShi1220/NexToU.svg?logo=github&label=Stars)](https://github.com/PengchengShi1220/NexToU) |
| 10 | SAM-Adapter | Duke | 94.1 | [![arXiv](https://img.shields.io/badge/arXiv-2404.09957-b31b1b.svg)](https://arxiv.org/abs/2404.09957) | [![GitHub stars](https://img.shields.io/github/stars/mazurowski-lab/finetune-SAM.svg?logo=github&label=Stars)](https://github.com/mazurowski-lab/finetune-SAM) |
| 11 | UNesT | NVIDIA | 93.6 | [![arXiv](https://img.shields.io/badge/arXiv-2303.10745-b31b1b.svg)](https://arxiv.org/abs/2303.10745) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)](https://github.com/Project-MONAI/tutorials) |
| 12 | Swin UNETR | NVIDIA | 93.5 | [![arXiv](https://img.shields.io/badge/arXiv-2211.11537-b31b1b.svg)](https://arxiv.org/abs/2211.11537) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)](https://github.com/Project-MONAI/tutorials) |
| 13 | UNETR | NVIDIA | 90.5 | [![arXiv](https://img.shields.io/badge/arXiv-2111.04004-b31b1b.svg)](https://arxiv.org/abs/2111.04004) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)](https://github.com/Project-MONAI/tutorials) |

</div>
</details>

<details>
<summary style="margin-left: 25px;">Pancreas - MedNeXt 🏆 </summary>
<div style="margin-left: 25px;">

| rank | model  | organization | DSC | paper | github |
|:---|:---|:---|:---:|:---:|:---:|
| 🏆 | MedNeXt | DKFZ | 83.3 | [![arXiv](https://img.shields.io/badge/arXiv-2303.09975-b31b1b.svg)](https://arxiv.org/pdf/2303.09975) | [![GitHub stars](https://img.shields.io/github/stars/MIC-DKFZ/MedNeXt.svg?logo=github&label=Stars)](https://github.com/MIC-DKFZ/MedNeXt) |
| 2 | STU-Net-B | Shanghai AI Lab | 83.2 | [![arXiv](https://img.shields.io/badge/arXiv-2304.06716-b31b1b.svg)](https://arxiv.org/pdf/2304.06716) | [![GitHub stars](https://img.shields.io/github/stars/uni-medical/STU-Net.svg?logo=github&label=Stars)](https://github.com/uni-medical/STU-Net) |
| 3 | nnU-Net ResEncL | DKFZ | 82.9 | [![arXiv](https://img.shields.io/badge/arXiv-1809.10486-b31b1b.svg)](https://arxiv.org/abs/1809.10486) | [![GitHub stars](https://img.shields.io/github/stars/MIC-DKFZ/nnUNet.svg?logo=github&label=Stars)](https://github.com/MIC-DKFZ/nnUNet) |
| 4 | UniSeg | NPU | 82.7 | [![arXiv](https://img.shields.io/badge/arXiv-2304.03493-b31b1b.svg)](https://arxiv.org/abs/2304.03493) | [![GitHub stars](https://img.shields.io/github/stars/yeerwen/UniSeg.svg?logo=github&label=Stars)](https://github.com/yeerwen/UniSeg) |
| 5 | Diff-UNet | HKUST | 81.9 | [![arXiv](https://img.shields.io/badge/arXiv-2303.10326-b31b1b.svg)](https://arxiv.org/abs/2303.10326) | [![GitHub stars](https://img.shields.io/github/stars/ge-xing/Diff-UNet.svg?logo=github&label=Stars)](https://github.com/ge-xing/Diff-UNet) |
| 6 | U-Net & CLIP | CityU | 80.8 | [![arXiv](https://img.shields.io/badge/arXiv-2301.00785-b31b1b.svg)](https://arxiv.org/abs/2301.00785) | [![GitHub stars](https://img.shields.io/github/stars/ljwztc/CLIP-Driven-Universal-Model.svg?logo=github&label=Stars)](https://github.com/ljwztc/CLIP-Driven-Universal-Model) |
| 7 | Swin UNETR & CLIP | CityU | 80.2 | [![arXiv](https://img.shields.io/badge/arXiv-2301.00785-b31b1b.svg)](https://arxiv.org/abs/2301.00785) | [![GitHub stars](https://img.shields.io/github/stars/ljwztc/CLIP-Driven-Universal-Model.svg?logo=github&label=Stars)](https://github.com/ljwztc/CLIP-Driven-Universal-Model) |
| 8 | NexToU | HIT | 80.2 | [![arXiv](https://img.shields.io/badge/arXiv-2305.15911-b31b1b.svg)](https://arxiv.org/abs/2305.15911) | [![GitHub stars](https://img.shields.io/github/stars/PengchengShi1220/NexToU.svg?logo=github&label=Stars)](https://github.com/PengchengShi1220/NexToU) |
| 9 | SegVol | BAAI | 79.1 | [![arXiv](https://img.shields.io/badge/arXiv-2311.13385-b31b1b.svg)](https://arxiv.org/abs/2311.13385) | [![GitHub stars](https://img.shields.io/github/stars/BAAI-DCAI/SegVol.svg?logo=github&label=Stars)](https://github.com/BAAI-DCAI/SegVol) |
| 10 | Swin UNETR | NVIDIA | 68.5 | [![arXiv](https://img.shields.io/badge/arXiv-2211.11537-b31b1b.svg)](https://arxiv.org/abs/2211.11537) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)](https://github.com/Project-MONAI/tutorials) |
| 11 | UNesT | NVIDIA | 68.3 | [![arXiv](https://img.shields.io/badge/arXiv-2303.10745-b31b1b.svg)](https://arxiv.org/abs/2303.10745) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)](https://github.com/Project-MONAI/tutorials) |
| 12 | UNETR | NVIDIA | 55.1 | [![arXiv](https://img.shields.io/badge/arXiv-2111.04004-b31b1b.svg)](https://arxiv.org/abs/2111.04004) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)](https://github.com/Project-MONAI/tutorials) |
| 13 | SAM-Adapter | Duke | 50.2 | [![arXiv](https://img.shields.io/badge/arXiv-2404.09957-b31b1b.svg)](https://arxiv.org/abs/2404.09957) | [![GitHub stars](https://img.shields.io/github/stars/mazurowski-lab/finetune-SAM.svg?logo=github&label=Stars)](https://github.com/mazurowski-lab/finetune-SAM) |

</div>
</details>

<details>
<summary style="margin-left: 25px;">Postcava - STU-Net-B & MedNeXt 🏆 </summary>
<div style="margin-left: 25px;">

| rank | model  | organization | DSC | paper | github |
|:---|:---|:---|:---:|:---:|:---:|
| 🏆 | STU-Net-B | Shanghai AI Lab | 81.3 | [![arXiv](https://img.shields.io/badge/arXiv-2304.06716-b31b1b.svg)](https://arxiv.org/pdf/2304.06716) | [![GitHub stars](https://img.shields.io/github/stars/uni-medical/STU-Net.svg?logo=github&label=Stars)](https://github.com/uni-medical/STU-Net) |
| 🏆 | MedNeXt | DKFZ | 81.3 | [![arXiv](https://img.shields.io/badge/arXiv-2303.09975-b31b1b.svg)](https://arxiv.org/pdf/2303.09975) | [![GitHub stars](https://img.shields.io/github/stars/MIC-DKFZ/MedNeXt.svg?logo=github&label=Stars)](https://github.com/MIC-DKFZ/MedNeXt) |
| 3 | UniSeg | NPU | 81.2 | [![arXiv](https://img.shields.io/badge/arXiv-2304.03493-b31b1b.svg)](https://arxiv.org/abs/2304.03493) | [![GitHub stars](https://img.shields.io/github/stars/yeerwen/UniSeg.svg?logo=github&label=Stars)](https://github.com/yeerwen/UniSeg) |
| 4 | Diff-UNet | HKUST | 80.8 | [![arXiv](https://img.shields.io/badge/arXiv-2303.10326-b31b1b.svg)](https://arxiv.org/abs/2303.10326) | [![GitHub stars](https://img.shields.io/github/stars/ge-xing/Diff-UNet.svg?logo=github&label=Stars)](https://github.com/ge-xing/Diff-UNet) |
| 5 | nnU-Net ResEncL | DKFZ | 80.5 | [![arXiv](https://img.shields.io/badge/arXiv-1809.10486-b31b1b.svg)](https://arxiv.org/abs/1809.10486) | [![GitHub stars](https://img.shields.io/github/stars/MIC-DKFZ/nnUNet.svg?logo=github&label=Stars)](https://github.com/MIC-DKFZ/nnUNet) |
| 6 | U-Net & CLIP | CityU | 78.5 | [![arXiv](https://img.shields.io/badge/arXiv-2301.00785-b31b1b.svg)](https://arxiv.org/abs/2301.00785) | [![GitHub stars](https://img.shields.io/github/stars/ljwztc/CLIP-Driven-Universal-Model.svg?logo=github&label=Stars)](https://github.com/ljwztc/CLIP-Driven-Universal-Model) |
| 7 | NexToU | HIT | 78.1 | [![arXiv](https://img.shields.io/badge/arXiv-2305.15911-b31b1b.svg)](https://arxiv.org/abs/2305.15911) | [![GitHub stars](https://img.shields.io/github/stars/PengchengShi1220/NexToU.svg?logo=github&label=Stars)](https://github.com/PengchengShi1220/NexToU) |
| 8 | SegVol | BAAI | 77.8 | [![arXiv](https://img.shields.io/badge/arXiv-2311.13385-b31b1b.svg)](https://arxiv.org/abs/2311.13385) | [![GitHub stars](https://img.shields.io/github/stars/BAAI-DCAI/SegVol.svg?logo=github&label=Stars)](https://github.com/BAAI-DCAI/SegVol) |
| 9 | Swin UNETR & CLIP | CityU | 76.8 | [![arXiv](https://img.shields.io/badge/arXiv-2301.00785-b31b1b.svg)](https://arxiv.org/abs/2301.00785) | [![GitHub stars](https://img.shields.io/github/stars/ljwztc/CLIP-Driven-Universal-Model.svg?logo=github&label=Stars)](https://github.com/ljwztc/CLIP-Driven-Universal-Model) |
| 10 | Swin UNETR | NVIDIA | 69.9 | [![arXiv](https://img.shields.io/badge/arXiv-2211.11537-b31b1b.svg)](https://arxiv.org/abs/2211.11537) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)](https://github.com/Project-MONAI/tutorials) |
| 11 | UNesT | NVIDIA | 66.2 | [![arXiv](https://img.shields.io/badge/arXiv-2303.10745-b31b1b.svg)](https://arxiv.org/abs/2303.10745) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)](https://github.com/Project-MONAI/tutorials) |
| 12 | UNETR | NVIDIA | 53.9 | [![arXiv](https://img.shields.io/badge/arXiv-2111.04004-b31b1b.svg)](https://arxiv.org/abs/2111.04004) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)](https://github.com/Project-MONAI/tutorials) |
| 13 | SAM-Adapter | Duke | 48.0 | [![arXiv](https://img.shields.io/badge/arXiv-2404.09957-b31b1b.svg)](https://arxiv.org/abs/2404.09957) | [![GitHub stars](https://img.shields.io/github/stars/mazurowski-lab/finetune-SAM.svg?logo=github&label=Stars)](https://github.com/mazurowski-lab/finetune-SAM) |


</div>
</details>

<details>
<summary style="margin-left: 25px;">Spleen - nnU-Net ResEncL 🏆 </summary>
<div style="margin-left: 25px;">

| rank | model  | organization | DSC | paper | github |
|:---|:---|:---|:---:|:---:|:---:|
| 🏆 | nnU-Net ResEncL | DKFZ | 95.2 | [![arXiv](https://img.shields.io/badge/arXiv-1809.10486-b31b1b.svg)](https://arxiv.org/abs/1809.10486) | [![GitHub stars](https://img.shields.io/github/stars/MIC-DKFZ/nnUNet.svg?logo=github&label=Stars)](https://github.com/MIC-DKFZ/nnUNet) |
| 2 | MedNeXt | DKFZ | 95.2 | [![arXiv](https://img.shields.io/badge/arXiv-2303.09975-b31b1b.svg)](https://arxiv.org/pdf/2303.09975) | [![GitHub stars](https://img.shields.io/github/stars/MIC-DKFZ/MedNeXt.svg?logo=github&label=Stars)](https://github.com/MIC-DKFZ/MedNeXt) |
| 3 | STU-Net-B | Shanghai AI Lab | 95.1 | [![arXiv](https://img.shields.io/badge/arXiv-2304.06716-b31b1b.svg)](https://arxiv.org/pdf/2304.06716) | [![GitHub stars](https://img.shields.io/github/stars/uni-medical/STU-Net.svg?logo=github&label=Stars)](https://github.com/uni-medical/STU-Net) |
| 4 | Diff-UNet | HKUST | 95.0 | [![arXiv](https://img.shields.io/badge/arXiv-2303.10326-b31b1b.svg)](https://arxiv.org/abs/2303.10326) | [![GitHub stars](https://img.shields.io/github/stars/ge-xing/Diff-UNet.svg?logo=github&label=Stars)](https://github.com/ge-xing/Diff-UNet) |
| 5 | UniSeg | NPU | 94.9 | [![arXiv](https://img.shields.io/badge/arXiv-2304.03493-b31b1b.svg)](https://arxiv.org/abs/2304.03493) | [![GitHub stars](https://img.shields.io/github/stars/yeerwen/UniSeg.svg?logo=github&label=Stars)](https://github.com/yeerwen/UniSeg) |
| 6 | SegVol | BAAI | 94.5 | [![arXiv](https://img.shields.io/badge/arXiv-2311.13385-b31b1b.svg)](https://arxiv.org/abs/2311.13385) | [![GitHub stars](https://img.shields.io/github/stars/BAAI-DCAI/SegVol.svg?logo=github&label=Stars)](https://github.com/BAAI-DCAI/SegVol) |
| 7 | NexToU | HIT | 94.7 | [![arXiv](https://img.shields.io/badge/arXiv-2305.15911-b31b1b.svg)](https://arxiv.org/abs/2305.15911) | [![GitHub stars](https://img.shields.io/github/stars/PengchengShi1220/NexToU.svg?logo=github&label=Stars)](https://github.com/PengchengShi1220/NexToU) |
| 8 | U-Net & CLIP | CityU | 94.3 | [![arXiv](https://img.shields.io/badge/arXiv-2301.00785-b31b1b.svg)](https://arxiv.org/abs/2301.00785) | [![GitHub stars](https://img.shields.io/github/stars/ljwztc/CLIP-Driven-Universal-Model.svg?logo=github&label=Stars)](https://github.com/ljwztc/CLIP-Driven-Universal-Model) |
| 9 | Swin UNETR & CLIP | CityU | 94.1 | [![arXiv](https://img.shields.io/badge/arXiv-2301.00785-b31b1b.svg)](https://arxiv.org/abs/2301.00785) | [![GitHub stars](https://img.shields.io/github/stars/ljwztc/CLIP-Driven-Universal-Model.svg?logo=github&label=Stars)](https://github.com/ljwztc/CLIP-Driven-Universal-Model) |
| 10 | SAM-Adapter | Duke | 90.5 | [![arXiv](https://img.shields.io/badge/arXiv-2404.09957-b31b1b.svg)](https://arxiv.org/abs/2404.09957) | [![GitHub stars](https://img.shields.io/github/stars/mazurowski-lab/finetune-SAM.svg?logo=github&label=Stars)](https://github.com/mazurowski-lab/finetune-SAM) |
| 11 | Swin UNETR | NVIDIA | 87.9 | [![arXiv](https://img.shields.io/badge/arXiv-2211.11537-b31b1b.svg)](https://arxiv.org/abs/2211.11537) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)](https://github.com/Project-MONAI/tutorials) |
| 12 | UNesT | NVIDIA | 86.7 | [![arXiv](https://img.shields.io/badge/arXiv-2303.10745-b31b1b.svg)](https://arxiv.org/abs/2303.10745) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)](https://github.com/Project-MONAI/tutorials) |
| 13 | UNETR | NVIDIA | 76.5 | [![arXiv](https://img.shields.io/badge/arXiv-2111.04004-b31b1b.svg)](https://arxiv.org/abs/2111.04004) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)](https://github.com/Project-MONAI/tutorials) |

</div>
</details>

<details>
<summary style="margin-left: 25px;">Stomach - STU-Net-B & MedNeXt & nnU-Net ResEncL 🏆 </summary>
<div style="margin-left: 25px;">

| rank | model  | organization | DSC | paper | github |
|:---|:---|:---|:---:|:---:|:---:|
| 🏆 | STU-Net-B | Shanghai AI Lab | 93.5 | [![arXiv](https://img.shields.io/badge/arXiv-2304.06716-b31b1b.svg)](https://arxiv.org/pdf/2304.06716) | [![GitHub stars](https://img.shields.io/github/stars/uni-medical/STU-Net.svg?logo=github&label=Stars)](https://github.com/uni-medical/STU-Net) |
| 🏆 | MedNeXt | DKFZ | 93.5 | [![arXiv](https://img.shields.io/badge/arXiv-2303.09975-b31b1b.svg)](https://arxiv.org/pdf/2303.09975) | [![GitHub stars](https://img.shields.io/github/stars/MIC-DKFZ/MedNeXt.svg?logo=github&label=Stars)](https://github.com/MIC-DKFZ/MedNeXt) |
| 🏆 | nnU-Net ResEncL | DKFZ | 93.4 | [![arXiv](https://img.shields.io/badge/arXiv-1809.10486-b31b1b.svg)](https://arxiv.org/abs/1809.10486) | [![GitHub stars](https://img.shields.io/github/stars/MIC-DKFZ/nnUNet.svg?logo=github&label=Stars)](https://github.com/MIC-DKFZ/nnUNet) |
| 4 | UniSeg | NPU | 93.3 | [![arXiv](https://img.shields.io/badge/arXiv-2304.03493-b31b1b.svg)](https://arxiv.org/abs/2304.03493) | [![GitHub stars](https://img.shields.io/github/stars/yeerwen/UniSeg.svg?logo=github&label=Stars)](https://github.com/yeerwen/UniSeg) |
| 5 | Diff-UNet | HKUST | 93.1 | [![arXiv](https://img.shields.io/badge/arXiv-2303.10326-b31b1b.svg)](https://arxiv.org/abs/2303.10326) | [![GitHub stars](https://img.shields.io/github/stars/ge-xing/Diff-UNet.svg?logo=github&label=Stars)](https://github.com/ge-xing/Diff-UNet) |
| 6 | NexToU | HIT | 92.7 | [![arXiv](https://img.shields.io/badge/arXiv-2305.15911-b31b1b.svg)](https://arxiv.org/abs/2305.15911) | [![GitHub stars](https://img.shields.io/github/stars/PengchengShi1220/NexToU.svg?logo=github&label=Stars)](https://github.com/PengchengShi1220/NexToU) |
| 7 | SegVol | BAAI | 92.5 | [![arXiv](https://img.shields.io/badge/arXiv-2311.13385-b31b1b.svg)](https://arxiv.org/abs/2311.13385) | [![GitHub stars](https://img.shields.io/github/stars/BAAI-DCAI/SegVol.svg?logo=github&label=Stars)](https://github.com/BAAI-DCAI/SegVol) |
| 8 | U-Net & CLIP | CityU | 92.4 | [![arXiv](https://img.shields.io/badge/arXiv-2301.00785-b31b1b.svg)](https://arxiv.org/abs/2301.00785) | [![GitHub stars](https://img.shields.io/github/stars/ljwztc/CLIP-Driven-Universal-Model.svg?logo=github&label=Stars)](https://github.com/ljwztc/CLIP-Driven-Universal-Model) |
| 9 | Swin UNETR & CLIP | CityU | 92.2 | [![arXiv](https://img.shields.io/badge/arXiv-2301.00785-b31b1b.svg)](https://arxiv.org/abs/2301.00785) | [![GitHub stars](https://img.shields.io/github/stars/ljwztc/CLIP-Driven-Universal-Model.svg?logo=github&label=Stars)](https://github.com/ljwztc/CLIP-Driven-Universal-Model) |
| 10 | SAM-Adapter | Duke | 88.0 | [![arXiv](https://img.shields.io/badge/arXiv-2404.09957-b31b1b.svg)](https://arxiv.org/abs/2404.09957) | [![GitHub stars](https://img.shields.io/github/stars/mazurowski-lab/finetune-SAM.svg?logo=github&label=Stars)](https://github.com/mazurowski-lab/finetune-SAM) |
| 11 | UNesT | NVIDIA | 87.6 | [![arXiv](https://img.shields.io/badge/arXiv-2303.10745-b31b1b.svg)](https://arxiv.org/abs/2303.10745) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)](https://github.com/Project-MONAI/tutorials) |
| 12 | Swin UNETR | NVIDIA | 84.1 | [![arXiv](https://img.shields.io/badge/arXiv-2211.11537-b31b1b.svg)](https://arxiv.org/abs/2211.11537) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)](https://github.com/Project-MONAI/tutorials) |
| 13 | UNETR | NVIDIA | 74.2 | [![arXiv](https://img.shields.io/badge/arXiv-2111.04004-b31b1b.svg)](https://arxiv.org/abs/2111.04004) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)](https://github.com/Project-MONAI/tutorials) |


</div>
</details>

# In-depth Result Analysis

<div align="center">
  <img src="utils/JHHAnalysiswlCaptions.png" alt="JHH Analysis" width="1200">
</div>

<details>
<summary style="margin-left: 25px;">* </summary>
<div style="margin-left: 25px;">

Each cell in the significance heatmap above indicates a one-sided statistical test. Red indicates that the x-axis AI algorithm is significantly superior to the y-axis algorithm in terms of DSC, for one organ.
  
</div>
</details>

We provide *DSC and NSD per CT scan* for each checkpoint in test sets #2 and #3, and a code tutorial for easy:
  - Per-organ performance analysis
  - Performance comparison across demographic groups (age, sex, race, scanner, diagnosis, etc.)
  - Pair-wise statistical tests and significance heatmaps
  - Boxplots

You can easily modify our code to compare your custom model to our checkpoints, or to analyze performance in custom demographic groups (e.g., hispanic men aged 20-25).

<details>
<summary style="margin-left: 25px;">Code tutorial </summary>
<div style="margin-left: 25px;">

Per-sample results are in CSV files inside the folders totalsegmentator_results and dapatlas_results.

<details>
<summary style="margin-left: 25px;">File structure </summary>
<div style="margin-left: 25px;">

```
totalsegmentator_results
    ├── Diff-UNet
    │   ├── dsc.csv
    │   └── nsd.csv
    ├── LHU-Net
    │   ├── dsc.csv
    │   └── nsd.csv
    ├── MedNeXt
    │   ├── dsc.csv
    │   └── nsd.csv
    ├── ...
dapatlas_results
    ├── Diff-UNet
    │   ├── dsc.csv
    │   └── nsd.csv
    ├── LHU-Net
    │   ├── dsc.csv
    │   └── nsd.csv
    ├── MedNeXt
    │   ├── dsc.csv
    │   └── nsd.csv
    ├── ...
```

</div>
</details>

#### 1. Clone the GitHub repository

```bash
git clone https://github.com/MrGiovanni/Touchstone
cd Touchstone
```

#### 2. Create environments

```bash
conda env create -f environment.yml
source activate touchstone
python -m ipykernel install --user --name touchstone --display-name "touchstone"
```

#### 3. Reproduce analysis figures in our paper

#### Figure 1 - Dataset statistics:
```bash
cd notebooks
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=touchstone TotalSegmentatorMetadata.ipynb
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=touchstone DAPAtlasMetadata.ipynb
#results: plots are saved inside Touchstone/outputs/plotsTotalSegmentator/ and Touchstone/outputs/plotsDAPAtlas/
```

#### Figure 2 - Potential confrounders significantly impact AI performance:
```bash
cd ../plot
python AggregatedBoxplot.py --stats
#results: Touchstone/outputs/summary_groups.pdf
```

#### Appendix D.2.3 - Statistical significance maps:
```bash
#statistical significance maps (Appendix D.2.3):
python PlotAllSignificanceMaps.py
python PlotAllSignificanceMaps.py --organs second_half
python PlotAllSignificanceMaps.py --nsd
python PlotAllSignificanceMaps.py --organs second_half --nsd
#results: Touchstone/outputs/heatmaps
```

#### Appendix D.4 and D.5 - Box-plots for per-group and per-organ results, with statistical tests:
```bash
cd ../notebooks
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=touchstone GroupAnalysis.ipynb
#results: Touchstone/outputs/box_plots
```

#### 4. Custom Analysis

<details>
<summary style="margin-left: 25px;">Define custom demographic groups (e.g., hispanic men aged 20-25) and compare AI performance on them </summary>
<div style="margin-left: 25px;">

The csv results files in totalsegmentator_results/ and dapatlas_results/ contain per-sample dsc and nsd scores. Rich meatdata for each one of those samples (sex, age, scanner, diagnosis,...) are available in metaTotalSeg.csv and 'Clinical Metadata FDG PET_CT Lesions.csv', for TotalSegmentator and DAP Atlas, respectively. The code in TotalSegmentatorMetadata.ipynb and DAPAtlasMetadata.ipynb extracts this meatdata into simplfied group lists (e.g., a list of all samples representing male patients), and saves these lists in the folders plotsTotalSegmentator/ and plotsDAPAtlas/. You can modify the code to generate custom sample lists (e.g., all men aged 30-35). To compare a set of groups, the filenames of all lists in the set should begin with the same name. For example, comp1_list_a.pt, comp1_list_b.pt, comp1_list_C.pt can represent a set of 3 groups. Then, PlotGroup.py can draw boxplots and perform statistical tests comparing the AI algorithm's results (dsc and nsd) for the samples inside the different custom lists you created. In our example, you just just need to specify --group_name comp1 when running PlotGroup.py:

```bash
python utils/PlotGroup.py --ckpt_root totalsegmentator_results/ --group_root outputs/plotsTotalSegmentator/ --group_name comp1 --organ liver --stats
```

</div>
</details>



</div>
</details>

# Citation

Please cite the following papers if you find our leaderboard or dataset helpful.

```
@article{li2024abdomenatlas,
  title={AbdomenAtlas: A large-scale, detailed-annotated, \& multi-center dataset for efficient transfer learning and open algorithmic benchmarking},
  author={Li, Wenxuan and Qu, Chongyu and Chen, Xiaoxi and Bassi, Pedro RAS and Shi, Yijia and Lai, Yuxiang and Yu, Qian and Xue, Huimin and Chen, Yixiong and Lin, Xiaorui and others},
  journal={Medical Image Analysis},
  pages={103285},
  year={2024},
  publisher={Elsevier}
}

@inproceedings{li2024well,
  title={How Well Do Supervised Models Transfer to 3D Image Segmentation?},
  author={Li, Wenxuan and Yuille, Alan and Zhou, Zongwei},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024}
}

@article{qu2023abdomenatlas,
  title={Abdomenatlas-8k: Annotating 8,000 CT volumes for multi-organ segmentation in three weeks},
  author={Qu, Chongyu and Zhang, Tiezheng and Qiao, Hualin and Tang, Yucheng and Yuille, Alan L and Zhou, Zongwei and others},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2023}
}
```

# Acknowledgement

This work was supported by the Lustgarten Foundation for Pancreatic Cancer Research and the McGovern Foundation. Paper content is covered by patents pending.

<div align="center">
  <img src="utils/partner_logo.jpg" alt="Touchstone Benchmark" width="1200">
</div>
