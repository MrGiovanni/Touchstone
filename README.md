<div align="center">
  <img src="utils/logo_white_bg.jpg" alt="Touchstone Benchmark" width="300">
</div>

<h1 align="center" style="font-size: 60px; margin-bottom: 4px">Touchstone Benchmark</h1>

<p align="center">
    <a href='https://www.cs.jhu.edu/~zongwei/advert/TutorialBenchmarkV1.pdf'>
        <img src='https://img.shields.io/badge/Participate%20--%20Touchstone%201.0-%233F51B5?style=for-the-badge' alt='Participate - Touchstone 1.0'>
    </a>
    <a href='https://www.cs.jhu.edu/~zongwei/advert/Call4Benchmark.pdf'>
        <img src='https://img.shields.io/badge/Participate%20--%20Touchstone%202.0-%23F44336?style=for-the-badge' alt='Participate - Touchstone 2.0'>
    </a>
</p>

<div align="center">


[![touchstone leaderboard](https://img.shields.io/badge/Touchstone-Leaderboard-cyan.svg)](https://github.com/MrGiovanni/Touchstone/tree/main?tab=readme-ov-file#touchstone-10-leaderboard)
[![touchstone dataset](https://img.shields.io/badge/Touchstone-Dataset-cyan.svg)](https://github.com/MrGiovanni/Touchstone/tree/main?tab=readme-ov-file#touchstone-10-dataset)
[![touchstone model](https://img.shields.io/badge/Touchstone-Model-cyan.svg)](https://github.com/MrGiovanni/Touchstone/tree/main?tab=readme-ov-file#touchstone-10-model) <br/>
![visitors](https://visitor-badge.laobi.icu/badge?page_id=MrGiovanni/Touchstone&left_color=%2363C7E6&right_color=%23CEE75F)
[![GitHub Repo stars](https://img.shields.io/github/stars/MrGiovanni/Touchstone?style=social)](https://github.com/MrGiovanni/Touchstone/stargazers) 
<a href="https://twitter.com/bodymaps317">
        <img src="https://img.shields.io/twitter/follow/BodyMaps?style=social" alt="Follow on Twitter" />
</a><br/>
**Subscribe us: https://groups.google.com/u/2/g/bodymaps**  

</div>


We present Touchstone, a large-scale  medical segmentation benchmark based on annotated **5,195** CT volumes from **76** hospitals for training, and **6,933** CT volumes from **8** additional hospitals for testing. We invite AI inventors to train their models on AbdomenAtlas, and we independently evaluate their algorithms. We have already collaborated with **14** influential research teams, and we remain accepting new submissions.

# Paper

<b>Touchstone Benchmark: Are We on the Right Way for Evaluating AI Algorithms for Medical Segmentation?</b> <br/>
[Pedro R. A. S. Bassi](https://scholar.google.com.hk/citations?user=NftgL6gAAAAJ)<sup>1</sup>, [Wenxuan Li](https://scholar.google.com/citations?hl=en&user=tpNZM2YAAAAJ)<sup>1</sup>, [Yucheng Tang](https://scholar.google.com.hk/citations?hl=en&user=0xheliUAAAAJ)<sup>2</sup>, [Fabian Isensee](https://scholar.google.com.hk/citations?hl=en&user=PjerEe4AAAAJ)<sup>3</sup>, ..., [Alan Yuille](https://www.cs.jhu.edu/~ayuille/)<sup>1</sup>, [Zongwei Zhou](https://www.zongweiz.com/)<sup>1</sup> <br/>
<sup>1</sup>Johns Hopkins University, <sup>2</sup>NVIDIA, <sup>3</sup>DKFZ <br/>
NeurIPS 2024 <br/>
[JHU CS News](https://www.cs.jhu.edu/news/a-touchstone-of-medical-artificial-intelligence/) <br/>

<a href='https://www.zongweiz.com/dataset'><img src='https://img.shields.io/badge/Project-Page-Green'></a> <a href='https://www.cs.jhu.edu/~zongwei/publication/bassi2024touchstone.pdf'><img src='https://img.shields.io/badge/Paper-PDF-purple'></a> <a href='document/jhu_seminar_slides.pdf'><img src='https://img.shields.io/badge/Slides-Seminar-orange'></a> <a href='document/rsna2024_abstract.pdf'><img src='https://img.shields.io/badge/Abstract-RSNA-purple'></a> <a href='document/rsna2024_slides.pdf'><img src='https://img.shields.io/badge/Slides-RSNA-orange'></a> [![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://youtu.be/tPnTCFUTjjY)

# Touchstone 1.0 Leaderboard 

| rank | model  | organization | average DSC | paper | github
|:---|:---|:---|:---:|:---:|:---:|
| 🏆 | MedNeXt | DKFZ | 89.2 | [![arXiv](https://img.shields.io/badge/arXiv-2303.09975-b31b1b.svg)](https://arxiv.org/pdf/2303.09975) | [![GitHub stars](https://img.shields.io/github/stars/MIC-DKFZ/MedNeXt.svg?logo=github&label=Stars)](https://github.com/MIC-DKFZ/MedNeXt) |
| 🏆 | MedFormer | Rutgers | 89.0 | [![arXiv](https://img.shields.io/badge/arXiv-2203.00131-b31b1b.svg)](https://arxiv.org/abs/2203.00131) | [![GitHub stars](https://img.shields.io/github/stars/yhygao/CBIM-Medical-Image-Segmentation.svg?logo=github&label=Stars)](https://github.com/yhygao/CBIM-Medical-Image-Segmentation) |
| 3 | STU-Net-B | Shanghai AI Lab | 89.0 | [![arXiv](https://img.shields.io/badge/arXiv-2304.06716-b31b1b.svg)](https://arxiv.org/pdf/2304.06716) | [![GitHub stars](https://img.shields.io/github/stars/uni-medical/STU-Net.svg?logo=github&label=Stars)](https://github.com/uni-medical/STU-Net) |
| 4 | nnU-Net U-Net | DKFZ | 88.9 | [![arXiv](https://img.shields.io/badge/arXiv-1809.10486-b31b1b.svg)](https://arxiv.org/abs/1809.10486) | [![GitHub stars](https://img.shields.io/github/stars/MIC-DKFZ/nnUNet.svg?logo=github&label=Stars)](https://github.com/MIC-DKFZ/nnUNet) |
| 5 | nnU-Net ResEncL | DKFZ | 88.8 | [![arXiv](https://img.shields.io/badge/arXiv-1809.10486-b31b1b.svg)](https://arxiv.org/abs/1809.10486) | [![GitHub stars](https://img.shields.io/github/stars/MIC-DKFZ/nnUNet.svg?logo=github&label=Stars)](https://github.com/MIC-DKFZ/nnUNet) |
| 6 | UniSeg | NPU | 88.8 | [![arXiv](https://img.shields.io/badge/arXiv-2304.03493-b31b1b.svg)](https://arxiv.org/abs/2304.03493) | [![GitHub stars](https://img.shields.io/github/stars/yeerwen/UniSeg.svg?logo=github&label=Stars)](https://github.com/yeerwen/UniSeg) |
| 7 | Diff-UNet | HKUST | 88.5 | [![arXiv](https://img.shields.io/badge/arXiv-2303.10326-b31b1b.svg)](https://arxiv.org/abs/2303.10326) | [![GitHub stars](https://img.shields.io/github/stars/ge-xing/Diff-UNet.svg?logo=github&label=Stars)](https://github.com/ge-xing/Diff-UNet) |
| 8 | LHU-Net | UR | 88.0 | [![arXiv](https://img.shields.io/badge/arXiv-2404.05102-b31b1b.svg)](https://arxiv.org/abs/2404.05102) | [![GitHub stars](https://img.shields.io/github/stars/xmindflow/LHUNet.svg?logo=github&label=Stars)](https://github.com/xmindflow/LHUNet) |
| 9 | NexToU | HIT | 87.8 | [![arXiv](https://img.shields.io/badge/arXiv-2305.15911-b31b1b.svg)](https://arxiv.org/abs/2305.15911) | [![GitHub stars](https://img.shields.io/github/stars/PengchengShi1220/NexToU.svg?logo=github&label=Stars)](https://github.com/PengchengShi1220/NexToU) |
| 10 | SegVol | BAAI | 87.1 | [![arXiv](https://img.shields.io/badge/arXiv-2311.13385-b31b1b.svg)](https://arxiv.org/abs/2311.13385) | [![GitHub stars](https://img.shields.io/github/stars/BAAI-DCAI/SegVol.svg?logo=github&label=Stars)](https://github.com/BAAI-DCAI/SegVol) |
| 11 | U-Net & CLIP | CityU | 87.1 | [![arXiv](https://img.shields.io/badge/arXiv-2301.00785-b31b1b.svg)](https://arxiv.org/abs/2301.00785) | [![GitHub stars](https://img.shields.io/github/stars/ljwztc/CLIP-Driven-Universal-Model.svg?logo=github&label=Stars)](https://github.com/ljwztc/CLIP-Driven-Universal-Model) |
| 12 | Swin UNETR & CLIP | CityU | 86.7 | [![arXiv](https://img.shields.io/badge/arXiv-2301.00785-b31b1b.svg)](https://arxiv.org/abs/2301.00785) | [![GitHub stars](https://img.shields.io/github/stars/ljwztc/CLIP-Driven-Universal-Model.svg?logo=github&label=Stars)](https://github.com/ljwztc/CLIP-Driven-Universal-Model) |
| 13 | UNesT | NVIDIA | 84.9 | [![arXiv](https://img.shields.io/badge/arXiv-2303.10745-b31b1b.svg)](https://arxiv.org/abs/2303.10745) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)](https://github.com/Project-MONAI/tutorials) |
| 14 | Swin UNETR | NVIDIA | 84.8 | [![arXiv](https://img.shields.io/badge/arXiv-2211.11537-b31b1b.svg)](https://arxiv.org/abs/2211.11537) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)](https://github.com/Project-MONAI/tutorials) |
| 15 | UNETR | NVIDIA| 83.3 | [![arXiv](https://img.shields.io/badge/arXiv-2111.04004-b31b1b.svg)](https://arxiv.org/abs/2111.04004) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)](https://github.com/Project-MONAI/tutorials) |
| 16 | UCTransNet | Northeastern University | 81.1 | [![arXiv](https://img.shields.io/badge/arXiv-2211.11537-b31b1b.svg)]([https://arxiv.org/abs/2211.11537](https://arxiv.org/pdf/2109.04335)) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)]([https://github.com/McGregorWwww/UCTransNet](https://github.com/McGregorWwww/UCTransNet)) |
| 17 | SAM-Adapter | Duke | 73.4 | [![arXiv](https://img.shields.io/badge/arXiv-2404.09957-b31b1b.svg)](https://arxiv.org/abs/2404.09957) | [![GitHub stars](https://img.shields.io/github/stars/mazurowski-lab/finetune-SAM.svg?logo=github&label=Stars)](https://github.com/mazurowski-lab/finetune-SAM) |


<details>
<summary style="margin-left: 25px;">Aorta - NexToU & UCTransNet 🏆 </summary>
<div style="margin-left: 25px;">

| rank | model  | organization | DSC | paper | github |
|:---|:---|:---|:---:|:---:|:---:|
| 🏆 | UCTransNet | Northeastern University | 86.5 | [![arXiv](https://img.shields.io/badge/arXiv-2211.11537-b31b1b.svg)]([https://arxiv.org/abs/2211.11537](https://arxiv.org/pdf/2109.04335)) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)]([https://github.com/McGregorWwww/UCTransNet](https://github.com/McGregorWwww/UCTransNet)) |
| 🏆 | NexToU | HIT | 86.4 | [![arXiv](https://img.shields.io/badge/arXiv-2305.15911-b31b1b.svg)](https://arxiv.org/pdf/2305.15911) | [![GitHub stars](https://img.shields.io/github/stars/PengchengShi1220/NexToU.svg?logo=github&label=Stars)](https://github.com/PengchengShi1220/NexToU) |
| 3 | MedNeXt | DKFZ | 83.1 | [![arXiv](https://img.shields.io/badge/arXiv-2303.09975-b31b1b.svg)](https://arxiv.org/pdf/2303.09975) | [![GitHub stars](https://img.shields.io/github/stars/MIC-DKFZ/MedNeXt.svg?logo=github&label=Stars)](https://github.com/MIC-DKFZ/MedNeXt) |
| 4 | nnU-Net U-Net | DKFZ | 82.8 | [![arXiv](https://img.shields.io/badge/arXiv-1809.10486-b31b1b.svg)](https://arxiv.org/abs/1809.10486) | [![GitHub stars](https://img.shields.io/github/stars/MIC-DKFZ/nnUNet.svg?logo=github&label=Stars)](https://github.com/MIC-DKFZ/nnUNet) |
| 5 | UniSeg | NPU | 82.3 | [![arXiv](https://img.shields.io/badge/arXiv-2304.03493-b31b1b.svg)](https://arxiv.org/abs/2304.03493) | [![GitHub stars](https://img.shields.io/github/stars/yeerwen/UniSeg.svg?logo=github&label=Stars)](https://github.com/yeerwen/UniSeg) |
| 6 | MedFormer | Rutgers | 82.1 | [![arXiv](https://img.shields.io/badge/arXiv-2203.00131-b31b1b.svg)](https://arxiv.org/abs/2203.00131) | [![GitHub stars](https://img.shields.io/github/stars/yhygao/CBIM-Medical-Image-Segmentation.svg?logo=github&label=Stars)](https://github.com/yhygao/CBIM-Medical-Image-Segmentation) |
| 7 | STU-Net-B | Shanghai AI Lab | 82.1 | [![arXiv](https://img.shields.io/badge/arXiv-2304.06716-b31b1b.svg)](https://arxiv.org/pdf/2304.06716) | [![GitHub stars](https://img.shields.io/github/stars/uni-medical/STU-Net.svg?logo=github&label=Stars)](https://github.com/uni-medical/STU-Net) |
| 8 | nnU-Net ResEncL | DKFZ | 81.4 | [![arXiv](https://img.shields.io/badge/arXiv-1809.10486-b31b1b.svg)](https://arxiv.org/abs/1809.10486) | [![GitHub stars](https://img.shields.io/github/stars/MIC-DKFZ/nnUNet.svg?logo=github&label=Stars)](https://github.com/MIC-DKFZ/nnUNet) |
| 9 | Diff-UNet | HKUST | 81.2 | [![arXiv](https://img.shields.io/badge/arXiv-2303.10326-b31b1b.svg)](https://arxiv.org/abs/2303.10326) | [![GitHub stars](https://img.shields.io/github/stars/ge-xing/Diff-UNet.svg?logo=github&label=Stars)](https://github.com/ge-xing/Diff-UNet) |
| 10 | SegVol | BAAI | 80.2 | [![arXiv](https://img.shields.io/badge/arXiv-2311.13385-b31b1b.svg)](https://arxiv.org/abs/2311.13385) | [![GitHub stars](https://img.shields.io/github/stars/BAAI-DCAI/SegVol.svg?logo=github&label=Stars)](https://github.com/BAAI-DCAI/SegVol) |
| 11 | LHU-Net | UR | 79.5 | [![arXiv](https://img.shields.io/badge/arXiv-2404.05102-b31b1b.svg)](https://arxiv.org/abs/2404.05102) | [![GitHub stars](https://img.shields.io/github/stars/xmindflow/LHUNet.svg?logo=github&label=Stars)](https://github.com/xmindflow/LHUNet) |
| 12 | Swin UNETR & CLIP | CityU | 78.1 | [![arXiv](https://img.shields.io/badge/arXiv-2301.00785-b31b1b.svg)](https://arxiv.org/abs/2301.00785) | [![GitHub stars](https://img.shields.io/github/stars/ljwztc/CLIP-Driven-Universal-Model.svg?logo=github&label=Stars)](https://github.com/ljwztc/CLIP-Driven-Universal-Model) |
| 13 | UNesT | NVIDIA | 77.7 | [![arXiv](https://img.shields.io/badge/arXiv-2303.10745-b31b1b.svg)](https://arxiv.org/abs/2303.10745) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)](https://github.com/Project-MONAI/tutorials) |
| 14| Swin UNETR | NVIDIA | 77.2 | [![arXiv](https://img.shields.io/badge/arXiv-2211.11537-b31b1b.svg)](https://arxiv.org/abs/2211.11537) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)](https://github.com/Project-MONAI/tutorials) |
| 15 | U-Net & CLIP | CityU | 77.1 | [![arXiv](https://img.shields.io/badge/arXiv-2301.00785-b31b1b.svg)](https://arxiv.org/abs/2301.00785) | [![GitHub stars](https://img.shields.io/github/stars/ljwztc/CLIP-Driven-Universal-Model.svg?logo=github&label=Stars)](https://github.com/ljwztc/CLIP-Driven-Universal-Model) |
| 16 | UNETR | NVIDIA | 76.5 | [![arXiv](https://img.shields.io/badge/arXiv-2111.04004-b31b1b.svg)](https://arxiv.org/abs/2111.04004) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)](https://github.com/Project-MONAI/tutorials) |
| 17 | SAM-Adapter | Duke | 62.8 | [![arXiv](https://img.shields.io/badge/arXiv-2404.09957-b31b1b.svg)](https://arxiv.org/abs/2404.09957) | [![GitHub stars](https://img.shields.io/github/stars/mazurowski-lab/finetune-SAM.svg?logo=github&label=Stars)](https://github.com/mazurowski-lab/finetune-SAM) |


</div>
</details>

<details>
<summary style="margin-left: 25px;">Gallbladder - STU-Net-B & MedFormer 🏆 </summary>
<div style="margin-left: 25px;">

| rank | model  | organization | DSC | paper | github |
|:---|:---|:---|:---:|:---:|:---:|
| 🏆 | STU-Net-B | Shanghai AI Lab | 85.5 | [![arXiv](https://img.shields.io/badge/arXiv-2304.06716-b31b1b.svg)](https://arxiv.org/pdf/2304.06716) | [![GitHub stars](https://img.shields.io/github/stars/uni-medical/STU-Net.svg?logo=github&label=Stars)](https://github.com/uni-medical/STU-Net) |
| 🏆 | MedFormer | Rutgers | 85.3 | [![arXiv](https://img.shields.io/badge/arXiv-2203.00131-b31b1b.svg)](https://arxiv.org/abs/2203.00131) | [![GitHub stars](https://img.shields.io/github/stars/yhygao/CBIM-Medical-Image-Segmentation.svg?logo=github&label=Stars)](https://github.com/yhygao/CBIM-Medical-Image-Segmentation) |
| 3 | MedNeXt | DKFZ | 85.3 | [![arXiv](https://img.shields.io/badge/arXiv-2303.09975-b31b1b.svg)](https://arxiv.org/pdf/2303.09975) | [![GitHub stars](https://img.shields.io/github/stars/MIC-DKFZ/MedNeXt.svg?logo=github&label=Stars)](https://github.com/MIC-DKFZ/MedNeXt) |
| 4 | nnU-Net ResEncL | DKFZ | 84.9 | [![arXiv](https://img.shields.io/badge/arXiv-1809.10486-b31b1b.svg)](https://arxiv.org/abs/1809.10486) | [![GitHub stars](https://img.shields.io/github/stars/MIC-DKFZ/nnUNet.svg?logo=github&label=Stars)](https://github.com/MIC-DKFZ/nnUNet) |
| 5 | nnU-Net U-Net | DKFZ | 84.7 | [![arXiv](https://img.shields.io/badge/arXiv-1809.10486-b31b1b.svg)](https://arxiv.org/abs/1809.10486) | [![GitHub stars](https://img.shields.io/github/stars/MIC-DKFZ/nnUNet.svg?logo=github&label=Stars)](https://github.com/MIC-DKFZ/nnUNet) |
| 6 | UniSeg | NPU | 84.7 | [![arXiv](https://img.shields.io/badge/arXiv-2304.03493-b31b1b.svg)](https://arxiv.org/abs/2304.03493) | [![GitHub stars](https://img.shields.io/github/stars/yeerwen/UniSeg.svg?logo=github&label=Stars)](https://github.com/yeerwen/UniSeg) |
| 7 | LHU-Net | UR | 83.9 | [![arXiv](https://img.shields.io/badge/arXiv-2404.05102-b31b1b.svg)](https://arxiv.org/abs/2404.05102) | [![GitHub stars](https://img.shields.io/github/stars/xmindflow/LHUNet.svg?logo=github&label=Stars)](https://github.com/xmindflow/LHUNet) |
| 8 | Diff-UNet | HKUST | 83.8 | [![arXiv](https://img.shields.io/badge/arXiv-2303.10326-b31b1b.svg)](https://arxiv.org/abs/2303.10326) | [![GitHub stars](https://img.shields.io/github/stars/ge-xing/Diff-UNet.svg?logo=github&label=Stars)](https://github.com/ge-xing/Diff-UNet) |
| 9 | NexToU | HIT | 82.3 | [![arXiv](https://img.shields.io/badge/arXiv-2305.15911-b31b1b.svg)](https://arxiv.org/abs/2305.15911) | [![GitHub stars](https://img.shields.io/github/stars/PengchengShi1220/NexToU.svg?logo=github&label=Stars)](https://github.com/PengchengShi1220/NexToU) |
| 10 | U-Net & CLIP | CityU | 82.1 | [![arXiv](https://img.shields.io/badge/arXiv-2301.00785-b31b1b.svg)](https://arxiv.org/abs/2301.00785) | [![GitHub stars](https://img.shields.io/github/stars/ljwztc/CLIP-Driven-Universal-Model.svg?logo=github&label=Stars)](https://github.com/ljwztc/CLIP-Driven-Universal-Model) |
| 11 | Swin UNETR & CLIP | CityU | 80.2 | [![arXiv](https://img.shields.io/badge/arXiv-2301.00785-b31b1b.svg)](https://arxiv.org/abs/2301.00785) | [![GitHub stars](https://img.shields.io/github/stars/ljwztc/CLIP-Driven-Universal-Model.svg?logo=github&label=Stars)](https://github.com/ljwztc/CLIP-Driven-Universal-Model) |
| 12 | SegVol | BAAI | 79.3 | [![arXiv](https://img.shields.io/badge/arXiv-2311.13385-b31b1b.svg)](https://arxiv.org/abs/2311.13385) | [![GitHub stars](https://img.shields.io/github/stars/BAAI-DCAI/SegVol.svg?logo=github&label=Stars)](https://github.com/BAAI-DCAI/SegVol) |
| 13 | UCTransNet | Northeastern University | 77.8 | [![arXiv](https://img.shields.io/badge/arXiv-2211.11537-b31b1b.svg)]([https://arxiv.org/abs/2211.11537](https://arxiv.org/pdf/2109.04335)) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)]([https://github.com/McGregorWwww/UCTransNet](https://github.com/McGregorWwww/UCTransNet)) |
| 14 | Swin UNETR | NVIDIA | 76.9 | [![arXiv](https://img.shields.io/badge/arXiv-2211.11537-b31b1b.svg)](https://arxiv.org/abs/2211.11537) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)](https://github.com/Project-MONAI/tutorials) |
| 15 | UNesT | NVIDIA | 75.1 | [![arXiv](https://img.shields.io/badge/arXiv-2303.10745-b31b1b.svg)](https://arxiv.org/abs/2303.10745) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)](https://github.com/Project-MONAI/tutorials) |
| 16 | UNETR | NVIDIA | 74.7 | [![arXiv](https://img.shields.io/badge/arXiv-2111.04004-b31b1b.svg)](https://arxiv.org/abs/2111.04004) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)](https://github.com/Project-MONAI/tutorials) |
| 17 | SAM-Adapter | Duke | 49.4 | [![arXiv](https://img.shields.io/badge/arXiv-2404.09957-b31b1b.svg)](https://arxiv.org/abs/2404.09957) | [![GitHub stars](https://img.shields.io/github/stars/mazurowski-lab/finetune-SAM.svg?logo=github&label=Stars)](https://github.com/mazurowski-lab/finetune-SAM) |



</div>
</details>

<details>
<summary style="margin-left: 25px;">KidneyL - Diff-UNet 🏆 </summary>
<div style="margin-left: 25px;">

| rank | model  | organization | DSC | paper | github |
|:---|:---|:---|:---:|:---:|:---:|
| 🏆 | Diff-UNet | HKUST | 91.9 | [![arXiv](https://img.shields.io/badge/arXiv-2303.10326-b31b1b.svg)](https://arxiv.org/abs/2303.10326) | [![GitHub stars](https://img.shields.io/github/stars/ge-xing/Diff-UNet.svg?logo=github&label=Stars)](https://github.com/ge-xing/Diff-UNet) |
| 2 | MedFormer | Rutgers | 91.9 | [![arXiv](https://img.shields.io/badge/arXiv-2203.00131-b31b1b.svg)](https://arxiv.org/abs/2203.00131) | [![GitHub stars](https://img.shields.io/github/stars/yhygao/CBIM-Medical-Image-Segmentation.svg?logo=github&label=Stars)](https://github.com/yhygao/CBIM-Medical-Image-Segmentation) |
| 3 | nnU-Net ResEncL | DKFZ | 91.9 | [![arXiv](https://img.shields.io/badge/arXiv-1809.10486-b31b1b.svg)](https://arxiv.org/abs/1809.10486) | [![GitHub stars](https://img.shields.io/github/stars/MIC-DKFZ/nnUNet.svg?logo=github&label=Stars)](https://github.com/MIC-DKFZ/nnUNet) |
| 4 | STU-Net-B | Shanghai AI Lab | 91.9 | [![arXiv](https://img.shields.io/badge/arXiv-2304.06716-b31b1b.svg)](https://arxiv.org/pdf/2304.06716) | [![GitHub stars](https://img.shields.io/github/stars/uni-medical/STU-Net.svg?logo=github&label=Stars)](https://github.com/uni-medical/STU-Net) |
| 5 | nnU-Net U-Net | DKFZ | 91.9 | [![arXiv](https://img.shields.io/badge/arXiv-1809.10486-b31b1b.svg)](https://arxiv.org/abs/1809.10486) | [![GitHub stars](https://img.shields.io/github/stars/MIC-DKFZ/nnUNet.svg?logo=github&label=Stars)](https://github.com/MIC-DKFZ/nnUNet) |
| 6 | LHU-Net | UR | 91.8 | [![arXiv](https://img.shields.io/badge/arXiv-2404.05102-b31b1b.svg)](https://arxiv.org/abs/2404.05102) | [![GitHub stars](https://img.shields.io/github/stars/xmindflow/LHUNet.svg?logo=github&label=Stars)](https://github.com/xmindflow/LHUNet) |
| 7 | MedNeXt | DKFZ | 91.8 | [![arXiv](https://img.shields.io/badge/arXiv-2303.09975-b31b1b.svg)](https://arxiv.org/pdf/2303.09975) | [![GitHub stars](https://img.shields.io/github/stars/MIC-DKFZ/MedNeXt.svg?logo=github&label=Stars)](https://github.com/MIC-DKFZ/MedNeXt) |
| 8 | SegVol | BAAI | 91.8 | [![arXiv](https://img.shields.io/badge/arXiv-2311.13385-b31b1b.svg)](https://arxiv.org/abs/2311.13385) | [![GitHub stars](https://img.shields.io/github/stars/BAAI-DCAI/SegVol.svg?logo=github&label=Stars)](https://github.com/BAAI-DCAI/SegVol) |
| 9 | UniSeg | NPU | 91.5 | [![arXiv](https://img.shields.io/badge/arXiv-2304.03493-b31b1b.svg)](https://arxiv.org/abs/2304.03493) | [![GitHub stars](https://img.shields.io/github/stars/yeerwen/UniSeg.svg?logo=github&label=Stars)](https://github.com/yeerwen/UniSeg) |
| 10 | U-Net & CLIP | CityU | 91.1 | [![arXiv](https://img.shields.io/badge/arXiv-2301.00785-b31b1b.svg)](https://arxiv.org/abs/2301.00785) | [![GitHub stars](https://img.shields.io/github/stars/ljwztc/CLIP-Driven-Universal-Model.svg?logo=github&label=Stars)](https://github.com/ljwztc/CLIP-Driven-Universal-Model) |
| 11 | Swin UNETR & CLIP | CityU | 91.0 | [![arXiv](https://img.shields.io/badge/arXiv-2301.00785-b31b1b.svg)](https://arxiv.org/abs/2301.00785) | [![GitHub stars](https://img.shields.io/github/stars/ljwztc/CLIP-Driven-Universal-Model.svg?logo=github&label=Stars)](https://github.com/ljwztc/CLIP-Driven-Universal-Model) |
| 12 | UNesT | NVIDIA | 90.1 | [![arXiv](https://img.shields.io/badge/arXiv-2303.10745-b31b1b.svg)](https://arxiv.org/abs/2303.10745) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)](https://github.com/Project-MONAI/tutorials) |
| 13 | Swin UNETR | NVIDIA | 89.7 | [![arXiv](https://img.shields.io/badge/arXiv-2211.11537-b31b1b.svg)](https://arxiv.org/abs/2211.11537) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)](https://github.com/Project-MONAI/tutorials) |
| 14 | NexToU | HIT | 89.6 | [![arXiv](https://img.shields.io/badge/arXiv-2305.15911-b31b1b.svg)](https://arxiv.org/abs/2305.15911) | [![GitHub stars](https://img.shields.io/github/stars/PengchengShi1220/NexToU.svg?logo=github&label=Stars)](https://github.com/PengchengShi1220/NexToU) |
| 15 | UNETR | NVIDIA | 89.2 | [![arXiv](https://img.shields.io/badge/arXiv-2111.04004-b31b1b.svg)](https://arxiv.org/abs/2111.04004) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)](https://github.com/Project-MONAI/tutorials) |
| 16 | SAM-Adapter | Duke | 87.3 | [![arXiv](https://img.shields.io/badge/arXiv-2404.09957-b31b1b.svg)](https://arxiv.org/abs/2404.09957) | [![GitHub stars](https://img.shields.io/github/stars/mazurowski-lab/finetune-SAM.svg?logo=github&label=Stars)](https://github.com/mazurowski-lab/finetune-SAM) |
| 17 | UCTransNet | Northeastern University | 86.9 | [![arXiv](https://img.shields.io/badge/arXiv-2211.11537-b31b1b.svg)]([https://arxiv.org/abs/2211.11537](https://arxiv.org/pdf/2109.04335)) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)]([https://github.com/McGregorWwww/UCTransNet](https://github.com/McGregorWwww/UCTransNet)) |
                                                                                                      
</div>
</details>

<details>
<summary style="margin-left: 25px;">KidneyR - Diff-UNet 🏆 </summary>
<div style="margin-left: 25px;">

| rank | model  | organization | DSC | paper | github |
|:---|:---|:---|:---:|:---:|:---:|
| 🏆 | Diff-UNet | HKUST | 92.8 | [![arXiv](https://img.shields.io/badge/arXiv-2303.10326-b31b1b.svg)](https://arxiv.org/abs/2303.10326) | [![GitHub stars](https://img.shields.io/github/stars/ge-xing/Diff-UNet.svg?logo=github&label=Stars)](https://github.com/ge-xing/Diff-UNet) |
| 2 | MedFormer | Rutgers | 92.8 | [![arXiv](https://img.shields.io/badge/arXiv-2203.00131-b31b1b.svg)](https://arxiv.org/abs/2203.00131) | [![GitHub stars](https://img.shields.io/github/stars/yhygao/CBIM-Medical-Image-Segmentation.svg?logo=github&label=Stars)](https://github.com/yhygao/CBIM-Medical-Image-Segmentation) |
| 3 | nnU-Net U-Net | DKFZ | 92.7 | [![arXiv](https://img.shields.io/badge/arXiv-1809.10486-b31b1b.svg)](https://arxiv.org/abs/1809.10486) | [![GitHub stars](https://img.shields.io/github/stars/MIC-DKFZ/nnUNet.svg?logo=github&label=Stars)](https://github.com/MIC-DKFZ/nnUNet) |
| 4 | MedNeXt | DKFZ | 92.6 | [![arXiv](https://img.shields.io/badge/arXiv-2303.09975-b31b1b.svg)](https://arxiv.org/pdf/2303.09975) | [![GitHub stars](https://img.shields.io/github/stars/MIC-DKFZ/MedNeXt.svg?logo=github&label=Stars)](https://github.com/MIC-DKFZ/MedNeXt) |
| 5 | nnU-Net ResEncL | DKFZ | 92.6 | [![arXiv](https://img.shields.io/badge/arXiv-1809.10486-b31b1b.svg)](https://arxiv.org/abs/1809.10486) | [![GitHub stars](https://img.shields.io/github/stars/MIC-DKFZ/nnUNet.svg?logo=github&label=Stars)](https://github.com/MIC-DKFZ/nnUNet) |
| 6 | LHU-Net | UR | 92.5 | [![arXiv](https://img.shields.io/badge/arXiv-2404.05102-b31b1b.svg)](https://arxiv.org/abs/2404.05102) | [![GitHub stars](https://img.shields.io/github/stars/xmindflow/LHUNet.svg?logo=github&label=Stars)](https://github.com/xmindflow/LHUNet) |
| 7 | STU-Net-B | Shanghai AI Lab | 92.5 | [![arXiv](https://img.shields.io/badge/arXiv-2304.06716-b31b1b.svg)](https://arxiv.org/pdf/2304.06716) | [![GitHub stars](https://img.shields.io/github/stars/uni-medical/STU-Net.svg?logo=github&label=Stars)](https://github.com/uni-medical/STU-Net) |
| 8 | SegVol | BAAI | 92.5 | [![arXiv](https://img.shields.io/badge/arXiv-2311.13385-b31b1b.svg)](https://arxiv.org/abs/2311.13385) | [![GitHub stars](https://img.shields.io/github/stars/BAAI-DCAI/SegVol.svg?logo=github&label=Stars)](https://github.com/BAAI-DCAI/SegVol) |
| 9 | UniSeg | NPU | 92.2 | [![arXiv](https://img.shields.io/badge/arXiv-2304.03493-b31b1b.svg)](https://arxiv.org/abs/2304.03493) | [![GitHub stars](https://img.shields.io/github/stars/yeerwen/UniSeg.svg?logo=github&label=Stars)](https://github.com/yeerwen/UniSeg) |
| 10 | U-Net & CLIP | CityU | 91.9 | [![arXiv](https://img.shields.io/badge/arXiv-2301.00785-b31b1b.svg)](https://arxiv.org/abs/2301.00785) | [![GitHub stars](https://img.shields.io/github/stars/ljwztc/CLIP-Driven-Universal-Model.svg?logo=github&label=Stars)](https://github.com/ljwztc/CLIP-Driven-Universal-Model) |
| 11 | Swin UNETR & CLIP | CityU | 91.7 | [![arXiv](https://img.shields.io/badge/arXiv-2301.00785-b31b1b.svg)](https://arxiv.org/abs/2301.00785) | [![GitHub stars](https://img.shields.io/github/stars/ljwztc/CLIP-Driven-Universal-Model.svg?logo=github&label=Stars)](https://github.com/ljwztc/CLIP-Driven-Universal-Model) |
| 12 | UNesT | NVIDIA | 90.9 | [![arXiv](https://img.shields.io/badge/arXiv-2303.10745-b31b1b.svg)](https://arxiv.org/abs/2303.10745) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)](https://github.com/Project-MONAI/tutorials) |
| 13 | SAM-Adapter | Duke | 90.4 | [![arXiv](https://img.shields.io/badge/arXiv-2404.09957-b31b1b.svg)](https://arxiv.org/abs/2404.09957) | [![GitHub stars](https://img.shields.io/github/stars/mazurowski-lab/finetune-SAM.svg?logo=github&label=Stars)](https://github.com/mazurowski-lab/finetune-SAM) |
| 14 | NexToU | HIT | 90.1 | [![arXiv](https://img.shields.io/badge/arXiv-2305.15911-b31b1b.svg)](https://arxiv.org/abs/2305.15911) | [![GitHub stars](https://img.shields.io/github/stars/PengchengShi1220/NexToU.svg?logo=github&label=Stars)](https://github.com/PengchengShi1220/NexToU) |
| 15 | UNETR | NVIDIA | 90.1 | [![arXiv](https://img.shields.io/badge/arXiv-2111.04004-b31b1b.svg)](https://arxiv.org/abs/2111.04004) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)](https://github.com/Project-MONAI/tutorials) |
| 16 | Swin UNETR | NVIDIA | 89.8 | [![arXiv](https://img.shields.io/badge/arXiv-2211.11537-b31b1b.svg)](https://arxiv.org/abs/2211.11537) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)](https://github.com/Project-MONAI/tutorials) |
| 17 | UCTransNet | Northeastern University | 86.5 | [![arXiv](https://img.shields.io/badge/arXiv-2211.11537-b31b1b.svg)]([https://arxiv.org/abs/2211.11537](https://arxiv.org/pdf/2109.04335)) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)]([https://github.com/McGregorWwww/UCTransNet](https://github.com/McGregorWwww/UCTransNet)) |

</div>
</details>

<details>
<summary style="margin-left: 25px;">Liver - MedFormer 🏆 </summary>
<div style="margin-left: 25px;">

| rank | model  | organization | DSC | paper | github |
|:---|:---|:---|:---:|:---:|:---:|
| 🏆 | MedFormer | Rutgers | 96.4 | [![arXiv](https://img.shields.io/badge/arXiv-2203.00131-b31b1b.svg)](https://arxiv.org/abs/2203.00131) | [![GitHub stars](https://img.shields.io/github/stars/yhygao/CBIM-Medical-Image-Segmentation.svg?logo=github&label=Stars)](https://github.com/yhygao/CBIM-Medical-Image-Segmentation) |
| 2 | MedNeXt | DKFZ | 96.3 | [![arXiv](https://img.shields.io/badge/arXiv-2303.09975-b31b1b.svg)](https://arxiv.org/pdf/2303.09975) | [![GitHub stars](https://img.shields.io/github/stars/MIC-DKFZ/MedNeXt.svg?logo=github&label=Stars)](https://github.com/MIC-DKFZ/MedNeXt) |
| 3 | nnU-Net ResEncL | DKFZ | 96.3 | [![arXiv](https://img.shields.io/badge/arXiv-1809.10486-b31b1b.svg)](https://arxiv.org/abs/1809.10486) | [![GitHub stars](https://img.shields.io/github/stars/MIC-DKFZ/nnUNet.svg?logo=github&label=Stars)](https://github.com/MIC-DKFZ/nnUNet) |
| 4 | LHU-Net | UR | 96.2 | [![arXiv](https://img.shields.io/badge/arXiv-2404.05102-b31b1b.svg)](https://arxiv.org/abs/2404.05102) | [![GitHub stars](https://img.shields.io/github/stars/xmindflow/LHUNet.svg?logo=github&label=Stars)](https://github.com/xmindflow/LHUNet) |
| 5 | nnU-Net U-Net | DKFZ | 96.2 | [![arXiv](https://img.shields.io/badge/arXiv-1809.10486-b31b1b.svg)](https://arxiv.org/abs/1809.10486) | [![GitHub stars](https://img.shields.io/github/stars/MIC-DKFZ/nnUNet.svg?logo=github&label=Stars)](https://github.com/MIC-DKFZ/nnUNet) |
| 6 | Diff-UNet | HKUST | 96.2 | [![arXiv](https://img.shields.io/badge/arXiv-2303.10326-b31b1b.svg)](https://arxiv.org/abs/2303.10326) | [![GitHub stars](https://img.shields.io/github/stars/ge-xing/Diff-UNet.svg?logo=github&label=Stars)](https://github.com/ge-xing/Diff-UNet) |
| 7 | STU-Net-B | Shanghai AI Lab | 96.2 | [![arXiv](https://img.shields.io/badge/arXiv-2304.06716-b31b1b.svg)](https://arxiv.org/pdf/2304.06716) | [![GitHub stars](https://img.shields.io/github/stars/uni-medical/STU-Net.svg?logo=github&label=Stars)](https://github.com/uni-medical/STU-Net) |
| 8 | UniSeg | NPU | 96.1 | [![arXiv](https://img.shields.io/badge/arXiv-2304.03493-b31b1b.svg)](https://arxiv.org/abs/2304.03493) | [![GitHub stars](https://img.shields.io/github/stars/yeerwen/UniSeg.svg?logo=github&label=Stars)](https://github.com/yeerwen/UniSeg) |
| 9 | U-Net & CLIP | CityU | 96.0 | [![arXiv](https://img.shields.io/badge/arXiv-2301.00785-b31b1b.svg)](https://arxiv.org/abs/2301.00785) | [![GitHub stars](https://img.shields.io/github/stars/ljwztc/CLIP-Driven-Universal-Model.svg?logo=github&label=Stars)](https://github.com/ljwztc/CLIP-Driven-Universal-Model) |
| 10 | SegVol | BAAI | 96.0 | [![arXiv](https://img.shields.io/badge/arXiv-2311.13385-b31b1b.svg)](https://arxiv.org/abs/2311.13385) | [![GitHub stars](https://img.shields.io/github/stars/BAAI-DCAI/SegVol.svg?logo=github&label=Stars)](https://github.com/BAAI-DCAI/SegVol) |
| 11 | Swin UNETR & CLIP | CityU | 95.8 | [![arXiv](https://img.shields.io/badge/arXiv-2301.00785-b31b1b.svg)](https://arxiv.org/abs/2301.00785) | [![GitHub stars](https://img.shields.io/github/stars/ljwztc/CLIP-Driven-Universal-Model.svg?logo=github&label=Stars)](https://github.com/ljwztc/CLIP-Driven-Universal-Model) |
| 12 | NexToU | HIT | 95.7 | [![arXiv](https://img.shields.io/badge/arXiv-2305.15911-b31b1b.svg)](https://arxiv.org/abs/2305.15911) | [![GitHub stars](https://img.shields.io/github/stars/PengchengShi1220/NexToU.svg?logo=github&label=Stars)](https://github.com/PengchengShi1220/NexToU) |
| 13 | SAM-Adapter | Duke | 94.1 | [![arXiv](https://img.shields.io/badge/arXiv-2404.09957-b31b1b.svg)](https://arxiv.org/abs/2404.09957) | [![GitHub stars](https://img.shields.io/github/stars/mazurowski-lab/finetune-SAM.svg?logo=github&label=Stars)](https://github.com/mazurowski-lab/finetune-SAM) |
| 14 | UNesT | NVIDIA | 95.3 | [![arXiv](https://img.shields.io/badge/arXiv-2303.10745-b31b1b.svg)](https://arxiv.org/abs/2303.10745) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)](https://github.com/Project-MONAI/tutorials) |
| 15 | Swin UNETR | NVIDIA | 95.2 | [![arXiv](https://img.shields.io/badge/arXiv-2211.11537-b31b1b.svg)](https://arxiv.org/abs/2211.11537) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)](https://github.com/Project-MONAI/tutorials) |
| 16 | UNETR | NVIDIA | 95.0 | [![arXiv](https://img.shields.io/badge/arXiv-2111.04004-b31b1b.svg)](https://arxiv.org/abs/2111.04004) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)](https://github.com/Project-MONAI/tutorials) |
| 17 | UCTransNet | Northeastern University | 93.6 | [![arXiv](https://img.shields.io/badge/arXiv-2211.11537-b31b1b.svg)]([https://arxiv.org/abs/2211.11537](https://arxiv.org/pdf/2109.04335)) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)]([https://github.com/McGregorWwww/UCTransNet](https://github.com/McGregorWwww/UCTransNet)) |

</div>
</details>

<details>
<summary style="margin-left: 25px;">Pancreas - MedNeXt 🏆 </summary>
<div style="margin-left: 25px;">

| rank | model  | organization | DSC | paper | github |
|:---|:---|:---|:---:|:---:|:---:|
| 🏆 | MedNeXt | DKFZ | 83.3 | [![arXiv](https://img.shields.io/badge/arXiv-2303.09975-b31b1b.svg)](https://arxiv.org/pdf/2303.09975) | [![GitHub stars](https://img.shields.io/github/stars/MIC-DKFZ/MedNeXt.svg?logo=github&label=Stars)](https://github.com/MIC-DKFZ/MedNeXt) |
| 2 | STU-Net-B | Shanghai AI Lab | 83.2 | [![arXiv](https://img.shields.io/badge/arXiv-2304.06716-b31b1b.svg)](https://arxiv.org/pdf/2304.06716) | [![GitHub stars](https://img.shields.io/github/stars/uni-medical/STU-Net.svg?logo=github&label=Stars)](https://github.com/uni-medical/STU-Net) |
| 3 | MedFormer | Rutgers | 83.1 | [![arXiv](https://img.shields.io/badge/arXiv-2203.00131-b31b1b.svg)](https://arxiv.org/abs/2203.00131) | [![GitHub stars](https://img.shields.io/github/stars/yhygao/CBIM-Medical-Image-Segmentation.svg?logo=github&label=Stars)](https://github.com/yhygao/CBIM-Medical-Image-Segmentation) |
| 4 | nnU-Net ResEncL | DKFZ | 82.9 | [![arXiv](https://img.shields.io/badge/arXiv-1809.10486-b31b1b.svg)](https://arxiv.org/abs/1809.10486) | [![GitHub stars](https://img.shields.io/github/stars/MIC-DKFZ/nnUNet.svg?logo=github&label=Stars)](https://github.com/MIC-DKFZ/nnUNet) |
| 5 | UniSeg | NPU | 82.7 | [![arXiv](https://img.shields.io/badge/arXiv-2304.03493-b31b1b.svg)](https://arxiv.org/abs/2304.03493) | [![GitHub stars](https://img.shields.io/github/stars/yeerwen/UniSeg.svg?logo=github&label=Stars)](https://github.com/yeerwen/UniSeg) |
| 6 | nnU-Net U-Net | DKFZ | 82.3 | [![arXiv](https://img.shields.io/badge/arXiv-1809.10486-b31b1b.svg)](https://arxiv.org/abs/1809.10486) | [![GitHub stars](https://img.shields.io/github/stars/MIC-DKFZ/nnUNet.svg?logo=github&label=Stars)](https://github.com/MIC-DKFZ/nnUNet) |
| 7 | Diff-UNet | HKUST | 81.9 | [![arXiv](https://img.shields.io/badge/arXiv-2303.10326-b31b1b.svg)](https://arxiv.org/abs/2303.10326) | [![GitHub stars](https://img.shields.io/github/stars/ge-xing/Diff-UNet.svg?logo=github&label=Stars)](https://github.com/ge-xing/Diff-UNet) |
| 8 | LHU-Net | UR | 81.0 | [![arXiv](https://img.shields.io/badge/arXiv-2404.05102-b31b1b.svg)](https://arxiv.org/abs/2404.05102) | [![GitHub stars](https://img.shields.io/github/stars/xmindflow/LHUNet.svg?logo=github&label=Stars)](https://github.com/xmindflow/LHUNet) |
| 9 | U-Net & CLIP | CityU | 80.8 | [![arXiv](https://img.shields.io/badge/arXiv-2301.00785-b31b1b.svg)](https://arxiv.org/abs/2301.00785) | [![GitHub stars](https://img.shields.io/github/stars/ljwztc/CLIP-Driven-Universal-Model.svg?logo=github&label=Stars)](https://github.com/ljwztc/CLIP-Driven-Universal-Model) |
| 10 | Swin UNETR & CLIP | CityU | 80.2 | [![arXiv](https://img.shields.io/badge/arXiv-2301.00785-b31b1b.svg)](https://arxiv.org/abs/2301.00785) | [![GitHub stars](https://img.shields.io/github/stars/ljwztc/CLIP-Driven-Universal-Model.svg?logo=github&label=Stars)](https://github.com/ljwztc/CLIP-Driven-Universal-Model) |
| 11 | NexToU | HIT | 80.2 | [![arXiv](https://img.shields.io/badge/arXiv-2305.15911-b31b1b.svg)](https://arxiv.org/abs/2305.15911) | [![GitHub stars](https://img.shields.io/github/stars/PengchengShi1220/NexToU.svg?logo=github&label=Stars)](https://github.com/PengchengShi1220/NexToU) |
| 12 | SegVol | BAAI | 79.1 | [![arXiv](https://img.shields.io/badge/arXiv-2311.13385-b31b1b.svg)](https://arxiv.org/abs/2311.13385) | [![GitHub stars](https://img.shields.io/github/stars/BAAI-DCAI/SegVol.svg?logo=github&label=Stars)](https://github.com/BAAI-DCAI/SegVol) |
| 13 | UNesT | NVIDIA | 76.2 | [![arXiv](https://img.shields.io/badge/arXiv-2303.10745-b31b1b.svg)](https://arxiv.org/abs/2303.10745) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)](https://github.com/Project-MONAI/tutorials) |
| 14 | Swin UNETR | NVIDIA | 75.6 | [![arXiv](https://img.shields.io/badge/arXiv-2211.11537-b31b1b.svg)](https://arxiv.org/abs/2211.11537) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)](https://github.com/Project-MONAI/tutorials) |
| 15 | UNETR | NVIDIA | 72.3 | [![arXiv](https://img.shields.io/badge/arXiv-2111.04004-b31b1b.svg)](https://arxiv.org/abs/2111.04004) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)](https://github.com/Project-MONAI/tutorials) |
| 16 | UCTransNet | Northeastern University | 59.0 | [![arXiv](https://img.shields.io/badge/arXiv-2211.11537-b31b1b.svg)]([https://arxiv.org/abs/2211.11537](https://arxiv.org/pdf/2109.04335)) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)]([https://github.com/McGregorWwww/UCTransNet](https://github.com/McGregorWwww/UCTransNet)) |
| 17 | SAM-Adapter | Duke | 50.2 | [![arXiv](https://img.shields.io/badge/arXiv-2404.09957-b31b1b.svg)](https://arxiv.org/abs/2404.09957) | [![GitHub stars](https://img.shields.io/github/stars/mazurowski-lab/finetune-SAM.svg?logo=github&label=Stars)](https://github.com/mazurowski-lab/finetune-SAM) |


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
| 4 | nnU-Net U-Net | DKFZ | 81.0 | [![arXiv](https://img.shields.io/badge/arXiv-1809.10486-b31b1b.svg)](https://arxiv.org/abs/1809.10486) | [![GitHub stars](https://img.shields.io/github/stars/MIC-DKFZ/nnUNet.svg?logo=github&label=Stars)](https://github.com/MIC-DKFZ/nnUNet) |
| 5 | Diff-UNet | HKUST | 80.8 | [![arXiv](https://img.shields.io/badge/arXiv-2303.10326-b31b1b.svg)](https://arxiv.org/abs/2303.10326) | [![GitHub stars](https://img.shields.io/github/stars/ge-xing/Diff-UNet.svg?logo=github&label=Stars)](https://github.com/ge-xing/Diff-UNet) |
| 6 | MedFormer | Rutgers | 80.7 | [![arXiv](https://img.shields.io/badge/arXiv-2203.00131-b31b1b.svg)](https://arxiv.org/abs/2203.00131) | [![GitHub stars](https://img.shields.io/github/stars/yhygao/CBIM-Medical-Image-Segmentation.svg?logo=github&label=Stars)](https://github.com/yhygao/CBIM-Medical-Image-Segmentation) |
| 7 | nnU-Net ResEncL | DKFZ | 80.5 | [![arXiv](https://img.shields.io/badge/arXiv-1809.10486-b31b1b.svg)](https://arxiv.org/abs/1809.10486) | [![GitHub stars](https://img.shields.io/github/stars/MIC-DKFZ/nnUNet.svg?logo=github&label=Stars)](https://github.com/MIC-DKFZ/nnUNet) |
| 8 | LHU-Net | UR | 79.4 | [![arXiv](https://img.shields.io/badge/arXiv-2404.05102-b31b1b.svg)](https://arxiv.org/abs/2404.05102) | [![GitHub stars](https://img.shields.io/github/stars/xmindflow/LHUNet.svg?logo=github&label=Stars)](https://github.com/xmindflow/LHUNet) |
| 9 | U-Net & CLIP | CityU | 78.5 | [![arXiv](https://img.shields.io/badge/arXiv-2301.00785-b31b1b.svg)](https://arxiv.org/abs/2301.00785) | [![GitHub stars](https://img.shields.io/github/stars/ljwztc/CLIP-Driven-Universal-Model.svg?logo=github&label=Stars)](https://github.com/ljwztc/CLIP-Driven-Universal-Model) |
| 10 | NexToU | HIT | 78.1 | [![arXiv](https://img.shields.io/badge/arXiv-2305.15911-b31b1b.svg)](https://arxiv.org/abs/2305.15911) | [![GitHub stars](https://img.shields.io/github/stars/PengchengShi1220/NexToU.svg?logo=github&label=Stars)](https://github.com/PengchengShi1220/NexToU) |
| 11 | SegVol | BAAI | 77.8 | [![arXiv](https://img.shields.io/badge/arXiv-2311.13385-b31b1b.svg)](https://arxiv.org/abs/2311.13385) | [![GitHub stars](https://img.shields.io/github/stars/BAAI-DCAI/SegVol.svg?logo=github&label=Stars)](https://github.com/BAAI-DCAI/SegVol) |
| 12 | Swin UNETR & CLIP | CityU | 76.8 | [![arXiv](https://img.shields.io/badge/arXiv-2301.00785-b31b1b.svg)](https://arxiv.org/abs/2301.00785) | [![GitHub stars](https://img.shields.io/github/stars/ljwztc/CLIP-Driven-Universal-Model.svg?logo=github&label=Stars)](https://github.com/ljwztc/CLIP-Driven-Universal-Model) |
| 13 | Swin UNETR | NVIDIA | 75.4 | [![arXiv](https://img.shields.io/badge/arXiv-2211.11537-b31b1b.svg)](https://arxiv.org/abs/2211.11537) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)](https://github.com/Project-MONAI/tutorials) |
| 14 | UNesT | NVIDIA | 74.4 | [![arXiv](https://img.shields.io/badge/arXiv-2303.10745-b31b1b.svg)](https://arxiv.org/abs/2303.10745) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)](https://github.com/Project-MONAI/tutorials) |
| 15 | UNETR | NVIDIA | 71.5 | [![arXiv](https://img.shields.io/badge/arXiv-2111.04004-b31b1b.svg)](https://arxiv.org/abs/2111.04004) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)](https://github.com/Project-MONAI/tutorials) |
| 15 | UCTransNet | Northeastern University | 68.1 | [![arXiv](https://img.shields.io/badge/arXiv-2211.11537-b31b1b.svg)]([https://arxiv.org/abs/2211.11537](https://arxiv.org/pdf/2109.04335)) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)]([https://github.com/McGregorWwww/UCTransNet](https://github.com/McGregorWwww/UCTransNet)) |
| 17 | SAM-Adapter | Duke | 48.0 | [![arXiv](https://img.shields.io/badge/arXiv-2404.09957-b31b1b.svg)](https://arxiv.org/abs/2404.09957) | [![GitHub stars](https://img.shields.io/github/stars/mazurowski-lab/finetune-SAM.svg?logo=github&label=Stars)](https://github.com/mazurowski-lab/finetune-SAM) |




</div>
</details>

<details>
<summary style="margin-left: 25px;">Spleen - MedFormer 🏆 </summary>
<div style="margin-left: 25px;">

| rank | model  | organization | DSC | paper | github |
|:---|:---|:---|:---:|:---:|:---:|
| 🏆 | MedFormer | Rutgers | 95.5 | [![arXiv](https://img.shields.io/badge/arXiv-2203.00131-b31b1b.svg)](https://arxiv.org/abs/2203.00131) | [![GitHub stars](https://img.shields.io/github/stars/yhygao/CBIM-Medical-Image-Segmentation.svg?logo=github&label=Stars)](https://github.com/yhygao/CBIM-Medical-Image-Segmentation) |
| 2 | nnU-Net ResEncL | DKFZ | 95.2 | [![arXiv](https://img.shields.io/badge/arXiv-1809.10486-b31b1b.svg)](https://arxiv.org/abs/1809.10486) | [![GitHub stars](https://img.shields.io/github/stars/MIC-DKFZ/nnUNet.svg?logo=github&label=Stars)](https://github.com/MIC-DKFZ/nnUNet) |
| 3 | MedNeXt | DKFZ | 95.2 | [![arXiv](https://img.shields.io/badge/arXiv-2303.09975-b31b1b.svg)](https://arxiv.org/pdf/2303.09975) | [![GitHub stars](https://img.shields.io/github/stars/MIC-DKFZ/MedNeXt.svg?logo=github&label=Stars)](https://github.com/MIC-DKFZ/MedNeXt) |
| 4 | nnU-Net U-Net | DKFZ | 95.1 | [![arXiv](https://img.shields.io/badge/arXiv-1809.10486-b31b1b.svg)](https://arxiv.org/abs/1809.10486) | [![GitHub stars](https://img.shields.io/github/stars/MIC-DKFZ/nnUNet.svg?logo=github&label=Stars)](https://github.com/MIC-DKFZ/nnUNet) |
| 5 | STU-Net-B | Shanghai AI Lab | 95.1 | [![arXiv](https://img.shields.io/badge/arXiv-2304.06716-b31b1b.svg)](https://arxiv.org/pdf/2304.06716) | [![GitHub stars](https://img.shields.io/github/stars/uni-medical/STU-Net.svg?logo=github&label=Stars)](https://github.com/uni-medical/STU-Net) |
| 6 | Diff-UNet | HKUST | 95.0 | [![arXiv](https://img.shields.io/badge/arXiv-2303.10326-b31b1b.svg)](https://arxiv.org/abs/2303.10326) | [![GitHub stars](https://img.shields.io/github/stars/ge-xing/Diff-UNet.svg?logo=github&label=Stars)](https://github.com/ge-xing/Diff-UNet) |
| 7 | LHU-Net | UR | 94.9 | [![arXiv](https://img.shields.io/badge/arXiv-2404.05102-b31b1b.svg)](https://arxiv.org/abs/2404.05102) | [![GitHub stars](https://img.shields.io/github/stars/xmindflow/LHUNet.svg?logo=github&label=Stars)](https://github.com/xmindflow/LHUNet) |
| 8 | UniSeg | NPU | 94.9 | [![arXiv](https://img.shields.io/badge/arXiv-2304.03493-b31b1b.svg)](https://arxiv.org/abs/2304.03493) | [![GitHub stars](https://img.shields.io/github/stars/yeerwen/UniSeg.svg?logo=github&label=Stars)](https://github.com/yeerwen/UniSeg) |
| 9 | SegVol | BAAI | 94.5 | [![arXiv](https://img.shields.io/badge/arXiv-2311.13385-b31b1b.svg)](https://arxiv.org/abs/2311.13385) | [![GitHub stars](https://img.shields.io/github/stars/BAAI-DCAI/SegVol.svg?logo=github&label=Stars)](https://github.com/BAAI-DCAI/SegVol) |
| 10 | NexToU | HIT | 94.7 | [![arXiv](https://img.shields.io/badge/arXiv-2305.15911-b31b1b.svg)](https://arxiv.org/abs/2305.15911) | [![GitHub stars](https://img.shields.io/github/stars/PengchengShi1220/NexToU.svg?logo=github&label=Stars)](https://github.com/PengchengShi1220/NexToU) |
| 11 | U-Net & CLIP | CityU | 94.3 | [![arXiv](https://img.shields.io/badge/arXiv-2301.00785-b31b1b.svg)](https://arxiv.org/abs/2301.00785) | [![GitHub stars](https://img.shields.io/github/stars/ljwztc/CLIP-Driven-Universal-Model.svg?logo=github&label=Stars)](https://github.com/ljwztc/CLIP-Driven-Universal-Model) |
| 12 | Swin UNETR & CLIP | CityU | 94.1 | [![arXiv](https://img.shields.io/badge/arXiv-2301.00785-b31b1b.svg)](https://arxiv.org/abs/2301.00785) | [![GitHub stars](https://img.shields.io/github/stars/ljwztc/CLIP-Driven-Universal-Model.svg?logo=github&label=Stars)](https://github.com/ljwztc/CLIP-Driven-Universal-Model) |
| 13 | UNesT | NVIDIA | 93.2 | [![arXiv](https://img.shields.io/badge/arXiv-2303.10745-b31b1b.svg)](https://arxiv.org/abs/2303.10745) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)](https://github.com/Project-MONAI/tutorials) |
| 14 | Swin UNETR | NVIDIA | 92.7 | [![arXiv](https://img.shields.io/badge/arXiv-2211.11537-b31b1b.svg)](https://arxiv.org/abs/2211.11537) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)](https://github.com/Project-MONAI/tutorials) |
| 15 | UNETR | NVIDIA | 91.7 | [![arXiv](https://img.shields.io/badge/arXiv-2111.04004-b31b1b.svg)](https://arxiv.org/abs/2111.04004) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)](https://github.com/Project-MONAI/tutorials) |
| 16 | SAM-Adapter | Duke | 90.5 | [![arXiv](https://img.shields.io/badge/arXiv-2404.09957-b31b1b.svg)](https://arxiv.org/abs/2404.09957) | [![GitHub stars](https://img.shields.io/github/stars/mazurowski-lab/finetune-SAM.svg?logo=github&label=Stars)](https://github.com/mazurowski-lab/finetune-SAM) |
| 17 | UCTransNet | Northeastern University | 90.2 | [![arXiv](https://img.shields.io/badge/arXiv-2211.11537-b31b1b.svg)]([https://arxiv.org/abs/2211.11537](https://arxiv.org/pdf/2109.04335)) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)]([https://github.com/McGregorWwww/UCTransNet](https://github.com/McGregorWwww/UCTransNet)) |


</div>
</details>

<details>
<summary style="margin-left: 25px;">Stomach - STU-Net-B 🏆 </summary>
<div style="margin-left: 25px;">

| rank | model  | organization | DSC | paper | github |
|:---|:---|:---|:---:|:---:|:---:|
| 🏆 | STU-Net-B | Shanghai AI Lab | 93.5 | [![arXiv](https://img.shields.io/badge/arXiv-2304.06716-b31b1b.svg)](https://arxiv.org/pdf/2304.06716) | [![GitHub stars](https://img.shields.io/github/stars/uni-medical/STU-Net.svg?logo=github&label=Stars)](https://github.com/uni-medical/STU-Net) |
| 2 | MedNeXt | DKFZ | 93.5 | [![arXiv](https://img.shields.io/badge/arXiv-2303.09975-b31b1b.svg)](https://arxiv.org/pdf/2303.09975) | [![GitHub stars](https://img.shields.io/github/stars/MIC-DKFZ/MedNeXt.svg?logo=github&label=Stars)](https://github.com/MIC-DKFZ/MedNeXt) |
| 3 | nnU-Net ResEncL | DKFZ | 93.4 | [![arXiv](https://img.shields.io/badge/arXiv-1809.10486-b31b1b.svg)](https://arxiv.org/abs/1809.10486) | [![GitHub stars](https://img.shields.io/github/stars/MIC-DKFZ/nnUNet.svg?logo=github&label=Stars)](https://github.com/MIC-DKFZ/nnUNet) |
| 4 | MedFormer | Rutgers | 93.4 | [![arXiv](https://img.shields.io/badge/arXiv-2203.00131-b31b1b.svg)](https://arxiv.org/abs/2203.00131) | [![GitHub stars](https://img.shields.io/github/stars/yhygao/CBIM-Medical-Image-Segmentation.svg?logo=github&label=Stars)](https://github.com/yhygao/CBIM-Medical-Image-Segmentation) |
| 5 | UniSeg | NPU | 93.3 | [![arXiv](https://img.shields.io/badge/arXiv-2304.03493-b31b1b.svg)](https://arxiv.org/abs/2304.03493) | [![GitHub stars](https://img.shields.io/github/stars/yeerwen/UniSeg.svg?logo=github&label=Stars)](https://github.com/yeerwen/UniSeg) |
| 6 | nnU-Net U-Net | DKFZ | 93.3 | [![arXiv](https://img.shields.io/badge/arXiv-1809.10486-b31b1b.svg)](https://arxiv.org/abs/1809.10486) | [![GitHub stars](https://img.shields.io/github/stars/MIC-DKFZ/nnUNet.svg?logo=github&label=Stars)](https://github.com/MIC-DKFZ/nnUNet) |
| 7 | Diff-UNet | HKUST | 93.1 | [![arXiv](https://img.shields.io/badge/arXiv-2303.10326-b31b1b.svg)](https://arxiv.org/abs/2303.10326) | [![GitHub stars](https://img.shields.io/github/stars/ge-xing/Diff-UNet.svg?logo=github&label=Stars)](https://github.com/ge-xing/Diff-UNet) |
| 8 | LHU-Net | UR | 93.0 | [![arXiv](https://img.shields.io/badge/arXiv-2404.05102-b31b1b.svg)](https://arxiv.org/abs/2404.05102) | [![GitHub stars](https://img.shields.io/github/stars/xmindflow/LHUNet.svg?logo=github&label=Stars)](https://github.com/xmindflow/LHUNet) |
| 9 | NexToU | HIT | 92.7 | [![arXiv](https://img.shields.io/badge/arXiv-2305.15911-b31b1b.svg)](https://arxiv.org/abs/2305.15911) | [![GitHub stars](https://img.shields.io/github/stars/PengchengShi1220/NexToU.svg?logo=github&label=Stars)](https://github.com/PengchengShi1220/NexToU) |
| 10 | SegVol | BAAI | 92.5 | [![arXiv](https://img.shields.io/badge/arXiv-2311.13385-b31b1b.svg)](https://arxiv.org/abs/2311.13385) | [![GitHub stars](https://img.shields.io/github/stars/BAAI-DCAI/SegVol.svg?logo=github&label=Stars)](https://github.com/BAAI-DCAI/SegVol) |
| 11 | U-Net & CLIP | CityU | 92.4 | [![arXiv](https://img.shields.io/badge/arXiv-2301.00785-b31b1b.svg)](https://arxiv.org/abs/2301.00785) | [![GitHub stars](https://img.shields.io/github/stars/ljwztc/CLIP-Driven-Universal-Model.svg?logo=github&label=Stars)](https://github.com/ljwztc/CLIP-Driven-Universal-Model) |
| 12 | Swin UNETR & CLIP | CityU | 92.2 | [![arXiv](https://img.shields.io/badge/arXiv-2301.00785-b31b1b.svg)](https://arxiv.org/abs/2301.00785) | [![GitHub stars](https://img.shields.io/github/stars/ljwztc/CLIP-Driven-Universal-Model.svg?logo=github&label=Stars)](https://github.com/ljwztc/CLIP-Driven-Universal-Model) |
| 13 | UNesT | NVIDIA | 90.9 | [![arXiv](https://img.shields.io/badge/arXiv-2303.10745-b31b1b.svg)](https://arxiv.org/abs/2303.10745) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)](https://github.com/Project-MONAI/tutorials) |
| 14 | Swin UNETR | NVIDIA | 90.5 | [![arXiv](https://img.shields.io/badge/arXiv-2211.11537-b31b1b.svg)](https://arxiv.org/abs/2211.11537) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)](https://github.com/Project-MONAI/tutorials) |
| 15 | UNETR | NVIDIA | 88.8 | [![arXiv](https://img.shields.io/badge/arXiv-2111.04004-b31b1b.svg)](https://arxiv.org/abs/2111.04004) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)](https://github.com/Project-MONAI/tutorials) |
| 16 | SAM-Adapter | Duke | 88.0 | [![arXiv](https://img.shields.io/badge/arXiv-2404.09957-b31b1b.svg)](https://arxiv.org/abs/2404.09957) | [![GitHub stars](https://img.shields.io/github/stars/mazurowski-lab/finetune-SAM.svg?logo=github&label=Stars)](https://github.com/mazurowski-lab/finetune-SAM) |
| 17 | UCTransNet | Northeastern University | 81.9 | [![arXiv](https://img.shields.io/badge/arXiv-2211.11537-b31b1b.svg)]([https://arxiv.org/abs/2211.11537](https://arxiv.org/pdf/2109.04335)) | [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/tutorials.svg?logo=github&label=Stars)]([https://github.com/McGregorWwww/UCTransNet](https://github.com/McGregorWwww/UCTransNet)) |


</div>
</details>

# Touchstone 1.0 Dataset
 
### Training set
- Touchstone 1.0: [AbdomenAtlas1.0Mini](https://huggingface.co/datasets/AbdomenAtlas/AbdomenAtlas1.0Mini) (*N*=5,195)
- Touchstone 2.0: [AbdomenAtlas1.1Mini](https://huggingface.co/datasets/AbdomenAtlas/_AbdomenAtlas1.1Mini) (*N*=9,262)

### Test set 

- Proprietary [JHH dataset](https://www.sciencedirect.com/science/article/pii/S2211568419301391) (*N*=5,172)
- Public [TotalSegmentator V2](https://github.com/wasserth/TotalSegmentator) dataset (*N*=1,228)

<div align="center">
  <img src="utils/fig_metadata.png" alt="metadata" width="100%">
</div>

*Figure 1. Metadata distribution in the test set.*

# Touchstone 1.0 Model

> [!NOTE]
> We are releasing the trained AI models evaluated in Touchstone right here. Stay tuned!

| rank | model                  | average DSC | parameter | infer. speed | download |
|:---|:---|:---|:---|:---|:---|
| 🏆 | MedNeXt               | 89.2        | 61.8M     | ★☆☆☆☆           |          |
| 🏆 | MedFormer             |  89.0       | 38.5M     | ★★★☆☆           |          |
| 3 | STU-Net-B             | 89.0        | 58.3M     | ★★☆☆☆           | <a href="https://github.com/uni-medical/STU-Net/tree/main/AbdomenAtlas" style="margin: 2px;"> <img alt="checkpoint" src="https://img.shields.io/badge/⚡_checkpoint-instruction-FF4040?style=flat-square&labelColor=2C3E50" style="display: inline-block; vertical-align: middle;"/> |
| 4 | nnU-Net U-Net         | 88.9        | 102.0M    | ★★★★☆           | <a href="https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/dataset_conversion/Dataset224_AbdomenAtlas1.0.py" style="margin: 2px;"> <img alt="checkpoint" src="https://img.shields.io/badge/⚡_checkpoint-instruction-FF4040?style=flat-square&labelColor=2C3E50" style="display: inline-block; vertical-align: middle;"/> |
| 5 | nnU-Net ResEncL       | 88.8        | 102.0M    | ★★★★☆           | <a href="https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/dataset_conversion/Dataset224_AbdomenAtlas1.0.py" style="margin: 2px;"> <img alt="checkpoint" src="https://img.shields.io/badge/⚡_checkpoint-instruction-FF4040?style=flat-square&labelColor=2C3E50" style="display: inline-block; vertical-align: middle;"/> |
| 6 | UniSeg                | 88.8        | 31.0M     | ☆☆☆☆☆           |          |
| 7 | Diff-UNet             | 88.5        | 434.0M    | ★★★☆☆           |          |
| 8 | LHU-Net               | 88.0        | 8.6M      | ★★★★★           | <a href="https://github.com/xmindflow/LHUNet/tree/main/AbdomenAtlas1.0" style="margin: 2px;"> <img alt="checkpoint" src="https://img.shields.io/badge/⚡_checkpoint-instruction-FF4040?style=flat-square&labelColor=2C3E50" style="display: inline-block; vertical-align: middle;"/> |
| 9 | NexToU                | 87.8        | 81.9M     | ★★★★☆           | <a href="https://github.com/PengchengShi1220/NexToU/blob/NexToU_nnunetv2/NexToU_Touchstone%20Benchmark.md" style="margin: 2px;"> <img alt="checkpoint" src="https://img.shields.io/badge/⚡_checkpoint-instruction-FF4040?style=flat-square&labelColor=2C3E50" style="display: inline-block; vertical-align: middle;"/> |
| 10   | SegVol               | 87.1        | 181.0M    | ★★★★☆           | <a href="https://github.com/BAAI-DCAI/SegVol/blob/main/readme_AbdomenAtlas.md" style="margin: 2px;"> <img alt="checkpoint" src="https://img.shields.io/badge/⚡_checkpoint-instruction-FF4040?style=flat-square&labelColor=2C3E50" style="display: inline-block; vertical-align: middle;"/>         |
| 11  | U-Net & CLIP         | 87.1        | 19.1M     | ★★★☆☆           |          |
| 12  | Swin UNETR & CLIP    | 86.7        | 62.2M     | ★★★☆☆           |          |
| 13  | Swin UNETR           | 84.8        | 72.8M     | ★★★★★           |          |
| 14  | UNesT                | 84.9        | 87.2M     | ★★★★★           |          |
| 15  | UNETR                | 83.3        | 101.8M    | ★★★★★           |          |
| 16  | UCTransNet           | 81.1        | 68.0M     | ★★★★☆           |          |
| 17  | SAM-Adapter          | 73.4        | 11.6M     | ★★★★☆           |  <a href="https://drive.google.com/drive/folders/1VuIwR-STOjD5NUJtU40aMDmJOQX_onZw" style="margin: 2px;"> <img alt="checkpoint" src="https://img.shields.io/badge/⚡_checkpoint-instruction-FF4040?style=flat-square&labelColor=2C3E50" style="display: inline-block; vertical-align: middle;"/>         |

# Evaluation Code

<details>
<summary style="margin-left: 25px;">Click to expand </summary>
<div style="margin-left: 25px;">

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

If you are including a new segmentation model in the evaluation, organize its results following the structure in the CSV files inside the folders totalsegmentator_results and dapatlas_results (see below). Also, include its name in the model_ranking list in [plot/PlotGroup.py](plot/PlotGroup.py).

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
</details>


# Citation

Please cite the following papers if you find our study helpful.

```
@article{bassi2024touchstone,
  title={Touchstone Benchmark: Are We on the Right Way for Evaluating AI Algorithms for Medical Segmentation?},
  author={Bassi, Pedro RAS and Li, Wenxuan and Tang, Yucheng and Isensee, Fabian and Wang, Zifu and Chen, Jieneng and Chou, Yu-Cheng and Kirchhoff, Yannick and Rokuss, Maximilian and Huang, Ziyan and Ye, Jin and He, Junjun and Wald, Tassilo and Ulrich, Constantin and Baumgartner, Michael and Roy, Saikat and Maier-Hein, Klaus H. and Jaeger, Paul and Ye, Yiwen and Xie, Yutong and Zhang, Jianpeng and Chen, Ziyang and Xia, Yong and Xing, Zhaohu and Zhu, Lei and Sadegheih, Yousef and Bozorgpour, Afshin and Kumari, Pratibha and Azad, Reza and Merhof, Dorit and Shi, Pengcheng and Ma, Ting and Du, Yuxin and Bai, Fan and Huang, Tiejun and Zhao, Bo and Wang, Haonan and Li, Xiaomeng and Gu, Hanxue and Dong, Haoyu and Yang, Jichen and Mazurowski, Maciej A. and Gupta, Saumya and Wu, Linshan and Zhuang, Jiaxin and Chen, Hao and Roth, Holger and Xu, Daguang and Blaschko, Matthew B. and Decherchi, Sergio and Cavalli, Andrea and Yuille, Alan L. and Zhou, Zongwei},
  journal={Conference on Neural Information Processing Systems},
  year={2024},
  utl={https://github.com/MrGiovanni/Touchstone}
}

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
  <img src="utils/partner_logo.png" alt="Touchstone Benchmark" width="1200">
</div>
