# Touchstone

Standard benchmarks often have underlying problems such as in-distribution and small-size test sets, oversimplified metrics, unfair comparisons, and short-term outcome pressure. Thus, good performance on standard benchmarks does not guarantee success in real-world scenarios. We address this misalignment issue with Touchstone, a large-scale collaborative benchmark for medical segmentation. **Touchstone is based on annotated CT datasets of unprecedented scale: 5,195 training volumes from 76 medical institutions around the world, and 6,933 testing volumes from 8 additional hospitals.** This extensive and diverse test set not only makes the benchmark results **more statistically meaningful** than existing ones, but also systematically **tests AI algorithms in varied out-of-distribution scenarios**. For coparison fainess, we **invite AI creators** to train their algorithms on the publicly available training set. Our team, as a third party, independently evaluates these algorithms on the test set (mostly private) and **reports their pros/cons from multiple perspectives**. We already collaborated with **14 influential research teams** in the field of medical segmentation, evaluating their AI algorithms. With long term commitment, **we remain accepting new submissions**. Moreover, we will soon launch the second edition of Touchstone, with an even larger training dataset and more annotated structures.


## How to Participate?

If you are the creator of an original segmentation algorithm, we invite you to participate!

- First edition of Touchstone: [benchmark tutorial](https://docs.google.com/document/d/1NxOdpVyEiRhbTOl_1IszsW7Ayij36imGfkG62ppQgk4/edit?usp=sharing)

Please read the tutorial above and send us your trained checkpoint, along with your model's name and citation (email pedro.salvadorbassi2@unibo.it). We will evaluate it add it to our future online leaderbord.

- Second edition of Touchstone: [call-for-benchmark](https://www.cs.jhu.edu/~zongwei/advert/Call4Benchmark.pdf)

The second edition of Touchstone will feature an even larger training dataset, with 9,262 CT volumes and 25 fully-annotated anatomical structures!

Please contact us (pedro.salvadorbassi2@unibo.it) for more information and opportunities to collaborate in the future Touchstone publications.

## Benchmark Setup

- Training set (first edition): [AbdomenAtlas1.0Mini](https://huggingface.co/datasets/AbdomenAtlas/AbdomenAtlas1.0Mini) (*N*=5,195)
- Training set (second edition): AbdomenAtlas1.1Mini (*N*=9,262)
- Test set #1: proprietary JHH dataset (*N*=5,172)
- Test set #2: public [TotalSegmentator V2](https://github.com/wasserth/TotalSegmentator) dataset (*N*=1,228)
- Test set #3: public [DAP Atlas](https://github.com/alexanderjaus/AtlasDataset) dataset (*N*=533)

## Evaluate Benchmark Checkpoints

*Note: currently for internal use*

There is a [tutorial]() providing all the checkpoints and test sets (ask Zongwei Zhou).


## Analyze Benchmark Results

We provide per-sample results for each checkpoint in test sets #2 and #3. These results are saved as csv files, structured as follows:

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

##### 1. Clone the GitHub repository

```bash
git clone https://github.com/MrGiovanni/Touchstone
cd Touchstone
```

##### 2. Create environments

```bash
conda env create -f environment.yml
source activate touchstone
python -m ipykernel install --user --name touchstone --display-name "touchstone"
```

##### 3. Reproduce analysis figures in our paper

```bash
#check datasets' meatadata, create lists of demographic groups, and plots in Figure 1:
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=touchstone TotalSegmentatorMetadata.ipynb
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=touchstone DAPAtlasMetadata.ipynb
#result: plots are saved inside plotsTotalSegmentator/ and plotsDAPAtlas/

#per-group analysis: Figure 2 
python PlotImage.py --stats
#result: summary_groups.pdf

#statistical significance maps (Appendix D.2.3):
python PlotAllHeatmaps.py
python PlotAllHeatmaps.py --organs second_half
python PlotAllHeatmaps.py --nsd
python PlotAllHeatmaps.py --organs second_half --nsd
#results: pdf images named significance_heatmaps_...

#detailed box-plots (results per-group and per-organ) and corresponding statistical tests (figures in Appendix D.4 and D.5):
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=touchstone GroupMetrics.ipynb
#results are saved inside folders named box_plots... and in pdf files beginning as "boxplots_per_class"
```

##### 4. Custom Analysis

The csv results files in totalsegmentator_results/ and dapatlas_results/ contain per-sample dsc and nsd scores. Rich meatdata for each one of those samples (sex, age, scanner, diagnosis,...) are available in metaTotalSeg.csv and 'Clinical Metadata FDG PET_CT Lesions.csv', for TotalSegmentator and DAP Atlas, respectively. The code in TotalSegmentatorMetadata.ipynb and DAPAtlasMetadata.ipynb extracts this meatdata into simplfied group lists (e.g., a list of all samples representing male patients), and saves these lists in the folders plotsTotalSegmentator/ and plotsDAPAtlas/. You can modify the code to generate custom sample lists (e.g., all men aged 30-35). To compare a set of groups, the filenames of all lists in the set should begin with the same name. For example, comp1_list_a.pt, comp1_list_b.pt, comp1_list_C.pt can represent a set of 3 groups. Then, PlotGroup.py can draw boxplots and perform statistical tests comparing the AI algorithm's results (dsc and nsd) for the samples inside the different custom lists you created. In our example, you just just need to specify --group_name comp1 when running PlotGroup.py:

```bash
python PlotGroup.py --ckpt_root totalsegmentator_results/ --group_root plotsTotalSegmentator/ \
                    --group_name comp1 --organ liver --stats
```

## Citation

If you use this code or use our datasets for your research, please cite our papers:

```
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

## Acknowledgement

This work was supported by the Lustgarten Foundation for Pancreatic Cancer Research and the McGovern Foundation. Paper content is covered by patents pending.