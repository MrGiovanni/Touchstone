<div align="center">
  <img src="utils/AbdomenAtlas.png" alt="Touchstone Benchmark" width="200">
  <br>
  <h1><strong>Touchstone Benchmark</strong></h1>
</div>

We present Touchstone, a large-scale  medical segmentation benchmark based on annotated **5,195** CT volumes from **76** hospitals for training, and **6,933** CT volumes from **8** additional hospitals for testing. We invite AI inventor to train their models on AbdomenAtlas, and we independently evaluate the algorithms. We already collaborated with **14** influential research teams, and we remain accepting new submissions.


## How to Participate?

If you are the creator of an original segmentation algorithm, we invite you to participate!

- First edition of Touchstone: [benchmark tutorial](https://livejohnshopkins.sharepoint.com/:w:/r/sites/BodyMaps/Shared%20Documents/Collaboration/TutorialBenchmarkV1.docx?d=w7adb80080293445fb845f41103be6fc5&csf=1&web=1&e=iGafoB)

Please read the tutorial above and send us your trained checkpoint, along with your model's name and citation (email psalvad2@jh.edu). We will evaluate it add it to our future online leaderbord.

- Second edition of Touchstone: [call-for-benchmark](https://www.cs.jhu.edu/~zongwei/advert/Call4Benchmark.pdf)

We will soon launch the second edition of Touchstone. It will feature an even larger training dataset, with 9,262 CT volumes and 25 fully-annotated anatomical structures!

Please contact us (psalvad2@jh.edu) for more information and opportunities to collaborate in the future Touchstone publications.

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

<p align="center">
  <img src="utils/DiceTotalSegmentator.png" alt="Touchstone Benchmark" width="700">
</p>


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
cd notebooks
#check datasets' meatadata, create lists of demographic groups, and plots in Figure 1:
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=touchstone TotalSegmentatorMetadata.ipynb
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=touchstone DAPAtlasMetadata.ipynb
#results: plots are saved inside Touchstone/outputs/plotsTotalSegmentator/ and Touchstone/outputs/plotsDAPAtlas/

#per-group analysis: Figure 2 
cd ../plot
python AggregatedBoxplot.py --stats
#results: Touchstone/outputs/summary_groups.pdf

#statistical significance maps (Appendix D.2.3):
python PlotAllSignificanceMaps.py
python PlotAllSignificanceMaps.py --organs second_half
python PlotAllSignificanceMaps.py --nsd
python PlotAllSignificanceMaps.py --organs second_half --nsd
#results: Touchstone/outputs/heatmaps

#detailed box-plots (results per-group and per-organ) and corresponding statistical tests (figures in Appendix D.4 and D.5):
cd ../notebooks
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=touchstone GroupAnalysis.ipynb
#results: Touchstone/outputs/box_plots
```

##### 4. Custom Analysis

The csv results files in totalsegmentator_results/ and dapatlas_results/ contain per-sample dsc and nsd scores. Rich meatdata for each one of those samples (sex, age, scanner, diagnosis,...) are available in metaTotalSeg.csv and 'Clinical Metadata FDG PET_CT Lesions.csv', for TotalSegmentator and DAP Atlas, respectively. The code in TotalSegmentatorMetadata.ipynb and DAPAtlasMetadata.ipynb extracts this meatdata into simplfied group lists (e.g., a list of all samples representing male patients), and saves these lists in the folders plotsTotalSegmentator/ and plotsDAPAtlas/. You can modify the code to generate custom sample lists (e.g., all men aged 30-35). To compare a set of groups, the filenames of all lists in the set should begin with the same name. For example, comp1_list_a.pt, comp1_list_b.pt, comp1_list_C.pt can represent a set of 3 groups. Then, PlotGroup.py can draw boxplots and perform statistical tests comparing the AI algorithm's results (dsc and nsd) for the samples inside the different custom lists you created. In our example, you just just need to specify --group_name comp1 when running PlotGroup.py:

```bash
python utils/PlotGroup.py --ckpt_root totalsegmentator_results/ --group_root outputs/plotsTotalSegmentator/ --group_name comp1 --organ liver --stats
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