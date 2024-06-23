# Touchstone

In the Touchstone project, we invited AI creators to train their models on AbdomenAtlas, the world’s largest CT dataset. We benchmarked the state-of-the-art in medical segmentation, with some key contributions: a training set (AbdomenAtlas) with an unprecedented size and diversity; large scale out-of-distribution test sets; directly inviting AI creators to train their algorithms, thus fostering comparison fairness, and highlighting innovative methods; and leveraging multiple evaluation perspectives (e.g., accuracy, speed, and performance consistency across diverse demographic groups).

## Data
Training: 

AbdomenAtlas: https://huggingface.co/datasets/AbdomenAtlas/AbdomenAtlas1.0Mini

Testing:

TotalSegmentator: https://github.com/wasserth/TotalSegmentator

DAPAtlas: https://github.com/alexanderjaus/AtlasDataset

JHH: private

## Requirements
Python (3.8), PyTorch (2.2.1), numpy (1.19.5), nibabel (5.2.1), scipy (1.8.1), monai (1.3.0), pandas (2.2.1), matplotlib (3.1.3), seaborn (0.11.1), statsmodels (0.13.5), statannotations (0.6.0)

## Usage
**1- Evaluate predictions to get dice and NSD scores**, this step creates the per-sample csv files in DAPAtlas, TotalSegmentator and JHH:
```
#below, provide the path to the dataset with annotations, the path to the predictions saved in #step 1, and the name of the checkpoint (arbitrary)
python EvalCPU.py --pred_path /path/to/predictions/ \
--dataset_path /path/to/dataset/ \
--checkpoint_name checkpoint_name --num_workers 20 --dataset_name dataset_name
```

**2- Create group lists**

From the dataset’s metadata, create lists of samples for each sub-group you wish to analyze. These lists should be lists of strings saved as .pt (use torch.save). Each string represents the name of a sample, formatted as the names in the column ‘name’ of the csv files generated in step 1. The name of the .pt file should indicate the group it refers to. Examples of these .pt files are inside ./plotsDAPAtlas and ./plotsTotalSegmentator. All sub-groups inside a “super-group” should also have the name of the larger group in the .pt file name. Eg., ages_18-29_autopet.pt and ages_30-39_autopet.pt are sub-groups of the super-group “ages”, and both files begin with “ages_”.

TotalSegmentatorMetadata.ipynb - Creates TotalSegmentator demographic groups lists

DAPAtlasMetadata.ipynb - Creates DAP Atlas demographic groups lists


**3- Create box plots and perform statistical tests by group**

From the per-sample csv files from Step 1, and the group lists (.pt) from step 2, you can already generate box plots, using the script PlotGroup.py. The same code performs statistical tests (Kruskal–Wallis tests, followed by post-hoc Mann-Whitney U Tests with Bonferroni correction). Usage:
```
python PlotGroup.py --ckpt_root path/to/ckpt/directory/containing/the/csvs \
                    --group_root /path/to/directory/containing/group/lists \
                    --group_name super_group --organ organ --stats
```

super_group: name of the super-group you want to analyze, like ‘ages’, or ‘diagnosis’. The code will search for all .pt files including this name inside /path/to/directory/containing/group/lists.
if you set “–group all”, the code will produce box plots for all samples

organ: the organ the plot will consider. if “--organ mean”, the code will produce plots for the mean performances across all organs.

Statistical tests are automatically performed when you run PlotGroup.py with the option  --stats. These tests DO NOT check for differences between AI algorithms, it only checks for differences between groups, for the same algorithm. E.g.: it compares the nnunet performance in patients with cancer and healthy ones, but it does not compare the nnunet to the segvol. To compare models, see next item.


**4- Statistical significance heatmaps**

Creates heatmaps (appendix) according to pair-wise statistical comparisons between AI algorithms. Uses one-sided Wilcoxon signed rank test with Holm’s adjustment for multiplicity at 5% significance level.
```
python StatisticalHeatmap.py --ckpt_root path/to/ckpt/directory/containing/the/csvs
```

## Future contents

Training and test scripts, as well as saved inferences and per-sample scores are not yet available, since releaseing them requires the authorization of each one of the AI creators. JHH per-sample metadata is not public.

Predictions/ - Saved predictions and labels for all models and datasets - not yet public

DAPAtlas/ - per-sample dice and nsd csv files - not yet public

JHH/ - per-sample dice and nsd csv files for JHH - not yet public

TotalSegmentator/ - per-sample dice and nsd csv files - not yet public
