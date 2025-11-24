import PlotGroup as pg
from argparse import Namespace
from PlotGroup import read_models_and_groups, create_long_format_dataframe
import argparse
import scipy.stats as stats
from itertools import combinations
import numpy as np
import pandas as pd
import warnings
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import matplotlib.pyplot as plt

# Function to perform one-sided Wilcoxon signed-rank test
def wilcoxon_one_sided(x, y):
    try:
        res = wilcoxon(x,y, alternative='greater',nan_policy='raise')
    except ValueError as e:
        if str(e) == "zero_method 'wilcox' and 'pratt' do not work if x - y is zero for all elements.":
            return np.array([100.0])
        else:
            # Re-raise the exception if it doesn't match
            raise
    return res.pvalue

# Suppress specific warning
warnings.filterwarnings("ignore", category=UserWarning)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Statistical tests')
    parser.add_argument('--organ', type=str, help='model name',default='mean')
    parser.add_argument('--title', type=str, help='title',default='')
    parser.add_argument('--ckpt_root', type=str, help='Path to the directory containing model result CSV files',default='/run/media/pedro/e911bf59-fe8e-4ddb-8938-5dc4f40b094f/Checkpoints/Metrics/TotalSegmentator/')
    parser.add_argument('--nsd', action='store_true', help='Plot dice if not set', default=False)
    parser.add_argument('--test_set_only', action='store_true', help='Plot dice if not set', default=False)
    parser.add_argument('--split_path', default='/run/media/pedro/e911bf59-fe8e-4ddb-8938-5dc4f40b094f/metaTotalSeg.csv', help='Location of TotalSegmentator metadata')
    return parser.parse_args()


def rank(results, args):
    """Rank models by median performance, optimized for both 'mean' and specific organs.
    
    Uses efficient computation and avoids try-except in loop.
    """
    means = {}
    
    for model, df in results.items():
        if args.organ == 'mean':
            # Check once if 'Average' column exists
            if 'Average' in df.columns:
                means[model] = df.drop(columns=['Average']).mean(numeric_only=True, axis=1).median()
            else:
                means[model] = df.mean(numeric_only=True, axis=1).median()
        else:
            means[model] = df[args.organ].median()
    
    return sorted(means, key=means.get, reverse=True)
    
def allign(df1, df2):
    """Align two dataframes by common 'name' values, optimized for performance.
    
    Removes NaN values, duplicates, and sorts by 'name' to ensure proper alignment.
    """
    # Step 1: Remove rows with NaN values and duplicates
    df1_clean = df1.dropna().drop_duplicates(subset=['name']).reset_index(drop=True)
    df2_clean = df2.dropna().drop_duplicates(subset=['name']).reset_index(drop=True)
    
    # Step 2: Find intersection using set operations for efficiency
    common_names = set(df1_clean['name']) & set(df2_clean['name'])
    
    # Step 3 & 4: Filter and sort in one operation per dataframe
    df1_subset = (df1_clean[df1_clean['name'].isin(common_names)]
                  .sort_values(by='name')
                  .reset_index(drop=True))
    df2_subset = (df2_clean[df2_clean['name'].isin(common_names)]
                  .sort_values(by='name')
                  .reset_index(drop=True))
    
    # Verify alignment
    assert (df1_subset['name'] == df2_subset['name']).all(), "DataFrames not properly aligned"
    
    # Return without 'name' column
    return df1_subset.drop(columns=['name']), df2_subset.drop(columns=['name'])
    
def HeatmapOfSignificance(args,ax=None):
    flag=(ax is None)
    
    #Use for only per-group comparisons
    p_args = Namespace()
    p_args.group_name='all'
    p_args.ckpt_root=args.ckpt_root
    #p_args.group_root=args.group_root
    p_args.nsd=args.nsd
    p_args.organ=args.organ
    p_args.th=10
    p_args.test_set_only=args.test_set_only
    p_args.mean_and_best=False
    p_args.just_mean=False
    p_args.split_path=args.split_path
    results, groups_lists, order, num_groups, num_algos = read_models_and_groups(p_args)
    groups = rank(results, args)
    
    # Extract relevant organ data for each model - optimize with comprehension
    for model in results:
        if args.organ == 'mean':
            if 'Average' in results[model].columns:
                results[model] = results[model][['name', 'Average']]
            else:
                # Create Average column if it doesn't exist
                results[model] = results[model].copy()
                results[model]['Average'] = results[model].drop(columns=['name']).mean(axis=1)
                results[model] = results[model][['name', 'Average']]
        else:
            results[model] = results[model][['name', args.organ]]
    
    # Generate all pairwise comparisons (bidirectional)
    comparisons = [(g1, g2) for g1, g2 in combinations(groups, 2)]
    
    # Perform pair-wise tests - optimize to avoid intermediate lists
    p_values = []
    comparison_pairs = []
    
    for (group1, group2) in comparisons:
        df1, df2 = allign(results[group1], results[group2])
        # Test both directions
        p1 = wilcoxon_one_sided(df1, df2)
        p_values.append(p1.item())
        comparison_pairs.append((group1, group2))
        
        p2 = wilcoxon_one_sided(df2, df1)
        p_values.append(p2.item())
        comparison_pairs.append((group2, group1))
    
    # Correct for multiple comparisons using Holm's method
    _, corrected_p_values, _, _ = multipletests(p_values, method='holm')
    
    # Create a DataFrame to store the results
    significance_matrix = pd.DataFrame(np.nan, index=list(reversed(groups)), columns=groups)
    
    # Fill in the matrix with corrected p-values - vectorized approach
    for (group1, group2), p in zip(comparison_pairs, corrected_p_values):
        significance_matrix.loc[group2, group1] = 1 if p < 0.05 else -1
    
    # Create a custom color map
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['blue', 'white', 'yellow'])
    
    # Plotting the significance map using a heatmap
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        plt.sca(ax)
        
    ax = sns.heatmap(significance_matrix, annot=False, cmap=cmap, center=0,
                     xticklabels=groups, yticklabels=list(reversed(groups)), linewidths=0.5, linecolor='gray',
                     cbar=False,ax=ax)

    # Diagonal line to separate significant and non-significant areas
    plt.plot([0, len(groups)], [len(groups), 0], color='black', lw=1)

    plt.title(args.title)
    #plt.xlabel('Algorithm')
    #plt.ylabel('Algorithm')
    # Rotate x-axis labels by 45 degrees
    #ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right')
    #ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right')
    
    
    if flag:
        plt.show()

def HeatmapOfSignificanceNoCorrection(args,ax=None):
    flag=(ax is None)
    
    #Use for only per-group comparisons
    p_args = Namespace()
    p_args.group_name='all'
    p_args.ckpt_root=args.ckpt_root
    #p_args.group_root=args.group_root
    p_args.nsd=args.nsd
    p_args.organ=args.organ
    p_args.th=10
    p_args.test_set_only=args.test_set_only
    p_args.mean_and_best=False
    p_args.just_mean=False
    p_args.split_path=args.split_path
    results, groups_lists, order, num_groups, num_algos = read_models_and_groups(p_args)
    groups=rank(results,args)
    for model in results:#get only organ we want
        if args.organ=='mean':
            try:
                results[model]['mean']=results[model].drop(columns=['Average','name']).mean(axis=1)
            except:
                results[model]['mean']=results[model].drop(columns=['name']).mean(axis=1)
            results[model]=results[model][['name', 'mean']]
        else:
            results[model] = results[model][['name', args.organ]]
    
    # Generate all pairwise comparisons (bidirectional)
    comparisons = [(g1, g2) for g1, g2 in combinations(groups, 2)]
    
    # Perform pair-wise tests
    p_values = []
    comparison_pairs = []
    
    for (group1, group2) in comparisons:
        df1, df2 = allign(results[group1], results[group2])
        # Test both directions
        p1 = wilcoxon_one_sided(df1, df2)
        p_values.append(p1.item())
        comparison_pairs.append((group1, group2))
        print(f'{group1} > {group2} p: {p1.item()}')
        
        p2 = wilcoxon_one_sided(df2, df1)
        p_values.append(p2.item())
        comparison_pairs.append((group2, group1))
        print(f'{group2} > {group1} p: {p2.item()}')
    
    # No correction applied in this version
    corrected_p_values = p_values
    
    # Create significance matrix
    significance_matrix = pd.DataFrame(np.nan, index=list(reversed(groups)), columns=groups)
    
    # Fill in the matrix with p-values
    for (group1, group2), p in zip(comparison_pairs, corrected_p_values):
        significance_matrix.loc[group2, group1] = 1 if p < 0.05 else -1
    
    # Create a custom color map
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['blue', 'white', 'yellow'])
    
    # Plotting the significance map using a heatmap
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        plt.sca(ax)
    
    ax = sns.heatmap(significance_matrix, annot=False, cmap=cmap, center=0,
                     xticklabels=groups, yticklabels=list(reversed(groups)), 
                     linewidths=0.5, linecolor='gray', cbar=False, ax=ax)

    # Diagonal line to separate significant and non-significant areas
    plt.plot([0, len(groups)], [len(groups), 0], color='black', lw=1)

    plt.title(args.title)
    #plt.xlabel('Algorithm')
    #plt.ylabel('Algorithm')
    # Rotate x-axis labels by 45 degrees
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    
    if flag:
        plt.show()      
        
if __name__ == "__main__":
    args = parse_arguments()
    HeatmapOfSignificance(args)