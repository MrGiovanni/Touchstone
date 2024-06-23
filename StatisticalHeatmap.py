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


def rank(results,args):
    #changes begin here
    means={}    
    for model in results:
        if args.organ=='mean':
            #means[model]=results[model]['Average'].mean()
            try:
                means[model]=results[model].drop(
                    columns=['Average']).mean(numeric_only=True,axis=1).median()
            except:
                means[model]=results[model].mean(numeric_only=True,axis=1).median()
        else:
            means[model]=results[model][args.organ].median()
    sorted_keys_descending = sorted(means, key=means.get, reverse=True)
    #print(means)
    #changes end here
    
    return sorted_keys_descending
    
def allign(df1,df2):
    #print(df1,df2)
    #Step 1: Remove rows with NaN values
    df1_clean = df1.dropna().reset_index(drop=True).drop_duplicates(subset=['name'])
    df2_clean = df2.dropna().reset_index(drop=True).drop_duplicates(subset=['name'])
    #print(df1_clean)

    # Step 2: Find the intersection of 'name' values
    common_names = set(df1_clean['name']).intersection(set(df2_clean['name']))

    # Step 3: Subset both DataFrames to only include rows with these common 'name' values
    df1_subset = df1_clean[df1_clean['name'].isin(common_names)].reset_index(drop=True)
    df2_subset = df2_clean[df2_clean['name'].isin(common_names)].reset_index(drop=True)

    # Step 4: Ensure that both DataFrames have the same order of rows by sorting
    df1_subset = df1_subset.sort_values(by='name').reset_index(drop=True)
    df2_subset = df2_subset.sort_values(by='name').reset_index(drop=True)

    # Verify that both DataFrames have the same order of 'name' values
    #print(df1_subset['name'],df2_subset['name'])
    assert (df1_subset['name']==df2_subset['name']).all()
    #print(df1_subset['name'],df2_subset['name'])
    df1_subset,df2_subset=df1_subset.drop(columns=['name']),df2_subset.drop(columns=['name'])
    #print(df1_subset,df2_subset)
    return df1_subset,df2_subset
    
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
    groups=rank(results,args)
    for model in results:#get only organ we want
        if args.organ=='mean':
            #try:
            #    results[model]['mean']=results[model].drop(columns=['Average','name']).mean(axis=1)
            #except:
            #    results[model]['mean']=results[model].drop(columns=['name']).mean(axis=1)
            #results[model]=results[model][['name', 'mean']]
            try:
                results[model]=results[model][['name', 'Average']]
            except:
                #print('Problem: no Average in ',model)
                #print(results[model])
                results[model]['Average']=results[model].drop(columns=['name']).mean(axis=1)
                #print(results[model].drop(columns=['name']).mean(axis=1))
                results[model]=results[model][['name', 'Average']]
                #print(results[model])
        else:
            results[model]=results[model][['name', args.organ]]
    
    
    comparisons = list(combinations(groups, 2))
    
    # Perform pair-wise tests
    p_values = []
    tmp=[]
    for (group1, group2) in comparisons:
        #print(group1,group2)
        df1, df2=allign(results[group1], results[group2])
        p1 = wilcoxon_one_sided(df1, df2)
        p_values.append(p1.item())
        tmp.append((group1, group2))
        p2 = wilcoxon_one_sided(df2, df1)
        p_values.append(p2.item())
        tmp.append((group2, group1))
    comparisons=tmp
        
    #print(p_values)

    for i,p in enumerate(p_values,0):
        group1, group2=comparisons[i]
        #print(group1,'>', group2,'p:',p)
        
    # Correct for multiple comparisons using Holm's method
    _, corrected_p_values, _, _ = multipletests(p_values, method='holm')
    #print(len(p_values),len(corrected_p_values))
    #corrected_p_values=p_values
    
    for i,p in enumerate(corrected_p_values,0):
        group1, group2=comparisons[i]
        #print(group1,'>', group2,'p:',p)
        
        
    # Create a DataFrame to store the results
    significance_matrix = pd.DataFrame(np.nan, index=list(reversed(groups)), columns=groups)
 

    # Fill in the matrix with corrected p-values
    for (group1, group2), p in zip(comparisons, corrected_p_values):
        if p < 0.05:
            significance_matrix.loc[group2, group1] = 1  # Yellow 
            #significance_matrix.loc[group1, group2] = -1  # Blue 
        else:
            #significance_matrix.loc[group1, group2] = -1
            significance_matrix.loc[group2, group1] = -1

    # Create a custom color map
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(['blue', 'white', 'yellow'])

    # Revert the order of the y-axis labels
    #reversed_groups = groups[::-1]

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
            results[model]=results[model][['name', args.organ]]
    
    
    comparisons = list(combinations(groups, 2))
    
   # Perform pair-wise tests
    p_values = []
    tmp=[]
    for (group1, group2) in comparisons:
        df1, df2=allign(results[group1], results[group2])
        p1 = wilcoxon_one_sided(df1, df2)
        p_values.append(p1.item())
        tmp.append((group1, group2))
        p2 = wilcoxon_one_sided(df2, df1)
        p_values.append(p2.item())
        tmp.append((group2, group1))
    comparisons=tmp
        
    #print(p_values)

    for i,p in enumerate(p_values,0):
        group1, group2=comparisons[i]
        print(group1,'>', group2,'p:',p)
        
    # Correct for multiple comparisons using Holm's method
    #p_crr={}
    #for model in groups:
    #    pc=[[comp,p_values[i]] for i,comp in enumerate(comparisons,0) if comp[0]==model]
    #    p=[pval for comp,pval in pc]
    #    _, corrected_p_values, _, _ = multipletests(p, method='holm')
    #    
    #    for i,comp in enumerate(comparisons,0):
    #        for j,(comp2,_) in enumerate(pc,0):
    #            if comp2==comp:
    #                p_values[i]=corrected_p_values[j]
    #    
    #    corrected_p_values=p_values
        
    corrected_p_values=p_values
    # Create a DataFrame to store the results
    significance_matrix = pd.DataFrame(np.nan, index=list(reversed(groups)), columns=groups)
 

    # Fill in the matrix with corrected p-values
    for (group1, group2), p in zip(comparisons, corrected_p_values):
        if p < 0.05:
            significance_matrix.loc[group2, group1] = 1  # Yellow 
            #significance_matrix.loc[group1, group2] = -1  # Blue 
        else:
            #significance_matrix.loc[group1, group2] = -1
            significance_matrix.loc[group2, group1] = -1

    # Create a custom color map
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(['blue', 'white', 'yellow'])

    # Revert the order of the y-axis labels
    #reversed_groups = groups[::-1]

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
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    
    if flag:
        plt.show()      
        
if __name__ == "__main__":
    args = parse_arguments()
    HeatmapOfSignificance(args)