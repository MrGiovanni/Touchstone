import argparse
import pandas as pd
import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
from statannotations.Annotator import Annotator
from itertools import combinations

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate dot and boxplots with confidence intervals.')
    parser.add_argument('--ckpt_root', type=str, help='Path to the directory containing model result CSV files')
    parser.add_argument('--group_root', type=str, help='Path to the directory containing group sample lists')
    parser.add_argument('--group_name', type=str, help='Group name to filter the sample lists')
    parser.add_argument('--nsd', action='store_true', help='Plot dice if not set', default=False)
    parser.add_argument('--organ', help='Organ to plot, or mean', default='mean')
    parser.add_argument('--split_path', default='../utils/metaTotalSeg.csv', help='Location of TotalSegmentator metadata')
    parser.add_argument('--test_set_only', action='store_true', help='Tests only on totalSegmentator test set', default=False)
    parser.add_argument('--mean_and_best', action='store_true', help='Plot only results for nnU-Net and for average of all models', default=False)
    parser.add_argument('--just_mean', action='store_true', help='Plot only results for average of all models', default=False)
    parser.add_argument('--th', help='exclude groups with less samples than th',default=5)
    parser.add_argument('--orientation', help='Plot orientation, h or v or auto',default='auto')
    parser.add_argument('--stats', action='store_true', help='Plot only results for nnU-Net and for average of all models', default=False)
    parser.add_argument('--font', default=11)
    parser.add_argument('--fig_length', default='10')
    
    
    
    return parser.parse_args()

#gives model order in plot
model_ranking=['Average AI Algorithm','STU-Net','nnU-Net U-Net',
               'nnU-Net ResEncL','MedNeXt','UniSeg','Diff-UNet','LHU-Net','U-Net & CLIP', 
               'NexToU','SegResNet','SwinUNETR & CLIP','SegVol',
               'UCTransNet','UNEST','SwinUNETR','UNETR','SAM-Adapter','CleanNet']

#palette = sns.color_palette('bright', 30)
cmap = plt.get_cmap('tab20')
palette = [cmap(i % 20) for i in range(len(model_ranking))]
model_color_dict = dict(zip(model_ranking, palette))

def find_color(model):
    """Find color for a model, checking if it contains any model_ranking key.
    
    Optimized: Direct lookup if exact match exists, otherwise substring search.
    """
    # Try direct lookup first (O(1))
    if model in model_color_dict:
        return model_color_dict[model]
    
    # Fall back to substring search (for cases where model contains ranking name)
    for ranking_model, color in model_color_dict.items():
        if ranking_model in model:
            return color
    
    raise ValueError(f'Unrecognized model: {model}')

def Kruskal_Wallis(df):
    """Perform Kruskal-Wallis test followed by pairwise Mann-Whitney U tests.
    
    Optimized to cache grouped data and use vectorized operations where possible.
    """
    groups = df['Group'].unique()
    
    # Group once and convert to list - cache the result
    grouped_dict = {group: df[df['Group'] == group]['Value'].values 
                    for group in groups}
    
    # Prepare data for Kruskal-Wallis test
    values = list(grouped_dict.values())
    h_statistic, p_value = stats.kruskal(*values)
    
    if p_value > 0.05:
        return None  # no significant result
    
    # Post-hoc tests: Wilcoxon rank sum tests/Mann-Whitney U test
    results = []
    
    # Perform pairwise tests using cached grouped data
    for (group1, group2) in combinations(groups, 2):
        stat, p_value = stats.mannwhitneyu(grouped_dict[group1], grouped_dict[group2], 
                                           alternative='two-sided')
        results.append((group1, group2, p_value))
    
    # Convert results to a DataFrame
    results_df = pd.DataFrame(results, columns=['Group1', 'Group2', 'P-Value'])
    
    # Apply FDR correction using Benjamini-Hochberg method
    pvals_corrected = stats.false_discovery_control(results_df['P-Value'], method='bh')
    results_df.loc[:, 'P-Value Adjusted'] = pvals_corrected

    significant_results = results_df[results_df['P-Value Adjusted'] < 0.05]
    return significant_results


def Kruskal_Wallis_Pure(df):
    """Simplified Kruskal-Wallis test that only returns True/False for significance.
    
    Optimized version without post-hoc tests.
    """
    groups = df['Group'].unique()
    
    # More efficient: use groupby and get values directly
    grouped_data = df.groupby('Group')['Value'].apply(list)
    
    # Prepare the data for the Kruskal-Wallis test
    values = list(grouped_data)
    h_statistic, p_value = stats.kruskal(*values)
    
    return p_value < 0.05


def rename_model(string):
    """Map model string names to standardized names using pattern matching.
    
    Optimized with early returns and ordered checks from most to least specific.
    """
    string_lower = string.lower()
    
    # Most specific patterns first (avoid false matches)
    if 'suprem_swinunetr' in string_lower or 'swin_unetr_clip' in string_lower or 'swin unetr and clip' in string_lower:
        return 'SwinUNETR & CLIP'
    
    if 'suprem_unet' in string_lower or 'u-net_clip' in string_lower or 'u-net and clip' in string_lower:
        return 'U-Net & CLIP'
    
    # nnU-Net variants (order matters - check U-Net variant before ResEncL)
    if 'nnu-net_u-net' in string_lower or 'nnu-net u-net' in string_lower or ('riginal' in string and 'nnunet' in string_lower):
        return 'nnU-Net U-Net'
    
    if 'resencl' in string_lower or ('riginal' not in string and 'nnunet' in string_lower):
        return 'nnU-Net ResEncL'
    
    # SwinUNETR (check after CLIP variants)
    if ('swinunetr' in string_lower or 'swin_unetr' in string_lower or 'swin unetr' in string_lower) and 'suprem' not in string_lower and 'clip' not in string_lower:
        return 'SwinUNETR'
    
    # UNETR variants
    if ('unest' in string_lower) and 'suprem' not in string_lower and 'clip' not in string_lower:
        return 'UNEST'
    
    if ('unetr' in string_lower) and 'suprem' not in string_lower and 'clip' not in string_lower:
        return 'UNETR'
    
    # Simple pattern mappings
    simple_patterns = {
        ('yiwen', 'uniseg'): 'UniSeg',
        ('zhaohu', 'diff-unet'): 'Diff-UNet',
        ('uctransnet',): 'UCTransNet',
        ('segvol', 'bozhao'): 'SegVol',
        ('saikat', 'mednext'): 'MedNeXt',
        ('segresnet', 'suprem_segresnet'): 'SegResNet',
        ('nextou',): 'NexToU',
        ('lhunet', 'lhu-net'): 'LHU-Net',
        ('stu_base', 'stunetbase', 'stu-net-b', 'stu-net'): 'STU-Net',
        ('sam',): 'SAM-Adapter',
        ('cleannet',): 'CleanNet',
    }
    
    for patterns, result in simple_patterns.items():
        if any(pattern in string_lower for pattern in patterns):
            return result
    
    return string
    
def rename_group(string, args):
    """Extract group name from string based on group type."""
    group_name = args.group_name
    
    if group_name == 'all':
        return ''
    
    # Use more efficient extraction patterns
    if group_name == 'ages':
        start = string.rfind('ages')
        return string[start:start+10].replace('_', ' ') if start != -1 else string
    
    # Use a dictionary for prefix-based extractions
    prefix_patterns = {
        'diagnosis': 'diagnosis_',
        'cancer_diagnosis': 'cancer_diagnosis_',
        'sex': 'sex_',
        'race': 'race_',
        'institute': 'institute_',
        'manufacturer': 'manufacturer_',
        'scanner_model': 'scanner_model_'
    }
    
    if group_name == 'manufacturer':
        # Special case: use direct mapping for manufacturers
        manufacturer_map = {'ge': 'GE', 'siemens': 'Siemens', 'philips': 'Philips'}
        string_lower = string.lower()
        for key, value in manufacturer_map.items():
            if key in string_lower:
                return value
        # Fallback to prefix extraction
        group_name = 'manufacturer'
    
    if group_name in prefix_patterns:
        prefix = prefix_patterns[group_name]
        start = string.find(prefix)
        if start != -1:
            start += len(prefix)
            end = string.rfind('_')
            # Handle race special case with wrong offset
            if group_name == 'race':
                return string[start:].replace('_', ' ')
            return string[start:end].replace('_', ' ') if end > start else string[start:].replace('_', ' ')
    
    return string

def intersect(list1, list2):
    # Use set intersection for O(n) complexity instead of O(n²)
    return len(set(list1) & set(list2))

def mean_model_performance(df_dict, groups_lists=None, args=None):
    """Compute mean model performance across all models.
    
    Optimized to reduce redundant operations and use efficient lookups.
    """
    # Combine all dataframes and compute mean per sample
    combined_df = pd.concat(df_dict.values(), axis=0)
    df = combined_df.groupby('name').mean(numeric_only=True).reset_index()
    
    if groups_lists is not None:  # not for all and ages
        long_df = convert_to_long_format(df, model_name='avg', args=args)
        long_df = long_df.dropna(subset=['Value'])
        
        # Convert sample lists to sets for O(1) lookup
        means = {}
        for group_name, sample_list in groups_lists.items():
            sample_set = set(sample_list) if not isinstance(sample_list, set) else sample_list
            group_df = long_df[long_df['name'].isin(sample_set)]
            means[group_name] = group_df['Value'].mean()
        
        # Sort by mean performance
        return sorted(means, key=means.get, reverse=True)
    else:
        return df

def order_models(models):
    # Use set for O(1) lookup instead of O(n) for each model
    models_set = set(models)
    ranking_set = set(model_ranking)
    
    # Check for unranked models first
    unranked = models_set - ranking_set
    if unranked:
        raise ValueError(f'Unranked model(s): {unranked}, please add to model_ranking list inside this code, in the correct position, according to the overall ranking')
    
    # Filter ranking to only include models present in the input
    return [model for model in model_ranking if model in models_set]



def read_models_and_groups(args):
    """Load model results and group lists with optimized file I/O."""
    th = int(args.th)
    
    # Load model results - filter .DS_Store early
    metric_file = 'nsd.csv' if args.nsd else 'dsc.csv'
    
    # Get list of directories, filtering out .DS_Store
    model_dirs = [f for f in os.listdir(args.ckpt_root) if '.DS_Store' not in f]
    model_files = [os.path.join(file, metric_file) for file in model_dirs]
    model_names = [rename_model(file) for file in model_dirs]
    
    # Load CSVs efficiently
    if args.test_set_only:
        split = pd.read_csv(args.split_path, sep=';')
        test_image_ids = split.loc[split['split'] == 'test', 'image_id'].tolist()
        # Convert to set for O(1) lookup
        test_image_ids_set = set(test_image_ids)
        # Read CSV once and filter
        results = {}
        for model, file in zip(model_names, model_files):
            df = pd.read_csv(os.path.join(args.ckpt_root, file))
            results[model] = df[df['name'].isin(test_image_ids_set)]
    else:
        results = {model: pd.read_csv(os.path.join(args.ckpt_root, file))
                   for model, file in zip(model_names, model_files)}
    
    if args.mean_and_best:
        results = {'Average AI Algorithm': mean_model_performance(results),
                   'nnU-Net': results['nnU-Net']}
        model_names = ['Average AI Algorithm', 'nnU-Net']
    if args.just_mean:
        results = {'Average AI Algorithm': mean_model_performance(results)}
        model_names = ['Average AI Algorithm']
    
    # Get first result key efficiently
    first_key = next(iter(results))
    samples = results[first_key]['name'].tolist()
    
    # Get no_nan_samples
    no_nan_samples = convert_to_long_format(results[first_key],
                                            model_name=first_key,
                                            args=args).dropna(subset=['Value'])['name'].tolist()
    
    if args.group_name == 'all':  # 1 group with all samples
        groups_lists = {'all': samples}
        print('Samples: ', len(groups_lists['all']))
    else:  # per group-analysis
        # Load group lists - avoid loading files twice
        group_files = [file for file in os.listdir(args.group_root) 
                       if '.pt' in file and args.group_name in file]
        
        # Convert no_nan_samples to set for O(1) intersection check
        no_nan_samples_set = set(no_nan_samples)
        groups_lists = {}
        
        for file in group_files:
            file_path = os.path.join(args.group_root, file)
            samples_list = torch.load(file_path)
            # Use set intersection for efficiency
            if len(set(samples_list) & no_nan_samples_set) >= th:
                groups_lists[rename_group(os.path.splitext(file)[0], args)] = samples_list
    
    order = []
    group_names = list(groups_lists.keys())
    model_names = order_models(model_names)
    
    if args.group_name != 'all' and args.group_name != 'ages':
        # sort groups by average model performance
        group_names = mean_model_performance(results, groups_lists, args)
    else:
        group_names = sorted(group_names)
    
    # Build order list more efficiently
    if args.group_name != 'all':
        order = [f"{model_name}-{group_name}" 
                 for model_name in model_names 
                 for group_name in group_names]
    else:
        order = model_names.copy()
    
    num_groups = len(group_names)
    num_algos=len(model_names)
    #print(group_names)
    
    return results, groups_lists, order, num_groups, num_algos

def convert_to_long_format(df, model_name, args):
    """Convert DataFrame to long format optimized for the specified organ.
    
    Uses copy() to avoid SettingWithCopyWarning and optimizes column selection.
    """
    if args.organ == 'mean':  # data points are per-ct mean scores
        # More efficient: select numeric columns and compute mean
        result_df = df.copy()
        result_df['Average'] = result_df.iloc[:, 1:].mean(axis=1)
        df = result_df[['name', 'Average']]
    elif args.organ == 'all':  # data points are all per-organ values
        pass  # Use df as-is
    else:  # per-organ plot
        df = df[['name', args.organ]].copy()
    
    # Melt the DataFrame from wide to long format
    long_df = df.melt(id_vars=['name'], var_name='Organ', value_name='Value')
    long_df['Model'] = model_name
    return long_df

def create_long_format_dataframe(results, groups_lists, args):
    """Create combined long format dataframe from all models and groups.
    
    Optimized to use list comprehension and minimize DataFrame operations.
    """
    data = []
    
    # Convert sample lists to sets for O(1) lookup
    groups_lists_sets = {name: set(samples) for name, samples in groups_lists.items()}
    
    for model_name, df in results.items():
        long_df = convert_to_long_format(df, model_name, args)
        long_df = long_df.dropna(subset=['Value'])  # Drop rows with NaN values in 'Value'
        
        for group_name, sample_set in groups_lists_sets.items():
            combined_group_name = f"{model_name}-{group_name}" if args.group_name != 'all' else model_name
            # Use set for isin() - more efficient lookup
            group_df = long_df[long_df['name'].isin(sample_set)].copy()
            group_df['Group'] = combined_group_name
            data.append(group_df[['Group', 'Value']])

    # Concatenate all DataFrames into a single DataFrame
    final_df = pd.concat(data)
    
    return final_df


def break_title(title, fig_width):
    """Break title into multiple lines based on figure width.
    
    Optimized to use a more efficient line-breaking algorithm.
    """
    # Adjust max_char_in_line based on figure width
    char_per_inch = 8  # Approximate number of characters per inch
    max_char_in_line = int(fig_width * char_per_inch)
    
    if len(title) <= max_char_in_line:
        return title
    
    # Break title into multiple lines at word boundaries
    parts = []
    while len(title) > max_char_in_line:
        # Find last space within limit
        split_idx = title[:max_char_in_line].rfind(' ')
        if split_idx == -1:
            # No space found, force split at max_char_in_line
            split_idx = max_char_in_line
        parts.append(title[:split_idx])
        title = title[split_idx:].lstrip()  # Remove leading whitespace
    
    if title:  # Add remaining text
        parts.append(title)
    
    return '\n'.join(parts)
    

def second_last_rfind(s, char):
    # Find the last occurrence of the character
    last_occurrence = s.rfind(char)
    if last_occurrence == -1:
        return -1  # Character not found at all
    # Find the second last occurrence by slicing the string up to the last occurrence
    second_last_occurrence = s.rfind(char, 0, last_occurrence)
    return second_last_occurrence

def remove_model(value):
    
    if 'Average AI Algorithm' in value:
        return value.replace('Average AI Algorithm','Avg.')
    # Example transformation: append '_modified' to each group name
    return value

def find_model(value):
    """Find which model a value string belongs to.
    
    Returns the model name if found, None otherwise.
    """
    for m in model_ranking + ['Avg.', 'Average AI Algorithm']:
        if m in value:
            return m
    return None

organDict={ 'spleen':'spleen',
            'kidney_right':'kidneyR',
            'kidney_left':'kidneyL',
            'gall_bladder':'gallbladder',
            'liver':'liver',
            'stomach':'stomach',
            'aorta':'aorta',
            'postcava':'IVC',
            'pancreas':'pancreas',
            'mean':'average'}

def create_boxplot(long_df, group_order, num_groups, args, num_algos, ax=None,save=False,
                   hide_model=False,limits=None,omit_metric=False,significance_test=True,
                   colorful=True,title_style=None,rotation=45,font=13,fig_length=10):
    
    if 'totalsegmentator_results' in args.ckpt_root:
        dataset='TotalSegmentator'
    elif 'dapatlas_results' in args.ckpt_root:
        dataset='DAP Atlas'
    elif 'PrivateGT' in args.ckpt_root or 'privateGT' in args.ckpt_root or 'JHH' in args.ckpt_root:
        dataset='JHH'
    else:
        dataset=''
        
    #this one rotates, the old one does not
    fig_width=len(group_order)*num_algos/36
    
    # Determine the plot orientation based on the number of groups
    if args.orientation=='h' or (args.orientation=='auto' and fig_width<10):  # You can adjust this threshold
    #if args.group_name=='all':
        if num_algos<=2:
            fig_width=fig_width*36/9
        fig_width=max(fig_width,2)
        orientation = 'h'
        figsize = (fig_length, fig_width)  # Height based on number of groups
        xlabel = 'Value'
        ylabel = 'Group'
        w=10*0.9
        r=0
    elif args.orientation=='v' or (args.orientation=='auto' and fig_width>=10):
        fig_width=max(fig_width,3)
        orientation = 'v'
        figsize = (fig_width, fig_length)  # Width based on number of groups
        xlabel = 'Group'
        ylabel = 'Value'
        w=fig_width*0.9
        r=rotation
    else:
        raise ValueError('Unrecognized args.orientation, use h, v or auto')
        
    #reorder
    category_type = pd.CategoricalDtype(categories=group_order, ordered=True)
    long_df['Group'] = long_df['Group'].astype(category_type)
    long_df.sort_values('Group', inplace=True)
    
    if hide_model:
        long_df['Group'] = long_df['Group'].apply(remove_model)
    
    # Optimize color palette generation
    if not colorful:
        # Define color mapping for datasets
        color_dict = {
            "TotalSegmentator": "#FFA500",  # Orange
            "DAP Atlas": "#0000FF",  # Blue
            "JHH": "#008000"  # Green
        }
        # Use single color for non-colorful plots
        color_palette = [color_dict.get(dataset, "#808080")]  # Default to gray
    elif args.group_name != 'all':
        color_palette = [find_color(i) for i in group_order]
    else:
        color_palette = [model_color_dict[i] for i in group_order]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        plt.sca(ax)
        
    ax=sns.boxplot(
        x=xlabel, 
        y=ylabel, 
        data=long_df, 
        palette=color_palette, 
        #order=group_order if orientation == 'v' else None, 
        order=None,#reordered above
        fliersize=1, 
        width=0.7, 
        orient=orientation,
        ax=ax
    )

    metric = 'NSD' if args.nsd else 'dice score'

    organ = args.organ.replace('_', ' ')
    title = f'{organ} {metric}'
    group_name = 'cancer diagnosis' if args.group_name == 'cancer_diagnosis' else args.group_name
    
    if orientation == 'v':
        if not hide_model:
            if group_name != 'all':
                plt.xlabel('AI Algorithm-Group',fontsize=font)
            else:
                plt.xlabel('AI Algorithm',fontsize=font)
        else:
             plt.xlabel('')
        if not omit_metric:
            plt.ylabel(metric,fontsize=font)
        else:
            plt.ylabel('')
            
    else:
        if not hide_model:
            if group_name != 'all':
                plt.ylabel('AI Algorithm-Group',fontsize=font)
            else:
                plt.ylabel('AI Algorithm',fontsize=font)
        else:
             plt.ylabel('')
        if not omit_metric:
            plt.xlabel(metric,fontsize=font)
        else:
            plt.xlabel('')
            print('no METRIC')
            
        
    
    if group_name != 'all':
        title += ' by ' + group_name

    
        
    title += ' in ' + dataset
    if args.test_set_only:
        title += ' official test set'
        
    if title_style=='group':    
        title=group_name
        
    if title_style=='organ_dataset':    
        title=organDict[args.organ]+' - '+dataset
        
    if title_style=='organ':    
        title=organDict[args.organ]
        
    title=title.replace('PrivateGT','JHH')
    title=title.replace('DAPAtlas','DAP Atlas')
    
    
    try:
        plt.title(break_title(title,fig_width=w), fontsize=max(19,font))
    except:
        plt.title(break_title(title,fig_width=w), fontsize=max(19,font))
    

    plt.xticks(rotation=r, ha='right', fontsize=font)
    plt.tight_layout()
    
    if orientation=='h' and hide_model:
         plt.yticks(rotation=45, ha='right', fontsize=font)
    
    if orientation=='v':
        # Set more divisions on the y-axis
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.tight_layout()
        if limits is not None:
            ax.set_ylim(limits[0], limits[1])
        else:
            # Adjust y-axis limits to remove the bottom empty space
            y_min = long_df['Value'].min()
            buffer = (long_df['Value'].max() - y_min) * 0.05  # Create a buffer of 10% of the range
            y_min=max(y_min - buffer,0)
            ax.set_ylim(y_min, 1.0)  # Assuming your data values range between 0 and 1
        plt.xticks(fontsize=font)
    if orientation=='h':
        # Set more divisions on the y-axis
        plt.xticks(np.arange(0, 1.1, 0.2), fontsize=font)
        plt.tight_layout()
        if limits is not None:
            ax.set_xlim(limits[0], limits[1])
        else:
            # Adjust y-axis limits to remove the bottom empty space
            x_min = long_df['Value'].min()
            buffer = (long_df['Value'].max() - x_min) * 0.05  # Create a buffer of 10% of the range
            x_min=max(x_min - buffer,0)
            ax.set_xlim(x_min, 1.0)  # Assuming your data values range between 0 and 1
        plt.yticks(fontsize=font)

    if significance_test and args.group_name != 'all':
        if Kruskal_Wallis_Pure(long_df):
            # Get unique groups once
            unique_groups = long_df['Group'].unique()
            # Generate combinations and filter in one pass
            group_comb = [(g1, g2) for g1, g2 in combinations(unique_groups, 2)
                         if find_model(g1) == find_model(g2)]
            
            if group_comb:  # Only create annotator if there are valid combinations
                annotator = Annotator(ax, group_comb, x=xlabel, y=ylabel, 
                                    data=long_df, order=None, orient=orientation)
                annotator.configure(test='Mann-Whitney', text_format='star', loc='inside',
                                  comparisons_correction='Bonferroni', hide_non_significant=True,
                                  text_offset=0, line_height=0.01, fontsize=13)
                annotator.apply_and_annotate()
    
    if args.just_mean:
        # Modify individual ytick labels to remove 'Avg.-'
        new_labels = [label.get_text().replace('Avg.-', '') for label in ax.get_yticklabels()]

        # Set the new y-tick labels
        ax.set_yticklabels(new_labels, rotation=0, ha='right')
    if hide_model:
        new_labels = ['' for label in ax.get_yticklabels()]

        # Set the new y-tick labels
        if orientation=='v':
            ax.set_xticklabels(new_labels, rotation=0, ha='right')
        if orientation=='h':
            ax.set_yticklabels(new_labels, rotation=0, ha='right')
        
    
    folder = '../outputs/box_plots/box_plots_' + dataset
    if args.test_set_only:
        folder += '_test_set'
    os.makedirs(folder, exist_ok=True)
    if args.mean_and_best:
        title+=' mean and NNU-Net'
    if args.just_mean:
        title+=' mean'
    if save:
        title='Boxplot of '+title
        plt.savefig(folder + '/' + title.replace('/', ' ').replace('\n', ' ') + '.pdf', dpi=300,
                   bbox_inches='tight')
        plt.show()

    
if __name__ == "__main__":
    args = parse_arguments()

    results, groups_lists, order, num_groups, num_algos = read_models_and_groups(args)
    print('Read models and groups')

    long_df = create_long_format_dataframe(results, groups_lists, args)
    print('Created long format DataFrame')
    
    
    
    # Optionally, save the DataFrame to a CSV file
    long_df.to_csv("combined_long_format.csv", index=False)
    print('Saved long format DataFrame to combined_long_format.csv')
    
    create_boxplot(long_df, group_order=order, num_groups=num_groups,args=args,
                   num_algos=num_algos,significance_test=args.stats,
                   font=int(args.font),fig_length=int(args.fig_length),save=True)
