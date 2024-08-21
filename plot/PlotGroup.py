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
#print(model_color_dict)

def find_color(model):
    for i,m in enumerate(model_ranking,0):
        if m in model:
            return palette[i]
    raise ValueError('Uncrecognized model: '+model)

def Kruskal_Wallis(df):
    
    groups=df['Group'].unique()
    
    grouped_data = df.groupby('Group')['Value'].apply(list)

    
    ## Prepare the data for the Kruskal-Wallis test
    values = [group for group in grouped_data]
    h_statistic, p_value = stats.kruskal(*values)

    if p_value>0.05:
        return None #no significant result
    
    
    #Post-hoc tests: Wilcoxon rank sum tests/Mann–Whitney U test
    results = []

    # Perform pairwise Wilcoxon rank sum tests
    for (group1, group2) in combinations(groups, 2):
        group1_values = df[df['Group'] == group1]['Value']
        group2_values = df[df['Group'] == group2]['Value']
        stat, p_value = stats.mannwhitneyu(group1_values, group2_values, alternative='two-sided')
        results.append((group1, group2, p_value))

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results, columns=['Group1', 'Group2', 'P-Value'])
    
    # Apply FDR correction using Benjamini-Hochberg method
    pvals_corrected = stats.false_discovery_control(results_df['P-Value'], method='bh')
    results_df.loc[:, 'P-Value Adjusted'] = pvals_corrected

    significant_results = results_df[results_df['P-Value Adjusted'] < 0.05]
    return significant_results


def Kruskal_Wallis_Pure(df):
    
    groups=df['Group'].unique()
    
    grouped_data = df.groupby('Group')['Value'].apply(list)

    
    ## Prepare the data for the Kruskal-Wallis test
    values = [group for group in grouped_data]
    h_statistic, p_value = stats.kruskal(*values)

    if p_value<0.05:
        return True
    else:
        return False
    

def rename_model(string):
    if 'yiwen' in string or 'uniseg' in string or 'UniSeg' in string:
        return 'UniSeg'
    elif 'zhaohu' in string or 'Diff-UNet' in string:
        return 'Diff-UNet'
    elif 'UCTransNet' in string or 'uctransnet' in string:
        return 'UCTransNet'
    elif 'SegVol' in string or 'BoZhao' in string:
        return 'SegVol'
    elif 'Saikat' in string or 'mednext' in string or 'MedNeXt' in string:
        return 'MedNeXt'
    elif 'SegResNet' in string or 'SuPreM_segresnet' in string:
        return 'SegResNet'
    elif 'nextou' in string or 'NexToU' in string:
        return 'NexToU'
    elif 'SuPreM_UNet' in string or 'SuPreM_unet' in string or 'U-Net_CLIP' in string or 'U-Net and CLIP' in string:
        return 'U-Net & CLIP'
    elif 'SuPreM_swinunetr' in string or 'Swin_UNETR_CLIP' in string or 'Swin UNETR and CLIP' in string:
        return 'SwinUNETR & CLIP'
    elif 'LHUNet' in string or 'LHU-Net' in string:
        return 'LHU-Net'
    elif 'ResEncL' in string or ('riginal' not in string and ('nnUNet' in string or 'nnunet' in string)):
        return 'nnU-Net ResEncL'
    elif 'nnU-Net_U-Net' in string or 'nnU-Net U-Net' in string or ('riginal' in string and ('nnUNet' in string or 'nnunet' in string)):
        return 'nnU-Net U-Net'
    elif ('swinunetr' in string or 'Swin_UNETR' in string or 'Swin UNETR' in string) and 'SuPreM' not in string and 'CLIP' not in string:
        return 'SwinUNETR'
    elif 'STU_base' in string or 'STUNetBase' in string or 'STU-Net-B' in string or 'STU-Net' in string:
        return 'STU-Net'
    elif 'SAM' in string:
        return 'SAM-Adapter'
    elif ('unetr' in string or 'UNETR' in string) and 'SuPreM' not in string and 'CLIP' not in string:
        return 'UNETR'
    elif ('UNEST' in string or 'unest' in string or 'UNesT' in string) and 'SuPreM' not in string and 'CLIP' not in string:
        return 'UNEST'
    elif 'CleanNet' in string:
        return 'CleanNet'
    else:
        return string
    
def rename_group(string,args):    
    if args.group_name=='ages':
        return string[string.rfind('ages'):string.rfind('ages')+10].replace('_',' ')
    elif args.group_name=='diagnosis':
        return string[string.rfind('diagnosis_')+len('diagnosis_'):\
                      string.rfind('_')].replace('_',' ')
    elif args.group_name=='cancer_diagnosis':
        return string[string.find('cancer_diagnosis_')+len('cancer_diagnosis_'):\
                      string.rfind('_')].replace('_',' ')
    elif args.group_name=='sex':
        return string[string.rfind('sex_')+len('sex_'):\
                      string.rfind('_')].replace('_',' ')
    elif args.group_name=='race':
        return string[string.rfind('race_')+len('sex_'):].replace('_',' ')
    elif args.group_name=='institute':
        return string[string.rfind('institute_'):string.rfind('_')].replace('_',' ')
    elif args.group_name=='manufacturer':
        if 'ge' in string:
            return 'GE'
        elif 'siemens' in string:
            return 'Siemens'
        elif 'philips' in string:
            return 'Philips'
        else:
            return string[string.rfind('manufacturer_')+len('manufacturer_'):\
                          string.rfind('_')].replace('_',' ')
    elif args.group_name=='all':    
        return ''
    elif args.group_name=='scanner_model':
        return string[string.rfind('scanner_model_')+len('scanner_model_'):string.rfind('_')].replace('_',' ')
    else:
        return string

def intersect(list1,list2):
    # Convert lists to sets
    set1 = set(list1)
    set2 = set(list2)
    # Find the intersection of both sets
    intersection = set1.intersection(set2)
    # Count the number of elements in the intersection
    return len(intersection)

def mean_model_performance(df_dict,groups_lists=None,args=None):
    #df_dict: results per model
    combined_df = pd.concat(df_dict.values(), axis=0)
    # Group by 'names' and compute the mean across all original DataFrames
    df = combined_df.groupby('name').mean().reset_index()
    
    if groups_lists is not None:#not for all and ages
        long_df = convert_to_long_format(df, model_name='avg',args=args)
        long_df = long_df.dropna(subset=['Value'])  # Drop rows with NaN values in 'Value'
        means={}
        for group_name, sample_list in groups_lists.items():
            group_df = long_df[long_df['name'].isin(sample_list)]
            means[group_name]=group_df['Value'].mean()
        group_order=sorted(means, key=lambda k: means[k], reverse=True)
        return group_order
    else:
        return df

def order_models(models):
    tmp=[]
    for model in model_ranking:
        if model in models:
            tmp.append(model)
            
    for model in models:
        if model not in model_ranking:
            raise ValueError('Unranked model: ', model, ', please add it to model_ranking list inside this code, in the correct position, according to the overall raking')
    
    return tmp



def read_models_and_groups(args):
    #th: exclude groups with less samples than th
    th=int(args.th)
    
    
    # Load model results
    #remove yiwen from dap atlas
    if not args.nsd:
        model_files = [os.path.join(file,'dsc.csv') for file in os.listdir(args.ckpt_root)]
    else:
        model_files = [os.path.join(file,'nsd.csv') for file in os.listdir(args.ckpt_root)]

    model_names = [rename_model(file[:file.rfind('/')]) for file in model_files]
    
    if args.test_set_only:
        split=pd.read_csv(args.split_path,sep=';')
        test_image_ids = split.loc[split['split'] == 'test', 'image_id'].tolist()
        results = {model: pd.read_csv(os.path.join(args.ckpt_root,file))\
                   [pd.read_csv(os.path.join(args.ckpt_root,file))['name'].isin(test_image_ids)]\
                   for model, file in zip(model_names, model_files)}
    else:
        results = {model: pd.read_csv(os.path.join(args.ckpt_root,file))\
                   for model, file in zip(model_names, model_files) if '.DS_Store' not in model}
        
    if args.mean_and_best:
        results={'Average AI Algorithm':mean_model_performance(results),
                 'nnU-Net':results['nnU-Net']}
        model_names = ['Average AI Algorithm','nnU-Net']
    if args.just_mean:
        results={'Average AI Algorithm':mean_model_performance(results)}
        model_names = ['Average AI Algorithm']
        
    samples=results[list(results.keys())[0]]['name'].to_list()
    
    
    no_nan_samples=convert_to_long_format(results[list(results.keys())[0]],
                                          model_name=list(results.keys())[0],
                                          args=args).dropna(subset=['Value'])['name'].to_list()

    if args.group_name=='all':#1 group with all samples
        groups_lists={'all':samples}
        print('Samples: ',len(groups_lists['all']))
    else:#per group-analysis
        # Load group lists
        group_files = [file for file in os.listdir(args.group_root) if '.pt' in file and args.group_name in file]
        groups_lists = {rename_group(os.path.splitext(file)[0],args): torch.load(os.path.join(args.group_root, file)) for file in group_files \
   if intersect(torch.load(os.path.join(args.group_root, file)),no_nan_samples)>=th}
    
    order=[]
    group_names=list(groups_lists.keys())
    model_names=order_models(model_names)
    if args.group_name!='all' and args.group_name!='ages':
        #sort groups by average model performance
        group_names=mean_model_performance(results,groups_lists,args)
    else:
        group_names=sorted(group_names)
        
    
    for model_name in model_names:
        for group_name in group_names:
            if args.group_name!='all':
                order.append(f"{model_name}-{group_name}")
            else:
                order.append(model_name)
                
        
    num_groups=len(group_names)
    num_algos=len(model_names)
    #print(group_names)
    
    return results, groups_lists, order, num_groups, num_algos

def convert_to_long_format(df, model_name,args):
    if args.organ=='mean':#data points are per-ct mean scores
        df['Average'] = df.iloc[:, 1:].mean(axis=1)
        # Create a new DataFrame with just the 'Name' and 'Average' columns
        df = df[['name', 'Average']]
    elif args.organ=='all':#data points are all per-organ values (points~number of organs x number of cts)
        pass
    else:#per-organ plot
        df = df[['name', args.organ]]
        
        
        
    # Melt the DataFrame from wide to long format
    long_df = df.melt(id_vars=['name'], var_name='Organ', value_name='Value')
    long_df['Model'] = model_name
    return long_df

def create_long_format_dataframe(results, groups_lists,args):
    data = []
    
    
    for model_name, df in results.items():
        long_df = convert_to_long_format(df, model_name,args)
        long_df = long_df.dropna(subset=['Value'])  # Drop rows with NaN values in 'Value'
        
        for group_name, sample_list in groups_lists.items():
            if args.group_name!='all':
                combined_group_name = f"{model_name}-{group_name}"
            else:
                combined_group_name = model_name
            group_df = long_df[long_df['name'].isin(sample_list)].copy()
            group_df['Group'] = combined_group_name#modified latter, was group_df['Group'] =
            data.append(group_df[['Group', 'Value']])

    # Concatenate all DataFrames into a single DataFrame
    final_df = pd.concat(data)
    
    return final_df


def break_title(title,fig_width):
    # Adjust max_char_in_line based on figure width
    char_per_inch = 8  # Approximate number of characters per inch
    max_char_in_line = int((fig_width * char_per_inch)//1)
    
    # Break title into multiple lines if necessary
    parts = []
    while len(title) > max_char_in_line:
        part = title[:max_char_in_line]
        next_space = part.rfind(' ')
        if next_space != -1:
            parts.append(part[:next_space])
            title = title[next_space+1:]
        else:
            parts.append(part)
            title = title[max_char_in_line:]
    parts.append(title)
    title = '\n'.join(parts)
    return title
    

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
    for m in model_ranking+['Avg.','Average AI Algorithm']:
        if m in value:
            return m

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
        

    if args.group_name!='all':
        color_palette=[find_color(i) for i in group_order]
    else:
        color_palette=[model_color_dict[i] for i in group_order]
        
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        plt.sca(ax)
        
    if not colorful:
        color_dict = {
        "TotalSegmentator": ["#FFA500"],  # Orange
        "DAP Atlas": ["#0000FF"],  # Blue
        "JHH": ["#008000"]   # Green
        }
        for key in color_dict:
            if key in dataset:
                color_palette=color_dict[key]
        
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

    if significance_test:
        if Kruskal_Wallis_Pure(long_df) and args.group_name!='all':
            group_comb=[item for item in combinations(long_df['Group'].unique(), 2)]
            group_comb=[item for item in group_comb if find_model(item[0])==find_model(item[1])]
                
            #print(group_comb)
            
            annotator = Annotator(ax, group_comb, x=xlabel, 
            y=ylabel, 
            data=long_df, 
            order=None,#reordered above
            orient=orientation)
            annotator.configure(test='Mann-Whitney', text_format='star', loc='inside',
                               comparisons_correction='Bonferroni',hide_non_significant=True,
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
