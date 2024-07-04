import matplotlib.pyplot as plt
from PlotGroup import read_models_and_groups, create_long_format_dataframe, create_boxplot
import matplotlib.gridspec as gridspec
from argparse import Namespace
import argparse
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate dot and boxplots with confidence intervals.')
    parser.add_argument('--ckpt_root_TotalSegmentator', type=str, help='Path to the directory containing model result CSV files',default='../totalsegmentator_results/')
    parser.add_argument('--ckpt_root_JHH', type=str, help='Path to the directory containing model result CSV files',default='PrivateGT/')
    parser.add_argument('--ckpt_root_DAPAtlas', type=str, help='Path to the directory containing model result CSV files',default='../dapatlas_results/')
    parser.add_argument('--group_root_TotalSegmentator', type=str, help='Path to the directory containing group sample lists',default='../outputs/plotsTotalSegmentator/')
    parser.add_argument('--group_root_DAPAtlas', type=str, help='Path to the directory containing group sample lists',default='../outputs/plotsDAPAtlas/')
    parser.add_argument('--group_root_JHH', type=str, help='Path to the directory containing group sample lists',default='plotJHH/')
    parser.add_argument('--nsd', action='store_true', help='Plot dice if not set', default=False)
    parser.add_argument('--stats', action='store_true', help='Plot only results for nnU-Net and for average of all models', default=False)
    return parser.parse_args()


def create_multiple_boxplots(args):
    fig = plt.figure(figsize=(21, 29.7))
    
    gs = gridspec.GridSpec(45, 2, figure=fig, hspace=0.45, wspace=0.3)

    # Define subplots in custom locations
    ts_age = fig.add_subplot(gs[0:8, 0])
    ts_institute = fig.add_subplot(gs[9:15, 0])
    
    ts_diag = fig.add_subplot(gs[0:8, 1])
    ts_sex = fig.add_subplot(gs[9:11, 1])
    ts_manu = fig.add_subplot(gs[12:15, 1])
    
    dap_age = fig.add_subplot(gs[16:23, 0])
    
    dap_diag = fig.add_subplot(gs[17:19, 1])
    dap_sex = fig.add_subplot(gs[20:22, 1])
    
    JHH_age = fig.add_subplot(gs[24:31, 0])
    
    JHH_diag = fig.add_subplot(gs[23:25, 1])
    JHH_sex  = fig.add_subplot(gs[26:28, 1])
    JHH_race  = fig.add_subplot(gs[29:32, 1])
    
    # Add "DAP Atlas" text in cell 16
    #fig.text(0, 0, "DAP Atlas", fontsize=20, ha='left', va='center', rotation=90)
    
    configs=[{'group_name':'ages','ckpt_root':args.ckpt_root_TotalSegmentator,
              'group_root':args.group_root_TotalSegmentator,'subplot':ts_age,
              'limits':None},
             {'group_name':'diagnosis','ckpt_root':args.ckpt_root_TotalSegmentator,
              'group_root':args.group_root_TotalSegmentator,'subplot':ts_diag,
              'limits':None},
             {'group_name':'manufacturer','ckpt_root':args.ckpt_root_TotalSegmentator,
              'group_root':args.group_root_TotalSegmentator,'subplot':ts_manu,
              'limits':None},
             {'group_name':'sex','ckpt_root':args.ckpt_root_TotalSegmentator,
              'group_root':args.group_root_TotalSegmentator,'subplot':ts_sex,
              'limits':None},
             {'group_name':'institute','ckpt_root':args.ckpt_root_TotalSegmentator,
              'group_root':args.group_root_TotalSegmentator,'subplot':ts_institute,
              'limits':None},
             {'group_name':'ages','ckpt_root':args.ckpt_root_DAPAtlas,
              'group_root':args.group_root_DAPAtlas,'subplot':dap_age,
              'limits':[0.75,1]},
             {'group_name':'cancer_diagnosis','ckpt_root':args.ckpt_root_DAPAtlas,
              'group_root':args.group_root_DAPAtlas,'subplot':dap_diag,
              'limits':[0.75,1]},
             {'group_name':'sex','ckpt_root':args.ckpt_root_DAPAtlas,
              'group_root':args.group_root_DAPAtlas,'subplot':dap_sex,
              'limits':[0.75,1]},
             #{'group_name':'ages','ckpt_root':args.ckpt_root_JHH,
             # 'group_root':args.group_root_JHH,'subplot':JHH_age,
             # 'limits':[0.7,1]},
             #{'group_name':'sex','ckpt_root':args.ckpt_root_JHH,
             # 'group_root':args.group_root_JHH,'subplot':JHH_sex,
             # 'limits':[0.7,1]},
             #{'group_name':'race','ckpt_root':args.ckpt_root_JHH,
             # 'group_root':args.group_root_JHH,'subplot':JHH_race,
             # 'limits':[0.7,1]},
             #{'group_name':'cancer_diagnosis','ckpt_root':args.ckpt_root_JHH,
             # 'group_root':args.group_root_JHH,'subplot':JHH_diag,
             # 'limits':[0.7,1]}
            ]
    

    for config in configs:
        p_args = Namespace()
        p_args.group_name=config['group_name']
        p_args.ckpt_root=config['ckpt_root']
        p_args.group_root=config['group_root']
        p_args.nsd=args.nsd
        p_args.organ='mean'
        p_args.mean_and_best=False
        p_args.just_mean=True
        p_args.orientation='h'
        p_args.th=6
        p_args.test_set_only=False
        p_args.stats=args.stats
        
        results, groups_lists, order, num_groups, num_algos = read_models_and_groups(p_args)
        long_df = create_long_format_dataframe(results, groups_lists, p_args)
        create_boxplot(long_df, order, num_groups, p_args, num_algos, ax=config['subplot'],
                       save=False,limits=config['limits'],hide_model=True,omit_metric=True,
                       colorful=False)
        

        #configs.subplot.set_title(f'Plot {i + 1}')
        #configs.subplot.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    os.makedirs('../outputs/',exist_ok=True)
    plt.savefig('../outputs/summary_groups.pdf',bbox_inches='tight', pad_inches=0)
    #plt.show()
    plt.close()

if __name__ == "__main__":
    args = parse_arguments()
    create_multiple_boxplots(args)