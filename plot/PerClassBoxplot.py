import matplotlib.pyplot as plt
from PlotGroup import read_models_and_groups, create_long_format_dataframe, create_boxplot
import matplotlib.gridspec as gridspec
from argparse import Namespace
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate dot and boxplots with confidence intervals.')
    parser.add_argument('--dataset', type=str, help='name of dataset',default='TotalSegmentator')
    parser.add_argument('--ckpt_root_TotalSegmentator', type=str, help='Path to the directory containing model result CSV files',default='../totalsegmentator_results/')
    parser.add_argument('--ckpt_root_JHH', type=str, help='Path to the directory containing model result CSV files',default='PrivateGT/')
    parser.add_argument('--ckpt_root_DAPAtlas', type=str, help='Path to the directory containing model result CSV files',default='../dapatlas_results/')
    parser.add_argument('--group_root_TotalSegmentator', type=str, help='Path to the directory containing group sample lists',default='../outputs/plotsTotalSegmentator/')
    parser.add_argument('--group_root_DAPAtlas', type=str, help='Path to the directory containing group sample lists',default='../outputs/plotsDAPAtlas/')
    parser.add_argument('--group_root_JHH', type=str, help='Path to the directory containing group sample lists',default='plotJHH/')
    parser.add_argument('--nsd', action='store_true', help='Plot dice if not set', default=False)
    parser.add_argument('--stats', action='store_true', help='Plot only results for nnU-Net and for average of all models', default=False)
    parser.add_argument('--test_set_only', action='store_true', help='Plot only results for nnU-Net and for average of all models', default=False)
    parser.add_argument('--split_path', default='../utils/metaTotalSeg.csv', help='Location of TotalSegmentator metadata')
    return parser.parse_args()


def create_multiple_boxplots(args):
    fig = plt.figure(figsize=(21, 29.7))
    
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.1, wspace=0.1)

    if args.dataset=='TotalSegmentator':
        group_root=args.group_root_TotalSegmentator
        ckpt_root=args.ckpt_root_TotalSegmentator
    elif args.dataset=='DAPAtlas':
        group_root=args.group_root_DAPAtlas
        ckpt_root=args.ckpt_root_DAPAtlas
    elif args.dataset=='JHH':
        group_root=args.group_root_JHH
        ckpt_root=args.ckpt_root_JHH
        
    
    configs=[{'group_name':'all',
              'organ':'spleen',
              'subplot':fig.add_subplot(gs[0, 0]),
              'limits':None},
             {'group_name':'all',
              'organ':'kidney_right',
              'subplot':fig.add_subplot(gs[0, 1]),
              'limits':None},
             {'group_name':'all',
              'organ':'kidney_left',
              'subplot':fig.add_subplot(gs[0, 2]),
              'limits':None},
             {'group_name':'all',
              'organ':'gall_bladder',
              'subplot':fig.add_subplot(gs[1, 0]),
              'limits':None},
             {'group_name':'all',
              'organ':'liver',
              'subplot':fig.add_subplot(gs[1, 1]),
              'limits':None},
             {'group_name':'all',
              'organ':'stomach',
              'subplot':fig.add_subplot(gs[1, 2]),
              'limits':None},
             {'group_name':'all',
              'organ':'aorta',
              'subplot':fig.add_subplot(gs[2, 0]),
              'limits':None},
             {'group_name':'all',
              'organ':'postcava',
              'subplot':fig.add_subplot(gs[2, 1]),
              'limits':None},
             {'group_name':'all',
              'organ':'pancreas',
              'subplot':fig.add_subplot(gs[2, 2]),
              'limits':None}
            ]
    
    
    
    

    for config in configs:
        p_args = Namespace()
        p_args.group_name=config['group_name']
        p_args.ckpt_root=ckpt_root
        p_args.group_root=group_root
        p_args.nsd=args.nsd
        p_args.organ=config['organ']
        p_args.mean_and_best=False
        p_args.just_mean=False
        p_args.orientation='h'
        p_args.th=6
        p_args.test_set_only=args.test_set_only
        p_args.split_path=args.split_path
        p_args.stats=args.stats
        
        results, groups_lists, order, num_groups, num_algos = read_models_and_groups(p_args)
        long_df = create_long_format_dataframe(results, groups_lists, p_args)
        create_boxplot(long_df, order, num_groups, p_args, num_algos, ax=config['subplot'],
                       save=False,limits=config['limits'],omit_metric=True,
                       title_style='organ',colorful=False,rotation=0,
                       hide_model=not (config['organ']=='spleen' or config['organ']=='gall_bladder' \
                                   or config['organ']=='aorta'),font=17)
        

        #configs.subplot.set_title(f'Plot {i + 1}')
        #configs.subplot.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    filename='../outputs/box_plots/box_plots_per_class_'+args.dataset.replace(' ','_')
    if args.nsd:
        print('NSD')
        filename+='_nsd'
    if args.test_set_only:
        filename+='_official_test_set'
    filename+='.pdf'
    plt.savefig(filename,bbox_inches='tight', pad_inches=0)
    plt.show()

if __name__ == "__main__":
    args = parse_arguments()
    create_multiple_boxplots(args)