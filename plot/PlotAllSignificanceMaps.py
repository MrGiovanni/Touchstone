import matplotlib.pyplot as plt
from PlotGroup import read_models_and_groups, create_long_format_dataframe, create_boxplot
import matplotlib.gridspec as gridspec
from argparse import Namespace
import argparse
import SignificanceMaps as StatisticalHeatmap
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate dot and boxplots with confidence intervals.')
    parser.add_argument('--ckpt_root_TotalSegmentator', type=str, help='Path to the directory containing model result CSV files',default='../totalsegmentator_results/')
    parser.add_argument('--ckpt_root_DAPAtlas', type=str, help='Path to the directory containing model result CSV files',default='../dapatlas_results/')
    parser.add_argument('--ckpt_root_JHH', type=str, help='Path to the directory containing model result CSV files',default='../PrivateGT/')
    parser.add_argument('--group_root_TotalSegmentator', type=str, help='Path to the directory containing group sample lists',default='../outputs/plotsTotalSegmentator/')
    parser.add_argument('--group_root_DAPAtlas', type=str, help='Path to the directory containing group sample lists',default='../outputs/plotsDAPAtlas/')
    parser.add_argument('--group_root_JHH', type=str, help='Path to the directory containing group sample lists',default='../outputs/plotJHH/')
    parser.add_argument('--nsd', action='store_true', help='Plot dice if not set', default=False)
    parser.add_argument('--split_path', default='../utils/metaTotalSeg.csv', help='Location of TotalSegmentator metadata')
    parser.add_argument('--organs', type=str, help='list of organs',default='spleen kidneyR kidneyL gallbladder liver')
    return parser.parse_args()

organDict={ 'spleen':'spleen',
            'kidneyR':'kidney_right',
            'kidneyL':'kidney_left',
            'gallbladder':'gall_bladder',
            'liver':'liver',
            'stomach':'stomach',
            'aorta':'aorta',
            'IVC':'postcava',
            'pancreas':'pancreas',
            'average':'mean'}

def create_multiple_boxplots(args):
    organs= args.organs.split()
    
    fig = plt.figure(figsize=(20, 5*len(organs)))
    
    gs = gridspec.GridSpec(len(organs), 4, figure=fig, hspace=0.7, wspace=0.55)

    # Define subplots in custom locations
    configs=[]
    for i,organ in enumerate(organs,0):
        configs.append( {'title':'TotalSegmentator - '+organ,
                         'ckpt_root':args.ckpt_root_TotalSegmentator,
                         'group_root':args.group_root_TotalSegmentator,
                         'subplot':fig.add_subplot(gs[i, 0]),
                         'test_set_only':False,
                         'organ':organDict[organ]})
        configs.append( {'title':'TotalSegmentator Official Test Set - '+organ,
                         'ckpt_root':args.ckpt_root_TotalSegmentator,
                         'group_root':args.group_root_TotalSegmentator,
                         'subplot':fig.add_subplot(gs[i, 1]),
                         'test_set_only':True,
                         'organ':organDict[organ]})
        configs.append( {'title':'DAP Atlas - '+organ,
                         'ckpt_root':args.ckpt_root_DAPAtlas,
                         'group_root':args.group_root_DAPAtlas,
                         'subplot':fig.add_subplot(gs[i, 2]),
                         'test_set_only':False,
                         'organ':organDict[organ]})
        #configs.append( {'title':'JHH - '+organ,
        #                 'ckpt_root':args.ckpt_root_JHH,
        #                 'group_root':args.group_root_JHH,
        #                 'subplot':fig.add_subplot(gs[i, 3]),
        #                 'test_set_only':False,
        #                 'organ':organDict[organ]})
    
    

    for config in configs:
        p_args = Namespace()
        p_args.ckpt_root=config['ckpt_root']
        p_args.group_root=config['group_root']
        p_args.nsd=args.nsd
        p_args.organ=config['organ']
        p_args.mean_and_best=False
        p_args.test_set_only=config['test_set_only']
        p_args.title=config['title']
        p_args.split_path=args.split_path
        StatisticalHeatmap.HeatmapOfSignificance(p_args,ax=config['subplot'])
        
        

        #configs.subplot.set_title(f'Plot {i + 1}')
        #configs.subplot.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    os.makedirs('../outputs/heatmaps',exist_ok=True)
    filename='../outputs/heatmaps/significance_heatmaps_'+args.organs.replace(' ','_')
    if args.nsd:
        filename+='_nsd'
    filename+='.pdf'
    plt.savefig(filename,bbox_inches='tight', pad_inches=0)
    plt.close()

if __name__ == "__main__":
    args = parse_arguments()
    if args.organs=='first_half':
        args.organs='spleen kidneyR kidneyL gallbladder liver'
    if args.organs=='second_half':
        args.organs='stomach aorta IVC pancreas'# average'
    create_multiple_boxplots(args)