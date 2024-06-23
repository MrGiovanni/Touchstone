import torch
import numpy as np
import os
import argparse
import time
import csv
import nibabel as nib
from scipy import ndimage
import warnings
warnings.filterwarnings("ignore")
from monai.transforms import AsDiscrete
#from metrics import dice_score, surface_dice #pedro changed
from multiprocessing import Pool, Manager
import pandas as pd
import monai
import itertools
import scipy
import matplotlib.pyplot as plt
import torch.nn.functional as F

from scipy.ndimage import binary_erosion

def get_skips(args):
    import pickle
    if 'TotalSegmentator' in args.dataset_path:
        file_path='TotalSegmentator.pkl'
    elif 'DAP' in args.dataset_path:
        file_path='DAPAtlas.pkl'
    elif 'JHH' or 'PrivateGT' in args.dataset_path:
        file_path='JHH.pkl'
    else:
        raise ValueError('uncrecognized dataset for loading skips')
    # Open the pickle file in read-binary mode
    with open(file_path, 'rb') as file:
        # Load the contents of the file
        data = pickle.load(file)
    # Print the data to verify
    return data

def write_line_to_file(filename, line):
    with open(filename, 'a') as file:  # Open file in append mode
        file.write(line + '\n')

def count_connected_components(array):
    erosion = scipy.ndimage.binary_erosion(array.cpu(), structure=np.ones((3, 3)), iterations=1)
    _, num_features = scipy.ndimage.label(erosion)
    return num_features

def find_longest_consecutive_ones(array):
    max_length = 0
    max_start_index = -1
    max_end_index = -1

    current_length = 0
    current_start_index = -1

    for i in range(len(array)):
        if array[i] == 1:
            if current_length == 0:
                current_start_index = i
            current_length += 1
        else:
            if current_length > max_length:
                max_length = current_length
                max_start_index = current_start_index
                max_end_index = i - 1
            current_length = 0

    if current_length > max_length:
        max_length = current_length
        max_start_index = current_start_index
        max_end_index = len(array) - 1

    return max_start_index, max_end_index

def dice_score(predict, target):
    #try:
    #    assert predict.shape == target.shape
    #except:
        #write_line_to_file('AssertionBugs.txt',case)
    #    return float('nan')
    assert predict.shape == target.shape
    tp = torch.sum(torch.mul(predict, target))
    den = torch.sum(predict) + torch.sum(target) + 1
    dice = 2 * tp / den

    return dice


def Plot(array):
    # Plot multiple slices along the z-axis
    num_slices = array.shape[-1]
    rows = 4
    cols = 4

    # Calculate the total number of subplots needed
    total_plots = rows * cols

    # Determine slice indices to display
    slice_indices = np.linspace(0, num_slices - 1, total_plots, dtype=int)

    # Create the figure and subplots
    fig, axs = plt.subplots(rows, cols, figsize=(12, 12))

    # Plot each slice on a separate subplot
    for i, idx in enumerate(slice_indices):
        row = i // cols
        col = i % cols
        ax = axs[row, col]
        ax.imshow(array[:, :, idx], cmap='gray')
        ax.axis('off')
        connected=str(count_unconnected_objects(array[:, :, idx]))
        ax.set_title(f"Slice {idx}"+' '+connected)

    # Remove any unused subplots
    for j in range(total_plots, rows * cols):
        fig.delaxes(axs.flatten()[j])

    plt.tight_layout()
    plt.show()


def count_unconnected_objects(binary_array):
    #erosion
    binary_array=ndimage.binary_erosion(binary_array,structure=np.ones((3,3)), iterations=1)
    
    # Label connected components in the binary array
    labeled_array, num_features = scipy.ndimage.label(binary_array)

    # Create a mask to identify white objects (labeled regions)
    white_object_mask = (labeled_array > 0)  # Consider all labeled regions

    # Count the number of unique labels (excluding background label 0)
    num_objects = np.max(labeled_array)  # This gives the number of unique labels

    return num_objects


structures=['spleen','kidney_right','kidney_left',
            'gall_bladder','liver','stomach',
            'aorta','postcava','pancreas']
structures=sorted(structures)
#dice_metric = monai.metrics.DiceMetric(include_background=True, reduction="mean")


def check_first_column(csv_filename, target_string):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_filename, header=None)  # Assume no header (header=None)

    # Check if the target string exists in the first column (index 0)
    if len(df.columns) > 0 and (df[0] == target_string).any():
        return True
    else:
        return False
    
def is_binary_tensor(tensor):
    unique_values = torch.unique(tensor)
    print(unique_values)
    return (len(unique_values) <= 2) and (torch.all(unique_values == 0) or torch.all(unique_values == 1))

def eval_single_CSV(case,args):
    try:
        eval_single_CSV_real(case,args)
    except Exception as e:
        # Print the error message
        print(f"An error occurred: {e}")
        print('BAD CASE:',case)
        write_line_to_file('BUGS.txt',case)

def eval_single_CSV_real(case,args):
    if case in os.listdir(args.pred_path):
        folderPred=case
    else:
        folderPred=renaming[case]
        #print(folderPred)
    
    csv_dsc = open(args.checkpoint_name+'_dice.csv', 'a')
    csv_dsc_writer = csv.DictWriter(csv_dsc, fieldnames=['name'] + structures)
    csv_nsd = open(args.checkpoint_name+'_nsd.csv', 'a')
    csv_nsd_writer = csv.DictWriter(csv_nsd, fieldnames=['name'] + structures)

    dice_case_result = {'name': case}
    nsd_case_result = {'name': case}
    print('case:', case)
    for structure in structures:
        print('Structure:',structure)
        
        if structure in args.skips[case]:#skipp small size/missing
            if args.skips[case][structure]:
                dice_case_result[structure]=np.NaN
                nsd_case_result[structure]=np.NaN
                continue
                
        if args.just_aorta and structure!='aorta':
            continue
        if 'predictions' in os.listdir(args.pred_path+folderPred):
            x=nib.load(args.pred_path+folderPred+'/predictions/'+structure+'.nii.gz')
        else:
            x=nib.load(args.pred_path+folderPred+'/segmentations/'+structure+'.nii.gz')
        y=nib.load(args.dataset_path+case+'/segmentations/'+structure+'.nii.gz')
        spacing_mm = tuple(y.header['pixdim'][1:4])
        #spacing_pred = tuple(x.header['pixdim'][1:4])
        print('inside loop')

        x,y=x.get_fdata(),y.get_fdata()
        xTorch=torch.from_numpy(x).clone()
        yTorch=torch.from_numpy(y).clone()
        xTorch=torch.where(xTorch>=0.5,1.0,0.0)
        yTorch=torch.where(yTorch>=0.5,1.0,0.0)
        xTorch = xTorch.to(torch.uint8)
        yTorch = yTorch.to(torch.uint8)
        xTorch,yTorch=xTorch.to(device),yTorch.to(device)

        
        
        if args.permute:
            #Necessary for Yiwen
            all_permutations = list(itertools.permutations([0, 1, 2]))
            perms=[]
            #Permute prediction to match shape of label
            for permut in all_permutations:
                tmp=xTorch.permute(permut)
                if tmp.shape==y.shape:
                    perms.append(tmp)
            if len(perms)==1:
                xTorch=perms[0]
            else:
                #if multiple permutations match the label, select the one with highest dice
                dscs=[]
                for t in perms:
                    dice  = dice_score(t,yTorch)
                    dice = dice.item() if torch.is_tensor(dice) else dice
                    dscs.append(dice)
                xTorch=perms[np.argmax(dscs)]

        
        if structure == "aorta" and args.crop_aorta:
            components = np.array([count_connected_components(yTorch[:, :, i]) \
                                   for i in range(yTorch.shape[2])])
            start_index, end_index = find_longest_consecutive_ones(components)
            if start_index != -1:
                print('cropping')
                xTorch[:, :, :start_index] = 0
                xTorch[:, :, end_index:] = 0
                yTorch[:, :, :start_index] = 0
                yTorch[:, :, end_index:] = 0
                
        dice = dice_score(xTorch,yTorch)
        dice = dice.item() if torch.is_tensor(dice) else dice
                
        #nsdOld = surface_dice(torch.from_numpy(x),torch.from_numpy(y),spacing_mm,1)
        if args.fake_nsd:
            nsd=1e6#any absurd number
        else:
            spacingFloat = tuple(float(x) for x in spacing_mm)
            nsd=monai.metrics.compute_surface_dice(xTorch.unsqueeze(0).unsqueeze(0),
                                                   yTorch.unsqueeze(0).unsqueeze(0),
                                                    spacing=spacingFloat,
                                                    include_background=True,
                                                    class_thresholds=[1.5]).item()
            
        print('y:',yTorch.shape)
        
        #using args.skips, no need for erosion
        #y=yTorch.squeeze(0).squeeze(0).cpu().numpy()
        #size = ndimage.binary_erosion(y,structure=np.ones((3,3,3)), iterations=1).sum()
        #if size>10:
        #    dice_case_result[structure]=dice
        #    nsd_case_result[structure]=nsd
        #    print(case,structure,'- DICE:',dice,'; NSD:',nsd)
        #else:
        #    dice_case_result[structure]=np.NaN
        #    nsd_case_result[structure]=np.NaN
        #    print(case,structure,'- Too little foreground')
        
        dice_case_result[structure]=dice
        nsd_case_result[structure]=nsd
        
        print(case,structure,'- DICE:',dice,'; NSD:',nsd)
        # Explicitly delete tensors to free up memory
        del xTorch, yTorch
        torch.cuda.empty_cache()
        
    csv_dsc_writer.writerows([dice_case_result])
    csv_nsd_writer.writerows([nsd_case_result])
    csv_dsc.close()
    csv_nsd.close()

def evaluate_CSV(args):
    print('Evaluating')
    os.makedirs(args.dataset_name,exist_ok=True)
    if args.restart or not os.path.exists(args.checkpoint_name+'_dice.csv'):
        if os.path.exists(args.checkpoint_name+'_dice.csv'):
            os.remove(args.checkpoint_name+'_dice.csv')
        if os.path.exists(args.checkpoint_name+'_dice_means.csv'):
            os.remove(args.checkpoint_name+'_dice_means.csv')
        csv_dsc = open(args.checkpoint_name+'_dice.csv', 'a')
        fieldnames = ['name'] + structures
        csv_dsc_writer = csv.DictWriter(csv_dsc, fieldnames=fieldnames)
        csv_dsc_writer.writeheader()
        csv_dsc.close()
        
        if os.path.exists(args.checkpoint_name+'_nsd.csv'):
            os.remove(args.checkpoint_name+'_nsd.csv')
        if os.path.exists(args.checkpoint_name+'_nsd_means.csv'):
            os.remove(args.checkpoint_name+'_nsd_means.csv')
        csv_dsc = open(args.checkpoint_name+'_nsd.csv', 'a')
        csv_dsc_writer = csv.DictWriter(csv_dsc, fieldnames=fieldnames)
        csv_dsc_writer.writeheader()
        csv_dsc.close()
    if not args.subset_JHU:
        case_list=sorted(os.listdir(args.dataset_path))
    else:
        case_list = pd.read_csv('private_subset.csv').iloc[:, 0].tolist()
    if int(args.parts)>1:
        part=int(args.part)
        part_size=int(len(case_list)/int(args.parts))
        if part==(int(args.parts)-1):#last
            case_list=case_list[part*part_size:]
        else:
            case_list=case_list[part*part_size:(part+1)*part_size]
            
    print('CASES:',len(case_list))
    print('Data dir:',args.dataset_path)
    print('CASE LIST:',case_list)
    
    cases=[]
    for case in case_list:
        if os.path.isdir(args.dataset_path+case):
            cases.append(case)
    
    if not args.restart:
        dfd = pd.read_csv(args.checkpoint_name+'_dice.csv')['name'].to_list()
        dkj = pd.read_csv(args.checkpoint_name+'_nsd.csv')['name'].to_list()
        tmp=[]
        for case in cases:
            if ((case not in dfd) or (case not in dkj)):
                tmp.append(case)
        cases=tmp
                                             
    if args.num_workers>1:
        pool = Pool(processes=args.num_workers)
        for case in cases:
            pool.apply_async(eval_single_CSV, (case,args))
        pool.close()
        pool.join()
    else:
        for case in cases:
            eval_single_CSV(case,args)

def weighted_mean(df,weights):
    mean=0.0
    for structure in structures:
        mean+=df[structure].item()*weights[structure]
    return mean

def get_means(args,stds=False,folder='mean_results',test=False):
    folder=folder+'_'+args.dataset_name
    os.makedirs(args.dataset_name,exist_ok=True)
    dice_name,nsd_name=args.checkpoint_name+'_dice.csv',args.checkpoint_name+'_nsd.csv'
    dice=pd.read_csv(dice_name)
    nsd=pd.read_csv(nsd_name)
    print(dice)
    
    if test:
        try:
            split=pd.read_csv(args.dataset_path+'meta.csv',sep=';')
        except:
            split=pd.read_csv(args.dataset_path[:args.dataset_path[:-1].rfind('/')]+'/meta.csv',
                              sep=';')
        test_image_ids = split.loc[split['split'] == 'test', 'image_id'].tolist()
        dice=dice[dice['name'].isin(test_image_ids)]
        nsd=nsd[nsd['name'].isin(test_image_ids)]
        
    if args.subset_JHU:
        test_image_ids = pd.read_csv('private_subset.csv').iloc[:, 0].tolist()
        dice=dice[dice['name'].isin(test_image_ids)]
        nsd=nsd[nsd['name'].isin(test_image_ids)]
    
    if not stds:
        means_dice=dice.mean(numeric_only=True)
        means_nsd=nsd.mean(numeric_only=True)
    else:
        means_dice=dice.std(numeric_only=True)
        means_nsd=nsd.std(numeric_only=True)
    
    means_dice = pd.DataFrame({'name': ['std'], **means_dice}).reset_index(drop=True)
    means_nsd = pd.DataFrame({'name': ['std'], **means_nsd}).reset_index(drop=True)
    
    means_dice['Average']=means_dice.to_numpy()[0,1:].astype(float).mean()
    means_nsd['Average']=means_nsd.to_numpy()[0,1:].astype(float).mean()
    
    weights_dice=torch.load('DICE_weights.pt')
    weights_nsd=torch.load('NSD_weights.pt')
    
    means_dice['Weighted Average']=weighted_mean(means_dice,weights_dice)
    means_nsd['Weighted Average']=weighted_mean(means_nsd,weights_nsd)
    
    means_dice.drop('name', axis=1, inplace=True)
    means_nsd.drop('name', axis=1, inplace=True)
    
    means_dice=means_dice.round(3)
    means_nsd=means_nsd.round(3)
    print('dice:',means_dice)
    print('nsd:',means_nsd)
    
    if test:
        dest=folder+'_test_set'
    else:
        dest=folder
        
    os.makedirs(dest,exist_ok=True)
    if not stds:
        means_dice.to_csv(dest+'/'+dice_name[dice_name.find('/')+1:-4]+'_means.csv', index=False)
        means_nsd.to_csv(dest+'/'+nsd_name[nsd_name.find('/')+1:-4]+'_means.csv', index=False)
    else:
        means_dice.to_csv(dest+'/'+dice_name[dice_name.find('/')+1:-4]+'_stds.csv', index=False)
        means_nsd.to_csv(dest+'/'+nsd_name[nsd_name.find('/')+1:-4]+'_stds.csv', index=False)
        

        

parser = argparse.ArgumentParser()
## dataset
parser.add_argument('--checkpoint_name', help='name for saving the results')
parser.add_argument('--restart', action='store_true', help='Continues from previous run',default=False)
parser.add_argument('--pred_path', help='predictions path')
parser.add_argument('--dataset_path', help='dataset path')
parser.add_argument('--num_workers', default=1, type=int)
parser.add_argument('--permute', action='store_true', help='Tries permutating the prediction (for Yiwen ckpt)',default=False)
parser.add_argument('--crop_aorta', action='store_true', help='Tries permutating the prediction (for Yiwen ckpt)',default=False)
parser.add_argument('--just_aorta', action='store_true', help='Tries permutating the prediction (for Yiwen ckpt)',default=False)
parser.add_argument('--just_mean', action='store_true', help='Just uses saved csv to calculate means',default=False)
parser.add_argument('--dataset_name', default='TotalSegmentator', help='For output folder')
parser.add_argument('--device', default='cpu')
parser.add_argument('--parts', default='1')
parser.add_argument('--part', default='0')
parser.add_argument('--fake_nsd', action='store_true', help='Tries permutating the prediction (for Yiwen ckpt)',default=False)
parser.add_argument('--subset_JHU', action='store_true', help='Tries permutating the prediction (for Yiwen ckpt)',default=False)

args = parser.parse_args()

args.checkpoint_name=args.dataset_name+'/'+args.checkpoint_name

args.skips=get_skips(args)

if int(args.parts)>1:
    os.makedirs(args.dataset_name,exist_ok=True)
    args.checkpoint_name=args.checkpoint_name+'_part_'+args.part+'_of_'+args.parts
    flag=False
    torch.save(flag,args.checkpoint_name+'_finished.pt')
    
device=args.device
if args.dataset_path[-1]!='/':
    args.dataset_path+='/'
if args.pred_path[-1]!='/':
    args.pred_path+='/'
    
if 'cuda' in args.device:
    args.num_workers=1

if "TotalSegmentator" in args.dataset_name:
    renaming=pd.read_csv('renamingTotalSegmentator.csv', header=None, 
	             names=['c1', 'c2'])
    renaming=renaming.set_index('c2')['c1'].to_dict()
if "DAPAtlas" in args.dataset_name:
    renaming=pd.read_csv('renamingDAPAtlas.csv', header=None, 
	             names=['c1', 'c2'])
    renaming=renaming.set_index('c2')['c1'].to_dict()

if args.dataset_path[-1]!='/':
        args.dataset_path=args.dataset_path+'/'

if not args.just_mean:
    evaluate_CSV(args)
    
    
# Explicit synchronization check
torch.cuda.synchronize()

main_flag=True
if int(args.parts)>1:
    flag=True
    torch.save(flag,args.checkpoint_name+'_finished.pt')
    flags=[torch.load(args.checkpoint_name[:args.checkpoint_name.rfind('_part_')+6]+\
                      str(i)+'_of_'+args.parts+'_finished.pt') \
           for i in list(range(int(args.parts)))]
    if not all(flags):
        main_flag=False
        #only run means if all partts finished running!
    else:
        #concatenate dataframes, change checkpoint name, run means
        dice = [pd.read_csv(args.checkpoint_name[:args.checkpoint_name.rfind('_part_')+6]+\
                     str(i)+'_of_'+args.parts+'_dice.csv') \
                     for i in list(range(int(args.parts)))]
        nsd = [pd.read_csv(args.checkpoint_name[:args.checkpoint_name.rfind('_part_')+6]+\
               str(i)+'_of_'+args.parts+'_nsd.csv') \
               for i in list(range(int(args.parts)))]
        
        dice = pd.concat(dice, ignore_index=True)
        nsd = pd.concat(nsd, ignore_index=True)
        
        args.checkpoint_name=args.checkpoint_name[:args.checkpoint_name.rfind('_part_')]
        
        dice.to_csv(args.checkpoint_name+'_dice.csv', index=False)
        nsd.to_csv(args.checkpoint_name+'_nsd.csv', index=False)
    
if main_flag:
    get_means(args)
    get_means(args,stds=True)


    try:
        get_means(args,stds=False,test=True)
        get_means(args,stds=True,test=True)
    except:
        pass

    print('Finished')


    
            
    
