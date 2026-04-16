import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import torch
import click
import math
import os
import time

from evaluate import Evaluator
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm

from utils.masks import find_centre_slice, find_first_last_slice

from nnInteractive.nnInteractive.inference.inference_session import nnInteractiveInferenceSession

def initialize_session(model_path: Path, 
                       autozoom: bool):
    '''
    Initialize nnInteractive using the most recent model available.

    Parameters
    ----------
    model_path: Path
        Where the checkpoint file is located. Should have been downloaded from HuggingFace and put into a known location.
    autozoom: bool 
        Whether or not to use AutoZoom during inference. Please see devnotes.md for information on how 
        this setting can effect inference performance. 

    Returns
    ----------
    session:
        The session initialized using the inputted parameters
    ''' 
    print(autozoom)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    session = nnInteractiveInferenceSession(
    device=device,  # Set inference device
    use_torch_compile=False,  # Experimental: Not tested yet
    verbose=False,
    torch_n_threads=os.cpu_count(),  # Use available CPU cores
    do_autozoom=autozoom,  # Enables AutoZoom for better patching
    use_pinned_memory=True,  # Optimizes GPU memory transfers
    )

    session.initialize_from_trained_model_folder(model_path)
    return session

def run_recist_infer(image, 
                     recist, 
                     session): 
    '''
    Run inference on an image for segmentation given a line prompt (aka RECIST or RERECIST). 

    Parameters
    ----------
    image: 4D array
        The image in the format (1, x, y, z)
    recist: 3D array 
        An array containing the line measurement on one of the slices, the rest of the slices are 0. In the format (x, y, z)
    session: 
        An object from nnInteractive that has the initialized session. 
    Returns
    ----------
    results: array 
        The array holding the predicted segmentation in the format (x, y, z).
    '''
    # Ensure that the session gets reset in case there is weird behaviour with the same image being set with different recist prompt
    # (this shouldn't happen, this is just a precaution)
    session.reset_interactions() 
    
    # Check if input image dimensions are acceptable 
    if image.ndim != 4:
        raise ValueError("Input image must be 4D with shape (1, x, y, z)")
    
    # Set image and target buffer
    session.set_image(image)
    session.set_target_buffer(torch.zeros(image.shape[1:], dtype=torch.uint8))

    # Perform inference
    session.add_scribble_interaction(recist, include_interaction = True)

    results = session.target_buffer.clone()

    return results


def mid_slice_visual(image, 
                     mask_preds, 
                     gt_masks,  
                     mid_slice: int, 
                     full_savepath: Path, 
                     window_level: int = 40, 
                     window_width: int = 400): 
    '''
    Adjusted visualization from the inference_example_3D.ipynb example notebook that is in the BiomedParse repo. Saves a figure 
    showing the middle slice of the original image, the ground truth mask overlayed, and the predicted mask overlayed along 
    with the text prompt used to create the mask as the legend. 

    Parameters
    ----------
    image: 
        The 4D array containing the original image data (assumes (1, x, y, z) order)
    mask_preds: 
        The array containing the predicted mask values (same shape as gt_masks). Assumes (x, y, z) order.
    gt_masks: 
        The array containing the ground truth mask values (same shape as mask_preds). Assumes (z, x, y) order.
    text_prompts: dict 
        Contains all of the prompts used to create the predicted masks (for now only one text prompt in dict) 
    mid_slice: int 
        The middle slice of the segmentation 
    full_savepath: Path
        Should contain where to save the path and what to call the file outputted
    window_level: int 
        Window level for image visualiation. Set to 40 default. Is the abdominal window when the width is also left as default.
    window_width: int 
        The width of the window for image visualization. Set to 400 default. Is the abdominal window when the level is also left as default.
    '''
    #Calculate min and max HU for windowing 
    upper_val = window_level + window_width / 2 
    lower_val = window_level - window_width / 2 

    #Get middle slice for image and both masks
    slice_id = mid_slice
    slice_image = image[0,:,:,slice_id]
    slice_mask = mask_preds[:,:,slice_id]
    slice_gt   = gt_masks[slice_id]

    slice_mask = np.ma.masked_where(slice_mask == 0, slice_mask)
    slice_gt = np.ma.masked_where(slice_gt == 0, slice_gt)

    colours = ['c']
    cmap = mcolors.ListedColormap(colours)

    fig, axes = plt.subplots(1, 3, figsize=(8, 3))

    axes[0].imshow(np.clip(slice_image, lower_val, upper_val), cmap="gray")
    axes[0].set_title("Original Image Slice")
    axes[0].axis("off")

    axes[1].imshow(np.clip(slice_image, lower_val, upper_val), cmap='gray')
    axes[1].imshow(slice_gt, cmap=cmap, interpolation='nearest', alpha = 0.6)
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis("off")

    axes[2].imshow(np.clip(slice_image, lower_val, upper_val), cmap='gray')
    axes[2].imshow(slice_mask, cmap=cmap, interpolation='nearest', alpha = 0.6)
    axes[2].set_title("Predicted Mask")
    axes[2].axis("off")

    plt.tight_layout()
    plt.subplots_adjust(bottom=0)
    
    fig.savefig(full_savepath, bbox_inches = 'tight')

def pos_neg_true_visual(image, 
                        mask_preds, 
                        gt_masks, 
                        full_savepath: Path, 
                        window_level: int = 40, 
                        window_width: int = 400): 
    '''
    Visualization of the selected slices based on the ground truth, showing the true positive, false positive, and false
    negative areas within these slices. 

    Parameters
    ----------
    image: 
        The 4D array containing the original image data (assumes (1, x, y, z) order)
    mask_preds: 
        The array containing the predicted mask values (same shape as gt_masks). Assumes (x, y, z) order.
    gt_masks: 
        The array containing the ground truth mask values (same shape as mask_preds). Assumes (z, x, y) order.
    full_savepath: Path
        Should contain where to save the path and what to call the file outputted
    window_level: int 
        Window level for image visualiation. Set to 40 default. Is the abdominal window when the width is also left as default.
    window_width: int 
        The width of the window for image visualization. Set to 400 default. Is the abdominal window when the level is also left as default.
    '''
    #Calculate min and max HU for windowing 
    upper_val = window_level + window_width / 2 
    lower_val = window_level - window_width / 2 

    # Make the predicted mask a different number to represent a different colour 
    mask_alt = mask_preds * 2 

    # Add masks together so that false negative is 1, false positive is 2, and true positive is 3 
    comb_masks = mask_alt + gt_masks.transpose(1, 2, 0)
    comb_masks = np.ma.masked_where(comb_masks == 0, comb_masks)

    # Find the first and last slices that have mask in them 
    gt_min, gt_max = find_first_last_slice(gt_masks)

    num_nonzero_slices = gt_max - gt_min + 1 # need to add one to get true number. e.g. slices 0 - 5 have non zero (inclusive), true answer is 6 slices, but subtraction only will yield 5
    # Check to see if there are more than 5 slices within the ground truth mask and adjust the subplot information accordingly 
    if num_nonzero_slices < 5: 
        if num_nonzero_slices == 1: 
            return 0 #If there was only one slice in the segmentation, this is redundant with the mid slice view plot.
        subplot_slices = num_nonzero_slices
        slices_to_plot = range(gt_min, gt_max + 1)
    else: 
        subplot_slices = 5
        slices_to_plot = [gt_min, gt_min + math.floor(num_nonzero_slices/4), gt_min + math.floor(num_nonzero_slices/2), gt_min + math.floor(num_nonzero_slices * 3 / 4), gt_max]
    
    print(slices_to_plot)
    fig, axes = plt.subplots(1, subplot_slices, figsize = (15, 3)) 

    # Create colour map for mask 
    colours = ['red', 'green', 'blue']
    boundaries = [1, 2, 3, 4]
    cmap = mcolors.ListedColormap(colours)
    norm = mcolors.BoundaryNorm(boundaries, cmap.N)

    # Make legend info 
    legend_elem = [mpatches.Patch(color = 'red', label = 'False Negative'), 
                mpatches.Patch(color = 'green', label = 'False Positive'), 
                mpatches.Patch(color = 'blue', label = 'True Positive')]
    counter = 0
    for i in slices_to_plot: 
        axes[counter].imshow(np.clip(image[0,:,:,i], lower_val, upper_val), cmap = 'gray') 
        axes[counter].imshow(comb_masks[:,:,i], cmap = cmap, norm = norm, interpolation = 'nearest', alpha = 0.6)
        axes[counter].axis("off")
        axes[counter].text(0.5, -0.1, f"Slice {i}", size = 11, ha ="center", transform = axes[counter].transAxes)
        counter += 1

        plt.tight_layout()

    fig.legend(handles = legend_elem, loc = 'lower right', bbox_to_anchor=(0.67, -0.15), ncol=3, frameon=False, fontsize=11)
    fig.savefig(full_savepath, bbox_inches = 'tight')

def list_nonzero_seg_slices(seg: np.ndarray): 
    '''  
    From a given 3D segmentation array, list the slices that have nonzero values (mask) in them.

    Parameters
    ----------
    seg: np.ndarray
        A 3D array containing a mask (ground truth, predicted, etc.)
    
    Returns
    ----------
    nonzero_slices: list 
        Contains all of the slice numbers where there are nonzero values
    '''
    nonzero_slices = []
    for slice_idx in range(seg.shape[0]): 
        if np.count_nonzero(seg[slice_idx]) > 0: 
            nonzero_slices.append(slice_idx)
    return nonzero_slices

def calc_metrics(pred_mask: np.ndarray, 
                 gt_mask: np.ndarray, 
                 spacing: np.ndarray, 
                 filename: str): 
    '''
    Calculate performance metrics based on the predicted and ground truth masks and save into a dataframe. 

    Parameters
    ----------
    pred_mask: np.ndarray
        The mask that was predicted by the model. 
    gt_mask: np.ndarray
        The ground truth segmentation array. 
    spacing: np.ndarray
        The spacing associated with the ground truth mask. 
    filename: str 
        The name of the npz file that is being evaluated.
    
    Returns
    ----------
    metric_df: pd.DataFrame
        Contains the evaluation performance. 
    '''
    #Initialize evaluator 
    metric_eval = Evaluator() 
    metric_dict = metric_eval(preds = pred_mask, 
                              targets = gt_mask, 
                              spacing = spacing 
                              )
    
    metric_df = pd.DataFrame(metric_dict, index = [0])

    #Add columns for the range of segmentation values (both ground truth and predicted)
    first_gts, last_gts = find_first_last_slice(gt_mask)
    try: #If no mask was predicted, this will throw an error
        first_pred, last_pred = find_first_last_slice(pred_mask) 
    except ValueError: 
        print(f"Empty predicted segmentation for file: {filename}.")
        first_pred = 0
        last_pred = 0

    # Get the list of all slices that have segmentation in them for each mask 
    mask_pred_list = list_nonzero_seg_slices(pred_mask)
    gt_list = list_nonzero_seg_slices(gt_mask) 

    metric_df['GTSliceList'] = [gt_list]
    metric_df['PredSliceList'] = [mask_pred_list]

    gts_range = [first_gts, last_gts] 
    pred_range = [first_pred, last_pred] 

    metric_df['GTSliceRange'] = [gts_range]
    metric_df['PredSliceRange'] = [pred_range]
    metric_df['filename'] = filename # To ensure we can map the results back to the segmentations 

    # Get slice interval IoU 
    metric_df['SliceIoU'] = len(list(set.intersection(set(gt_list), set(mask_pred_list)))) / len(list(set.union(set(gt_list), set(mask_pred_list))))
    metric_df['MaskUniqueSlice'] = [list(set(mask_pred_list) - set(gt_list))] # Only slices that are in predicted mask and are not in ground truth mask
    metric_df['GTUniqueSlice'] = [list(set(gt_list) - set(mask_pred_list))] # Opposite of the line above
    
    return metric_df

def run_one_sample_inference(input_npz_file: Path, 
                            model_path: Path,
                            autozoom: bool,
                            results_folder: Path, 
                            visualizations: bool = True):
    '''
    From the existing files created for the MedSAM2-RECIST testing, create new npz files containing all relevant information and 
    output the results. Optional visualizations are available. 

    Parameters 
    ----------
    input_npz_file: Path 
        The path to a single npz file to be inferred on. Must have at least a "imgs", "gts", "recist", and "spacing" key. 
    model_path: Path
        Where the checkpoint file is located. Should have been downloaded from HuggingFace and put into a known location.
    autozoom: bool 
        Whether or not to use AutoZoom during inference. Please see devnotes.md for information on how 
        this setting can effect inference performance. 
    results_folder: Path 
        Where to save the predicted mask information and visualizations (if applicable)
    visualizations: bool 
        Whether or not to save any visualizations about the prediction. Automatically set to true.
    '''
    # Initialize session (Think this needs to be done for each sample if you're running in parallel, otherwise one open session is 
    # running all inferences, which could cause issues with the queue when resetting the interactions)
    session = initialize_session(model_path = model_path, 
                                autozoom = autozoom)

    # Get save filename from input path and ensure results_folder is a Path object
    save_filename = str(input_npz_file).split("/")[-1]
    results_folder = Path(results_folder)

    print(f"Loading in npz file: {save_filename}") 
    t_start = time.time()
    # Read in data 
    curr_npz = np.load(input_npz_file)
    image = curr_npz['imgs'][None].transpose(0, 2, 3, 1) #Adds fourth dimension that nnInteractive expects and changes order from (1, z, x, y) to (1, x, y, z)
    gt_masks = curr_npz['gts'] #This does not get inputted for inference and therefore does not get transposed
    recist = curr_npz['recist'].transpose(1,2,0) #Transpose from (z, x, y) to (x, y, z)
    spacing = curr_npz['spacing']

    t_end = time.time()
    t_elapsed = t_end - t_start
    print(f"---Time to load file {save_filename}: {t_elapsed:.2f}---")

    # Run inference on the image given the RECIST prompt 
    print(f"Running inference on {save_filename}")
    t_start = time.time()
    mask_preds = run_recist_infer(image = image, 
                                  recist = recist, 
                                  session = session) 
    t_end = time.time()
    t_elapsed = t_end - t_start
    print(f"---Time to complete inference for {save_filename}: {t_elapsed:.2f}---")

    # Save relevant information out into npz files 
    save_path = results_folder / save_filename 

    # Make sure results folder exists before saving
    if not results_folder.exists(): 
        results_folder.mkdir(parents = True, exist_ok = True) 

    print(f"Saving the results for: {save_filename}") 
    t_start = time.time()
    np.savez_compressed(save_path, 
                        imgs = image, 
                        gts = gt_masks, 
                        preds = mask_preds 
                        )
    t_end = time.time() 
    t_elapsed = t_end - t_start
    print(f"---Time to save results for {save_filename}: {t_elapsed:.2f}")

    # Calculate metrics 
    print(f"Calculating metrics for {save_filename}")
    t_start = time.time()
    metrics_df = calc_metrics(pred_mask = mask_preds.cpu().numpy().transpose(2, 0, 1), # Must transpose back into (z, x, y) for metrics calculations
                              gt_mask = gt_masks, 
                              spacing = spacing, 
                              filename = save_filename)
    t_end = time.time()
    t_elapsed = t_end - t_start
    print(f"---Time to calculate metrics for {save_filename}: {t_elapsed:.2f}")

    if visualizations: 
        print(f"---Creating visualizations for {save_filename}")
        t_start = time.time()
        # Create folder to put the visualizations 
        visual_folder = results_folder / Path('visualization') 

        if not visual_folder.exists(): 
            visual_folder.mkdir(parents = True, exist_ok = True) 
        
        # Save a visualization of the middle slice of the ground truth segmentation and the predicted segmentation at that slice
        mid_slice = find_centre_slice(gt_masks) #FUTURE NOTE: If we decide to do multiple segmentations in one array, this will need to change

        midview_folder = visual_folder / 'mid_seg_slice_view'

        if not midview_folder.exists(): 
            midview_folder.mkdir(parents = True, exist_ok = True) 
        
        midview_save = midview_folder / Path(save_filename.split('.')[0] + '_midseg.png') 

        mid_slice_visual(image = image, 
                         mask_preds = mask_preds, 
                         gt_masks = gt_masks, 
                         mid_slice = mid_slice, 
                         full_savepath = midview_save)

        # Save a visualization of the 5 most central slices and the true positive, false positive, and false negative regions
        pos_neg_folder = visual_folder / 'pos_neg_true_view'

        if not pos_neg_folder.exists(): 
            pos_neg_folder.mkdir(parents = True, exist_ok = True) 

        pos_neg_save = pos_neg_folder / Path(save_filename.split('.')[0] + '_posneg.png') 

        pos_neg_true_visual(image = image, 
                            mask_preds = mask_preds, 
                            gt_masks = gt_masks, 
                            full_savepath = pos_neg_save)
        
        t_end = time.time()
        t_elapsed = t_end - t_start
        print(f"---Time to create visualizations for {save_filename}: {t_elapsed:.2f}")

    return metrics_df

@click.command()
@click.option('--model_path')
@click.option('--npz_folder')
@click.option('--results_folder') 
@click.option('--n_jobs')
@click.option('--autozoom', type = bool, default = True)
def run_create_pred_eval(model_path: Path, 
                         npz_folder: Path,
                         results_folder: Path, 
                         n_jobs: int,
                         autozoom: bool): 
    if autozoom: 
        print(f"NOTICE: Autozoom feature is set to {str(autozoom)}. Please refer to devnotes.md to see how this affects inference.")
    
    metrics_results = Parallel(n_jobs = n_jobs)(delayed(run_one_sample_inference)(input_npz_file = file, 
                                                                                model_path = model_path,
                                                                                autozoom = autozoom,
                                                                                results_folder = results_folder,  
                                                                                ) for file in tqdm(Path(npz_folder).iterdir(), total = len(list(Path(npz_folder).iterdir()))) if str(file).endswith('.npz'))
    
    for result in metrics_results: 
        if 'all_metrics_df' not in locals(): 
            all_metrics_df = result 
        else: 
            all_metrics_df = pd.concat([all_metrics_df, result], ignore_index = True).reset_index(drop = True)
    
    all_metrics_df.to_csv(Path(results_folder) / 'metric_eval.csv', index = False)

if __name__ == "__main__": 
    run_create_pred_eval()