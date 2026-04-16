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

from utils.masks import find_centre_slice, find_first_last_slice, list_nonzero_seg_slices
from utils.visualization import mid_slice_visual, pos_neg_true_visual

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