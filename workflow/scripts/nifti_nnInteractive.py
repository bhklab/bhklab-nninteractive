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
import seaborn as sns
import SimpleITK as sitk 

from skimage.draw import line
from evaluate import Evaluator
from pathlib import Path
from joblib import Parallel, delayed
from skimage.measure import regionprops, label
from tqdm import tqdm

from skimage.measure import label

from nnInteractive.nnInteractive.inference.inference_session import nnInteractiveInferenceSession

from run_nnInteractive import initialize_session, run_recist_infer, find_first_last_slice, calc_metrics

def apply_windowing(img_array: np.ndarray,
                    window_level: int, 
                    window_width: int
                    ) -> np.ndarray:
    '''
    Window an image based on a window width (width of range of values to use) and a window level (where to center a window level). Otherwise known as clipping or clamping in image processing.
    
    Parameters
    ----------
    img_array: np.ndarray, 
        The image to be windowed 
    window_level: int
        Where to center the range defined in window_width
    window_width: int 
        How wide the of a range to include, centered on the level.  

    Returns 
    ----------
    windowed_img: np.ndarray
        The processed image with values clamped at the upper and lower value
    '''
    #Calculate upper and lower clamp values
    upper_val = window_level + window_width / 2 
    lower_val = window_level - window_width / 2 

    #Window image
    windowed_img = np.clip(img_array, lower_val, upper_val)

    return windowed_img 

def choose_windowing(disease_location: str): 
    '''  
    Determine which window level and width to use based on 
    the current dataset being used. 

    Parameters
    ----------
    disease_location: str
        The location of the disease corresponding to the window to be applied (e.g. Lung, Abodmen, Mediastinum, etc.)
    
    Returns 
    ----------
    window_level: int 
        The centering value of the window 
    window_width: int
        The width of the window
    '''
    match disease_location: 
        case 'abdomen': 
            window_level = 50
            window_width = 400
        case 'headneck': 
            window_level = 50
            window_width = 400
        case 'lung': 
            window_level = -600
            window_width = 1500
        case 'mediastinum': 
            window_level = 40
            window_width = 400
        case _: 
            raise ValueError(f"Invalid window name: {disease_location}. Please check spelling or add to this function with the correct window and level")
        
    return window_level, window_width 

def get_line_from_recist(recist_coords: np.array, 
                         slice_number: int, 
                         img_size: np.array):
    '''
    From the RECIST measurement coordinates, generate a line connecting both coordinates on the correct slice and return an np.ndarray the same shape as the image.
    Output to be compatible with the ['recist'] array of the .npz files needed for MedSAM2-RECIST.

    Parameters
    ----------
    recist_coords: array
        A list of coordinates in [x1, y1, x2, y2] format that defines the RECIST measurement 
    slice_number: int
        The slice that the measurement was taken on
    img_size: np.array
        The x, y, and z size of the image in [z_space, x_space, y_space] format
    
    Returns
    ----------
    recist_arr: np.ndarray
        A binary array of the same shape as the image with the pixels of the line = 1
    '''
    # Check to see if RECIST coordinates are in string form and if so, convert to list 
    if type(recist_coords) == str: 
        just_coords = recist_coords.strip("[]")
        recist_coords = list(map(float, just_coords.split()))
        print(recist_coords)
        
    #Generate an array in the same size as the image filled with all zeros 
    recist_arr = np.zeros((img_size[0], img_size[1], img_size[2]), dtype = int)
    
    #Round the coordinate values to their nearest integers 
    coords_round = np.rint(recist_coords).astype(int)

    #Draw line using coordinates 
    rr, cc = line(coords_round[0], coords_round[1], coords_round[2], coords_round[3])

    #Put line into the correct slice in the RECIST array of all zeros 
    recist_arr[slice_number][cc, rr] = 1

    return recist_arr

## Metrics and Visualization ##
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

def get_hist_data(seg: np.ndarray): 
    '''  
    Get the nonzero pixel counts for each slice into dictionary form. Counts to be used for histogram plot.

    Parameters
    ----------
    seg: np.ndarray 
        A 3D array containing a mask (ground truth, predicted, etc.) 
    
    Returns 
    ----------
    pix_slice_dict: dict
        A dictionary with the slice number as the keys and the nonzero pixel count as the corresponding values
    '''
    pix_slice_dict = dict() 
    for slice_idx in range(seg.shape[0]): 
        pix_count = np.count_nonzero(seg[slice_idx]) 
        pix_slice_dict[slice_idx] = pix_count

    return pix_slice_dict

def get_hist_data_df(seg): 
    '''  
    Get the nonzero pixel counts for each slice into dataframe form. To be used for the density plot to be 
    compatible with seaborn. 

    Parameters
    ----------
    seg: np.ndarray
        A 3D array containing a mask (ground truth, predicted, etc.)

    Returns
    ----------
    pix_slice_df: pd.DataFrame
        A dataframe containing the slice number and the corresponding count in their respective columns
    '''
    pix_slice_dict = {
        'slice_num': [],
        'pix_count': []
    }
    for slice_idx in range(seg.shape[0]): 
        pix_count = np.count_nonzero(seg[slice_idx]) 
        pix_slice_dict['slice_num'].append(slice_idx) 
        pix_slice_dict['pix_count'].append(pix_count) 

    pix_slice_df = pd.DataFrame(pix_slice_dict)

    return pix_slice_df

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
        The 3D array containing the original image data (assumes (z, x, y) order)
    mask_preds: 
        The array containing the predicted mask values (same shape as gt_masks). Assumes (z, x, y) order.
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
    comb_masks = mask_alt + gt_masks
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
        axes[counter].imshow(np.clip(image[i,:,:], lower_val, upper_val), cmap = 'gray') 
        axes[counter].imshow(comb_masks[i,:,:], cmap = cmap, norm = norm, interpolation = 'nearest', alpha = 0.6)
        axes[counter].axis("off")
        axes[counter].text(0.5, -0.1, f"Slice {i}", size = 11, ha ="center", transform = axes[counter].transAxes)
        counter += 1

        plt.tight_layout()

    fig.legend(handles = legend_elem, loc = 'lower right', bbox_to_anchor=(0.67, -0.15), ncol=3, frameon=False, fontsize=11)
    fig.savefig(full_savepath, bbox_inches = 'tight')

def plot_hist(gt_mask: np.ndarray, 
              pred_mask: np.ndarray, 
              full_savepath: Path, 
              prompt: str): 
    '''  
    Plot a histogram of the number of mask pixels in each of the slices. To give a quick
    view of where the model is segmenting vs. where the ground truth mask is. For
    a visual check to see how well the coordinates localize the segmentation to a 
    specific point. 

    Parameters
    ----------
    gt_mask: np.ndarray
        A 3D array containing the ground truth mask segmentation 
    pred_mask: np.ndarray 
        A 3D array containing the predicted mask segmentation 
    prompt: str
        The prompt (text, RECIST, etc.) used to generate the predicted mask 
    full_savepath: Path
        A path containing both the location for saving and the 
        name of the file to be saved.
    '''
    # Get histogram data of number of pixels in each slice 
    pred_hist_data = get_hist_data(pred_mask)
    gt_hist_data = get_hist_data(gt_mask)

    # Get count data and slice data in a form that is compatible with histogram
    pred_val, pred_weight = zip(*[(key, val) for key, val in pred_hist_data.items()])
    gt_val, gt_weight = zip(*[(key, val) for key, val in gt_hist_data.items()])

    # Create figure 
    fig, ax = plt.subplots(1, 2, figsize=(8,4))
    ax[0].hist(pred_val, weights = pred_weight, bins = gt_mask.shape[0]-1) 
    ax[0].set_ylim(0, max(max(gt_hist_data.values()), max(pred_hist_data.values())))
    ax[0].set_title('Predicted Mask')
    ax[0].set_ylabel('Pixel Count')
    ax[0].set_xlabel('Slice Number')
    ax[1].hist(gt_val, weights = gt_weight, bins = gt_mask.shape[0]-1)
    ax[1].set_ylim(0, max(max(gt_hist_data.values()), max(pred_hist_data.values())))
    ax[1].set_title('Ground Truth Mask')
    ax[1].set_xlabel('Slice Number')
    plt.figtext(0.5, -0.05, "Prompt: " + prompt, ha='center', va='top')

    # Save figure 
    fig.savefig(full_savepath, bbox_inches = 'tight')

def plot_density(gt_mask: np.ndarray, 
              pred_mask: np.ndarray,
              prompt: str, 
              full_savepath: Path): 
    '''  
    Make a density plot to showcase where most of the segmented pixels
    are located. Similar to the histogram, but this shows the density
    information of the ground truth and the predicted mask overlayed 
    on the same plot.

    Parameters
    ----------
    gt_mask: np.ndarray
        A 3D array containing the ground truth mask segmentation 
    pred_mask: np.ndarray 
        A 3D array containing the predicted mask segmentation 
    prompt: str
        The prompt used to generate the predicted mask 
    full_savepath: Path
        A path containing both the location for saving and the 
        name of the file to be saved.
    '''
    # Get data into a dataframe to be compatible with seaborn 
    gt_data = get_hist_data_df(gt_mask)
    pred_data = get_hist_data_df(pred_mask)

    # Add labels to each dataframe to identify which are ground truth 
    # and which are predicted
    gt_data["Mask Type"] = "Ground Truth"
    pred_data["Mask Type"] = "Predicted"

    # Combine dataframes into one for plotting
    all_data = pd.concat([gt_data, pred_data], axis = 0).reset_index(drop = True)
    # Check if the data has NaNs and if so, return and don't make the graph (it'll return an error otherwise)
    if all_data.isnull().values.any(): 
        print(f"NaNs present in the histogram data dataframe for: {full_savepath}. Cannot create density plot.")
        return 0

    # Create figure 
    try:
        plot = sns.displot(data = all_data, 
                x = "slice_num", 
                weights = "pix_count", 
                hue = "Mask Type",
                kind = "kde", 
                fill = True
                )
        plot.set(xlim=(0, gt_mask.shape[0]), xlabel = "Slice Number")

        plt.figtext(0.5, -0.05, "Prompt: " + str(prompt), ha='center', va='top')

        # Save figure
        plt.savefig(full_savepath, bbox_inches = 'tight')
    except ValueError:
        print(f"NaNs present during the calculation of density for: {full_savepath}. Cannot create density plot.")
        return 0
    
def slice_visual(image, 
                mask_preds, 
                gt_masks, 
                slice_idx: int, 
                full_savepath: Path, 
                text_prompts: dict = None): 
    '''
    Adjusted visualization from the inference_example_3D.ipynb example notebook that is in the BiomedParse repo. Saves a figure 
    showing the middle slice of the original image, the ground truth mask overlayed, and the predicted mask overlayed along 
    with the text prompt used to create the mask as the legend. 

    Parameters
    ----------
    image: 
        The array containing the original image data (same shape as mask_preds and gt_masks)
    mask_preds: 
        The array containing the predicted mask values (same shape as the image and gt_masks) 
    gt_masks: 
        The array containing the ground truth mask values (same shape as the image and mask_preds) 
    text_prompts: dict = None
        Contains all of the prompts used to create the predicted masks (for now only one text prompt in dict) 
    slice_idx: int 
        The slice index to view
    full_savepath: Path
        Should contain where to save the path and what to call the file outputted
    '''
    slice_id = slice_idx
    slice_image = image[slice_id]
    slice_mask = mask_preds[slice_id]
    slice_gt   = gt_masks[slice_id]

    # 1) Compute the mapping
    unique_ids = np.unique(np.concatenate((slice_mask, slice_gt)))
    id_map = {orig_id: new_i for new_i, orig_id in enumerate(unique_ids)}

    slice_mask_mapped = np.vectorize(id_map.get)(slice_mask)
    slice_gt_mapped   = np.vectorize(id_map.get)(slice_gt)

    slice_mask_mapped = np.ma.masked_where(slice_mask_mapped == 0, slice_mask_mapped)
    slice_gt_mapped = np.ma.masked_where(slice_gt_mapped == 0, slice_gt_mapped)

    # 2) Which IDs to show (drop background=0)
    mask_ids = unique_ids[1:]

    # 3) Labels for legend (only if there is a text prompt)
    if text_prompts is not None:
        legends = [text_prompts[str(i)] for i in mask_ids if str(i) in text_prompts]
        mask_ids = [i for i in mask_ids if str(i) in text_prompts]

    # 4) Build a *discrete* colormap of size len(unique_ids)
    #    so that cmap(k) gives exactly the k-th color.
    cmap = plt.get_cmap('tab20', len(unique_ids)-1)

    # 5) Create handles using integer lookup into the discrete cmap
    if text_prompts is not None: 
        handles = [
            mpatches.Patch(color=cmap(id_map[i]-1), label=txt)
            for i, txt in zip(mask_ids, legends)
        ]

    # 6) Plot
    fig, axes = plt.subplots(1, 3, figsize=(8, 3))

    axes[0].imshow(slice_image, cmap="gray")
    axes[0].set_title("Original Image Slice")
    axes[0].axis("off")

    axes[1].imshow(slice_image, cmap='gray')
    axes[1].imshow(slice_gt_mapped, cmap=cmap, interpolation='nearest', alpha = 0.6)
    axes[1].set_title("Ground Truth Masks")
    axes[1].axis("off")

    axes[2].imshow(slice_image, cmap='gray')
    axes[2].imshow(slice_mask_mapped, cmap=cmap, interpolation='nearest', alpha = 0.6)
    axes[2].set_title("Predicted Masks")
    axes[2].axis("off")

    # 7) Shared legend below, single column
    if text_prompts is not None: 
        fig.legend(handles, legends, loc='lower center', bbox_to_anchor=(0.29, -0.15), ncol=1, frameon=False, fontsize=11)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0)
    
    fig.savefig(full_savepath, bbox_inches = 'tight')

def run_inference_base(model_path: Path,
                       img_array: np.ndarray, 
                       recist_coords: np.ndarray, 
                       slice_num: int, 
                       autozoom: bool = True):
    """  
    Run inference on a given sample. Calculates the RECIST
    array from the given RECIST coordinates. 

    Parameters 
    ----------
    model_path: Path    
        Where to load the model session from
    img_array: np.ndarray
        Contains the imaging data (no windowing). Assumes normal NIfTI coordinate
        format (z, x, y)
    recist_coords: np.ndarray 
        A list of coordinates in [x1, y1, x2, y2] format
    slice_num: int
        The slice index that the RECIST coordinates are taken on
    autozoom: bool = True
        Whether or not to use the autozoom feature. Default True. 
        For more information on the affects of autozoom on reproducibility 
        and performance, please see the devnotes.md 

    Returns
    ----------
    pred_array: np.ndarray 
        Contains the predicted segmentation in (z, x, y) form
    duration: 
        How long the inference took for this sample. 
    """
    t_start = time.time()

    # Create the RECIST array 
    recist_zxy = get_line_from_recist(recist_coords = recist_coords, 
                                  slice_number = slice_num, 
                                  img_size = img_array.shape)
    
    # Preprocess image and RECIST prompt to be in the correct coordinate system for model 
    image = img_array[None].transpose(0, 2, 3, 1) #Adds fourth dimension that nnInteractive expects and changes order from (1, z, x, y) to (1, x, y, z)
    recist = recist_zxy.transpose(1, 2, 0) #Transpose from (z, x, y) to (x, y, z)

    # Perform inference 
    # Initialize session (Think this needs to be done for each sample if you're running in parallel, otherwise one open session is 
    # running all inferences, which could cause issues with the queue when resetting the interactions)
    session = initialize_session(model_path = model_path, 
                                autozoom = autozoom)
    mask_preds = run_recist_infer(image = image, 
                                  recist = recist, 
                                  session = session) 
    
    # Transpose mask prediction back to NIfTI coordinate format for saving and metric calculation
    mask_pred_zxy = mask_preds.cpu().numpy().transpose(2, 0, 1) # Must transpose back into (z, x, y) for metrics calculations

    t_end = time.time() 

    # Calculate time for inference
    duration = t_end - t_start 

    return mask_pred_zxy, duration
    
def run_infer_metric_vis(img_path: Path, 
                         model_path: Path,
                        gts_path: Path,
                        lesion_location: str,
                        disease_loc: str,
                        recist_coords: np.ndarray, 
                        slice_num: int,
                        visualization: bool = True, 
                        autozoom: bool = True): 
    """  
    Run inference, calculate metrics, and produce visualizations for one sample. 

    Parameters 
    ----------
    img_path: Path
        Where the CT image is located
    gts_path: Path 
        Where the ground truth segmentation is located
    lesion_location: str 
        Where the lesion is located (to be used for systematic windowing)
    recist_coords: np.ndarray 
        A list of coordinates in [x1, y1, x2, y2] format that defines the RECIST measurement
    slice_num: int 
        The slice that the recist_coords are taken on (after preprocessing)
    visualization: bool
        Whether or not to produce visualizations. Default is True. 
    autozoom: bool 
        Whether to use the autozoom feature for the nnInteractive model. Default is True.
    Returns 
    ----------
    durations: 
        The time it took to complete one inference. 
    metric_df: 
        The evaluation results from the current run.
    """
    ## Load image and ground truth segmentation ## 
    img = sitk.ReadImage(Path("data/procdata") / disease_loc / img_path) 
    gts = sitk.ReadImage(Path("data/procdata") / disease_loc / gts_path) 

    img_array = sitk.GetArrayFromImage(img) 
    gts_array = sitk.GetArrayFromImage(gts)
    
    ## Window image ## 
    win_lvl, win_width = choose_windowing(disease_location = lesion_location)

    print(f"Window level chosen: {win_lvl}. Window width: {win_width}. Disease location: {lesion_location}")
    img_win = apply_windowing(img_array = img_array, 
                              window_level = win_lvl, 
                              window_width = win_width)
    
    # Get appropriate save path for the images and visualizations (if applicable)
    base_savepath = Path("data/results") / disease_loc / "/".join(gts_path.split("/")[:-1]).replace("images", "predictions_nnInt")
    mask_name = gts_path.split("/")[-1].replace(".nii.gz", "_pred.nii.gz")
    image_savepath = base_savepath / mask_name 
    visual_savepath = base_savepath / 'visualization'

    if not base_savepath.exists(): 
        base_savepath.mkdir(parents = True, exist_ok = True)

    # Run inference 
    pred_seg, infer_dur = run_inference_base(model_path = model_path,
                                            img_array = img_array,  
                                            recist_coords = recist_coords, 
                                            slice_num = slice_num, 
                                            autozoom = autozoom)   
    # Save prediction 
    pred_seg_img = sitk.GetImageFromArray(pred_seg)
    pred_seg_img.SetSpacing(img.GetSpacing())
    pred_seg_img.SetOrigin(img.GetOrigin())
    pred_seg_img.SetDirection(img.GetDirection())

    sitk.WriteImage(pred_seg_img, image_savepath)

    # Calculate metrics 
    metrics_df = calc_metrics(pred_mask = pred_seg, 
                              gt_mask = gts_array, 
                              spacing = img.GetSpacing(), 
                              filename = base_savepath)
    
    durations = pd.DataFrame({'image': str(gts_path), 
                            'duration': infer_dur}, index = [0])
    
    # Export visualizations (if applicable) 
    if visualization: 
        if not visual_savepath.exists(): 
            visual_savepath.mkdir(parents = True, exist_ok = True)
        
        # Largest slice plot 
        large_slice_savepath = visual_savepath / mask_name.replace(".nii.gz", "_largeslice.png")

        slice_visual(image = img_win, 
                    mask_preds = pred_seg, 
                    gt_masks = gts_array, 
                    slice_idx = slice_num, 
                    full_savepath = large_slice_savepath)
        
        # True positive, false positive, false negative plot 
        pos_neg_savepath = visual_savepath / mask_name.replace(".nii.gz", "_posneg.png") 

        pos_neg_true_visual(image = img_win, 
                            mask_preds = pred_seg, 
                            gt_masks = gts_array, 
                            full_savepath = pos_neg_savepath)
        
        # Pixel count histogram 
        pix_hist_savepath = visual_savepath / mask_name.replace(".nii.gz", "_pixhist.png")

        plot_hist(gt_mask = gts_array, 
                  pred_mask = pred_seg, 
                  prompt = str(recist_coords), 
                  full_savepath = pix_hist_savepath)
        
        # Pixel-slice density plot 
        pix_dens_savepath = visual_savepath / mask_name.replace(".nii.gz", "_pixdens.png") 

        plot_density(gt_mask = gts_array, 
                     pred_mask = pred_seg, 
                     prompt = str(recist_coords), 
                     full_savepath = pix_dens_savepath)
    
    return metrics_df, durations    

@click.command()
@click.option('--index_path')
@click.option('--disease_loc')
@click.option('--model_path') 
@click.option('--n_jobs') 
@click.option('--visualizations', type = bool, default = True) 
@click.option('--autozoom', type = bool, default = True)
def execute_run(index_path: str, 
                disease_loc: str,
                model_path: str,
                n_jobs: int, 
                visualizations: bool = True, 
                autozoom: bool = True):
    """  
    Run inference, metric evaluation, and visualizations based
    on the patients within the index file. 

    Parameters 
    ----------
    index_path: str
        Path to the index file containing the relevant information for 
        the images, segmentations and RECIST coordinates
    disease_loc: str
        The location of the disease corresponding to the current file 
        structure (e.g. Abdomen, Lung, MultiSite, etc.)
    model_path: str
        Where the model is stored for initialization
    n_jobs: int
        How many jobs to run in parallel
    visualizations: bool
        Whether to output visualizations. Default is True
    autozoom: bool 
        Whether to use the nnInteractive autozoom feature. Default is 
        True 
    """
    # Load in index csv 
    index_df = pd.read_csv(index_path)

    # Inference, evaluation, and visuals
    pred_results, durations = zip(*Parallel(n_jobs = n_jobs)(delayed(run_infer_metric_vis)(img_path = row['image_path'], 
                         model_path = model_path,
                        gts_path = row['mask_path'],
                        lesion_location = row['lesion_location'],
                        disease_loc = disease_loc,
                        recist_coords = row['annotation_coords'], 
                        slice_num = row['largest_slice_index']) for _, row in tqdm(index_df.iterrows(), total = index_df.shape[0])))

    # Save all evaluation results 
    out_path = Path("data/results") / disease_loc / "/".join(index_df['image_path'].iloc[0].replace("images", "predictions_nnInt").split("/")[:3])
    if not out_path.exists(): 
        out_path.mkdir(parents = True, exist_ok = True)

    for evaluate in pred_results: 
        if 'all_metrics_df' not in locals(): 
            all_metrics_df = evaluate
        else: 
            all_metrics_df = pd.concat([all_metrics_df, evaluate], ignore_index = True).reset_index(drop = True)

    for dur in durations: 
        if 'duration_df' not in locals(): 
            duration_df = dur
        else: 
            duration_df = pd.concat([duration_df, dur], ignore_index = True).reset_index(drop = True)

    all_metrics_df.to_csv(Path(out_path) / 'metric_eval.csv', index = False)
    duration_df.to_csv(Path(out_path) / 'inference_times.csv', index = False)

if __name__ == "__main__": 
    execute_run()
