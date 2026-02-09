"""  
For testing iterative sessions for various spatial input prompts using nnInteractive. 

Input AAuRA index file (must contain paths to imaging and mask files in NIfTI format which have not been preprocessed)
Output prediction masks, metric evaluations, and visualizations (optional) 
"""

### Dependencies ###
import click 
import SimpleITK as sitk
import numpy as np 
import pandas as pd
import math
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors 
import matplotlib.patches as mpatches 
import torch

from tqdm import tqdm
from pathlib import Path
from skimage.measure import regionprops
from joblib import Parallel, delayed
from nifti_nnInteractive import get_line_from_recist, apply_windowing, choose_windowing, pos_neg_true_visual, calc_metrics
from run_nnInteractive import initialize_session, find_first_last_slice

### Functions ###
def transform_nifti_pair(img_nifti_path: Path, 
                         gts_nifti_path: Path): 
    '''
    From a given image and mask path pair, load and convert to an array (image will be converted to be
    a nnInteractive-compatible array shape (x, y, z)) and extract necessary metadata 
    (spacing, origin, direction).

    Parameters
    ----------
    img_nifti_path: Path
        Path to CT scan 
    gts_nifti_path: Path
        Path to current ground truth segmentation mask that pairs with the inputted CT scan.

    Returns
    ----------
    img_array: np.ndarray
        The CT image converted to an array with (1, x, y, z) shape
    gts_array: np.ndarray
        The ground truth segmentation converted to an array with (z, x, y) shape. 
        Info pertaining to the mask (like RECIST measurements) will be converted
        into the appropriate format after being calculated. 
    spacing: 
        The spacing of the CT image 
    direction: 
        The direction of the CT image 
    origin: 
        The origin of the CT image 
    '''
    # Read in images 
    img = sitk.ReadImage(img_nifti_path) 
    gts = sitk.ReadImage(gts_nifti_path)

    # Convert to arrays 
    img_arr_zxy = sitk.GetArrayFromImage(img) 
    img_array = img_arr_zxy[None].transpose(0, 2, 3, 1) # Adds additional dimension required and changes shape from (1,z,x,y) to (1,x,y,z)

    gts_array = sitk.GetArrayFromImage(gts) 

    # Obtain imaging metadata info 
    spacing = img.GetSpacing()
    direction = img.GetDirection()
    origin = img.GetOrigin()

    return img_array, gts_array, spacing, direction, origin

def find_max_area_slice(gts_array: np.ndarray): 
    '''  
    Find maximum area slice of a given ground truth segmentation. Will be the slice 
    to calculate RERECIST on. 

    Parameters
    ----------
    gts_array: np.ndarray
        The ground truth segmentation array in (z,x,y) form. 

    Returns
    ----------
    max_area_slice: int 
        The index of the slice with the largest tumour area.
    '''
    slice_sums = np.sum(gts_array, axis = (1, 2)) # Get the pixel area of each slice 
    max_area_slice = np.argmax(slice_sums) # Get the slice with the largest pixel area 

    return max_area_slice

def array_to_coords(data):
    for y, line in enumerate(data):
        for x, point in enumerate(line):
            if point == 1:
                yield(x, y)

def get_prompt_points(gt2D: np.array, 
                      spacing):
    '''  
    From the ground truth segmentation with the largest pixel area, generate the four 
    points that define the corners of a bounding box whose sides are orthogonal and 
    parallel to the RECIST line along with the midpoint of the RECIST line.

    Parameters
    ----------
    gt2D: np.array
        The ground truth segmentation at the largest area slice. Expects (z, x, y) format.
    spacing: 
        The spacing of the imaging used to create the ground truth segmentation. 
        To be used in case the sampled points are out of range. 

    Returns
    ----------
    negative_pts: np.array
        Holds the coordinate information for the four corners of the bounding box 
        and two points sampled along the minor axis with the length of a buffered
        semi major axis. In [x1, y1, x2, y2, x3, y3, x4, y4, x_min1, y_min1, x_min2, y_min2] 
        form. 
    center_pt: np.array
        Holds the coordinate information for the midpoint of the RECIST line in 
        [x_cent, y_cent] form.
    recist_pts: np.array
        Holds the information for the found RERECIST line in the form 
        [x_r1, y_r1, x_r2, y_r2]
    pts_25_75: np.array
        Holds the information for two points sampled along 25% and 75% of the length
        of the RECIST line
    major_axis_length: float 
        The length of the major axis. Returned for further calculations (e.g. bounding box prompt)
    '''
    try: 
        props = regionprops(gt2D)[0]
        y_cent, x_cent = props.centroid
        orientation = props.orientation
        semi_maj_axis_len = props.major_axis_length / 2

        x_r1 = x_cent - np.sin(orientation) * semi_maj_axis_len
        y_r1 = y_cent - np.cos(orientation) * semi_maj_axis_len

        x_r2 = x_cent + np.sin(orientation) * semi_maj_axis_len
        y_r2 = y_cent + np.cos(orientation) * semi_maj_axis_len

        # This is the point that is 25% of the way along the RECIST line from (x_r1, y_r1)
        x_r25 = x_cent - np.sin(orientation) * semi_maj_axis_len / 2 
        y_r25 = y_cent - np.cos(orientation) * semi_maj_axis_len / 2

        # This is the point that is 75% of the way long the RECIST line from (x_r1, y_r1) 
        x_r75 = x_cent + np.sin(orientation) * semi_maj_axis_len / 2
        y_r75 = y_cent + np.cos(orientation) * semi_maj_axis_len / 2

        # Coordinate pairs of points along minor axis but at the length of major axis (add a bit of buffer room since if sphere, these points would want to be included in the segmentation)
        x_min1 = min(spacing[1]-1, x_cent + np.cos(orientation) * semi_maj_axis_len * 1.2)
        y_min1 = min(spacing[2]-1, y_cent - np.sin(orientation) * semi_maj_axis_len * 1.2)

        x_min2 = min(spacing[1]-1, x_cent - np.cos(orientation) * semi_maj_axis_len * 1.2)
        y_min2 = min(spacing[2]-1, y_cent + np.sin(orientation) * semi_maj_axis_len * 1.2)

        # Corners of rotated bounding box
        x1 = min(spacing[1]-1, x_r1 + np.cos(orientation) * semi_maj_axis_len)
        y1 = min(spacing[2]-1, y_r1 - np.sin(orientation) * semi_maj_axis_len)

        x2 = min(spacing[1]-1, x_r1 - np.cos(orientation) * semi_maj_axis_len)
        y2 = min(spacing[2]-1, y_r1 + np.sin(orientation) * semi_maj_axis_len)

        x3 = min(spacing[1]-1, x_r2 + np.cos(orientation) * semi_maj_axis_len)
        y3 = min(spacing[2]-1, y_r2 - np.sin(orientation) * semi_maj_axis_len)

        x4 = min(spacing[1]-1, x_r2 - np.cos(orientation) * semi_maj_axis_len)
        y4 = min(spacing[2]-1, y_r2 + np.sin(orientation) * semi_maj_axis_len)

        negative_pts = np.array([x1, y1, x2, y2, x3, y3, x4, y4, x_min1, y_min1, x_min2, y_min2])
        
        recist_pts = np.array([x_r1, y_r1, x_r2, y_r2])

        center_pt = np.array([x_cent, y_cent])

        pts_25_75 = np.array([x_r25, y_r25, x_r75, y_r75])

    except Exception as e: 
         # Usually errors will arise here if there is an issue with region props calculation and the mask being too small to calculate anything from. 
         raise Exception(f'error {e} and sum of gts is {gt2D.sum()}')
    
    return negative_pts.astype(int), center_pt.astype(int), recist_pts.astype(int), pts_25_75.astype(int), props.major_axis_length 

def transform_prompt_points(negative_pts: np.array,
                            recist_pts: np.array, 
                            pts_25_75: np.array, 
                            max_area_slice: int, 
                            img_shape: list):
    '''   
    From the points extracted from the current imaging pair, get them into nnInteractive-ready form. 

    Parameters
    ----------
    negative_pts: np.array
        Holds the coordinate information for the four corners of the bounding box 
        and two points sampled along the minor axis with the length of a buffered
        semi major axis. In [x1, y1, x2, y2, x3, y3, x4, y4, x_min1, y_min1, x_min2, y_min2] 
        form. 
    center_pt: np.array
        Holds the coordinate information for the midpoint of the RECIST line in 
        [x_cent, y_cent] form.
    recist_pts: np.array
        Holds the information for the found RERECIST line in the form 
        [x_r1, y_r1, x_r2, y_r2]
    pts_25_75: np.array
        Holds the information for two points sampled along 25% and 75% of the length
        of the RECIST line 
    max_area_slice: int
        The index of the slice with the largest tumour area.
    img_shape: list 
        The shape of the imaging array. To be used to make the RERECIST scribble. 

    Returns
    ----------
    RERECIST_SCRIB: np.ndarray 
        A 3D array with the RERECIST line on the maximum tumour area slice in (x, y, z) form. 
    BBOX_ROTATED: list 
        List of tuples representing the four corners of the bounding box of the rotated ellipse encompassing the tumour
        NOTE: All points in (y, x, z) form. 
    MIN_AX_PTS: list
        List of tuples representing the two points sampled along the minor axis with length 1.2*major_axis_length/2
        from the center point. NOTE: All points in (y, x, z) form.
    PTS_25_75: list 
        List of tuples representing two points sampled along 25% length and 75% length of the RERECIST line. 
        NOTE: All points in (y, x, z) form.
    ''' 
    # Get RERECIST scribble 
    recist_line = get_line_from_recist(recist_coords = recist_pts, slice_number = max_area_slice, img_size = img_shape)
    RERECIST_SCRIB = recist_line.transpose(1, 2, 0)

    # Get the rotated bounding box corners 
    BBOX_ROTATED = [(negative_pts[1], negative_pts[0], max_area_slice), 
                    (negative_pts[3], negative_pts[2], max_area_slice), 
                    (negative_pts[5], negative_pts[4], max_area_slice), 
                    (negative_pts[7], negative_pts[6], max_area_slice)]
    
    # Get minor axis sampled points 
    MIN_AX_PTS = [(negative_pts[9], negative_pts[8], max_area_slice), 
                  (negative_pts[11], negative_pts[10], max_area_slice)]
    
    # Get two points sampled along the RERECIST line 
    PTS_25_75 = [(pts_25_75[1], pts_25_75[0], max_area_slice), 
                 (pts_25_75[3], pts_25_75[2], max_area_slice)]
    
    return RERECIST_SCRIB, BBOX_ROTATED, MIN_AX_PTS, PTS_25_75
    
def get_centered_bbox(center_pt: np.array, 
                      major_axis_length: float, 
                      max_area_slice: int): 
    '''  
    Get a square bounding box centered on the midpoint of the RERECIST line. 

    Parameters
    ----------
    center_pt: np.array
        Contains the midpoint of the RERECIST line in [x_cent, y_cent] form.
    major_axis_length: float
        Length of the RERECIST line.
    max_area_slice: int
        The index of the slice with the largest area.

    Returns
    -----------
    bbox_2d: np.ndarray
        A bounding box with shape compatible with nnInteractive prompt input. 
        NOTE: In readme.md, bounding boxes are defined as [[x1, x2], [y1, y2], [z1, z2]], 
        but in my experience they are actually [[y1, y2], [x1, x2], [z1, z2]]. The latter
        is what is implemented here.
    '''
    #Get top left and bottom right corners of the bounding box 
    x_tl = center_pt[0] - major_axis_length / 2 
    y_tl = center_pt[1] - major_axis_length / 2 

    x_br = center_pt[0] + major_axis_length / 2 
    y_br = center_pt[1] + major_axis_length / 2

    # Note that the nnInteractive examples say the bounding box is in (x, y, z) order, but
    # in my experience, it actually expects (y, x, z) order 
    bbox_2d = [[min(512, int(y_tl)), min(512, int(y_br))], [min(512, int(x_tl)), min(512, int(x_br))], [max_area_slice, max_area_slice + 1]]

    return bbox_2d

def set_session_img(image: np.ndarray, 
                    session): 
    '''  
    Initialize the current image with the session to prepare for multiple prompt testing. 

    Parameters
    ----------
    image: np.ndarray
        CT image in (1, x, y, z) form 
    session: 
        An object from nnInteractive that holds the initialized session 

    Returns 
    ----------
    session: 
        The session object now with the current imaging and target buffers set
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

    return session 

def infer_rerecist(session_img,
                   rerecist: np.ndarray): 
    '''  
    Run inference using the current session initialized with image and matching RERECIST 
    measurement. Assumes any session resetting will be handled outside of this function. 

    Parameters
    ----------
    session_img: 
        An nnInteractive object of the session that has been initialized and has image and target
        buffer set. 
    rerecist: np.ndarray
        The RERECIST information in an nnInteractive-ready format. 

    Returns
    ----------
    mask_pred_zxy: np.ndarray
        The predicted mask transposed back into the NIfTI format (z, x, y)
    '''
    # Do inference with RERECIST prompt
    session_img.add_scribble_interaction(rerecist, include_interaction=True)
    results = session_img.target_buffer.clone()

    # Transpose outputs back to NIfTI coordinate format for saving and metric calculation 
    mask_pred_zxy = results.cpu().numpy().transpose(2, 0, 1)

    return mask_pred_zxy

def infer_bbox_2d(session_img, 
                  bbox_2d: list): 
    '''  
    Run inference using a 2D bounding box on the largest tumour area slice. Assumes any session resetting will
    be handled outside of this function. 

    Parameters
    ----------
    session_img: 
        An nnInteractive object of the session that has been initialized and has image and target buffer set. 
    bbox_2d: list
        A bounding box with shape compatible with nnInteractive prompt input. 
    '''
    # Do inference using the bounding box provided 
    session_img.add_bbox_interaction(bbox_coords = bbox_2d, include_interaction = True)
    results = session_img.target_buffer.clone()

    # Transpose outputs back to NIfTI coordinate format for saving and metric calculation 
    mask_pred_zxy = results.cpu().numpy().transpose(2, 0, 1)

    return mask_pred_zxy

def infer_25_75_pts(session_img, 
                    pts_25_75: list): 
    '''  
    Run inference using the two points sampled along 1/4 and 3/4 length of the RERECIST line. Assumes any 
    session resetting will be handled outside of this function. 

    Parameters 
    ----------
    session_img: 
        An nnInteractive object of the session that has been initialized and has image and target buffer set. 
    pts_25_75:
        List of tuples representing two points sampled along 25% length and 75% length of the RERECIST line. 
        NOTE: All points in (y, x, z) form.
    
    Returns
    ----------
    mask_pred_zxy: np.ndarray
        The predicted mask transposed back into the NIfTI format (z, x, y)
    '''
    # Add each of the sampled points as positive interactions for inference 
    session_img.add_point_interaction(pts_25_75[0], include_interaction=True) 
    session_img.add_point_interaction(pts_25_75[1], include_interaction=True) 

    results = session_img.target_buffer.clone() 

    # Transpose outputs back to NIfTI coordinate format for saving and metric calculation 
    mask_pred_zxy = results.cpu().numpy().transpose(2, 0, 1)

    return mask_pred_zxy

def infer_bbox_rotated(session_img, 
                       bbox_rotated_pts: list):
    '''  
    Run inference using the four corners of the rotated bounding box. Assumes any session resetting will 
    be handled outside of this function. TO BE RUN AFTER THERE IS A SEGMENTATION AS THESE ARE NEGATIVE 
    POINTS.

    Parameters
    ----------
    session_img: 
        An nnInteractive object of the session that has been initialized and has image and target buffer set. 
    bbox_rotated_pts: list 
        List of tuples representing the four corners of the bounding box of the rotated ellipse encompassing the tumour
        NOTE: All points in (y, x, z) form.

    Returns
    ----------
    mask_pred_zxy: np.ndarray
        The predicted mask transposed back into the NIfTI format (z, x, y)
    '''
    # Add each of the four corners as an negative interaction for inference 
    session_img.add_point_interaction(bbox_rotated_pts[0], include_interaction=False)
    session_img.add_point_interaction(bbox_rotated_pts[1], include_interaction=False)
    session_img.add_point_interaction(bbox_rotated_pts[2], include_interaction=False)
    session_img.add_point_interaction(bbox_rotated_pts[3], include_interaction=False)

    results = session_img.target_buffer.clone()

    # Transpose outputs back to NIfTI coordinate format for saving and metric calculation 
    mask_pred_zxy = results.cpu().numpy().transpose(2, 0, 1)

    return mask_pred_zxy

def infer_min_ax_pts(session_img, 
                     min_ax_pts: list): 
    '''  
    Run inference using the sampled points from minor axis orientation (length 20% longer than semi-major axis length).
    Assumes any session resetting will be handled outside of this function. TO BE RUN AFTER THERE IS A
    SEGEMNTATION AS THESE ARE NEGATIVE POINTS. 

    Parameters
    ----------
    session_img: 
        An nnInteractive object of the session that has been initialized and has image and target buffer set. 
    min_ax_pts: list 
        List of tuples representing the two points sampled along the minor axis with length 1.2*major_axis_length/2
        from the center point. NOTE: All points in (y, x, z) form.
    
    Returns
    ----------
    mask_pred_zxy: np.ndarray
        The predicted mask transposed back into the NIfTI format (z, x, y)
    '''
    # Add the minor axis sampled points as negative interactions
    session_img.add_point_interaction(min_ax_pts[0], include_interaction=False)
    session_img.add_point_interaction(min_ax_pts[1], include_interaction=False)

    results = session_img.target_buffer.clone() 

    # Transpose outputs back to NIfTI coordinate format for saving and metric calculation 
    mask_pred_zxy = results.cpu().numpy().transpose(2, 0, 1)

    return mask_pred_zxy

def slice_visual_nnint(image, 
                        mask_preds, 
                        gt_masks, 
                        slice_idx: int, 
                        full_savepath: Path, 
                        prompts: list, 
                        prompt_list: list):
    '''
    Adjusted visualization from the inference_example_3D.ipynb example notebook that is in the BiomedParse repo. Saves a figure 
    showing the specified slice of the original image, the ground truth mask overlayed, and the predicted mask overlayed with the 
    prompts used to create the mask visible.  

    Parameters
    ----------
    image: 
        The array containing the original image data (same shape as mask_preds and gt_masks)
    mask_preds: 
        The array containing the predicted mask values (same shape as the image and gt_masks) 
    gt_masks: 
        The array containing the ground truth mask values (same shape as the image and mask_preds) 
    slice_idx: int 
        The slice index to view
    full_savepath: Path
        Should contain where to save the path and what to call the file outputted
    prompts: list
        A list of prompts used in the inference (e.g. [RERECIST_SCRIB, BBOX_ROTATED])
    prompt_names: list 
        A list of prompt names (as strings) that are in the prompt list (e.g. [rerecist, bbox_rotated]). 
        Must match order of prompt list. Accepts all lowercase versions of the prompt variables. 
        Options accepted are: rerecist_scrib, bbox_rotated, bbox_2d, min_ax_pts, pts_25_75
    '''
    slice_image = image[slice_idx]
    slice_mask = mask_preds[slice_idx]
    slice_gt   = gt_masks[slice_idx]

    # 1) Compute the mapping
    unique_ids = np.unique(np.concatenate((slice_mask, slice_gt)))
    id_map = {orig_id: new_i for new_i, orig_id in enumerate(unique_ids)}

    slice_mask_mapped = np.vectorize(id_map.get)(slice_mask)
    slice_gt_mapped   = np.vectorize(id_map.get)(slice_gt)

    slice_mask_mapped = np.ma.masked_where(slice_mask_mapped == 0, slice_mask_mapped)
    slice_gt_mapped = np.ma.masked_where(slice_gt_mapped == 0, slice_gt_mapped)

    # 2) Build a *discrete* colormap of size len(unique_ids)
    #    so that cmap(k) gives exactly the k-th color.
    cmap = plt.get_cmap('tab20', len(unique_ids)-1)

    # 3) Plot
    fig, axes = plt.subplots(1, 3, figsize=(8, 3))

    axes[0].imshow(slice_image, cmap="gray")
    axes[0].set_title("Original Image Slice")
    axes[0].axis("off")

    axes[1].imshow(slice_image, cmap='gray')
    axes[1].imshow(slice_gt_mapped, cmap=cmap, interpolation='nearest', alpha = 0.5)
    axes[1].set_title("Ground Truth Masks")
    axes[1].axis("off")

    axes[2].imshow(slice_image, cmap='gray')
    axes[2].imshow(slice_mask_mapped, cmap=cmap, interpolation='nearest', alpha = 0.5)
    # Plot the prompts given 
    if 'rerecist_scrib' in prompt_list: 
        idx = prompt_list.index('rerecist_scrib')
        rerecist_info = prompts[idx]
        rerecist_zxy = rerecist_info[0, :, :, :].transpose(2, 0, 1)
        line_data = list(array_to_coords(rerecist_zxy[slice_idx]))
        x, y = zip(*line_data)
        axes[2].plot(x, y, 'r')
    if 'bbox_rotated' in prompt_list: 
        idx = prompt_list.index('bbox_rotated') 
        bbox_rotated = prompts[idx]
        axes[2].scatter(bbox_rotated[1], bbox_rotated[0], c = 'c', marker = '.', s = 6)
        axes[2].scatter(bbox_rotated[3], bbox_rotated[2], c = 'c', marker = '.', s = 6)
        axes[2].scatter(bbox_rotated[5], bbox_rotated[4], c = 'c', marker = '.', s = 6)
        axes[2].scatter(bbox_rotated[7], bbox_rotated[6], c = 'c', marker = '.', s = 6)
    if 'bbox_2d' in prompt_list: 
        idx = prompt_list.index['bbox_2d'] 
        bbox_2d = prompts[idx]
        bbox_pt = (bbox_2d[1][0], bbox_2d[0][0]) # (x1, y1) pair
        w = bbox_2d[1][1] - bbox_2d[1][0] # x2 - x1 for width
        h = bbox_2d[0][1] - bbox_2d[0][0] # y2 - y1 for height
        bbox = mpatches.Rectangle(bbox_pt, w, h, linewidth = 2, edgecolor='red', facecolor=None, fill = False)
        axes[2].add_patch(bbox)
    if 'min_ax_pts' in prompt_list:
        idx = prompt_list.index['min_ax_pts']
        min_ax_pts = prompts[idx]
        axes[2].scatter(min_ax_pts[1], min_ax_pts[0], c = 'c', marker = '.', s = 6)
        axes[2].scatter(min_ax_pts[3], min_ax_pts[2], c = 'c', marker = '.', s = 6)
    if 'pts_25_75' in prompt_list: 
        idx = prompt_list.index['pts_25_75']
        pts_27_75 = prompts[idx]
        axes[2].scatter(pts_27_75[1], pts_27_75[0], c = 'y', marker = '.', s = 6)
        axes[2].scatter(pts_27_75[3], pts_27_75[2], c = 'y', marker = '.', s = 6)

    axes[2].set_title("Predicted Masks")
    axes[2].axis("off")

    plt.tight_layout()
    plt.subplots_adjust(bottom=0)
    
    fig.savefig(full_savepath, bbox_inches = 'tight')

def pred_to_img(mask_pred: np.ndarray, 
                spacing, 
                origin, 
                direction): 
    mask_pred_img = sitk.GetImageFromArray(mask_pred, dtype = sitk.sitkUInt8)
    mask_pred_img.SetSpacing(spacing) 
    mask_pred_img.SetOrigin(origin) 
    mask_pred_img.SetDirection(direction) 

    return mask_pred_img

def run_one_patient(model_path: Path, 
                    img_path: Path, 
                    gts_path: Path, 
                    disease_loc: str, 
                    lesion_loc: str, 
                    autozoom: bool,
                    visualization: bool = True):
    '''   
    Run combinations of iterative segmentation for one patient. Combinations are: 
    * RERECIST 
    * RERECIST with rotated bounding box corners as negative interactions 
    * RERECIST with rotated bounding box corners and offset minor axis points as negative interactions 
    * Centered bounding box with sides = major axis length
    * Two sampled points along RERECIST line as positive interactions 
    * Two sampled points along RERECIST line as positive interactions with rotated bounding box corners as 
      negative interactions 
    * Two sampled points along RERECIST line as positive interactions with rotated bounding box corners and
      offset minor axis points as negative interactions
    
    Parameters 
    ----------
    model_path: Path
        Path to model. Will be used for initialization of sessions. 
    img_path: Path
        Path to the CT imaging 
    gts_path: Path 
        Path to the ground truth segmentation
    disease_loc: str
        the location of the disease corresponding to the current file structure 
        (e.g. Abdomen, Lung, MultiSite, etc.) 
    lesion_loc: str
        Where the lesion is located anatomically (e.g. abdomen, headneck, lung, mediastinum). 
        To be used for systematic windowing for visualizations only. 
    autozoom: bool 
        Whether or not to use autozoom during inference. 
    visualization: bool 
        Whether to export visualizations. Default is True. 

    Returns
    ----------
    metrics_df: pd.DataFrame
        The evaluation of performance for all tested prompts. 
    '''
    # Get appropriate save path for the images and visualizations (if applicable)
    base_savepath = Path("data/results") / disease_loc / "/".join(str(gts_path).split("/")[:-1]).replace("images", "nnInt_prompt_test")
    
    visual_savepath = base_savepath / 'visualization'

    if not base_savepath.exists(): 
        base_savepath.mkdir(parents = True, exist_ok = True)

    # Transform data into workable format 
    img_array, gts_array, spacing, direction, origin = transform_nifti_pair(img_nifti_path = img_path, 
                                                                            gts_nifti_path = gts_path)

    # Get all points and prompts needed for inference 
    max_area_slice = find_max_area_slice(gts_array = gts_array) 
    negative_pts, center_pt, recist_pts, pts_25_75, maj_axis_len = get_prompt_points(gt2D = gts_array[max_area_slice], 
                                                                                     spacing = spacing)
    BBOX_2D = get_centered_bbox(center_pt = center_pt, 
                                major_axis_length = maj_axis_len, 
                                max_area_slice = max_area_slice)
    
    RERECIST_SCRIB, BBOX_ROTATED, MIN_AX_PTS, PTS_25_75 = transform_prompt_points(negative_pts = negative_pts, 
                                                                                  recist_pts = recist_pts, 
                                                                                  pts_25_75 = pts_25_75, 
                                                                                  max_area_slice = max_area_slice, 
                                                                                  img_shape = gts_array.shape)
    
    # Initiate session 
    curr_session = initialize_session(model_path = model_path, 
                                      autozoom = autozoom) 
    curr_session_img = set_session_img(image = img_array,
                                       session = curr_session)
    
    # Begin inference testing #
    # RERECIST prompt only
    rerecist = infer_rerecist(session_img = curr_session_img, 
                               rerecist = RERECIST_SCRIB)
    rere_metrics = calc_metrics(pred_mask = rerecist, 
                                gt_mask = gts_array, 
                                spacing = spacing, 
                                filename = str(gts_path))
    rere_metrics['prompt_type'] = 'RERECIST'
    rerecist_img = pred_to_img(mask_pred = rerecist, 
                               spacing = spacing, 
                               origin = origin, 
                               direction = direction)
    rere_mask_name = str(gts_path).split("/")[-1].replace(".nii.gz", "_pred_RERECIST.nii.gz")
    rere_savepath = base_savepath / rere_mask_name 
    sitk.WriteImage(rerecist_img, rere_savepath)

    # RERECIST prompt with 4 negative points from rotated bounding box 
    rere_bbox = infer_bbox_rotated(session_img = curr_session_img, 
                                   bbox_rotated_pts = BBOX_ROTATED)
    rere_bbox_metrics = calc_metrics(pred_mask = rere_bbox, 
                                     gt_mask = gts_array, 
                                     spacing = spacing, 
                                     filename = str(gts_path))
    rere_bbox_metrics['prompt_type'] = 'RERECIST_BBOX'
    rere_bbox_img = pred_to_img(mask_pred = rere_bbox, 
                                spacing = spacing, 
                                origin = origin, 
                                direction = direction)
    rere_bbox_name = str(gts_path).split("/")[-1].replace(".nii.gz", "_pred_RERECIST_BBOX.nii.gz")
    rere_bbox_savepath = base_savepath / rere_bbox_name 
    sitk.WriteImage(rere_bbox_img, rere_bbox_savepath)

    # RERECIST prompt with 4 negative points from rotated bounding box and 2 negative points from 
    # augmented minor axis points 
    rere_bbox_minax = infer_min_ax_pts(session_img = curr_session_img, 
                                       min_ax_pts = MIN_AX_PTS) 
    rere_bbox_minax_metrics = calc_metrics(pred_mask = rere_bbox_minax, 
                                           gt_mask = gts_array, 
                                           spacing = spacing, 
                                           filename = str(gts_path))
    rere_bbox_minax_metrics['prompt_type'] = 'RERECIST_BBOX_MINAX'
    rere_bbox_minax_img = pred_to_img(mask_pred = rere_bbox_minax, 
                                      spacing = spacing, 
                                      origin = origin, 
                                      direction = direction)
    rere_bbox_minax_name = str(gts_path).split("/")[-1].replace(".nii.gz", "_pred_RERECIST_BBOX_MINAX.nii.gz")
    rere_bbox_minax_savepath = base_savepath / rere_bbox_minax_name
    sitk.WriteImage(rere_bbox_minax_img, rere_bbox_minax_savepath)

    # Reset interactions and target buffer before inference using just centered bounding box 
    curr_session_img.set_target_buffer(torch.zeros(img_array.shape[1:], dtype=torch.uint8))
    curr_session_img.reset_interactions()

    bbox_2d = infer_bbox_2d(session_img = curr_session_img, 
                            bbox_2d = BBOX_2D)
    bbox_2d_metrics = calc_metrics(pred_mask = bbox_2d, 
                                   gt_mask = gts_array, 
                                   spacing = spacing, 
                                   filename = str(gts_path))
    bbox_2d_metrics['prompt_type'] = 'BBOX_2D'
    bbox_2d_img = pred_to_img(mask_pred = bbox_2d, 
                              spacing = spacing, 
                              origin = origin, 
                              direction = direction)
    bbox_2d_name = str(gts_path).split("/")[-1].replace(".nii.gz", "_pred_BBOX_2D.nii.gz")
    bbox_2d_savepath = base_savepath / bbox_2d_name
    sitk.WriteImage(bbox_2d_img, bbox_2d_savepath)

    # Reset interactions and target buffer before inference using two sampled RERECIST points 
    curr_session_img.set_target_buffer(torch.zeros(img_array.shape[1:], dtype=torch.uint8))
    curr_session_img.reset_interactions()

    pos_pts = infer_25_75_pts(session_img = curr_session_img, 
                              pts_25_75 = PTS_25_75)
    pos_pts_metrics = calc_metrics(pred_mask = pos_pts, 
                                   gt_mask = gts_array, 
                                   spacing = spacing, 
                                   filename = str(gts_path))  
    pos_pts_metrics['prompt_type'] = 'PTS_25_75'
    pos_pts_img = pred_to_img(mask_pred = pos_pts, 
                              spacing = spacing, 
                              origin = origin, 
                              direction = direction) 
    pos_pts_name = str(gts_path).split("/")[-1].replace(".nii.gz", "_pred_PTS_25_75.nii.gz")
    pos_pts_savepath = base_savepath / pos_pts_name
    sitk.WriteImage(pos_pts_img, pos_pts_savepath)

    # Sampled RERECIST points with 4 negative points from rotated bounding box
    pos_pts_bbox = infer_bbox_rotated(session_img = curr_session_img, 
                                      bbox_rotated_pts = BBOX_ROTATED)
    pos_pts_bbox_metrics = calc_metrics(pred_mask = pos_pts_bbox, 
                                        gt_mask = gts_array, 
                                        spacing = spacing, 
                                        filename = str(gts_path))
    pos_pts_bbox_metrics['prompt_type'] = 'PTS_25_75_BBOX'
    pos_pts_bbox_img = pred_to_img(mask_pred = pos_pts_bbox, 
                                   spacing = spacing, 
                                   origin = origin, 
                                   direction = direction)
    pos_pts_bbox_name = str(gts_path).split("/")[-1].replace(".nii.gz", "_pred_PTS_25_75_BBOX.nii.gz")
    pos_pts_bbox_savepath = base_savepath / pos_pts_bbox_name
    sitk.WriteImage(pos_pts_bbox_img, pos_pts_bbox_savepath)

    # Sampled RERECIST points with 4 negative points from rotated bounding box and 2 negative points from 
    # augmented minor axis points 
    pos_pts_bbox_minax = infer_min_ax_pts(session_img = curr_session_img, 
                                          min_ax_pts = MIN_AX_PTS)
    pos_pts_bbox_minax_metrics = calc_metrics(pred_mask = pos_pts_bbox, 
                                              gt_mask = gts_array, 
                                              spacing = spacing, 
                                              filename = str(gts_path))
    pos_pts_bbox_minax_metrics['prompt_type'] = 'PTS_25_75_BBOX_MINAX'
    pos_pts_bbox_minax_img = pred_to_img(mask_pred = pos_pts_bbox_minax, 
                                         spacing = spacing, 
                                         origin = origin, 
                                         direction = direction)
    pos_pts_bbox_minax_name = str(gts_path).split("/")[-1].replace(".nii.gz", "_pred_PTS_25_75_BBOX_MINAX.nii.gz")
    pos_pts_bbox_minax_savepath = base_savepath / pos_pts_bbox_minax_name
    sitk.WriteImage(pos_pts_bbox_minax_img, pos_pts_bbox_minax_savepath)

    # Concatenate all results together 
    metrics_df = pd.concat([rere_metrics, 
                            rere_bbox_metrics, 
                            rere_bbox_minax_metrics,
                            bbox_2d_metrics, 
                            pos_pts_metrics, 
                            pos_pts_bbox_metrics, 
                            pos_pts_bbox_minax_metrics], ignore_index = True)
     
    # Do visualizations (if flagged) 
    # rerecist, bbox_rotated, bbox_2d, min_ax_pts, pts_25_75
    if visualization: 
        # Created nested dictionary of all prompts to be looped over
        visualization_dict = {'RERECIST': {'prediction': rerecist, 
                                           'prompt_type': ['rerecist_scrib'],
                                           'prompts': [RERECIST_SCRIB]}, 
                              'RERECIST_BBOX': {'prediction': rere_bbox, 
                                                'prompt_type': ['rerecist_scrib', 'bbox_rotated'], 
                                                'prompts': [RERECIST_SCRIB, BBOX_ROTATED]}, 
                              'RERECIST_BBOX_MINAX': {'prediction': rere_bbox_minax, 
                                                      'prompt_type': ['rerecist_scrib', 'bbox_rotated', 'min_ax_pts'], 
                                                      'prompts': [RERECIST_SCRIB, BBOX_ROTATED, MIN_AX_PTS]}, 
                              'BBOX_2D': {'prediction': bbox_2d, 
                                          'prompt_type': ['bbox_2d'], 
                                          'prompts': [BBOX_2D]}, 
                              'POS_PTS': {'prediction': pos_pts, 
                                          'prompt_type': ['pts_25_75'], 
                                          'prompts': [PTS_25_75]}, 
                              'POS_PTS_BBOX': {'prediction': pos_pts_bbox, 
                                               'prompt_type': ['pts_25_75', 'bbox_rotated'], 
                                               'prompts': [PTS_25_75, BBOX_ROTATED]}, 
                              'POS_PTS_BBOX_MINAX': {'prediction': pos_pts_bbox_minax, 
                                                     'prompt_type': ['pts_25_75', 'bbox_rotated', 'min_ax_pts'], 
                                                     'prompts': [PTS_25_75, BBOX_ROTATED, MIN_AX_PTS]}
                              }
    
        # Get all visualization settings (e.g. windowing) 
        win_lvl, win_width = choose_windowing(disease_location = lesion_loc)
        img_win = apply_windowing(img_array = img_array[0,:, :, :].transpose(2, 0, 1), 
                                  window_level = win_lvl, 
                                  window_width = win_width)
        print(f"Window level chosen: {win_lvl}. Window width: {win_width}. Disease location: {lesion_loc}")

        visual_savepath = base_savepath / 'visualization'
        if not visual_savepath.exists(): 
            visual_savepath.mkdir(parents = True, exist_ok = True)

        for key, value in visualization_dict.items: 
            slice_vis_savepath = visual_savepath / Path(str(key) + '_slice_view.png')
            pos_neg_vis_savepath = visual_savepath / Path(str(key) + '_pos_neg_view.png') 

            slice_visual_nnint(image = img_win, 
                               mask_preds = value['prediction'], 
                               gt_masks = gts_array,
                               slice_idx = max_area_slice, 
                               full_savepath = slice_vis_savepath, 
                               prompts = value['prompts'], 
                               prompt_list = value['prompt_type'])
            
            pos_neg_true_visual(image = img_win, 
                                mask_preds = value['prediction'], 
                                gt_masks = gts_array, 
                                full_savepath = pos_neg_vis_savepath, 
                                window_level = win_lvl, 
                                window_width = win_width) # TODO: fix this in other file later to not need the window level here
            
    return metrics_df

@click.command() 
@click.option('--index_path')
@click.option('--disease_loc')
@click.option('--model_path') 
@click.option('--n_jobs') 
@click.option('--visualizations', type = bool, default = True) 
@click.option('--autozoom', type = bool, default = True)
def run_nnint_prompt_test(index_path: str,
                          disease_loc: str, 
                          model_path: str, 
                          n_jobs: int, 
                          visualizations: bool, 
                          autozoom: bool): 
    '''  
    Run inference, metric evaluation, and visualizations on multiple iterative prompts based on patients within the 
    AAuRA index file. 

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
    '''
    # Load in index csv 
    index_df = pd.read_csv(index_path) 

    # Inference, evaluation, and visuals. Parallelized at patient level 
    pred_results = Parallel(n_jobs = n_jobs)(delayed(run_one_patient)(model_path = model_path, 
                                                                     img_path = Path("data/procdata") / disease_loc / row['image_path'], 
                                                                     gts_path = Path("data/procdata") / disease_loc / row['mask_path'], 
                                                                     disease_loc = disease_loc, 
                                                                     lesion_loc = row['lesion_location'], 
                                                                     autozoom = autozoom, 
                                                                     visualization = visualizations)
                                            for _, row in tqdm(index_df.iterrows(), total = index_df.shape[0]))
    
    # Save all evaluation results 
    out_path = Path("data/results") / disease_loc / "/".join(index_df['image_path'].iloc[0].replace("images", "nnInt_prompt_test").split("/")[:3])

    if not out_path.exists(): 
        out_path.mkdir(parents = True, exist_ok = True)

    for result in pred_results: 
        if 'all_metrics_df' not in locals(): 
            all_metrics_df = result
        else: 
            all_metrics_df = pd.concat([all_metrics_df, result], ignore_index = True).reset_index(drop = True)

    all_metrics_df.to_csv(Path(out_path) / 'metric_eval.csv', index = False)

if __name__ == "__main__": 
    run_nnint_prompt_test()