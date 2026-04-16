"""  
Running nnInteractive inference using the sampled positive and negative points.

Input AAuRA index file (must contain paths to imaging and mask files in NIfTI format which have not been preprocessed)
Output prediction masks, metric evaluations, and visualizations (optional) 
"""

### Dependencies ###
import click 
import SimpleITK as sitk
import numpy as np 
import pandas as pd
import torch

from tqdm import tqdm
from pathlib import Path
from joblib import Parallel, delayed
from nifti_nnInteractive import calc_metrics
from run_nnInteractive import initialize_session

from utils.loaders import transform_nifti_pair
from utils.masks import find_max_area_slice, pred_to_img
from utils.scans import apply_windowing, choose_windowing
from utils.prompts import get_prompt_points, transform_prompt_points
from utils.visualization import slice_visual_nnint, pos_neg_true_visual

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
    session_img: 
        An nnInteractive object of the session that has been initialized and has image and target buffer set. Includes the current
        added prompts
    '''
    # Do inference with RERECIST prompt
    session_img.add_scribble_interaction(rerecist, include_interaction=True)
    results = session_img.target_buffer.clone()

    # Transpose outputs back to NIfTI coordinate format for saving and metric calculation 
    mask_pred_zxy = results.cpu().numpy().transpose(2, 0, 1)

    return mask_pred_zxy, session_img

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
    
    Returns
    ----------
    mask_pred_zxy: np.ndarray
        The predicted mask transposed back into the NIfTI format (z, x, y)
    session_img: 
        An nnInteractive object of the session that has been initialized and has image and target buffer set. Includes the current
        added prompts
    '''
    # Do inference using the bounding box provided 
    session_img.add_bbox_interaction(bbox_coords = bbox_2d, include_interaction = True)
    results = session_img.target_buffer.clone()

    # Transpose outputs back to NIfTI coordinate format for saving and metric calculation 
    mask_pred_zxy = results.cpu().numpy().transpose(2, 0, 1)

    return mask_pred_zxy, session_img

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
    session_img: 
        An nnInteractive object of the session that has been initialized and has image and target buffer set. Includes the current
        added prompts
    '''
    # Add each of the sampled points as positive interactions for inference 
    session_img.add_point_interaction(pts_25_75[0], include_interaction=True) 
    session_img.add_point_interaction(pts_25_75[1], include_interaction=True) 

    results = session_img.target_buffer.clone() 

    # Transpose outputs back to NIfTI coordinate format for saving and metric calculation 
    mask_pred_zxy = results.cpu().numpy().transpose(2, 0, 1)

    return mask_pred_zxy, session_img

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
    session_img: 
        An nnInteractive object of the session that has been initialized and has image and target buffer set. Includes the current
        added prompts
    '''
    # Add each of the four corners as an negative interaction for inference 
    session_img.add_point_interaction(bbox_rotated_pts[0], include_interaction=False)
    session_img.add_point_interaction(bbox_rotated_pts[1], include_interaction=False)
    session_img.add_point_interaction(bbox_rotated_pts[2], include_interaction=False)
    session_img.add_point_interaction(bbox_rotated_pts[3], include_interaction=False)

    results = session_img.target_buffer.clone()

    # Transpose outputs back to NIfTI coordinate format for saving and metric calculation 
    mask_pred_zxy = results.cpu().numpy().transpose(2, 0, 1)

    return mask_pred_zxy, session_img

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
    session_img: 
        An nnInteractive object of the session that has been initialized and has image and target buffer set. Includes the current
        added prompts
    '''
    # Add the minor axis sampled points as negative interactions
    session_img.add_point_interaction(min_ax_pts[0], include_interaction=False)
    session_img.add_point_interaction(min_ax_pts[1], include_interaction=False)

    results = session_img.target_buffer.clone() 

    # Transpose outputs back to NIfTI coordinate format for saving and metric calculation 
    mask_pred_zxy = results.cpu().numpy().transpose(2, 0, 1)

    return mask_pred_zxy, session_img


def run_one_patient(model_path: Path, 
                    img_path: Path, 
                    gts_path: Path, 
                    disease_loc: str, 
                    lesion_loc: str, 
                    autozoom: bool,
                    visualization: bool = True):
    '''   
    Run combinations of iterative segmentation for one patient. Combinations are: 
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
    base_savepath = Path("data/results") / disease_loc / Path("/".join(str(gts_path).split("/")[:-1]).replace("images", "nnInt_pts_run"))
    
    visual_savepath = base_savepath / 'visualization'

    if not base_savepath.exists(): 
        base_savepath.mkdir(parents = True, exist_ok = True)

    # Transform data into workable format 
    img_array, gts_array, spacing, direction, origin = transform_nifti_pair(img_nifti_path = Path("data/procdata") / disease_loc / img_path, 
                                                                            gts_nifti_path = Path("data/procdata") / disease_loc / gts_path)

    # Get all points and prompts needed for inference 
    max_area_slice = find_max_area_slice(gts_array = gts_array) 
    negative_pts, _, recist_pts, pts_25_75, _ = get_prompt_points(gt2D = gts_array[max_area_slice], 
                                                                                     spacing = img_array.shape)
    
    _, BBOX_ROTATED, MIN_AX_PTS, PTS_25_75 = transform_prompt_points(negative_pts = negative_pts, 
                                                                                  recist_pts = recist_pts, 
                                                                                  pts_25_75 = pts_25_75, 
                                                                                  max_area_slice = max_area_slice, 
                                                                                  img_shape = gts_array.shape)
    
    # Initiate session 
    curr_session = initialize_session(model_path = model_path, 
                                      autozoom = autozoom) 
    curr_session_img = set_session_img(image = img_array,
                                       session = curr_session)
    
    # Reset interactions and target buffer before inference using two sampled RERECIST points 
    curr_session_img.set_target_buffer(torch.zeros(img_array.shape[1:], dtype=torch.uint8))
    curr_session_img.reset_interactions()

    _, pos_session = infer_25_75_pts(session_img = curr_session_img, 
                              pts_25_75 = PTS_25_75)

    # Sampled RERECIST points with 4 negative points from rotated bounding box
    _, pos_bbox_session = infer_bbox_rotated(session_img = pos_session, 
                                      bbox_rotated_pts = BBOX_ROTATED)
    
    # Sampled RERECIST points with 4 negative points from rotated bounding box and 2 negative points from 
    # augmented minor axis points 
    pos_pts_bbox_minax, _ = infer_min_ax_pts(session_img = pos_bbox_session, 
                                          min_ax_pts = MIN_AX_PTS)
    pos_pts_bbox_minax_metrics = calc_metrics(pred_mask = pos_pts_bbox_minax, 
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
     
    # Do visualizations (if flagged) 
    # rerecist, bbox_rotated, bbox_2d, min_ax_pts, pts_25_75
    if visualization: 
        # Created nested dictionary of all prompts to be looped over
        visualization_dict = {'POS_PTS_BBOX_MINAX': {'prediction': pos_pts_bbox_minax, 
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

        for key, value in visualization_dict.items(): 
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
            
    return pos_pts_bbox_minax_metrics

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
                                                                     img_path = row['image_path'], 
                                                                     gts_path = row['mask_path'], 
                                                                     disease_loc = disease_loc, 
                                                                     lesion_loc = row['lesion_location'], 
                                                                     autozoom = autozoom, 
                                                                     visualization = visualizations)
                                            for _, row in tqdm(index_df.iterrows(), total = index_df.shape[0]))
    
    # Save all evaluation results 
    out_path = Path("data/results") / disease_loc / "/".join(index_df['image_path'].iloc[0].replace("images", "nnInt_pts_run").split("/")[:3])

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