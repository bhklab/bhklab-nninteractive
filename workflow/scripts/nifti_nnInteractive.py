import click
import numpy as np
import pandas as pd
import SimpleITK as sitk 
import time

from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm

from .run_nnInteractive import initialize_session, run_recist_infer, calc_metrics

from utils.annotations import get_line_from_recist
from utils.plots import plot_hist, plot_density
from utils.scans import apply_windowing, choose_windowing
from utils.visualization import pos_neg_true_visual, slice_visual


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
