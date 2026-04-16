# Just to get the metrics calculated for the 25_75 point with bounding box and minor axis points (bug that doesn't calc it)

import pandas as pd 
import SimpleITK as sitk
import click

from pathlib import Path 
from nifti_nnInteractive import calc_metrics

@click.command()
@click.option('--mit_folder') 
@click.option('--dataset')
def get_metrics_25_75(mit_folder: Path, 
                      dataset: str): 
    '''  
    Calculate the metrics that didn't get calculated the first run. 

    Parameters
    ----------
    mit_folder: Path
        Full path to the folder directly under the nnInt_prompt_test folder 
    dataset: str
        The dataset that is referenced in the mit_folder. For making unique save names.
    '''
    for path in Path(mit_folder).iterdir(): 
        if path.is_dir():
            for seg in path.iterdir(): 
                if seg.is_dir(): 
                    for pred in seg.iterdir():
                        if 'pred_PTS_25_75_BBOX_MINAX.nii.gz' in str(pred): 
                            curr_seg = sitk.ReadImage(pred)
                            curr_seg_arr = sitk.GetArrayFromImage(curr_seg)

                            temp_path = str(pred).replace("results", "procdata")
                            temp_path = temp_path.replace("nnInt_prompt_test", "images")
                            gts_path = temp_path.replace("_pred_PTS_25_75_BBOX_MINAX", "")

                            gts = sitk.ReadImage(gts_path) 
                            spacing = gts.GetSpacing()
                            gts_array = sitk.GetArrayFromImage(gts)

                            curr_metrics = calc_metrics(pred_mask = curr_seg_arr, 
                                                gt_mask = gts_array, 
                                                spacing = spacing, 
                                                filename = str(gts_path))
                            curr_metrics['prompt_type'] = 'PTS_25_75_BBOX_MINAX'

                            if 'all_metrics' not in locals(): 
                                all_metrics = curr_metrics 
                            else: 
                                all_metrics = pd.concat([all_metrics, curr_metrics], ignore_index=True)
    
    all_metrics.to_csv(dataset + '_pts_25_75_bbox_minax_metrics.csv', index = False)
                
if __name__ == '__main__': 
    get_metrics_25_75()
