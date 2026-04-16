import numpy as np
import pandas as pd
import SimpleITK as sitk

def list_nonzero_seg_slices(seg: np.ndarray) -> list: 
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


def get_hist_data(seg: np.ndarray) -> dict: 
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


def get_hist_data_df(seg: np.ndarray) -> pd.DataFrame: 
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


def find_max_area_slice(gts_array: np.ndarray) -> int: 
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


def array_to_coords(data: np.ndarray):
    """Generator function to get the coordinates of all nonzero points in a 2D array."""
    for y, line in enumerate(data):
        for x, point in enumerate(line):
            if point == 1:
                yield(x, y)



def pred_to_img(mask_pred: np.ndarray, 
                spacing, 
                origin, 
                direction): 
    mask_pred_img = sitk.GetImageFromArray(mask_pred)
    mask_pred_img.SetSpacing(spacing) 
    mask_pred_img.SetOrigin(origin) 
    mask_pred_img.SetDirection(direction) 
    mask_pred_img = sitk.Cast(mask_pred_img, sitk.sitkUInt8)

    return mask_pred_img