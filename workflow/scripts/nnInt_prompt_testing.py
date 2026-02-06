"""  
For testing iterative sessions for various spatial input prompts using nnInteractive. 

Input AAuRA index file (must contain paths to imaging and mask files in NIfTI format which have not been preprocessed)
Output prediction masks, metric evaluations, and visualizations (optional) 
"""

### Dependencies ###
import click 
import SimpleITK as sitk
import numpy as np 
import math

from pathlib import Path
from skimage.measure import regionprops
from nifti_nnInteractive import get_line_from_recist

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
        The index of the slice with the largest area.
    '''
    slice_sums = np.sum(gts_array, axis = (1, 2)) # Get the pixel area of each slice 
    max_area_slice = np.argmax(slice_sums) # Get the slice with the largest pixel area 

    return max_area_slice

def get_prompt_points(gt2D: np.array, 
                      spacing):
    '''  
    From the ground truth segmentation with the largest pixel area, generate the four 
    points that define the corners of a bounding box whose sides are orthogonal and 
    parallel to the RECIST line along with the midpoint of the RECIST line.

    Parameters
    ----------
    gt2D: np.array
        The ground truth segmentation at the largest area slice
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

