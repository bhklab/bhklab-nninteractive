import numpy as np
from skimage.draw import line
from skimage.measure import regionprops

from masks import find_first_last_slice

def get_line_from_recist(recist_coords: np.ndarray, 
                         slice_number: int, 
                         img_size: np.ndarray):
    '''
    From the RECIST measurement coordinates, generate a line connecting both coordinates on the correct slice and return an np.ndarray the same shape as the image.
    Output to be compatible with the ['recist'] array of the .npz files needed for MedSAM2-RECIST.

    Parameters
    ----------
    recist_coords: array
        A list of coordinates in [x1, y1, x2, y2] format that defines the RECIST measurement 
    slice_number: int
        The slice that the measurement was taken on
    img_size: np.ndarray
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


def get_centered_bbox(center_pt: np.ndarray, 
                      major_axis_length: float, 
                      max_area_slice: int): 
    '''  
    Get a square bounding box centered on the midpoint of the RERECIST line. 

    Parameters
    ----------
    center_pt: np.ndarray
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


def get_bbox_from_line(recist_arr): 
    '''ASSUMES IT'S IN Z, X, Y FORM'''
    # Find slice with the line in it 
    first, _ = find_first_last_slice(recist_arr)
    first = int(first)
    recist_slice = recist_arr[first]
    
    #Get properties of recist line 
    line_props = regionprops(recist_slice)[0]
    y_cent, x_cent = line_props.centroid
    maj_ax_len = line_props.axis_major_length

    #Get top left and bottom right corners of the bounding box 
    x_tl = x_cent - maj_ax_len / 2 
    y_tl = y_cent - maj_ax_len / 2 

    x_br = x_cent + maj_ax_len / 2 
    y_br = y_cent + maj_ax_len / 2

    # Note that the nnInteractive examples say the bounding box is in (x, y, z) order, but
    # in my experience, it actually expects (y, x, z) order 
    bbox = [[min(512, int(y_tl)), min(512, int(y_br))], [min(512, int(x_tl)), min(512, int(x_br))], [first, first + 1]]

    return bbox


def get_slice_properties(mask_slice: np.ndarray) -> tuple[float, float, float, float, float]:
    """Utility function for prompt generation function to get properties of a given mask slice."""
    try: 
        props = regionprops(mask_slice)[0]
        y_cent, x_cent = props.centroid
        orientation = props.orientation
        semi_maj_axis_len = props.axis_major_length / 2
    except Exception as e: 
         # Usually errors will arise here if there is an issue with region props calculation and the mask being too small to calculate anything from. 
         raise Exception(f'error {e} and sum of mask slice is {mask_slice.sum()}')

    return x_cent, y_cent, orientation, semi_maj_axis_len


def get_center_pt(mask_slice: np.ndarray) -> np.ndarray:
    """Get the coordinates of the center point of the ROI in a mask array.
    
    Parameters
    ----------
    mask_slice: np.ndarray
        The ground truth segmentation at the largest area slice. Expects (z, x, y) format.
    
    Returns
    -------
    center_pt: np.ndarray
        Holds the coordinate information for the midpoint of the RECIST line in [x_cent, y_cent] form.
    """
    x_cent, y_cent, _orientation, _semi_maj_axis_len = get_slice_properties(mask_slice)
    center_pt = np.array([x_cent, y_cent])
    
    return center_pt


def get_recist_pts(mask_slice: np.ndarray) -> np.ndarray:
    """Get the coordinates of the endpoints of the an automatically generated RECIST line in a mask array. 
    
    Parameters
    ----------
    mask_slice: np.ndarray[int, int]
        The ground truth segmentation at the largest area slice. Expects (z, x, y) format.
    
    Returns
    -------
    recist_pts: np.ndarray[float, float, float, float]
        Holds the information for the found RERECIST line in the form [x_r1, y_r1, x_r2, y_r2]
    """
    x_cent, y_cent, orientation, semi_maj_axis_len = get_slice_properties(mask_slice)

    x_r1 = x_cent - np.sin(orientation) * semi_maj_axis_len
    y_r1 = y_cent - np.cos(orientation) * semi_maj_axis_len

    x_r2 = x_cent + np.sin(orientation) * semi_maj_axis_len
    y_r2 = y_cent + np.cos(orientation) * semi_maj_axis_len

    recist_pts = np.array([x_r1, y_r1, x_r2, y_r2])
    
    return recist_pts


def get_negative_pts(mask_slice: np.ndarray,
                     spacing: np.ndarray,
                     recist_pts: np.ndarray | None = None
                    ) -> np.ndarray:
    
    """Get the coordinates of the four corners of the bounding box and two points sampled along the minor axis with the length of a buffered semi major axis.

    Parameters
    ----------
    mask_slice: np.ndarray
        The ground truth segmentation at the largest area slice. Expects (z, x, y) format.
    spacing: 
        The size of the image used to create the ground truth segmentation. 
        To be used in case the sampled points are out of range. 

    Returns
    -------
    negative_pts: np.ndarray
        Holds the coordinate information for the four corners of the bounding box 
        and two points sampled along the minor axis with the length of a buffered
        semi major axis. In [x1, y1, x2, y2, x3, y3, x4, y4, x_min1, y_min1, x_min2, y_min2] 
        form. 
    """
    x_cent, y_cent, orientation, semi_maj_axis_len = get_slice_properties(mask_slice)

    if recist_pts is None:
        recist_pts = get_recist_pts(mask_slice)
    # Separate out the recist points
    x_r1, y_r1, x_r2, y_r2 = recist_pts

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

    return negative_pts


def get_recist_25_75_pts(mask_slice: np.ndarray) -> np.ndarray:
    """
    Get the coordinates of points 25% and 75% of the way along the maximum diameter line (RECIST). 

    Parameters
    ----------
    mask_slice: np.ndarray
        The ground truth segmentation at the largest area slice. Expects (z, x, y) format.
    
    Returns
    -------
    pts_25_75: np.ndarray
        Holds the information for two points sampled along 25% and 75% of the length
        of the maximum diameter line in the mask. In [x_r25, y_r25, x_r75, y_r75] form.
    """
    x_cent, y_cent, orientation, semi_maj_axis_len = get_slice_properties(mask_slice)

    # This is the point that is 25% of the way along the RECIST line from (x_r1, y_r1)
    x_r25 = x_cent - np.sin(orientation) * semi_maj_axis_len / 2 
    y_r25 = y_cent - np.cos(orientation) * semi_maj_axis_len / 2

    # This is the point that is 75% of the way long the RECIST line from (x_r1, y_r1) 
    x_r75 = x_cent + np.sin(orientation) * semi_maj_axis_len / 2
    y_r75 = y_cent + np.cos(orientation) * semi_maj_axis_len / 2

    pts_25_75 = np.array([x_r25, y_r25, x_r75, y_r75])

    return pts_25_75