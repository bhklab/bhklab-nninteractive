import numpy as np

from skimage.measure import regionprops

from .annotations import get_line_from_recist, get_center_pt, get_recist_pts, get_negative_pts, get_recist_25_75_pts

def get_prompt_points(gt2D: np.ndarray, 
                      spacing):
    '''  
    From the ground truth segmentation with the largest pixel area, generate the four 
    points that define the corners of a bounding box whose sides are orthogonal and 
    parallel to the RECIST line along with the midpoint of the RECIST line.

    Parameters
    ----------
    gt2D: np.ndarray
        The ground truth segmentation at the largest area slice. Expects (z, x, y) format.
    spacing: 
        The spacing of the imaging used to create the ground truth segmentation. 
        To be used in case the sampled points are out of range. 
        NOTE: THIS IS REALLY THE SIZE OF THE 3D IMAGE THE SEGMENTATION WAS DONE ON.

    Returns
    ----------
    negative_pts: np.ndarray
        Holds the coordinate information for the four corners of the bounding box 
        and two points sampled along the minor axis with the length of a buffered
        semi major axis. In [x1, y1, x2, y2, x3, y3, x4, y4, x_min1, y_min1, x_min2, y_min2] 
        form. 
    center_pt: np.ndarray
        Holds the coordinate information for the midpoint of the RECIST line in 
        [x_cent, y_cent] form.
    recist_pts: np.ndarray
        Holds the information for the found RERECIST line in the form 
        [x_r1, y_r1, x_r2, y_r2]
    pts_25_75: np.ndarray
        Holds the information for two points sampled along 25% and 75% of the length
        of the RECIST line
    major_axis_length: float 
        The length of the major axis. Returned for further calculations (e.g. bounding box prompt)
    '''
    try: 
        props = regionprops(gt2D)[0]

        center_pt = get_center_pt(gt2D)
        recist_pts = get_recist_pts(gt2D)
        pts_25_75 = get_recist_25_75_pts(gt2D)
        negative_pts = get_negative_pts(gt2D, spacing, recist_pts)

    except Exception as e: 
         # Usually errors will arise here if there is an issue with region props calculation and the mask being too small to calculate anything from. 
         raise Exception(f'error {e} and sum of gts is {gt2D.sum()}')
    
    return negative_pts.astype(int), center_pt.astype(int), recist_pts.astype(int), pts_25_75.astype(int), props.axis_major_length



def transform_prompt_points(negative_pts: np.ndarray,
                            recist_pts: np.ndarray, 
                            pts_25_75: np.ndarray, 
                            max_area_slice: int, 
                            img_shape: list):
    '''   
    From the points extracted from the current imaging pair, get them into nnInteractive-ready form. 

    Parameters
    ----------
    negative_pts: np.ndarray
        Holds the coordinate information for the four corners of the bounding box 
        and two points sampled along the minor axis with the length of a buffered
        semi major axis. In [x1, y1, x2, y2, x3, y3, x4, y4, x_min1, y_min1, x_min2, y_min2] 
        form. 
    center_pt: np.ndarray
        Holds the coordinate information for the midpoint of the RECIST line in 
        [x_cent, y_cent] form.
    recist_pts: np.ndarray
        Holds the information for the found RERECIST line in the form 
        [x_r1, y_r1, x_r2, y_r2]
    pts_25_75: np.ndarray
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