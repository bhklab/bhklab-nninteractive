from pathlib import Path
import SimpleITK as sitk

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