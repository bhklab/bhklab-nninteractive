import numpy as np


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
    Determine which window level and width to use based on the current dataset being used. 

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


