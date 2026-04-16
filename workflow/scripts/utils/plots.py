import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pathlib import Path

from .masks import get_hist_data, get_hist_data_df

def plot_hist(gt_mask: np.ndarray, 
              pred_mask: np.ndarray, 
              full_savepath: Path, 
              prompt: str): 
    '''  
    Plot a histogram of the number of mask pixels in each of the slices. To give a quick
    view of where the model is segmenting vs. where the ground truth mask is. For
    a visual check to see how well the coordinates localize the segmentation to a 
    specific point. 

    Parameters
    ----------
    gt_mask: np.ndarray
        A 3D array containing the ground truth mask segmentation 
    pred_mask: np.ndarray 
        A 3D array containing the predicted mask segmentation 
    prompt: str
        The prompt (text, RECIST, etc.) used to generate the predicted mask 
    full_savepath: Path
        A path containing both the location for saving and the 
        name of the file to be saved.
    '''
    # Get histogram data of number of pixels in each slice 
    pred_hist_data = get_hist_data(pred_mask)
    gt_hist_data = get_hist_data(gt_mask)

    # Get count data and slice data in a form that is compatible with histogram
    pred_val, pred_weight = zip(*[(key, val) for key, val in pred_hist_data.items()])
    gt_val, gt_weight = zip(*[(key, val) for key, val in gt_hist_data.items()])

    # Create figure 
    fig, ax = plt.subplots(1, 2, figsize=(8,4))
    ax[0].hist(pred_val, weights = pred_weight, bins = gt_mask.shape[0]-1) 
    ax[0].set_ylim(0, max(max(gt_hist_data.values()), max(pred_hist_data.values())))
    ax[0].set_title('Predicted Mask')
    ax[0].set_ylabel('Pixel Count')
    ax[0].set_xlabel('Slice Number')
    ax[1].hist(gt_val, weights = gt_weight, bins = gt_mask.shape[0]-1)
    ax[1].set_ylim(0, max(max(gt_hist_data.values()), max(pred_hist_data.values())))
    ax[1].set_title('Ground Truth Mask')
    ax[1].set_xlabel('Slice Number')
    plt.figtext(0.5, -0.05, "Prompt: " + prompt, ha='center', va='top')

    # Save figure 
    fig.savefig(full_savepath, bbox_inches = 'tight')

    plt.close() 

def plot_density(gt_mask: np.ndarray, 
              pred_mask: np.ndarray,
              prompt: str, 
              full_savepath: Path): 
    '''  
    Make a density plot to showcase where most of the segmented pixels
    are located. Similar to the histogram, but this shows the density
    information of the ground truth and the predicted mask overlayed 
    on the same plot.

    Parameters
    ----------
    gt_mask: np.ndarray
        A 3D array containing the ground truth mask segmentation 
    pred_mask: np.ndarray 
        A 3D array containing the predicted mask segmentation 
    prompt: str
        The prompt used to generate the predicted mask 
    full_savepath: Path
        A path containing both the location for saving and the 
        name of the file to be saved.
    '''
    # Get data into a dataframe to be compatible with seaborn 
    gt_data = get_hist_data_df(gt_mask)
    pred_data = get_hist_data_df(pred_mask)

    # Add labels to each dataframe to identify which are ground truth 
    # and which are predicted
    gt_data["Mask Type"] = "Ground Truth"
    pred_data["Mask Type"] = "Predicted"

    # Combine dataframes into one for plotting
    all_data = pd.concat([gt_data, pred_data], axis = 0).reset_index(drop = True)
    # Check if the data has NaNs and if so, return and don't make the graph (it'll return an error otherwise)
    if all_data.isnull().values.any(): 
        print(f"NaNs present in the histogram data dataframe for: {full_savepath}. Cannot create density plot.")
        return 0

    # Create figure 
    try:
        plot = sns.displot(data = all_data, 
                x = "slice_num", 
                weights = "pix_count", 
                hue = "Mask Type",
                kind = "kde", 
                fill = True
                )
        plot.set(xlim=(0, gt_mask.shape[0]), xlabel = "Slice Number")

        plt.figtext(0.5, -0.05, "Prompt: " + str(prompt), ha='center', va='top')

        # Save figure
        plt.savefig(full_savepath, bbox_inches = 'tight')

        plt.close()
    except ValueError:
        print(f"NaNs present during the calculation of density for: {full_savepath}. Cannot create density plot.")
        return 0