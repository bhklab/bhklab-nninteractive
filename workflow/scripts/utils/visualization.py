from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import math
import numpy as np

from .masks import find_first_last_slice, array_to_coords

def pos_neg_true_visual(image, 
                        mask_preds, 
                        gt_masks, 
                        full_savepath: Path, 
                        window_level: int = 40, 
                        window_width: int = 400): 
    '''
    Visualization of the selected slices based on the ground truth, showing the true positive, false positive, and false
    negative areas within these slices. 

    Parameters
    ----------
    image: 
        The 3D array containing the original image data (assumes (z, x, y) order)
    mask_preds: 
        The array containing the predicted mask values (same shape as gt_masks). Assumes (z, x, y) order.
    gt_masks: 
        The array containing the ground truth mask values (same shape as mask_preds). Assumes (z, x, y) order.
    full_savepath: Path
        Should contain where to save the path and what to call the file outputted
    window_level: int 
        Window level for image visualiation. Set to 40 default. Is the abdominal window when the width is also left as default.
    window_width: int 
        The width of the window for image visualization. Set to 400 default. Is the abdominal window when the level is also left as default.
    '''
    #Calculate min and max HU for windowing 
    upper_val = window_level + window_width / 2 
    lower_val = window_level - window_width / 2 

    # Make the predicted mask a different number to represent a different colour 
    mask_alt = mask_preds * 2 

    # Add masks together so that false negative is 1, false positive is 2, and true positive is 3 
    comb_masks = mask_alt + gt_masks
    comb_masks = np.ma.masked_where(comb_masks == 0, comb_masks)

    # Find the first and last slices that have mask in them 
    gt_min, gt_max = find_first_last_slice(gt_masks)

    num_nonzero_slices = gt_max - gt_min + 1 # need to add one to get true number. e.g. slices 0 - 5 have non zero (inclusive), true answer is 6 slices, but subtraction only will yield 5
    # Check to see if there are more than 5 slices within the ground truth mask and adjust the subplot information accordingly 
    if num_nonzero_slices < 5: 
        if num_nonzero_slices == 1: 
            return 0 #If there was only one slice in the segmentation, this is redundant with the mid slice view plot.
        subplot_slices = num_nonzero_slices
        slices_to_plot = range(gt_min, gt_max + 1)
    else: 
        subplot_slices = 5
        slices_to_plot = [gt_min, gt_min + math.floor(num_nonzero_slices/4), gt_min + math.floor(num_nonzero_slices/2), gt_min + math.floor(num_nonzero_slices * 3 / 4), gt_max]
    
    print(slices_to_plot)
    fig, axes = plt.subplots(1, subplot_slices, figsize = (15, 3)) 

    # Create colour map for mask 
    colours = ['red', 'green', 'blue']
    boundaries = [1, 2, 3, 4]
    cmap = mcolors.ListedColormap(colours)
    norm = mcolors.BoundaryNorm(boundaries, cmap.N)

    # Make legend info 
    legend_elem = [mpatches.Patch(color = 'red', label = 'False Negative'), 
                mpatches.Patch(color = 'green', label = 'False Positive'), 
                mpatches.Patch(color = 'blue', label = 'True Positive')]
    counter = 0
    for i in slices_to_plot: 
        axes[counter].imshow(np.clip(image[i,:,:], lower_val, upper_val), cmap = 'gray') 
        axes[counter].imshow(comb_masks[i,:,:], cmap = cmap, norm = norm, interpolation = 'nearest', alpha = 0.6)
        axes[counter].axis("off")
        axes[counter].text(0.5, -0.1, f"Slice {i}", size = 11, ha ="center", transform = axes[counter].transAxes)
        counter += 1

        plt.tight_layout()

    fig.legend(handles = legend_elem, loc = 'lower right', bbox_to_anchor=(0.67, -0.15), ncol=3, frameon=False, fontsize=11)
    fig.savefig(full_savepath, bbox_inches = 'tight')

    plt.close()


def slice_visual(image, 
                mask_preds, 
                gt_masks, 
                slice_idx: int, 
                full_savepath: Path, 
                text_prompts: dict = None): 
    '''
    Adjusted visualization from the inference_example_3D.ipynb example notebook that is in the BiomedParse repo. Saves a figure 
    showing the middle slice of the original image, the ground truth mask overlayed, and the predicted mask overlayed along 
    with the text prompt used to create the mask as the legend. 

    Parameters
    ----------
    image: 
        The array containing the original image data (same shape as mask_preds and gt_masks)
    mask_preds: 
        The array containing the predicted mask values (same shape as the image and gt_masks) 
    gt_masks: 
        The array containing the ground truth mask values (same shape as the image and mask_preds) 
    text_prompts: dict = None
        Contains all of the prompts used to create the predicted masks (for now only one text prompt in dict) 
    slice_idx: int 
        The slice index to view
    full_savepath: Path
        Should contain where to save the path and what to call the file outputted
    '''
    slice_id = slice_idx
    slice_image = image[slice_id]
    slice_mask = mask_preds[slice_id]
    slice_gt   = gt_masks[slice_id]

    # 1) Compute the mapping
    unique_ids = np.unique(np.concatenate((slice_mask, slice_gt)))
    id_map = {orig_id: new_i for new_i, orig_id in enumerate(unique_ids)}

    slice_mask_mapped = np.vectorize(id_map.get)(slice_mask)
    slice_gt_mapped   = np.vectorize(id_map.get)(slice_gt)

    slice_mask_mapped = np.ma.masked_where(slice_mask_mapped == 0, slice_mask_mapped)
    slice_gt_mapped = np.ma.masked_where(slice_gt_mapped == 0, slice_gt_mapped)

    # 2) Which IDs to show (drop background=0)
    mask_ids = unique_ids[1:]

    # 3) Labels for legend (only if there is a text prompt)
    if text_prompts is not None:
        legends = [text_prompts[str(i)] for i in mask_ids if str(i) in text_prompts]
        mask_ids = [i for i in mask_ids if str(i) in text_prompts]

    # 4) Build a *discrete* colormap of size len(unique_ids)
    #    so that cmap(k) gives exactly the k-th color.
    cmap = plt.get_cmap('tab20', len(unique_ids)-1)

    # 5) Create handles using integer lookup into the discrete cmap
    if text_prompts is not None: 
        handles = [
            mpatches.Patch(color=cmap(id_map[i]-1), label=txt)
            for i, txt in zip(mask_ids, legends)
        ]

    # 6) Plot
    fig, axes = plt.subplots(1, 3, figsize=(8, 3))

    axes[0].imshow(slice_image, cmap="gray")
    axes[0].set_title("Original Image Slice")
    axes[0].axis("off")

    axes[1].imshow(slice_image, cmap='gray')
    axes[1].imshow(slice_gt_mapped, cmap=cmap, interpolation='nearest', alpha = 0.6)
    axes[1].set_title("Ground Truth Masks")
    axes[1].axis("off")

    axes[2].imshow(slice_image, cmap='gray')
    axes[2].imshow(slice_mask_mapped, cmap=cmap, interpolation='nearest', alpha = 0.6)
    axes[2].set_title("Predicted Masks")
    axes[2].axis("off")

    # 7) Shared legend below, single column
    if text_prompts is not None: 
        fig.legend(handles, legends, loc='lower center', bbox_to_anchor=(0.29, -0.15), ncol=1, frameon=False, fontsize=11)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0)
    
    fig.savefig(full_savepath, bbox_inches = 'tight')

    plt.close()


def slice_visual_nnint(image, 
                        mask_preds, 
                        gt_masks, 
                        slice_idx: int, 
                        full_savepath: Path, 
                        prompts: list, 
                        prompt_list: list):
    '''
    Adjusted visualization from the inference_example_3D.ipynb example notebook that is in the BiomedParse repo. Saves a figure 
    showing the specified slice of the original image, the ground truth mask overlayed, and the predicted mask overlayed with the 
    prompts used to create the mask visible.  

    Parameters
    ----------
    image: 
        The array containing the original image data (same shape as mask_preds and gt_masks)
    mask_preds: 
        The array containing the predicted mask values (same shape as the image and gt_masks) 
    gt_masks: 
        The array containing the ground truth mask values (same shape as the image and mask_preds) 
    slice_idx: int 
        The slice index to view
    full_savepath: Path
        Should contain where to save the path and what to call the file outputted
    prompts: list
        A list of prompts used in the inference (e.g. [RERECIST_SCRIB, BBOX_ROTATED])
    prompt_names: list 
        A list of prompt names (as strings) that are in the prompt list (e.g. [rerecist, bbox_rotated]). 
        Must match order of prompt list. Accepts all lowercase versions of the prompt variables. 
        Options accepted are: rerecist_scrib, bbox_rotated, bbox_2d, min_ax_pts, pts_25_75
    '''
    slice_image = image[slice_idx]
    slice_mask = mask_preds[slice_idx]
    slice_gt   = gt_masks[slice_idx]

    # 1) Compute the mapping
    unique_ids = np.unique(np.concatenate((slice_mask, slice_gt)))
    id_map = {orig_id: new_i for new_i, orig_id in enumerate(unique_ids)}

    slice_mask_mapped = np.vectorize(id_map.get)(slice_mask)
    slice_gt_mapped   = np.vectorize(id_map.get)(slice_gt)

    slice_mask_mapped = np.ma.masked_where(slice_mask_mapped == 0, slice_mask_mapped)
    slice_gt_mapped = np.ma.masked_where(slice_gt_mapped == 0, slice_gt_mapped)

    # 2) Build a *discrete* colormap of size len(unique_ids)
    #    so that cmap(k) gives exactly the k-th color.
    cmap = plt.get_cmap('tab20', len(unique_ids)-1)

    # 3) Plot
    fig, axes = plt.subplots(1, 3, figsize=(8, 3))

    axes[0].imshow(slice_image, cmap="gray")
    axes[0].set_title("Original Image Slice")
    axes[0].axis("off")

    axes[1].imshow(slice_image, cmap='gray')
    axes[1].imshow(slice_gt_mapped, cmap=cmap, interpolation='nearest', alpha = 0.5)
    axes[1].set_title("Ground Truth Masks")
    axes[1].axis("off")

    axes[2].imshow(slice_image, cmap='gray')
    axes[2].imshow(slice_mask_mapped, cmap=cmap, interpolation='nearest', alpha = 0.5)
    # Plot the prompts given 
    if 'rerecist_scrib' in prompt_list: 
        idx = prompt_list.index('rerecist_scrib')
        rerecist_info = prompts[idx]
        rerecist_zxy = rerecist_info.transpose(2, 0, 1)
        line_data = list(array_to_coords(rerecist_zxy[slice_idx]))
        x, y = zip(*line_data)
        axes[2].plot(x, y, 'r')
    if 'bbox_rotated' in prompt_list: 
        idx = prompt_list.index('bbox_rotated') 
        bbox_rotated = prompts[idx]
        axes[2].scatter(bbox_rotated[0][1], bbox_rotated[0][0], c = 'c', marker = '.', s = 6)
        axes[2].scatter(bbox_rotated[1][1], bbox_rotated[1][0], c = 'c', marker = '.', s = 6)
        axes[2].scatter(bbox_rotated[2][1], bbox_rotated[2][0], c = 'c', marker = '.', s = 6)
        axes[2].scatter(bbox_rotated[3][1], bbox_rotated[3][0], c = 'c', marker = '.', s = 6)
    if 'bbox_2d' in prompt_list: 
        idx = prompt_list.index('bbox_2d') 
        bbox_2d = prompts[idx]
        bbox_pt = (bbox_2d[1][0], bbox_2d[0][0]) # (x1, y1) pair
        w = bbox_2d[1][1] - bbox_2d[1][0] # x2 - x1 for width
        h = bbox_2d[0][1] - bbox_2d[0][0] # y2 - y1 for height
        bbox = mpatches.Rectangle(bbox_pt, w, h, linewidth = 1, edgecolor='red', facecolor=None, fill = False)
        axes[2].add_patch(bbox)
    if 'min_ax_pts' in prompt_list:
        idx = prompt_list.index('min_ax_pts')
        min_ax_pts = prompts[idx]
        axes[2].scatter(min_ax_pts[0][1], min_ax_pts[0][0], c = 'c', marker = '.', s = 6)
        axes[2].scatter(min_ax_pts[1][1], min_ax_pts[1][0], c = 'c', marker = '.', s = 6)
    if 'pts_25_75' in prompt_list: 
        idx = prompt_list.index('pts_25_75')
        pts_27_75 = prompts[idx]
        axes[2].scatter(pts_27_75[0][1], pts_27_75[0][0], c = 'y', marker = '.', s = 6)
        axes[2].scatter(pts_27_75[1][1], pts_27_75[1][0], c = 'y', marker = '.', s = 6)

    axes[2].set_title("Predicted Masks")
    axes[2].axis("off")

    plt.tight_layout()
    plt.subplots_adjust(bottom=0)
    
    fig.savefig(full_savepath, bbox_inches = 'tight')

    plt.close()