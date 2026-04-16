# Developer Notes

## AI for Oncology work - Katy
#### 2026-04-16  
- Updated pyproject.toml to put the jupyter notebook dependencies in the dev environment/feature
- Loading in the LesionLocator images with SimpleITK, they end up upside down in the axial slice (front of body facing bottom of slice)  
    - interesting because nii.vue automatically fixes this when displaying
- Restructured the code from nifti_nnInteractive and nnInt_prompt_testing into organized utils scripts for:  
    - _annotations_: Anything related to generating lines, bounding boxes, points etc.
    - _loaders_: Handling loading of anything 
    - _masks_: Anything related to processing of the segmentation mask
    - _plots_: Any plotting functions that make graphs of some kind
    - _prompts_: Functions for prompt generation and transformation. Will be connected to annotations a lot.
    - _scans_: Anything related to processing of the scan (CT, MR, etc.)
    - _visualization_: Any functions to visualize the scans, masks, annotations, or prompts.
- For the get_negative_points function, there is an input variable called `spacing`, but when called in `run_one_patient` in nnInt_prompt_testing, the shape of the 3D array is used. This is misleading, that variable name should be changed.

## Purpose of This Section

This section is for documenting technical decisions, challenges, and solutions encountered during your project. These notes are valuable for:

- Future you (who will forget why certain decisions were made)
- Collaborators who join the project later
- People coming from your publication who want to reproduce your work
- Anyone who might want to extend your research

### Technical Challenges
The following documents the decisions made during project development and noteworthy information regarding the behaviour of the model
``` markdown
### Non-Deterministic Behaviour When Using AutoZoom ###
[2025-12-03] - A session which has the same input (same image and line prompt given) will produce visibly different segmentations using session.add_scribble_interaction() while do_autozoom=True in the session initiation. These differences are "small" but noticeable. Behaviour becomes deterministic when setting do_autozoom=False during session initiation. Doing so also reduces the inference time by ~5-6x, but performance visibly decreases. Experimentation will be set up to quantitatively compare the segmentation performance and runtime performance. 
```
### Model Dependencies 
Please see the nnInteractive documentation for more information regarding the model's specific dependencies: https://github.com/MIC-DKFZ/nnInteractive
These should all be in the pyproject.toml file. 
