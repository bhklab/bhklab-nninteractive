# Developer Notes

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
