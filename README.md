# machine_learning_final_project_group7
# README

This repository contains scripts and files designed for training and optimizing face identification models. Each script is tailored for specific data processing and modeling approaches. Below is a detailed explanation of each file and its purpose:

## Files Overview

### 1. `Parameter_Optimization_for_old_data`
This script is used for training and optimizing face identification models based on older datasets. The training process does not unify the background color of the images to white. It focuses on parameter tuning to achieve the best performance for datasets where the background is not normalized.

### 2. `Parameter_Optimization_for_new_data`
Similar to the first script, this file focuses on training and optimizing face identification models. . Like the previous script, it does not unify the background color to white. Parameter optimization is conducted to enhance model accuracy on newer datasets.

### 3. `task_three`
This folder contains two subdirectories, each addressing a different approach to training and data handling:

- **Subdirectory 1:** Implements face identification with data augmentation techniques. These techniques are not applied during model training to test the raw model's effectiveness on augmented data.
  
- **Subdirectory 2:** Implements a one-user-one-model approach, where individual models are trained for each user. This approach has shown to perform better compared to a single model trained for all users.

## Usage
Refer to the comments and instructions within each script or folder for guidance on execution and parameter adjustment. Each script requires its corresponding dataset to be properly configured before training.

## Notes
- Ensure all dependencies are installed before running the scripts.
- Verify dataset compatibility with the script for optimal performance.

