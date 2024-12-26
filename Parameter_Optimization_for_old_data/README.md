# Face Recognition with Parameter Optimization

This project implements a face recognition system using OpenCV's LBPH (Local Binary Patterns Histograms) algorithm. It includes scripts for training models, analyzing results, and identifying the best parameter configurations based on F1-score. The project files and instructions pertain specifically to the contents of the `Parameter_Optimization_for_old_data` folder.

## Getting Started

### Prerequisites
Ensure you have the following installed on your system:
- Python 3.8 or higher
- OpenCV
- NumPy
- Pandas

Install the required Python packages:
```bash
pip install opencv-python-headless numpy pandas
```

### Directory Structure
Ensure your directory contains the following structure:
```
Parameter_Optimization_for_old_data/
├── old_frames/  # Training videos split into frames
├── old_testing_photos/  # Testing photos for evaluation
├── main_for_old_frames_and_testing_photos.py  # Main script for training and testing
├── find_best_ver_for_old_data.py  # Script for analyzing results
├── haarcascade_frontalface_default.xml
├── version_results_for_old_data/  # Output folder for training results
└── version_analysis_f1_for_old_data/  # Output folder for analysis results
```

## Usage

### Step 1: Train Models and Generate Results
Run the following script to train models using different parameter combinations and save results:
```bash
python main_for_old_frames_and_testing_photos.py
```
This script will:
- Train models using frames extracted from `old_frames/`.
- Evaluate each model on `old_testing_photos/`.
- Save the results for each parameter combination in the `version_results_for_old_data/` folder. Each version folder (e.g., `ver1`, `ver2`, etc.) corresponds to a unique parameter combination.

### Step 2: Analyze Results and Identify Best Parameters
Run the following script to analyze the results and find the best parameters for each version:
```bash
python find_best_ver_for_old_data.py
```
This script will:
- Analyze the results stored in `version_results_for_old_data/`.
- Compute F1-score for different confidence thresholds.
- Save the analysis in the `version_analysis_f1_for_old_data/` folder.
- Generate a `version_summary.txt` file summarizing the best confidence threshold and average F1-score for each version.

### Step 3: Review Results
1. Open `version_analysis_f1_for_old_data/version_summary.txt` to view:
   - The best confidence threshold for each version.
   - The average F1-score for each version.

2. Navigate to `version_results_for_old_data/` to find the folder corresponding to the version with the best parameters.

3. Open the files within the selected version folder to see detailed results, including:
   - Parameter combinations used for training.
   - Prediction details and performance metrics (e.g., Error Rate, Recall, Precision, F1-score).

## Output Example
### `version_summary.txt`
```
Version: ver1
Best Confidence Threshold: 50
Average F1-Score: 0.87
---
Version: ver2
Best Confidence Threshold: 60
Average F1-Score: 0.90
---
...
```
### Version Folder Content (e.g., `ver1`)
```
Frame Count: 200
LBPH Parameters: radius=2, neighbors=10, grid_x=10, grid_y=10
...
Error Rate: 5.50%
Recall: 0.85
Precision: 0.89
F1-score: 0.87
TP: 45, FP: 5, TN: 40, FN: 10
```

## Notes
- Ensure the `old_frames/` and `old_testing_photos/` folders contain the necessary data before running the scripts.
- Adjust paths and parameters in the scripts as needed to suit your data and requirements.

## License
This project is open-source and available under the MIT License.
