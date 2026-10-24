# Alzheimer’s Disease Severity Prediction Using Machine Learning

## Project Overview
This project demonstrates an end-to-end Predictive Analytics workflow using the Knowledge Discovery in Databases (KDD) process. It classifies CDR-based severity groups from available demographic, cognitive, and brain-volume features in an OASIS cross-sectional dataset.

**Educational research prototype only — not a medical diagnostic tool.**

## Dataset
Download `oasis_cross-sectional.csv` from:
- ADNI Kaggle: https://www.kaggle.com/datasets/jboysen/mri-and-alzheimers
- OASIS resource: https://sites.wustl.edu/oasisbrains/

Place the dataset beside the notebook. The notebook also supports the corresponding OASIS cross-sectional Excel file if you update/confirm its filename in the loading cell.

## Target
The notebook groups CDR into:
- `0`: Normal
- `0.5`: Very mild impairment
- `>=1`: Mild/moderate group

These are project-defined groupings of the available CDR labels, not a clinical staging recommendation.

## Methodology (KDD)
1. Problem identification
2. Data selection and understanding
3. Data cleaning and preprocessing
4. Exploratory data analysis
5. Feature selection
6. Model development: Logistic Regression, Decision Tree, Random Forest, KNN, SVM
7. Evaluation: accuracy, macro precision, macro recall, macro F1, confusion matrix, ROC-AUC when appropriate
8. Interpretation using permutation feature importance

## Run Instructions
1. Install Python 3.10+ and Jupyter Notebook/Lab.
2. Install dependencies:
   `pip install pandas numpy matplotlib scikit-learn openpyxl jupyter`
3. Download the dataset and place it in this folder.
4. Open `Alzheimers_Severity_Prediction_KDD.ipynb`.
5. Run all cells from top to bottom.
6. Use the actual outputs from your run in the report and presentation.

## Important Limitations
- This is a small cross-sectional dataset.
- The task classifies recorded CDR severity; it does not predict future progression.
- MMSE is a cognitive assessment that may be strongly associated with CDR.
- Results are for educational purposes and should not be used for diagnosis.
