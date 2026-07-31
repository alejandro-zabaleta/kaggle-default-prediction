# Corporate Default Prediction — Kaggle Competition

Machine learning project developed during the **Applied Machine Learning to Solve 
Real-World Problems** course (MESIO UPC-UB Summer School, July 2026), as part of 
a Kaggle competition to predict corporate default from financial data.

## Problem

Predict whether a company will default (`TARGET`) based on a set of financial and 
categorical features, using ROC-AUC as the evaluation metric.

## Approach

The project follows an iterative modeling strategy, progressively combining 
simpler and more complex models:

1. **`01_baseline_linear_model.py`** — Data cleaning, feature engineering (handling 
   missing values, encoding categorical variables), and baseline linear models 
   (Lasso, Ridge) evaluated via k-fold cross-validation.
2. **`02_catboost_model.py`** — CatBoost gradient boosting model, tuned via 
   cross-validation over different numbers of estimators.
3. **`03_stacking_catboost.py`** — Stacking ensemble combining the out-of-fold 
   predictions from Lasso, Ridge, and CatBoost as inputs to a second-level 
   CatBoost model.

## Result

The final stacked model achieved an **AUC of 0.935** on the test set — an 
improvement over each individual base model.

## Tools

Python · Pandas · NumPy · Scikit-learn · CatBoost

## Notes

Data files are not included in this repository (competition dataset).
