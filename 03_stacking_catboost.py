################################################################################
############################### INITIALIZE #####################################
################################################################################
import numpy as np
import pandas as pd
import os
import warnings

Pos = []

warnings.simplefilter(action='ignore', category=FutureWarning) # Remove warning iteritems in Pool
# Warning "A value is trying to be set on a copy of a slice from a DataFrame.
# Try using .loc[row_indexer,col_indexer] = value instead" removing:
pd.options.mode.chained_assignment = None
pd.set_option("display.max_rows", 60, "display.max_columns", None)

# Read data
print(os.listdir("./BBDD Output"))
catboost_cv_train = pd.read_csv('./BBDD Output/catboost_cv_train.csv')
lasso_cv_train = pd.read_csv('./BBDD Output/lasso_cv_train.csv')
ridge_cv_train = pd.read_csv('./BBDD Output/ridge_cv_train.csv')
catboost_cv_test = pd.read_csv('./BBDD Output/catboost_cv_test.csv')
lasso_cv_test = pd.read_csv('./BBDD Output/lasso_cv_test.csv')
ridge_cv_test = pd.read_csv('./BBDD Output/ridge_cv_test.csv')

print(os.listdir("./DATA"))
train = pd.read_csv('./DATA/train.csv')
test = pd.read_csv('./DATA/test.csv')

train = train[train['ID'] != 'A1039']
train = train[train['ID'] != 'A2983']
train = train[train['ID'] != 'A3055']
train = train[train['ID'] != 'A4665']

train.reset_index(drop=True, inplace=True)


################################################################################
############################# LEVEL 1: MODEL ###################################
################################################################################

# New train and test
################################################################################

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def lodds(x):
    return np.log(x / (1 - x))


X1_train = pd.DataFrame({
    "Catboost": lodds(catboost_cv_train['catboost_pred']),
    "Lasso": lodds(lasso_cv_train['linear_pred']),
    "Ridge": lodds(ridge_cv_train['linear_pred'])
})

X1_test = pd.DataFrame({
    "Catboost": lodds(catboost_cv_test['catboost_pred']),
    "Lasso": lodds(lasso_cv_test['linear_pred']),
    "Ridge": lodds(ridge_cv_test['linear_pred'])
})

Y_train = train['TARGET']

# k-Fold Cross-Validation Function
################################################################################
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


def Model_cv(MODEL, k, X_train, X_test, y, RS, makepred=True, CatPos=None):
    # Create the k folds
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=RS)

    # first level train and test
    Level_1_train = pd.DataFrame(np.zeros((X_train.shape[0], 1)), columns=['train_yhat'])
    if makepred:
        Level_1_test = pd.DataFrame()

    # Main loop for each fold. Initialize counter
    count = 0
    for train_index, test_index in kf.split(X_train, Y_train):
        count += 1
        # Define train and test depending in which fold are we
        fold_train = X_train.loc[train_index.tolist(), :]
        fold_test = X_train.loc[test_index.tolist(), :]
        fold_ytrain = y[train_index.tolist()]
        fold_ytest = y[test_index.tolist()]

        # (k-1)-folds model adjusting
        if CatPos:
            # Prepare Pool
            pool_train = Pool(fold_train, fold_ytrain, cat_features=Pos)
            # (k-1)-folds model adjusting
            model_fit = MODEL.fit(X=pool_train)

        else:
            # (k-1)-folds model adjusting
            model_fit = MODEL.fit(fold_train, fold_ytrain)

        # Predict on the free fold to evaluate metric
        # and on train to have an overfitting-free prediction for the next level
        p_fold = MODEL.predict_proba(fold_test)[:, 1]
        p_fold_train = MODEL.predict_proba(fold_train)[:, 1]

        # Score in the free fold
        score = roc_auc_score(fold_ytest, p_fold)
        score_train = roc_auc_score(fold_ytrain, p_fold_train)
        print(k, '-cv, Fold ', count, '\t --test AUC: ', round(score, 4), '\t--train AUC: ', round(score_train, 4),
              sep='')
        # Save in Level_1_train the "free" predictions concatenated
        Level_1_train.loc[test_index.tolist(), 'train_yhat'] = p_fold

        # Predict in test to make the k model mean
        # Define name of the prediction (p_"iteration number")
        if makepred:
            name = 'p_' + str(count)
            # Predictin to real test
            real_pred = MODEL.predict_proba(X_test)[:, 1]
            # Name
            real_pred = pd.DataFrame({name: real_pred}, columns=[name])
            # Add to Level_1_test
            Level_1_test = pd.concat((Level_1_test, real_pred), axis=1)

    # Compute the metric of the total concatenated prediction (and free of overfitting) in train
    score_total = roc_auc_score(y, Level_1_train['train_yhat'])
    print('\n', k, '- cv, TOTAL AUC:', round((score_total) * 100, 4), '%')

    # mean of the k predictions in test
    if makepred:
        Level_1_test['model'] = Level_1_test.mean(axis=1)

    # Return train and test sets with predictions and the performance
    if makepred:
        return Level_1_train, pd.DataFrame({'test_yhat': Level_1_test['model']}), score_total
    else:
        return score_total


# LightGBM Level 1 Model
################################################################################
from catboost import CatBoostClassifier
from catboost import Pool

# Parameters of the CV
RS = 1234  # Seed for k-fold partition and model random part
n_folds = 5  # Number of folds

# Parameters of the model
# https://lightgbm.readthedocs.io/en/latest/Parameters.html
params = {'objective': 'Logloss',
          'learning_rate': 0.005,
          'min_data_in_leaf': 300,
          'subsample': 1,
          'rsm': 0.7,
          'l2_leaf_reg': 5,
          'random_seed': RS}

# We will define the model for various #trees and make a search of the optimum
iter = [1000, 1500, 2000]

print('\nCatboost Level 1 CV...')
print('########################################################')
scores = []
for nrounds in iter:

    print('\nn rounds: ', nrounds)

    # Define the model
    model_catboost_L1 = CatBoostClassifier()
    model_catboost_L1.set_params(**params)
    model_catboost_L1.set_params(n_estimators=nrounds)
    model_catboost_L1.set_params(verbose=False)

    s = Model_cv(model_catboost_L1, n_folds, X1_train, X1_test, Y_train, RS, makepred=False, CatPos=[])

    # Look if we are in the first test:
    if len(scores) == 0:
        max_score = float('-inf')
    else:
        max_score = max(scores)

    # If the score improves, we keep this one:
    if s >= max_score:
        print('BEST')

    # Append score
    scores.append(s)

# The best cross-validated score has been found in:
print('\n###########################################')
print('LASSO Level 0 AUC: ',round((roc_auc_score(Y_train, X1_train['Lasso'])) * 100, 4), '%')
print('Ridge Level 0 AUC: ',round((roc_auc_score(Y_train, X1_train['Ridge'])) * 100, 4), '%')
print('Catboost Level 0 AUC: ',round((roc_auc_score(Y_train, X1_train['Catboost'])) * 100, 4), '%')
print('###########################################')
print('Catboost Level 1 optimal rounds: ', iter[scores.index(max(scores))])
print('Catboost Level 1 optimal AUC: ', round(max(scores) * 100, 4), '%')
print('###########################################')


# 3) Level 1 model over the whole train with the optimal parameters and iterations from the CV
################################################################################
# Adjust optimal CV number of rounds to whole sample size:
nrounds = int(iter[scores.index(max(scores))] / (1 - 1 / n_folds))

print('\nCatboost Level 1 Fit with %d rounds...\n' % nrounds)
model_catboost_L1_TOTAL = CatBoostClassifier(n_estimators=nrounds,
                                             random_seed=RS,
                                             verbose=100)
model_catboost_L1_TOTAL.set_params(**params)
pool_train_L1 = Pool(X1_train, Y_train)
model_catboost_L1_TOTAL.fit(X=pool_train_L1)


################################################################################
################################### RESULTS ####################################
################################################################################
# Prediction
################################################################################
test['Pred'] = model_catboost_L1_TOTAL.predict_proba(X1_test)[:, 1]
outputs = pd.DataFrame(test[['ID', 'Pred']])

# Outputs to .csv
################################################################################
outputs.to_csv('./BBDD Output/outputs_stacking.csv', index=False)
print('END')
