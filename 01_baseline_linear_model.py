################################################################################
############################### INITIALIZE #####################################
################################################################################
import numpy as np
import pandas as pd
import os

Pool = None
Pos = None

# Warning "A value is trying to be set on a copy of a slice from a DataFrame.
# Try using .loc[row_indexer,col_indexer] = value instead" removing:
pd.options.mode.chained_assignment = None
pd.set_option("display.max_rows", 60, "display.max_columns", None)

# Read data
print(os.listdir("./DATA"))
train = pd.read_csv('./DATA/train.csv')
test = pd.read_csv('./DATA/test.csv')

os.makedirs("BBDD Output", exist_ok=True)

################################################################################
################################# FEATURE ######################################
############################### ENGINEERING ####################################
################################################################################
# Features types
Features=train.dtypes.reset_index()
Categorical=Features.loc[Features[0]=='str','index']

# Categorical to the begining
cols = train.columns.tolist()
pos=0
for col in Categorical:
	cols.insert(pos, cols.pop(cols.index(col)))
	pos+=1
train = train[cols]
cols.remove('TARGET')
test = test[cols]

# 1) Missings
################################################################################
# Function to print columns with at least n_miss missings
def miss(ds,n_miss):
	miss_list=list()
	for col in list(ds):
		if ds[col].isna().sum()>=n_miss:
			print(col,ds[col].isna().sum())
			miss_list.append(col)
	return miss_list
# Which columns have 1 missing at least...
print('\n################## TRAIN ##################')
m_tr=miss(train,1)
print('\n################## TEST ##################')
m_te=miss(test,1)

# Check that there are no missings only in test:
[i for i in m_te if i not in m_tr]

# 1.1) Missings in categorical features (fix it with an 'NA' string)
################################################################################
for col in Categorical:
	train.loc[train[col].isna(),col]='NA'
	test.loc[test[col].isna(),col]='NA'

# 1.2) Missings -> Drop some rows
################################################################################
# We can see a lot of colummns with 3 missings in train, look the data and...
# there are 4 observations that have many columns with missing values:
# A1039
# A2983
# A3055
# A4665

# Are all of them in train?
len(train.loc[train['ID']=='A1039',])
len(train.loc[train['ID']=='A2983',])
len(train.loc[train['ID']=='A3055',])
len(train.loc[train['ID']=='A4665',])
# ok, drop:
train = train[train['ID']!='A1039']
train = train[train['ID']!='A2983']
train = train[train['ID']!='A3055']
train = train[train['ID']!='A4665']

train.reset_index(drop=True,inplace=True)

print('\n################## TRAIN ##################')
m_tr=miss(train,1)
print('\n################## TEST ##################')
m_te=miss(test,1)

# Check that there are no missings only in test:
[i for i in m_te if i not in m_tr]

# 1.3) Missings -> n_tile, max or min
################################################################################
# Now, we consider columns with "many" missings:
print('\n################## TRAIN ##################')
m_tr=miss(train,30)

# And plot them to assign a percentile:

# Plot: Features with missing values to impute a value
# Bars = Population in each bucket (left axis)
# Line = Observed Default Frequency (ODF) (right axis)
import matplotlib
matplotlib.use("tkagg")

import matplotlib.pyplot as plt

def feat_graph(df,icol,binary_col,n_buckets):
	feat_data=df[[icol,binary_col]]
	feat_data['bucket']=pd.qcut(feat_data.iloc[:,0],q=n_buckets,labels=False,duplicates='drop')+1

	if len(feat_data.loc[feat_data[icol].isna(),'bucket'])>0:
		feat_data.loc[feat_data[icol].isna(),'bucket']=0

	hist_data_p=pd.DataFrame(feat_data[['bucket',binary_col]].groupby(['bucket'])[binary_col].mean()).reset_index()
	hist_data_N=pd.DataFrame(feat_data[['bucket',binary_col]].groupby(['bucket'])[binary_col].count()).reset_index()
	hist_data=pd.merge(hist_data_N,hist_data_p,how='left',on='bucket')
	hist_data.columns=['bucket','N','p']

	plt.figure()
	width = .70 # width of a bar
	hist_data['N'].plot(kind='bar', width = width, color='darkgray')
	hist_data['p'].plot(secondary_y=True,marker='o')

	ax = plt.gca()
	plt.xlim([-width, len(hist_data)-width/2])
	if len(feat_data.loc[feat_data[icol].isna(),'bucket'])>0:
		lab=['Missing']
		for i in range(1,n_buckets+1):
			lab.append('G'+str(i))
		ax.set_xticklabels(lab)
	else:
		lab=[]
		for i in range(1,n_buckets+1):
			lab.append('G'+str(i))
		ax.set_xticklabels(lab)

	plt.title(icol)
	plt.show(block=True)

# Plot
for icol in list(m_tr):
	feat_graph(train,icol,'TARGET',10)


# Decide assignation
assign_dict=dict.fromkeys(m_tr, 0)
assign_dict['X24']=0.8
assign_dict['X27']='min'
assign_dict['X28']=0.1
assign_dict['X37']=0.1
assign_dict['X41']=0.3
assign_dict['X45']=0.1
assign_dict['X53']=0.1
assign_dict['X54']=0.1
assign_dict['X60']='min'
assign_dict['X64']='min'

# Create missing indicators
miss_dummy_tr= pd.DataFrame(0, index=np.arange(len(train[m_tr])), columns=m_tr)
miss_dummy_te= pd.DataFrame(0, index=np.arange(len(test[m_tr])), columns=m_tr)

for col in m_tr:
	miss_dummy_tr.loc[train[col].isna(),col]=1
	miss_dummy_te.loc[test[col].isna(),col]=1
	v=assign_dict[col]
	print('imputing ',col,': ',v,sep='')
	if v=='min':
		value=train[col].min()-1
		train.loc[train[col].isna(),col]=value
		test.loc[test[col].isna(),col]=value
	elif v=='max':
		value=train[col].max()+1
		train.loc[train[col].isna(),col]=value
		test.loc[test[col].isna(),col]=value
	else:
		value=train[col].quantile(v)
		train.loc[train[col].isna(),col]=value
		test.loc[test[col].isna(),col]=value

miss_dummy_tr = miss_dummy_tr.add_suffix('_m')
miss_dummy_te = miss_dummy_te.add_suffix('_m')

# 1.4) Missings -> Exotic techniques
################################################################################
# The remaining missings will be imputed via Iterative Imputer:
# Models each feature with missing values as a function of other features, and
# uses that estimate for imputation

X_train=train.drop(columns=Categorical)
X_train.drop(columns='TARGET',inplace=True)
X_test=test.drop(columns=Categorical)

# Impute
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

filler=IterativeImputer()
X_train_filled = filler.fit_transform(X_train)
X_test_filled = filler.transform(X_test)

X_train_filled = pd.DataFrame(X_train_filled, columns=list(X_train))
X_test_filled = pd.DataFrame(X_test_filled, columns=list(X_test))

train=pd.concat([train[Categorical],X_train_filled,train['TARGET']],axis=1)
test=pd.concat([test[Categorical],X_test_filled],axis=1)

# Final check:
miss(train,1)
miss(test,1)

# # If we need to standardize data:
# from sklearn import preprocessing
# X_scaled = preprocessing.StandardScaler().fit_transform(X)

# 2) Correlations
################################################################################
# Let's see if certain columns are correlated
# or even that are the same with a "shift"
thresholdCorrelation = 0.6
def InspectCorrelated(df):
	corrMatrix = df.corr().abs() # Correlation Matrix
	upperMatrix = corrMatrix.where(np.triu(np.ones(corrMatrix.shape),k=1).astype(bool))
	correlColumns=[]
	for col in upperMatrix.columns:
		correls=upperMatrix.loc[upperMatrix[col]>thresholdCorrelation,col].keys()
		if (len(correls)>=1):
			correlColumns.append(col)
			print("\n",col,'->', end=" ")
			for i in correls:
				print(i, end=" ")
	print('\nSelected columns to drop:\n',correlColumns)
	return(correlColumns)

# Look at correlations in the original features
correlColumns=InspectCorrelated(train.iloc[:,len(Categorical):-1])
# Look at correlations in missing dummies
correlColumns_miss=InspectCorrelated(miss_dummy_tr)

# If we are ok, throw them:
train=train.drop(correlColumns,axis=1)
test=test.drop(correlColumns,axis=1)
miss_dummy_tr=miss_dummy_tr.drop(correlColumns_miss,axis=1)
miss_dummy_te=miss_dummy_te.drop(correlColumns_miss,axis=1)

# 3) Constants
################################################################################
# Let's see if there is some constant column:
def InspectConstant(df):
	consColumns=[]
	for col in list(df):
		if len(df[col].unique())<2:
			print(df[col].dtypes,'\t',col,len(df[col].unique()))
			consColumns.append(col)
	print('\nSelected columns to drop:\n',consColumns)
	return(consColumns)

consColumns=InspectConstant(train.iloc[:,len(Categorical):-1])

# If we are ok, throw them:
train=train.drop(consColumns,axis=1)
test=test.drop(consColumns,axis=1)

# 4) Alerts
################################################################################
from sklearn.tree import DecisionTreeClassifier

alerts_train=pd.DataFrame()
alerts_test=pd.DataFrame()
THRESHOLDS=list()
ACTIVATIONS=list()
TMRS=list()

print('\nAlerts...')
print('###########################################')
for FACTOR in list(train)[len(Categorical):-1]:

	# depth 1 tree
	dtree=DecisionTreeClassifier(max_depth=1)
	data_tree=train[[FACTOR,'TARGET']].loc[~train[FACTOR].isna()].reset_index(drop=True)
	dtree.fit(data_tree[[FACTOR]],data_tree[['TARGET']])
	# Optimal split
	threshold = dtree.tree_.threshold[0]
	# Alert creation
	alerts_train[FACTOR]=train[FACTOR]
	alerts_train[FACTOR+'_b']=np.zeros(len(train))
	alerts_test[FACTOR]=test[FACTOR]
	alerts_test[FACTOR+'_b']=np.zeros(len(test))

	orientation='<='
	if (len(alerts_train.loc[alerts_train[FACTOR]<=threshold,FACTOR+'_b']) < len(alerts_train.loc[alerts_train[FACTOR]>threshold,FACTOR+'_b'])):
		alerts_train.loc[alerts_train[FACTOR]<=threshold,FACTOR+'_b']=1
		alerts_test.loc[alerts_test[FACTOR]<=threshold,FACTOR+'_b']=1
	else:
		alerts_train.loc[alerts_train[FACTOR]>threshold,FACTOR+'_b']=1
		alerts_test.loc[alerts_test[FACTOR]>threshold,FACTOR+'_b']=1
		orientation='>'

	# ACTIVATIONS
	activ=int(alerts_train[FACTOR+'_b'].sum())

	# TMR
	TMO=pd.DataFrame(pd.concat([alerts_train[FACTOR+'_b'],train['TARGET']],axis=1).groupby([FACTOR+'_b'])['TARGET'].mean()).reset_index()
	TMR=float(TMO.loc[TMO[FACTOR+'_b']==1,'TARGET'].iloc[0])/train['TARGET'].mean()

	# Throw the original factor
	alerts_train.drop([FACTOR],axis=1,inplace=True)
	alerts_test.drop([FACTOR],axis=1,inplace=True)

	# Add THRESHOLDS, ACTIVATIONS and TMR to the sequence
	THRESHOLDS.append(orientation+str(round(threshold,3)))
	ACTIVATIONS.append(activ)
	TMRS.append(TMR*100)

# Severity table
severity=pd.DataFrame({'Alert':list(alerts_train),
						'Threshold':THRESHOLDS,
						'Activations (N)': ACTIVATIONS,
						'TMR (%)': TMRS})
severity['LOG TMR']=np.log(severity['TMR (%)']/100)
severity['ABS LOG TMR']=severity['LOG TMR'].abs()

severity=severity.sort_values(by='ABS LOG TMR',ascending=False).reset_index(drop=True)

# Correlations between alerts
# First, we order them by its importance
alerts_train=alerts_train[severity['Alert']]
alerts_test=alerts_test[severity['Alert']]

thresholdCorrelation = 0.7
correlColumns=InspectCorrelated(alerts_train)

# If we are ok, throw them:
alerts_train=alerts_train.drop(correlColumns,axis=1)
alerts_test=alerts_test.drop(correlColumns,axis=1)
for col in correlColumns:
	severity=severity[severity['Alert']!=col].reset_index(drop=True)

# Throw alerts with low activation
severity=severity.loc[severity['Activations (N)']>=30,].reset_index(drop=True)

# Throw alerts with low TMR (over 100) and high TMR under 100
severity=severity.loc[(severity['TMR (%)']>=200) | (severity['TMR (%)']<=50),].reset_index(drop=True)

print(severity.to_string())

# Final set of alerts
alerts_train=alerts_train[severity['Alert']]
alerts_test=alerts_test[severity['Alert']]

# Impute LOG TMR to alert activation
for col in list(alerts_train):
	mult=(severity.loc[severity['Alert']==col,['LOG TMR']]).values[0][0]
	alerts_train[col]=alerts_train[col]*mult
	alerts_test[col]=alerts_test[col]*mult

# Add ALERTS feature to train and test
# TRAIN
alerts_train['ALERTS']=alerts_train.sum(axis=1)
train['ALERTS']=alerts_train['ALERTS']
# TEST
alerts_test['ALERTS']=alerts_test.sum(axis=1)
test['ALERTS']=alerts_test['ALERTS']

# Finally add missing dummies to datasets
train=pd.concat([train,miss_dummy_tr],axis=1)
test=pd.concat([test,miss_dummy_te],axis=1)

# Reorder columns (TARGET at the end)
cols=list(train)
cols.insert(len(cols), cols.pop(cols.index('TARGET')))
train = train.reindex(columns= cols)

# 5) WoE Transformation
################################################################################
# Woe Function
def WoE(icol,binary_col,df_train,df_test,n_buckets=None):
	if n_buckets:
		df_train['bucket'], bins = pd.qcut(df_train[icol],q=n_buckets,labels=False,duplicates='drop',retbins=True)
		real_bins=len(bins)-1
		df_test['bucket'] = pd.cut(df_test[icol],bins=bins,labels=False,include_lowest=True)
		# If we are below the minimum or above the maximum in test assign the extreme buckets:
		df_test.loc[(df_test['bucket'].isna()) & (df_test[icol]>=max(bins)),'bucket']=real_bins-1
		df_test.loc[(df_test['bucket'].isna()) & (df_test[icol]<=min(bins)),'bucket']=0
		woe_table=df_train[['bucket',binary_col]].groupby(['bucket']).sum().reset_index()
	else:
		df_train['bucket']=df_train[icol]
		df_test['bucket']=df_test[icol]
		real_bins=len(df_train[icol].unique())
		woe_table=df_train[[icol,binary_col]].groupby([icol]).sum().reset_index()
		woe_table = woe_table.rename(columns={icol: 'bucket'})

	# GOOD & BAD Total
	BAD=df_train[binary_col].sum()
	GOOD=df_train.loc[~df_train[binary_col].isna(),binary_col].count()-BAD

	# We have at least 2 values
	if real_bins>=2:
		woe_table = woe_table.rename(columns={binary_col: 'BAD'}) # Defaults
		woe_table['TOTAL']=df_train[['bucket',binary_col]].groupby(['bucket']).count().reset_index()[binary_col] # Totales
		woe_table['GOOD']=(woe_table['TOTAL']-woe_table['BAD']).astype(int) # Buenos

		# WoE by bucket
		woe_table['WOE']=np.log(((woe_table['GOOD']+0.5)/GOOD)/((woe_table['BAD']+0.5)/BAD))

		# Add the new factor and remove the original
		df_train = pd.merge(df_train, woe_table[['bucket','WOE']], on='bucket', how='left')
		df_train = df_train.rename(columns={'WOE': icol+"_W"})
		df_train = df_train.drop(icol, axis=1)
		df_train = df_train.drop('bucket', axis=1)

		df_test = pd.merge(df_test, woe_table[['bucket','WOE']], on='bucket', how='left')
		# In case that for a Categorical variable (for Numerical variables this
		# is impossible since we have assigned every observation to a bin)
		# there are unseen categories in test (not found in train)
		# -> assign WoE = 0 (neutral WoE)
		df_test.loc[df_test['WOE'].isna(),'WOE']=0
		df_test = df_test.rename(columns={'WOE': icol+"_W"})
		df_test = df_test.drop(icol, axis=1)
		df_test = df_test.drop('bucket', axis=1)
	else:
		print('Column ',icol,' has less than 2 buckets -> Removed')
		df_train = df_train.drop(icol, axis=1)
		df_train = df_train.drop('bucket', axis=1)
		df_test = df_test.drop(icol, axis=1)
		df_test = df_test.drop('bucket', axis=1)

	return df_train, df_test

# List of features that we will treat as Categorical for WoE
As_Categorical=Categorical.tolist()
As_Categorical.remove('ID')
miss_dummies = [i for i in list(train) if '_m' in i]
for i in miss_dummies:
	As_Categorical.append(i)

# List of features that we will treat as Numerical for WoE
As_Numerical=list(train)
As_Numerical.remove('ID')
As_Numerical.remove('TARGET')
for i in As_Categorical:
	As_Numerical.remove(i)

# Initialize woe (or lineal) sets for modeling
train_woe=train.copy()
test_woe=test.copy()

# Transform Categorical
for icol in As_Categorical:
	train_woe, test_woe = WoE(icol=icol,
							  binary_col='TARGET',
							  df_train=train_woe,
							  df_test=test_woe,
							  n_buckets=None)
# Transform Numerical
for icol in As_Numerical:
	train_woe, test_woe = WoE(icol=icol,
							  binary_col='TARGET',
							  df_train=train_woe,
							  df_test=test_woe,
							  n_buckets=10)

cols=list(train_woe)
cols.insert(len(cols), cols.pop(cols.index('TARGET')))
train_woe = train_woe.reindex(columns= cols)

# Define final sets:
pred=list(train_woe)[1:-1]
X_train_woe=train_woe[pred].reset_index(drop=True)
Y_train=train_woe['TARGET'].reset_index(drop=True)
X_test_woe=test_woe[pred].reset_index(drop=True)

################################################################################
########################## MODELS LASSO and RIDGE ##############################
########################## TRAIN / TEST approach ###############################
################################################################################
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# 1) Train / test hyper-parameter optimization
################################################################################

# train / test partition
RS = 42 # Seed for partition and model random part
TS = 0.3 # Validation size

# Split
from sklearn.model_selection import train_test_split
x_tr, x_te, y_tr, y_te = train_test_split(X_train_woe, Y_train, test_size=TS, random_state=RS)

# Parameters of the model
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
params = {'l1_ratio': 1, # 1 for Lasso, 0 for Ridge
	'solver': 'liblinear',
	'random_state': RS}

MODEL=LogisticRegression()
MODEL.set_params(**params)

# We will define the model for various C's and make a search for the optimum
Cs=[1e-3,1e-2,1e-1,0.2,0.5,0.8,1.1,2,3,5,10]


print('\nLinear Train/Test...')
print('########################################################')
scores=[]

for c in Cs:
	MODEL.set_params(C=c)
	print('\nRegularization C: ',c)
	model_fit=MODEL.fit(x_tr, y_tr)
	p_te=MODEL.predict_proba(x_te)[:,1]
	p_tr=MODEL.predict_proba(x_tr)[:,1]
	score_te=roc_auc_score(y_te,p_te)
	score_tr=roc_auc_score(y_tr,p_tr)
	print('\t --test AUC: ', round(score_te,4), '\t--train auc: ', round(score_tr,4),sep='')

	# Look if we are in the first test:
	if len(scores)==0:
		max_score=float('-inf')
	else:
		max_score=max(scores)

	# If the score improves, we keep this one:
	if score_te>=max_score:
		print('BEST')
	# Append score
	scores.append(score_te)

# The best test score has been found in:
best_C=Cs[scores.index(max(scores))]
print('\n###########################################')
print('Linear optimal C: ',best_C)
print('Linear optimal GINI: ',round((max(scores)*2-1)*100,4),'%')
print('Linear optimal AUC: ',round(max(scores)*100,4),'%')
print('###########################################')

# 2) Model on all train with optimal hyper-parametrs
################################################################################
model_linear=LogisticRegression()
model_linear.set_params(**params)
model_linear.set_params(C=best_C)

model_linear.fit(X_train_woe, Y_train)
weights = pd.DataFrame({'feature':np.array(X_train_woe.columns),'coefs':np.transpose(np.array(model_linear.coef_))[:,0]})
weights=weights.sort_values(by='coefs',ascending=True).reset_index(drop=True)

################################################################################
# Results
# Prediction
test['Pred']=model_linear.predict_proba(X_test_woe)[:,1]
linear_submission=pd.DataFrame(test[['ID','Pred']])

# Outputs to .csv
# Lasso
linear_submission.to_csv("BBDD Output/lasso_submission.csv", index = False)
weights.to_csv("BBDD Output/lasso_features.csv", index = False)
# Ridge
# linear_submission.to_csv("BBDD Output/ridge_submission.csv", index = False)
# weights.to_csv("BBDD Output/ridge_features.csv", index = False)
################################################################################


################################################################################
########################### MODEL LASSO / RIDGE ################################
######################### k-Fold Cross-Validation ##############################
################################################################################

# 1) k-Fold Cross-Validation Function
################################################################################
from sklearn.model_selection import StratifiedKFold

def Model_cv(MODEL, k, X_train, X_test, y, RE, makepred=True, CatPos=None):
	# Create the k folds
	kf=StratifiedKFold(n_splits=k, shuffle=True, random_state=RE)

	# first level train and test
	Level_1_train = pd.DataFrame(np.zeros((X_train.shape[0],1)), columns=['train_yhat'])
	if makepred==True:
		Level_1_test = pd.DataFrame()

	# Main loop for each fold. Initialize counter
	count=0
	for train_index, test_index in kf.split(X_train, y):
		count+=1
		# Define train and test depending in which fold are we
		fold_train= X_train.loc[train_index.tolist(), :]
		fold_test=X_train.loc[test_index.tolist(), :]
		fold_ytrain=y[train_index.tolist()]
		fold_ytest=y[test_index.tolist()]

		# (k-1)-folds model adjusting
		if CatPos:
			# Prepare Pool
			pool_train=Pool(fold_train, fold_ytrain,cat_features=Pos)
			# (k-1)-folds model adjusting
			model_fit=MODEL.fit(X=pool_train)

		else:
			# (k-1)-folds model adjusting
			model_fit=MODEL.fit(fold_train, fold_ytrain)

		# Predict on the free fold to evaluate metric
		# and on train to have an overfitting-free prediction for the next level
		p_fold=MODEL.predict_proba(fold_test)[:,1]
		p_fold_train=MODEL.predict_proba(fold_train)[:,1]

		# Score in the free fold
		score=roc_auc_score(fold_ytest,p_fold)
		score_train=roc_auc_score(fold_ytrain,p_fold_train)
		print(k, '-cv, Fold ', count, '\t --test AUC: ', round(score,4), '\t--train AUC: ', round(score_train,4),sep='')
		# Save in Level_1_train the "free" predictions concatenated
		Level_1_train.loc[test_index.tolist(),'train_yhat'] = p_fold

		# Predict in test to make the k model mean
		# Define name of the prediction (p_"iteration number")
		if makepred==True:
			name = 'p_' + str(count)
			# Predictin to real test
			real_pred = MODEL.predict_proba(X_test)[:,1]
			# Name
			real_pred = pd.DataFrame({name:real_pred}, columns=[name])
			# Add to Level_1_test
			Level_1_test=pd.concat((Level_1_test,real_pred),axis=1)

	# Compute the metric of the total concatenated prediction (and free of overfitting) in train
	score_total=roc_auc_score(y,Level_1_train['train_yhat'])
	print('\n',k, '- cv, TOTAL AUC:', round((score_total)*100,4),'%')

	# mean of the k predictions in test
	if makepred==True:
		Level_1_test['model']=Level_1_test.mean(axis=1)

	# Return train and test sets with predictions and the performance
	if makepred==True:
		return Level_1_train, pd.DataFrame({'test_yhat':Level_1_test['model']}), score_total
	else:
		return score_total

# 2) k-fold Cross Validation execution
################################################################################
from sklearn.linear_model import LogisticRegression

# Parameters of the CV
RS=1234 # Seed for k-fold partition and model random part
n_folds=5 # Number of folds

# Parameters of the model
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
params = {'l1_ratio': 1, # 1 for Lasso, 0 for Ridge
		  'solver': 'liblinear',
		  'random_state': RS}

MODEL = LogisticRegression()
MODEL.set_params(**params)

# We will define the model for various C's and make a search for the optimum
# Cs=[0.1,0.2,0.3,0.4,0.5,0.7,1,1.3] # LASSO
Cs=[0.009,0.01,0.015,0.02,0.025,0.03,0.05,0.1,0.15] # Ridge

print('\Linear CV...')
print('########################################################')
scores=[]
for c in Cs:
	MODEL.set_params(C=c)
	print('\nRegularization C: ',c)
	Pred_train, Pred_test, s = Model_cv(MODEL,n_folds,X_train_woe,X_test_woe,Y_train,RS,makepred=True)

	# Look if we are in the first test:
	if len(scores)==0:
		max_score=float('-inf')
	else:
		max_score=max(scores)

	# If the score improves, we keep this one:
	if s>=max_score:
		print('BEST')
		Linear_train=Pred_train.copy()
		Linear_test=Pred_test.copy()

	# Append score
	scores.append(s)

# The best cross-validated score has been found in:
best_C=Cs[scores.index(max(scores))]
print('\n###########################################')
print('Linear optimal C: ',best_C)
print('Linear optimal GINI: ',round((max(scores)*2-1)*100,4),'%')
print('Linear optimal AUC: ',round(max(scores)*100,4),'%')
print('###########################################')

# 3) Model on all train with optimal hyper-parametrs
################################################################################
# best_C=0.074 # Manual C LASSO (no + Coefs)
# best_C=0.00019 # Manual C Ridge (no + Coefs)
model_linear=LogisticRegression()
model_linear.set_params(**params)
model_linear.set_params(C=best_C)

model_linear.fit(X_train_woe, Y_train)
weights = pd.DataFrame({'feature':np.array(X_train_woe.columns),'coefs':np.transpose(np.array(model_linear.coef_))[:,0]})
weights=weights.sort_values(by='coefs',ascending=True).reset_index(drop=True)
################################################################################
# Results
# Prediction
test['Pred']=model_linear.predict_proba(X_test_woe)[:,1]
linear_submission=pd.DataFrame(test[['ID','Pred']])

# Cv predictions
linear_cv_train=train[['ID']]
linear_cv_train['linear_pred']=Linear_train['train_yhat']
linear_cv_test=test[['ID']]
linear_cv_test['linear_pred']=Linear_test['test_yhat']

# Outputs to .csv
# Lasso
linear_submission.to_csv("BBDD Output/lasso_submission.csv", index = False)
linear_cv_train.to_csv("BBDD Output/lasso_cv_train.csv", index = False)
linear_cv_test.to_csv("BBDD Output/lasso_cv_test.csv", index = False)
weights.to_csv("BBDD Output/lasso_features.csv", index = False)
# Ridge
linear_submission.to_csv("BBDD Output/ridge_submission.csv", index = False)
linear_cv_train.to_csv("BBDD Output/ridge_cv_train.csv", index = False)
linear_cv_test.to_csv("BBDD Output/ridge_cv_test.csv", index = False)
weights.to_csv("BBDD Output/ridge_features.csv", index = False)
################################################################################
