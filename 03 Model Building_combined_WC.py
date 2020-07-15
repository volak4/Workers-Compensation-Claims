# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 04:03:14 2020

@author: volak
"""

#IMPORT LIBRARY
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



df.columns



######## LOAD DATA
path = "C:/Users/volak/Google Drive/01 Sharpest Mind/Volak Sin/00 Works Comp/Data/combined.csv"
path2 = "C:/Users/volak/Google Drive/01 Sharpest Mind/Volak Sin/00 Works Comp/Data/CAX_Comp_Claims_Public_Test - no Dependent.csv"
df = pd.read_csv(path)
df_test = pd.read_csv(path2)
######## SELECT FEATURES FOR MODEL
df_model = df.drop(['Unnamed: 0', 'Obs_ID','atty_firm_name','employ_status', 'how_injury_occur','jurisdiction','detail_cause','handling_office', 'injury_postal','length_how_injury',
              'time_injury','Dependent','policy_yr' ], axis=1 )

df_model.columns


####CREATE DUMMY VARIABLES AND CHECK DIMENSIONS
#get dummy data
df_temp = pd.get_dummies(df_model)

df_dum= df_temp.iloc[:15407,:]
df_dum_test= df_temp.iloc[15407:,:]


###Drop Outliers
dataset=df_dum
avg = dataset['DependentPlus1'].mean()
std = dataset['DependentPlus1'].std()
upper_outlier = avg + 3*std
lower_outlier = avg - 3*std
df_dum=dataset[dataset.DependentPlus1 > lower_outlier ]
df_dum=dataset[dataset.DependentPlus1 < upper_outlier ]


##Transform Target/Response


df_dum_test=df_dum_test.drop(['DependentPlus1'], axis=1)

df_dum.shape






######## TRAIN TEST SPLIT

from sklearn.model_selection import train_test_split

X = df_dum.drop('DependentPlus1', axis =1)
y = df_dum.DependentPlus1.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


####################
#### PCA & PRep ####
####################
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Applying PCA * requires feature scaling
from sklearn.decomposition import PCA
pca = PCA(n_components = 100) # number of principal components explain variance, use '0' first
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
X = np.concatenate((X_train,X_test),axis=0)






###########################################################
############### MODEL FITTING ############################# 
###########################################################

#######################
#######FIT BASIC MODEL#
#######################
## GLM GAMMA
import statsmodels.api as sm
model = sm.GLM(y_train, X_train, family=sm.families.Gamma(link = sm.genmod.families.links.log)).fit() '''Can't have outliers' '''
model.fit().summary()


from sklearn.linear_model import GammaRegressor
from sklearn.model_selection import cross_val_score
ga = GammaRegressor()
ga.fit(X_train, y_train

#np.mean(cross_val_score(lm,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3))
mse = np.mean(cross_val_score(ga,X_train,y_train, scoring = 'neg_mean_squared_error', cv = 5))
rmse = np.sqrt(mse*-1)
print(rmse)



##EXPORT PREDICTIONS
y_pred=model.predict(df_dum_test)
y_df = pd.DataFrame(data=y_pred)
df = pd.concat([df_test['Obs_ID'], y_df], axis=1, )
df.to_csv('ypredictedgammaloglink.csv')


#############################################################
######################### FIT XgBoost #######################
#############################################################






import xgboost
from sklearn.model_selection import RandomizedSearchCV
regressor=xgboost.XGBRegressor()


## Hyper Parameter Optimization


n_estimators = [100, 500, 900, 1100, 1500]
max_depth = [2, 3, 5, 6, 10]
booster=['gbtree','gblinear']
learning_rate=[0.02,0.03,0.04,0.06]
min_child_weight=[1,2,3,4]
base_score=[0.25,0.5,0.75,1]
gamma = [.03,.05,.07,.09]

# Define the grid of hyperparameters to search
hyperparameter_grid = {
    'n_estimators': n_estimators,
    'max_depth':max_depth,
    'learning_rate':learning_rate,
    'min_child_weight':min_child_weight,
    'booster':booster,
    'base_score':base_score,
    'gamma':gamma
    }


# Set up the random search with 4-fold cross validation
random_cv = RandomizedSearchCV(estimator=regressor,
            param_distributions=hyperparameter_grid,
            cv=5, n_iter=50,
            scoring = 'neg_mean_absolute_error',n_jobs = 4,
            verbose = 5, 
            return_train_score = True,
            random_state=42)

random_cv.fit(X_train,y_train)
random_cv.best_estimator_
''' output for best parameters
XGBRegressor(base_score=0.75, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0.07, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.02, max_delta_step=0, max_depth=6,
             min_child_weight=3, missing=nan, monotone_constraints='()',
             n_estimators=100, n_jobs=0, num_parallel_tree=1, random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
             tree_method='exact', validate_parameters=1, verbosity=None)
'''

#copy from above
regressor=xgboost.XGBRegressor(base_score=0.75, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0.07, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.02, max_delta_step=0, max_depth=6,
             min_child_weight=3,monotone_constraints='()',
             n_estimators=100, n_jobs=0, num_parallel_tree=1, random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
             tree_method='exact', validate_parameters=1, verbosity=None)



regressor.fit(X_train,y_train)

mse = np.mean(cross_val_score(regressor,X_train,y_train, scoring = 'neg_mean_squared_error', cv = 5))
rmse = np.sqrt(mse*-1)
print(rmse)


#########################
###### Boruta version1 ##
#########################
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy

# load X and y
# NOTE BorutaPy accepts numpy arrays only, hence the .values attribute
X = X_train
y = y_train
y = y.ravel()

# define random forest classifier, with utilising all cores and
# sampling in proportion to y labels
rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
rf = regressor

# define Boruta feature selection method
feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)

# find all relevant features - 5 features should be selected
feat_selector.fit(X, y)

# check selected features - first 5 features are selected
feat_selector.support_

# check ranking of features
feat_selector.ranking_

# call transform() on X to filter it down to selected features
X_filtered = feat_selector.transform(X)


############################
###### Boruta version 2.0 ##
############################
from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor
import numpy as np
###initialize Boruta
forest = RandomForestRegressor(
   n_jobs = -1, 
   max_depth = 5
)
boruta = BorutaPy(
   estimator = rf, 
   n_estimators = 'auto',
   max_iter = 100 # number of trials to perform
)
### fit Boruta (it accepts np.array, not pd.DataFrame)
boruta.fit(np.array(X), np.array(y))
### print results
green_area = X.columns[boruta.support_].to_list()
blue_area = X.columns[boruta.support_weak_].to_list()
print('features in the green area:', green_area)
print('features in the blue area:', blue_area)








####################
####### SAVE MODEL##
####################
import pickle
filename = 'xgboost_model_all.pkl'
pickle.dump(regressor, open(filename, 'wb'))


##EXPORT PREDICTIONS
y_pred=regressor.predict(df_dum_test)
y_df = pd.DataFrame(data=y_pred)
df = pd.concat([df_test['Obs_ID'], y_df], axis=1, )
df.to_csv('ypredictedXgBoost1all.csv')
