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
path = "C:/Users/volak/Google Drive/01 Sharpest Mind/Volak Sin/00 Works Comp/Data/check_train.csv"
path_test = "C:/Users/volak/Google Drive/01 Sharpest Mind/Volak Sin/00 Works Comp/Data/check_test.csv"
df = pd.read_csv(path)
df_test = pd.read_csv(path_test)

######## SELECT FEATURES FOR MODEL
df_model = df.drop(['Unnamed: 0', 'Obs_ID','atty_firm_name','employ_status', 'how_injury_occur','jurisdiction','detail_cause','handling_office', 'injury_postal','length_how_injury',
              'time_injury','LogDependent', 'Dependent','policy_yr' ], axis=1 )

df_model_test = df_test.drop(['Unnamed: 0', 'Obs_ID','atty_firm_name','employ_status', 'how_injury_occur','jurisdiction','detail_cause','handling_office', 'injury_postal','length_how_injury',
              'time_injury', 'Dependent','policy_yr' ], axis=1 )





####CREATE DUMMY VARIABLES AND CHECK DIMENSIONS
#get dummy data
df_dum = pd.get_dummies(df_model)
df_dum_test = pd.get_dummies(df_model_test)

df_dum_test.shape
df_dum.shape






######## TRAIN TEST SPLIT

from sklearn.model_selection import train_test_split

X = df_dum.drop('DependentPlus1', axis =1)
y = df_dum.DependentPlus1.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


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


####################
###### FIT XgBoost##
####################

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
regressor=xgboost.XGBRegressor(base_score=0.25, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
       max_depth=2, min_child_weight=1, missing=None, n_estimators=900,
       n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=1)



regressor.fit(X_train,y_train)

mse = np.mean(cross_val_score(regressor,X_train,y_train, scoring = 'neg_mean_squared_error', cv = 5))
rmse = np.sqrt(mse*-1)
print(rmse)

####################
####### SAVE MODEL##
####################
import pickle
filename = 'xgboost_model.pkl'
pickle.dump(regressor, open(filename, 'wb'))


##EXPORT PREDICTIONS
y_pred=regressor.predict(df_dum_test)
y_df = pd.DataFrame(data=y_pred)
df = pd.concat([df_test['Obs_ID'], y_df], axis=1, )
df.to_csv('ypredictedXgBoost1.csv')
