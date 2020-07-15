# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 13:58:51 2020

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
path2 = "C:/Users/volak/Google Drive/01 Sharpest Mind/Volak Sin/00 Works Comp/Data/check_test.csv"
train = pd.read_csv(path)
test = pd.read_csv(path2)
df_model = pd.concat([train,test],axis=0)
df_model =df_model.loc[:,~df_model.columns.duplicated()]
######## SELECT FEATURES FOR MODEL
df_model = df_model.drop(['Unnamed: 0', 'Obs_ID','atty_firm_name','employ_status', 'how_injury_occur','jurisdiction',
                    'detail_cause','handling_office', 'injury_postal','length_how_injury',
              'time_injury','LogDependent', 'DependentPlus1','policy_yr' ], axis=1 )



df_model.Obs_ID.value_counts()
df_model.columns
#############################################################################################
########################## MODEL ONE : FREQUENCY MODEL#######################################
#############################################################################################




####CREATE DUMMY VARIABLES AND CHECK DIMENSIONS
#get dummy data
df_dum = pd.get_dummies(df_model)

#### CREATE FREQUENCY COLUM
#df_model['freq']= df_model.Dependent.apply(lambda x: 1 if x > 0 else 0)
def model_freq(time):
    if time == 0 :
        return '0'
    elif time <=1278 :
        return '1'
    elif time >1278 :
        return '2'

df_dum['freq'] = df_dum['Dependent'].apply(model_freq)

df_dum.freq.value_counts()
df_dumm = df_dum.drop(['Dependent'],axis=1)

df_dum= df_dumm.iloc[:15211,:]
df_dum_test= df_dumm.iloc[15211:,:]

#df_dum_test = pd.DataFrame(data=df_dum_test)
df_dum_test.head()
######## TRAIN TEST SPLIT

from sklearn.model_selection import train_test_split

X = df_dum.drop('freq', axis =1)
y = df_dum.freq.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


####################
###### FIT XgBoost##
####################
import xgboost
from sklearn.model_selection import RandomizedSearchCV
regressor=xgboost.XGBRegressor()
classifier=xgboost.XGBClassifier()

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
random_cv = RandomizedSearchCV(estimator=classifier,
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
regressor=xgboost.XGBClassifier(base_score=0.25, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0.07, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.02, max_delta_step=0, max_depth=6,
              min_child_weight=1, monotone_constraints='()',
              n_estimators=100, n_jobs=0, num_parallel_tree=1,
              objective='multi:softprob', random_state=0, reg_alpha=0,
              reg_lambda=1, scale_pos_weight=None, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)

regressor.fit(X_train,y_train)


####### SAVE MODEL
import pickle
filename = 'xgboost_model.pkl'
pickle.dump(regressor, open(filename, 'wb'))


##EXPORT PREDICTIONS
y_pred=regressor.predict(df_dum_test)
y_df = pd.DataFrame(data=y_pred)
y_df.describe()
y_df=y_df.rename(columns={0:'prob'})
df_freq = pd.concat([df_test['Obs_ID'], y_df], axis=1, )
df_freq.to_csv('ypredicteddf_freq.csv')

df_freq.columns
#df_freq['prob'] = df_freq.prob.apply(lambda x: 1 if x<0.5 else 0)
df_freq.prob.value_counts()





#############################################################################################
########################## MODEL TWO : SEVERITY  MODEL#######################################
#############################################################################################
#### CREATE FREQUENCY COLUM
#df_model['freq']= df_model.Dependent.apply(lambda x: 1 if x > 0 else 0)
df_dum = pd.get_dummies(df_model)
def model_freq(time):
    if time == 0 :
        return '0'
    elif time <=1278 :
        return '1'
    elif time >1278 :
        return '2'

#df_model['freq'] = df_model['Dependent'].apply(model_freq)

df_dum['freq'] = df_dum['Dependent'].apply(model_freq)

df_dum.freq.value_counts()
df_dumm = df_dum

df_dum= df_dumm.iloc[:15211,:]
df_dum_test= df_dumm.iloc[15211:,:]

df_dum_test = df_dum_test.drop(['freq'],axis=1)
df_dum_test = df_dum_test.drop(['Dependent'],axis=1)
## CREATE DATAFRAME OF ONLY POSITIVE CLAIMS
df_freq_1 = df_dum.loc[df_dum['freq']=='1']
df_freq_2 = df_dum.loc[df_dum['freq']=='2']

df_freq_1 = df_freq_1.drop(['freq'],axis=1)
df_freq_2 = df_freq_2.drop(['freq'],axis=1)
df_freq_2.Dependent.describe()
df_freq_2.Dependent.value_counts()




####CREATE DUMMY VARIABLES AND CHECK DIMENSIONS
#get dummy data
#df_dum = pd.get_dummies(df_freq_1)
#df_dum = pd.get_dummies(df_freq_2)
#df_dum_test = pd.get_dummies(df_model_test)

#df_dum_test.shape
#df_dum.shape


#df_dum
#df_freq_1 = df_model.loc[df_model['freq']=='1']
#df_freq_2 = df_model.loc[df_model['freq']=='2']


######## TRAIN TEST SPLIT

from sklearn.model_selection import train_test_split

X = df_freq_2.drop('Dependent', axis =1)
y = df_freq_2.Dependent.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



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




parameters 2
XGBRegressor(base_score=0.75, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0.07, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.02, max_delta_step=0, max_depth=6,
             min_child_weight=3, monotone_constraints='()',
             n_estimators=100, n_jobs=0, num_parallel_tree=1, random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
             tree_method='exact', validate_parameters=1, verbosity=None)
'''

#copy from above
regressor=xgboost.XGBRegressor(base_score=0.75, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0.07, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.02, max_delta_step=0, max_depth=6,
             min_child_weight=3, monotone_constraints='()',
             n_estimators=100, n_jobs=0, num_parallel_tree=1, random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
             tree_method='exact', validate_parameters=1, verbosity=None)

regressor.fit(X_train,y_train)


####### SAVE MODEL
import pickle
filename = 'xgboost_model.pkl'
pickle.dump(regressor, open(filename, 'wb'))


##EXPORT PREDICTIONS
df_dum_test = df_dum_test.drop(['freq'],axis=1)
y_pred=regressor.predict(df_dum_test)
y_df = pd.DataFrame(data=y_pred)


y_df=y_df.rename(columns={0:'Severity_1'})
df_sev_1 = pd.concat([test['Obs_ID'], y_df], axis=1, )
df_freq_sev = pd.merge(df_freq,df_sev_1, on='Obs_ID', how='left')


y_df=y_df.rename(columns={0:'Severity_2'})
df_sev_2 = pd.concat([test['Obs_ID'], y_df], axis=1, )
df_freq_sev = pd.merge(df_freq_sev,df_sev_2, on='Obs_ID', how='left')

df_freq_sev.prob.value_counts()

#######MULTI COLUMNS DATA IMPUTINGE#####
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age

def actual_severity(cols):
    prob = cols[0]
    Severity_1 = cols[1]
    Severity_2 = cols[2]

    if prob == 1:
       return Severity_1
   
    elif prob == 2:
       return Severity_2
   
    else:
       return Severity_2




df_freq_sev['Severity'] = df_freq_sev[['prob','Severity_1','Severity_2']].apply(actual_severity, axis=1)
df_freq_sev['Severity'] = df_freq_sev.apply(lambda x: 0 if x['prob']==0
                                            else x['Severity_1'] if x['prob']==1 
                                            else x['Severity_2'])

train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)


df_freq_sev.to_csv('freqSeverityModel.csv')
