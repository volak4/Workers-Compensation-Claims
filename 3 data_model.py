# -*- coding: utf-8 -*-
"""
Created on Wed May 20 10:31:20 2020

@author: volak
"""


#IMPORT LIBRARY
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



df.columns



#LOAD DATA
path = "C:/Users/volak/Google Drive/01 Sharpest Mind/Volak Sin/00 Works Comp/Data/data_clean2.csv"
path_test = "C:/Users/volak/Google Drive/01 Sharpest Mind/Volak Sin/00 Works Comp/Data/data_clean_test.csv"
df = pd.read_csv(path)
df_test = pd.read_csv(path_test)

df.columns

df.type_loss.value_counts()


df.columns
'''
Index(['Unnamed: 0', 'Obs_ID', 'Dependent', 'ave_wkly_wage', 'body_part',
       'cause', 'claimant_age', 'atty_firm_name', 'gender', 'marital_status',
       'claim_st', 'depart_code', 'detail_cause', 'domestic_foreign',
       'employ_status', 'handling_office', 'how_injury_occur', 'injury_city',
       'injury_postal', 'injury_state', 'jurisdiction',
       'lost_time_or_medicalonly', 'nature_injury', '#dependents',
       'osha_injury_type', 'severity_index', 'time_injury', 'type_loss',
       'policy_yr', 'reforms_dummy', 'length_employed',
       'diff_carrier_employer', 'diff_employer_injury', 'shift']
'''
df['ave_wkly_wage'].head() = pd.to_numeric(df.ave_wkly_wage) 
df_con =['ave_wkly_wage', 'claimant_age', 'length_employed','hire_date_yr', 
       'diff_carrier_employer', 'diff_employer_injury', 'length_how_injury']
cat = [ 'body_part',
       'cause', 'gender', 
       'marital_status', 'claim_st', 'depart_code', 'detail_cause',
       'domestic_foreign', 
       'employ_status',  'handling_office', 
        'injury_state', 'jurisdiction',
       'lost_time_or_medicalonly', 'nature_injury', '#dependents',
       'osha_injury_type', 'severity_index', 'type_loss',
       'policy_yr', 'reforms_dummy' ]


#Choose revelant columns
#Test simple model
df_model = df[[ 'Dependent','body_part','cause','gender' ]]

df_model = df[['Dependent','cause','ave_wkly_wage','lost_time_or_medicalonly','osha_injury_type','gender','type_loss','length_employed','shift']]
df_model_test = df_test[['cause','lost_time_or_medicalonly','osha_injury_type','gender','type_loss','length_employed','shift']]



df_model = df[['Dependent', 'ave_wkly_wage', 'body_part',
       'cause', 'claimant_age', 'gender', 'marital_status',
        'depart_code',  'domestic_foreign',
       'employ_status', 'handling_office', 
       'injury_city',  'jurisdiction',
       'lost_time_or_medicalonly', 'nature_injury', '#dependents',
       'osha_injury_type', 'severity_index', 'type_loss',
       'policy_yr', 'reforms_dummy', 'length_employed',
       'diff_carrier_employer', 'diff_employer_injury', 'shift' ]]

df_model.columns


print(df.isnull().sum())



#get dummy data
df_dum = pd.get_dummies(df_model)
df_dum_test = pd.get_dummies(df_model_test)



########################################################
############# Dimensionality Reduction#################
#######################################################
#make sure data is in the same scale before reduction(Zscore common)

#######PRINCIPAL COMPONENT ANALYSIS#########
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
#use one of these methods
scaled_df = StandardScaler().fit_transform(df)
scaled_df = preprocessing.scale(df_con)
pca = PCA()
pca.fit(scaled_data)
pca_data = pca.transform(scaled_df)

#Create Skree Plot to see how many components
per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)
labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]
 
plt.bar(range(1,len(per_var)+1), height=per_var, tick_label=labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()
 
                                                  





df_model.columns
df_dum.shape
df_dum.head()




#Train test split
# train test split 
from sklearn.model_selection import train_test_split

X = df_dum.drop('Dependent', axis =1)
y = df_dum.Dependent.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



#Multipl Linear Regression
# multiple linear regression 
import statsmodels.api as sm

X_sm = X = sm.add_constant(X)
model = sm.OLS(y,X_sm)
model.fit().summary()


# Poisson regression code
exog, endog = sm.add_constant(X), y
mod = sm.GLM(endog, exog,
             family=sm.families.Poisson(link=sm.families.links.log))

mod.fit().summary()
res = mod.fit()

#Gaussian
mod = sm.GLM(endog, exog,
             family=sm.families.Gaussian(sm.families.links.log))
model.fit().summary()
res = mod.fit()



from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score

lm = LinearRegression()
lm.fit(X_train, y_train)

#np.mean(cross_val_score(lm,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3))
mse = np.mean(cross_val_score(lm,X_train,y_train, scoring = 'neg_mean_squared_error', cv = 3))
rmse = np.sqrt(mse*-1)
print(rmse)


# lasso regression 
lm_l = Lasso(alpha=1)
lm_l.fit(X_train,y_train)
mse =np.mean(cross_val_score(lm_l,X_train,y_train, scoring = 'neg_mean_squared_error', cv= 3))
rmse = np.sqrt(mse*-1)
print(rmse)

# FInd the right value of alphe
alpha = []
error = []

for i in range(1,100):
    alpha.append(i/100)
    lml = Lasso(alpha=(i/100))
    error.append(np.mean(cross_val_score(lml,X_train,y_train, scoring = 'neg_mean_squared_error', cv= 3)))
    
plt.plot(alpha,error)

err = tuple(zip(alpha,error))
df_err = pd.DataFrame(err, columns = ['alpha','error'])
df_err[df_err.error == max(df_err.error)]


# random forest 
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
regressor = RandomForestRegressor()
regressor.fit(X_train, y_train)

mse =np.mean(cross_val_score(rf,X_train,y_train,scoring = 'neg_mean_squared_error', cv= 3))
rmse = np.sqrt(mse*-1)
print(rmse)





#Tune Model using Grid Search

from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators':range(10,300,10), 'criterion':('mse','mae'), 'max_features':('auto','sqrt','log2')}

gs = GridSearchCV(rf,parameters,scoring='neg_mean_squared_error',cv=3)
gs.fit(X_train,y_train)

gs.best_score_
gs.best_estimator_



#Test Ensembles
tpred_lm = lm.predict(X_test)
tpred_lml = lm_l.predict(X_test)
tpred_rf = gs.best_estimator_.predict(X_test)

from sklearn.metrics import mean_square_error
mean_square_error(y_test,tpred_lm, squared=False)
mean_square_error(y_test,tpred_lml, squared=False)
mean_square_error(y_test,tpred_rf, squared=False)

mean_square_error(y_test,(tpred_lm+tpred_rf)/2, squared=False)





#######################################################################################
#################################XgBOOST###############################################
#######################################################################################
## Hyperparameter optimization using RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import xgboost

# Track how long the search takes
def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


regressor=xgboost.XGBRegressor()
## Hyper Parameter Optimization
xgboost.XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=100, 
                      verbosity=1, silent=None, objective='reg:linear', booster='gbtree', n_jobs=1, nthread=None, 
                      gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1, colsample_bynode=1, 
                      reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, random_state=0, seed=None, missing=None, importance_type='gain')


params={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 
}
    
    
random_search=RandomizedSearchCV(regressor,param_distributions=params,n_iter=5,scoring='mse',n_jobs=-1,cv=5,verbose=3)

from datetime import datetime
# Here we go
start_time = timer(None) # timing starts from this point for "start_time" variable
random_search.fit(X,Y)
timer(start_time) # timing ends here for "start_time" variable

random_search.best_estimator_
random_search.best_params_


regressor=xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.5, gamma=0.4, learning_rate=0.1,
       max_delta_step=0, max_depth=6, min_child_weight=7, missing=None,
       n_estimators=100, n_jobs=1, nthread=None,
       objective='binary:logistic', random_state=0, reg_alpha=0,
       reg_lambda=1, scale_pos_weight=1, seed=None, silent=True,
       subsample=1)


from sklearn.model_selection import cross_val_score
score=cross_val_score(classifier,X,Y,cv=10)

score.mean()








import xgboost as xgb
from sklearn.metrics import mean_squared_error
train = xgb.DMatrix(X_train, label=y_train)
test = xgb.DMatrix(X_test, label=y_test)
rtest =xgb.DMatrix(X1, )
#Hyperparamters Values
param = {
        'max_depth':4,
        'eta': 0.3,
        'objective':'reg:tweedie',
        'tweedie_variance_power': 1.75
        }
epochs = 10
model = xgb.train(param, train, epochs)
predictions = model.predict(test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(rmse)
X1 = df_dum_test
rtest =xgb.DMatrix(X1, )
y_pred2 = model.predict(rtest)

#XgBoost Regression - another method
'''
data_dmatrix = xgb.DMatrix(data=X,label=y)

xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)

xg_reg = xgb.XGBRegressor(param)

model = xg_reg.fit(X_train,y_train)

preds = xg_reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print(rmse)
'''

####################################################################################################################
#####################################Feature Importance#############################################################
####################################################################################################################
from xgboost import XGBRegressor
from matplotlib import pyplot
import operator

# get importance
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()

plot_importance(model)


########PLOTTING THE FEATURE IMPORTANCE GRAPH#############################
''' HAVEN'T GOTTEN TO WORK

def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()

features = []
create_feature_map(features)
importance = model.get_fscore(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1))

df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()

featp = df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')
fig_featp = featp.get_figure()
'''


#PLOTTING VERTCAL
pd.Series(model.feature_importances_,index=list(X_train.columns.values)).sort_values(ascending=True).plot(kind='barh',
         figsize=(50,400),title='XGBOOST FEATURE IMPORTANCE')



#POLTTING HORIZONTALLY
importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(model.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances.plot.bar(figsize=(400,5))



#################################### CREATE PREDICTION ON TEST AFTER FORMATING TEST ###############################


#############FITTING REGRESSOR ON TEST DATA
df.LogDependent.describe()

# Predicting the Test set results
X1 = df_dum_test
y_pred2 = regressor.predict(X1)
y_pred2= np.exp(y_pred2)
# MERGED AND EXPORT
y_df = pd.DataFrame(data=y_pred2)
df = pd.concat([df_test['Obs_ID'], y_df], axis=1, )

df.to_csv('ypredictedxgb.csv')

from google.colab import files
files.download("ypredicted.csv")


'''
from google.colab import files
files.download("ypredicted.csv")'''
