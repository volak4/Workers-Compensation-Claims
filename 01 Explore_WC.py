# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 17:35:15 2020

@author: volak
"""

import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
import seaborn as sns

path = "C:/Users/volak/Google Drive/01 Sharpest Mind/Volak Sin/00 Works Comp/Data/CAX_Comp_Claims_Train.csv"
path2 = "C:/Users/volak/Google Drive/01 Sharpest Mind/Volak Sin/00 Works Comp/Data/CAX_Comp_Claims_Public_Test - no Dependent.csv"
df = pd.read_csv(path)
dft = pd.read_csv(path2)



##################################################################################################################
########### DATA Explore - PART 0: Initial DATA Observations#####################################################
##################################################################################################################
df.shape
df.dtypes
df.info()
df.head()
df.columns
df.isnull().sum()

df.Dependent.median()
df[df.Dependent <= 0]['Dependent'].count()

##################Separating categorical and continous columns ##################################################
cat_cols = list(df.select_dtypes(include=['objects']).columns)
con_cols = list(df.select_dtypes(exclude=['objects']).columns)


################################################
######### Profile Report #######################
################################################

## Profile Report
profile = ProfileReport(df,title='Train Data Raw Summary', html={'style':{'full_width':True}})
profile.to_file(output_file="1_Summary_Raw_Train.html")

profile = ProfileReport(df2,title='Test Data Raw Summary', html={'style':{'full_width':True}})
profile.to_file(output_file="1_Summary_Raw_Test.html")


#######################################################
######### EXPLORE RELATIONSHIPS BETWEEN FEATURES ######
#######################################################

### Basic Descriptive Stats
df.describe().T

##### GROUP AND VIEW
df.groupby('body_injury').size().plot(kind='bar')  '''~ df.body_injury.value_counts() '''


##### Visualize entire data set
df.hist(bins=50, figsize=(20,15))
plt.show()



###### CORRELATION
corr = df.corr() 
sns.heatmap(corr, annot=True)






########PROFILE REPORT AFTER CHANGES
path = "C:/Users/volak/Google Drive/01 Sharpest Mind/Volak Sin/00 Works Comp/Data/p1_train.csv"
path2 = "C:/Users/volak/Google Drive/01 Sharpest Mind/Volak Sin/00 Works Comp/Data/p1_test.csv"
df = pd.read_csv(path)
df2 = pd.read_csv(path2)
df = df.drop(['Unnamed: 0', 'Obs_ID'], axis=1)
df2 = df2.drop(['Unnamed: 0', 'Obs_ID'], axis=1)
df.shape
df2.shape

profile = ProfileReport(df,title='Train Data Raw Summary', html={'style':{'full_width':True}})
profile.to_file(output_file="1_Summary_Raw_Train.html")

profile = ProfileReport(df2,title='Test Data Raw Summary', html={'style':{'full_width':True}})
profile.to_file(output_file="1_Summary_Raw_Test.html")




df.body_part.value_counts()
df2.body_part.value_counts()


