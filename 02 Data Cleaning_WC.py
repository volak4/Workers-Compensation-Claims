# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 19:25:58 2020

@author: volak
"""
#IMPORT LIBRARY
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


#LOAD DATA
path = "C:/Users/volak/Google Drive/01 Sharpest Mind/Volak Sin/00 Works Comp/Data/CAX_Comp_Claims_Train.csv"
path2 = "C:/Users/volak/Google Drive/01 Sharpest Mind/Volak Sin/00 Works Comp/Data/CAX_Comp_Claims_Public_Test - no Dependent.csv"
train = pd.read_csv(path)
test = pd.read_csv(path2)




train.shape
test.shape
df.info()
##################################################################################################################
########### DATA CLEANING - PART 0: Basic Cleaning   #############################################################
##################################################################################################################

## COMBINE TRAIN TEST
df = pd.concat([train,test],axis=0)
df.shape


# REMOVE Redundant Columns
df = df.drop(['Body Part Code', 'Cause Code', 'Claimant Gender Code','Claimant State Code','Detail Cause Code',
              'Domestic vs. Foreign? Code','Employment Status Code','Claimant Marital Status Code',
              'Injury/Illness State Code','Jurisdiction Code','Lost Time or Medical Only? Code',
              'OSHA Injury Type Code','Nature of Injury/Illness Code','Severity Index Code Code','Type of Loss Code'], axis=1)

# Rename columns to remove spaces
df.rename(columns={'Average Weekly Wage':'ave_wkly_wage'}, inplace = True)
df.rename(columns={'Body Part':'body_part'}, inplace = True)
df.rename(columns={'Cause':'cause'}, inplace = True)
df.rename(columns={'Claimant Age':'claimant_age'}, inplace = True)
df.rename(columns={'Claimant Atty Firm Name':'atty_firm_name'}, inplace = True)
df.rename(columns={'Claimant Gender':'gender'}, inplace = True)
df.rename(columns={'Claimant Hire Date':'hire_date'}, inplace = True)
df.rename(columns={'Claimant Marital Status':'marital_status'}, inplace = True)
df.rename(columns={'Claimant State':'claim_st'}, inplace = True)
df.rename(columns={'Department Code':'depart_code'}, inplace = True)
df.rename(columns={'Detail Cause':'detail_cause'}, inplace = True)
df.rename(columns={'Domestic vs. Foreign?':'domestic_foreign'}, inplace = True)
df.rename(columns={'Dt Reported to Carrier/TPA':'dt_report_carrier'}, inplace = True)
df.rename(columns={'Dt Reported to Employer':'dt_report_employer'}, inplace = True)
df.rename(columns={'Employment Status':'employ_status'}, inplace = True)
df.rename(columns={'Date of Injury/Illness':'injury_date'}, inplace = True)
df.rename(columns={'Handling Office Name':'handling_office'}, inplace = True)
df.rename(columns={'How Injury Occurred':'how_injury_occur'}, inplace = True)
df.rename(columns={'Injury/Illness City':'injury_city'}, inplace = True)
df.rename(columns={'Injury/Illness Postal':'injury_postal'}, inplace = True)
df.rename(columns={'Injury/Illness State':'injury_state'}, inplace = True)
df.rename(columns={'Jurisdiction':'jurisdiction'}, inplace = True)
df.rename(columns={'Lost Time or Medical Only?':'lost_time_or_medicalonly'}, inplace = True)
df.rename(columns={'Nature of Injury/Illness':'nature_injury'}, inplace = True)
df.rename(columns={'Number of Dependents':'#dependents'}, inplace = True)
df.rename(columns={'OSHA Injury Type':'osha_injury_type'}, inplace = True)
df.rename(columns={'Severity Index Code':'severity_index'}, inplace = True)
df.rename(columns={'Time of Injury/Illness':'time_injury'}, inplace = True)
df.rename(columns={'Type of Loss':'type_loss'}, inplace = True)
df.rename(columns={'Policy Year':'policy_yr'}, inplace = True)
df.rename(columns={'Reforms_dummy':'reforms_dummy'}, inplace = True)



###Correct Formatting : i)Numerical ii)Categorical/Ordinal iii) Date/Time  iv) Coordinates
df['ave_wkly_wage'] = df['ave_wkly_wage'].replace({',':''},regex=True).apply(pd.to_numeric,1)
df['hire_date'] = pd.to_datetime(df.hire_date)
df['dt_report_carrier'] = pd.to_datetime(df.dt_report_carrier)
df['dt_report_employer'] = pd.to_datetime(df.dt_report_employer)
df['injury_date'] = pd.to_datetime(df.injury_date)
#df['#dependents'] = pd.to_numeric(df['#dependents'])



##################################################################################################################
############################# DATA CLEANING - PART 1: FEATURE ENGINEERING ########################################
##################################################################################################################

######## HOW LONG THEY HAVE BEEN WORKING AT COMPANY
df['hire_date_yr']=df.hire_date.dt.year

######### NEW DATE FIELDS
df['length_employed'] = df.hire_date_yr.apply(lambda x : x if x <1 else 2015 -x )
df['diff_carrier_employer'] = (df['dt_report_carrier'] - df['dt_report_employer']).dt.days
df['diff_employer_injury'] = (df['dt_report_employer'] - df['injury_date']).dt.days

df=df.drop(['hire_date_yr','dt_report_carrier','dt_report_employer','injury_date','hire_date'],axis=1)

######### Time of Day of injury
df[df.time_injury >1900]['time_injury'].count()
def work_shift(time):
    if time <= 900 :
        return '1st'
    elif time <=1700 :
        return '2nd'
    elif time >1700 :
        return '3rd'
    else:
        return 'unk'

df['shift'] = df['time_injury'].apply(work_shift)
#df['shift'].value_counts()
df['length_how_injury'] =df['how_injury_occur'].apply(lambda x: len(x))


### Data Check with Profiling
df_train= df.iloc[:15407,:]
df_test= df.iloc[15407:,:]
df_train = df_train.drop(['Unnamed: 0', 'Obs_ID'], axis=1)
df_test = df_test.drop(['Unnamed: 0', 'Obs_ID'], axis=1)
df_train.to_csv('p1_train.csv')
df_test.to_csv('p1_test.csv')

##################################################################################################################
########### DATA CLEANING - PART 2: MISSING VALUES IMPUATION AND ENCODING ########################################
##################################################################################################################


print(df_save.isnull().sum())
print(df.isnull().sum())


df.columns
##################################################################
############# CREATE COLUMN TO REMEMBER MISSING###################
##################################################################


df.loc[df['ave_wkly_wage'].isnull(),'ave_wkly_wage_NaN'] = 0
df.loc[df['ave_wkly_wage'].notnull(), 'ave_wkly_wage_NaN'] = 1

df.loc[df['claimant_age'].isnull(),'claimant_age_NaN'] = 0
df.loc[df['atty_firm_name'].isnull(),'atty_firm_name_NaN'] = 0
df.loc[df['marital_status'].isnull(),'marital_status_NaN'] = 0
df.loc[df['depart_code'].isnull(),'depart_code_NaN'] = 0
df.loc[df['injury_postal'].isnull(),'injury_postal_NaN'] = 0
df.loc[df['injury_state'].isnull(),'injury_state_NaN'] = 0
df.loc[df['lost_time_or_medicalonly'].isnull(),'lost_time_or_medicalonly_NaN'] = 0
df.loc[df['#dependents'].isnull(),'#dependents_NaN'] = 0
df.loc[df['osha_injury_type'].isnull(),'osha_injury_type_NaN'] = 0
df.loc[df['severity_index'].isnull(),'severity_index_NaN'] = 0
df.loc[df['type_loss'].isnull(),'type_loss_NaN'] = 0
df.loc[df['reforms_dummy'].isnull(),'reforms_dummy_NaN'] = 0
df.loc[df['length_employed'].isnull(),'length_employed_NaN'] = 0
df.loc[df['diff_carrier_employer'].isnull(),'diff_carrier_employer_NaN'] = 0
df.loc[df['diff_employer_injury'].isnull(),'diff_employer_injury_NaN'] = 0

df.loc[df['claimant_age'].notnull(),'claimant_age_NaN'] = 1
df.loc[df['atty_firm_name'].notnull(),'atty_firm_name_NaN'] = 1
df.loc[df['marital_status'].notnull(),'marital_statusNaN'] = 1
df.loc[df['depart_code'].notnull(),'depart_code_NaN'] = 1
df.loc[df['injury_postal'].notnull(),'injury_postal_NaN'] = 1
df.loc[df['injury_state'].notnull(),'injury_state_NaN'] = 1
df.loc[df['lost_time_or_medicalonly'].notnull(),'lost_time_or_medicalonly_NaN'] = 1
df.loc[df['#dependents'].notnull(),'#dependents_NaN'] = 1
df.loc[df['osha_injury_type'].notnull(),'osha_injury_type_NaN'] = 1
df.loc[df['severity_index'].notnull(),'severity_index_NaN'] = 1
df.loc[df['type_loss'].notnull(),'type_loss_NaN'] = 1
df.loc[df['reforms_dummy'].notnull(),'reforms_dummy_NaN'] = 1
df.loc[df['length_employed'].notnull(),'length_employed_NaN'] = 1
df.loc[df['diff_carrier_employer'].notnull(),'diff_carrier_employer_NaN'] = 1
df.loc[df['diff_employer_injury'].notnull(),'diff_employer_injury_NaN'] = 1







##################################################################
#############  SIMPLE MISSING VALUES IMPUTATION  #################
##################################################################




#Contious Mean/Mode/Median
#median_values = df['ave_wkly_wage'].median(axis=0)
#df['ave_wkly_wage'] = df['ave_wkly_wage'].fillna(median_values, inplace=True)

age_array = df[df["claimant_age"]!=np.nan]["claimant_age"]
df['claimant_age'] =df["claimant_age"].replace(np.nan,age_array.median())

age_array = df[df["length_employed"]!=np.nan]["length_employed"]
df['length_employed'] =df["length_employed"].replace(np.nan,age_array.median())

age_array = df[df["diff_carrier_employer"]!=np.nan]["diff_carrier_employer"]
df['diff_carrier_employer'] =df["diff_carrier_employer"].replace(np.nan,age_array.median())

age_array = df[df["diff_employer_injury"]!=np.nan]["diff_employer_injury"]
df['diff_employer_injury'] =df["diff_employer_injury"].replace(np.nan,age_array.median())

age_array = df[df["ave_wkly_wage"]!=np.nan]["ave_wkly_wage"]
df['ave_wkly_wage'] =df["ave_wkly_wage"].replace(np.nan,age_array.median())

age_array = df[df["#dependents"]!=np.nan]["#dependents"]
df['#dependents'] =df["#dependents"].replace(np.nan,age_array.median())

#Categorical Missing/ nan 0
df['injury_city'] = df['injury_city'].fillna(df['injury_city'].mode()[0])
df['injury_postal'] = df['injury_postal'].fillna(df['injury_postal'].mode()[0])
df['injury_state'] = df['injury_state'].fillna(df['injury_state'].mode()[0])
df['lost_time_or_medicalonly'] = df['lost_time_or_medicalonly'].fillna(df['lost_time_or_medicalonly'].mode()[0])
df['osha_injury_type'] = df['osha_injury_type'].fillna(df['osha_injury_type'].mode()[0])
df['severity_index'] = df['severity_index'].fillna(df['severity_index'].mode()[0])
df['type_loss'] = df['type_loss'].fillna(df['type_loss'].mode()[0])
df['marital_status'] = df['marital_status'].fillna('unk')
df['depart_code'] = df['depart_code'].fillna('unk')

df['injury_postal'] = df['injury_postal'].fillna('unk')
df['reforms_dummy'] = df['reforms_dummy'].fillna('unk')
df['atty_firm_name'] = df['atty_firm_name'].fillna('unk')   
'''for atty firm name = word cloud '''

df.isnull().sum()
##################################################################
######### MULITPLE IMPUTATION WITH CHAINED EQUATION v1.0 #########
##################################################################
from fancyimpute import IterativeImputer
 
#copy df as backup
df_mice_imputed = df.copy(deep=True)

# Initialize IterativeImputer
mice_imputer = IterativeImputer
df_mice_imputed.iloc[:,:] = mice_imputer.fit_transform(df)


import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imp = IterativeImputer(max_iter=10, random_state=0)


#copy df as backup
df_mice_imputed = df.copy(deep=True)

# Initialize IterativeImputer
mice_imputer = IterativeImputer
df_mice_imputed.iloc[:,:] = mice_imputer.fit_transform(df)


##################################################################
######### MULITPLE IMPUTATION WITH CHAINED EQUATION v2.0 ##########
##################################################################
df.columns
# Multivariate feature imputation
df_impute = df.drop(['Obs_ID','atty_firm_name','employ_status', 'how_injury_occur','jurisdiction','detail_cause','handling_office', 'injury_postal','length_how_injury',
              'time_injury','Dependent','policy_yr' ], axis=1 )
df_impute=pd.get_dummies(df_impute)
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imp = IterativeImputer(max_iter=10, random_state=0)
imp.fit(df_impute)
imputed_df = imp.transform(df_impute)
imputed_df = pd.DataFrame(imputed_df, columns=df_impute.columns)

imputed_df.to_csv('combined_impute.csv')
# IterativeImputer(random_state=0)
X_test = [[np.nan, 2], [6, np.nan], [np.nan, 6]]
# the model learns that the second feature is double the first
print(np.round(imp.transform(X_test)))



##################################################################
#############  CORRELATED MISSING VALUES IMPUTATION  #############
##################################################################




############################################################################################
################### ENCODING AND MISSING VALUE IMPUTATIONS #################################
############################################################################################

df['body_part'].value_counts()
###################################
####TOP TEN CATEGORICAL ENCODING###
###################################
def body_ten(top):
    if top =='Finger(s)' :
        return 'Finger(s)'
    elif top =='Low Back Area' :
        return 'Low Back Area'
    elif top =='Knee' :
        return 'Knee'
    elif top =='Other Facial Soft Tissue' :
        return 'Other Facial Soft Tissue'
    elif top =='Eye(s)' :
        return 'Eye(s)'    
    elif top =='Ankle' :
        return 'Ankle'
    elif top =='Hand' :
        return 'Hand'    
    elif top =='Shoulder(s)' :
        return 'Shoulder(s)'
    elif top =='Foot' :
        return 'Foot'    
    elif top =='Lower Leg' :
        return 'Lower Leg'
    elif top =='unk' :
        return 'unk'     
    else:
        return 'others'
df['body_part'] = df['body_part'].apply(body_ten)

def employ_five(top):
    if top == 'Unknown/Other' :
        return 'Unknown/Other'
    elif top =='Full-Time' :
        return 'Full-Time'
    elif top =='Seasonal' :
        return 'Seasonal'
    elif top =='Part-Time' :
        return 'Part-Time'
    elif top =='Piece Worker' :
        return 'Piece Worker'       
    else:
        return 'others'
df['employ_status'] = df['employ_status'].apply(employ_five)



def state_ten(top):
    if top == 'California' :
        return 'California'
    elif top =='New York' :
        return 'New York'
    elif top =='New Mexico' :
        return 'New Mexico'
    elif top =='Georgia' :
        return 'Georgia'
    elif top =='Texas' :
        return 'Texas'    
    elif top =='North Carolina' :
        return 'North Carolina'
    elif top =='Louisiana' :
        return 'Louisiana'    
    elif top =='New Jersey' :
        return 'New Jersey'
    elif top =='Florida' :
        return 'Florida'    
    elif top =='Illinois' :
        return 'Illinois'
    elif top =='unk' :
        return 'unk'     
    else:
        return 'others'
df['claim_st'] = df['claim_st'].apply(state_ten)

#Convert Department code to str
#df['depart_code']=df['depart_code'].apply(str) There is a mismatch with train/test data so must list out
def depart_ten(top):
    if top == 21 :
        return '21'
    elif top ==17 :
        return '17'
    elif top ==8 :
        return '8'
    elif top ==3 :
        return '3'
    elif top ==6 :
        return '6'    
    elif top ==2 :
        return '2'
    elif top ==14 :
        return '14'    
    elif top ==18 :
        return '18'
    elif top ==11 :
        return '11'    
    elif top ==1 :
        return '1'
    elif top =='unk' :
        return 'unk'     
    else:
        return 'others'
df['depart_code'] = df['depart_code'].apply(depart_ten)
df.depart_code.value_counts()
'''Come back and find out if there are any relations with depart code and any other variables '''

df.injury_city.value_counts()


def city_ten(top):
    if top == 'LOS ANGELES' :
        return 'LOS ANGELES'
    elif top =='UNKNOWN' :
        return 'UNKNOWN'
    elif top =='BURBANK' :
        return 'BURBANK'
    elif top =='NEW ORLEANS' :
        return 'NEW ORLEANS'
    elif top =='NEW YORK' :
        return 'NEW YORK'    
    elif top == 'BROOKLYN' :
        return 'BROOKLYN'
    elif top =='WILMINGTON' :
        return 'WILMINGTON'    
    elif top =='CULVER CITY' :
        return 'CULVER CITY'
    elif top =='AUSTIN' :
        return 'AUSTIN'    
    elif top =='ATLANTA' :
        return 'ATLANTA'
    elif top =='unk' :
        return 'unk'     
    else:
        return 'others'
df['injury_city'] = df['injury_city'].apply(city_ten)


def state_ten(top):
    if top == 'California' :
        return 'California'
    elif top =='Louisiana' :
        return 'Louisiana'
    elif top =='New York' :
        return 'New York'
    elif top =='Georgia' :
        return 'Georgia'
    elif top =='North Carolina' :
        return 'North Carolina'    
    elif top == 'Texas' :
        return 'Texas'
    elif top =='Hawaii' :
        return 'Hawaii'    
    elif top =='Michigan' :
        return 'Michigan'
    elif top =='New Mexico' :
        return 'New Mexico'    
    elif top =='Illinois' :
        return 'Illinois'
    elif top =='unk' :
        return 'unk'     
    else:
        return 'others'
df['injury_state'] = df['injury_state'].apply(state_ten)

df.injury_state.value_counts()
'''Jurisdiction seems to be an overlap except 1000 more in CA juris than state '''


def nature_ten(top):
    if top == 'Strain' :
        return 'Strain'
    elif top =='Laceration' :
        return 'Laceration'
    elif top =='Specific Injury - All Other' :
        return 'Specific Injury - All Other'
    elif top =='Contusion' :
        return 'Contusion'
    elif top =='Sprain' :
        return 'Sprain'    
    elif top == 'Puncture' :
        return 'Puncture'
    elif top =='Foreign Body' :
        return 'Foreign Body'    
    elif top =='Inflammation' :
        return 'Inflammation'
    elif top =='Fracture' :
        return 'Fracture'    
    elif top =='Infection' :
        return 'Infection'
    elif top =='unk' :
        return 'unk'     
    else:
        return 'others'
df['nature_injury'] = df['nature_injury'].apply(nature_ten)

df.nature_injury.value_counts()


def severity_ten(top):
    if top == 'No Serious Injury Indicated' :
        return 'No Serious Injury'
    elif top =='Fatality' :
        return 'Fatality'
    elif top =='Fractured Bone(s)' :
        return 'Fractured Bone(s)'
    elif top =='Back Injury involving Surgery/Extended Disability' :
        return 'Back Injury involving Surgery/Extended Disability'
    elif top =='Involves AIDS, Herpes, TSS, Cancer, Other Diseases' :
        return 'Involves AIDS, Herpes,etc'    
    elif top =='unk' :
        return 'unk'     
    else:
        return 'others'
df['severity_index'] = df['severity_index'].apply(severity_ten)

#df.severity_index.value_counts()















###########################################################################################################
##################### DATA CLEANING - PART 3:  TARGET FEATURE TRANSFORMATION ##############################
###########################################################################################################

################# TARGET FEATURE TRANSFORMATION #################
#transform dependent to log
df['LogDependent'] = df.Dependent.apply(lambda x: 1 if x<=0 else x)
#df.LogDependent.describe()
#df[df.LogDependent <= 0]['LogDependent'].count()
df['LogDependent'] = np.log(df.LogDependent)

df['DependentPlus1'] = df.Dependent.apply(lambda x: 1 if x<=0 else x)
#df.DependentPlus1.describe()


##################################################################
###################### OUTLIERS ##################################
##################################################################
############ Dependent Drop Outliers - There are 196 outliers
dataset=df
avg = dataset['Dependent'].mean()
std = dataset['Dependent'].std()
upper_outlier = avg + 3*std
lower_outlier = avg - 3*std

#df[df.Dependent > upper_outlier]['Dependent'].count() 

###SPLIT BACK INTO TRAIN TEST
df_train= df.iloc[:15407,:]
df_test= df.iloc[15407:,:]
df_train.shape
df_test.shape

#round up to drop outliers, til reasonable
df_train=dataset[dataset.Dependent > lower_outlier ]
df_train=dataset[dataset.Dependent < upper_outlier ]



df.to_csv('combined.csv')

df_train.to_csv('check_train.csv')
df_test.to_csv('check_test.csv')
df.columns

'''looking for easy code to compare unique values of test and train data '''