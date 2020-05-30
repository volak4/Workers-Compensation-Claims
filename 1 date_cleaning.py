# -*- coding: utf-8 -*-
"""
Created on Wed May 20 16:15:17 2020

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
df = pd.read_csv(path)
df2 = pd.read_csv(path2)

#df=df2
##################################################################################################################
########### DATA CLEANING - PART 0: Initial DATA Observations#####################################################
##################################################################################################################
df.isnull().sum()
df.shape
df.info()
df.head()
df.columns

#catergorical value counts
df['bodypart'].value_counts()
df['continus'].describe()

#sum of missing data
# Compare descriptive stats of continous train/test data.
df.isnull().sum()





























##################################################################################################################
########### DATA CLEANING - PART 1: Before Exploratory Data Analysis###############################################
##################################################################################################################
df[df.Dependent <= 0]['Dependent'].count()

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

#Make sure dates and numbers are correct format
df['ave_wkly_wage'] = df['ave_wkly_wage'].replace({',':''},regex=True).apply(pd.to_numeric,1)
df['hire_date'] = pd.to_datetime(df.hire_date)
df['dt_report_carrier'] = pd.to_datetime(df.dt_report_carrier)
df['dt_report_employer'] = pd.to_datetime(df.dt_report_employer)
df['injury_date'] = pd.to_datetime(df.injury_date)





#df.Dependent.describe()



################# FEATURE ENGINEERINGING
#transform dependent to log
df['LogDependent'] = df.Dependent.apply(lambda x: 1 if x<=0 else x)
df.LogDependent.describe()
df[df.LogDependent <= 0]['LogDependent'].count()
df['LogDependent'] = np.log(df.LogDependent)



#HOW LONG THEY HAVE BEEN WORKING AT COMPANY
df['hire_date_yr']=df.hire_date.dt.year

# NEW DATE FIELDS
df['length_employed'] = df.hire_date_yr.apply(lambda x : x if x <1 else 2015 -x )
df['diff_carrier_employer'] = (df['dt_report_carrier'] - df['dt_report_employer']).dt.days
df['diff_employer_injury'] = (df['dt_report_employer'] - df['injury_date']).dt.days

df=df.drop(['hire_date_yr','dt_report_carrier','dt_report_employer','injury_date','hire_date'],axis=1)


df_save=df
df.to_csv('data_clean1.csv')
df=df_save

df.ave_wkly_wage.describe()
df.isnull().sum()



##################################################################################################################
########### DATA CLEANING - PART 2: After Exploratory Data Analysis###############################################
##################################################################################################################


print(df_save.isnull().sum())
print(df.isnull().sum())

##############################  MISSING VALUES




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





#Categorical Missing/ nan 0
df['injury_city'] = df['injury_city'].fillna(df['injury_city'].mode()[0])
df['injury_state'] = df['injury_state'].fillna(df['injury_state'].mode()[0])
df['lost_time_or_medicalonly'] = df['lost_time_or_medicalonly'].fillna(df['lost_time_or_medicalonly'].mode()[0])
df['osha_injury_type'] = df['osha_injury_type'].fillna(df['osha_injury_type'].mode()[0])
df['severity_index'] = df['severity_index'].fillna(df['severity_index'].mode()[0])
df['type_loss'] = df['type_loss'].fillna(df['type_loss'].mode()[0])
df['marital_status'] = df['marital_status'].fillna('unk')
df['depart_code'] = df['depart_code'].fillna('unk')
df['#dependents'] = df['#dependents'].fillna('unk')
df['injury_postal'] = df['injury_postal'].fillna('unk')
df['reforms_dummy'] = df['reforms_dummy'].fillna('unk')

# For now attorney firm name
df['atty_firm_name'] = df['atty_firm_name'].fillna('unk')

'''
  'Injury/Illness City' = mode
  'injury_state' = mode
   'lost_time_or_medicalonly' 11 = mode
   '#dependents'= unk
   'osha_injury_type'= mode
    'severity_index' = mode
     'type_loss' = mode'''
   
#Random 
''' didn't quite work'''
#df["ave_wkly_wage"].fillna(lambda x: random.choice(df[df["ave_wkly_wage"] != np.nan]), inplace =True)




print(df.isnull().sum())





#####################MORE AFTER EDA CHANGES

df['length_how_injury'] =df['how_injury_occur'].apply(lambda x: len(x))
df.describe()

df[df.diff_carrier_employer <0]['diff_carrier_employer'].count()

#Remove Negative Days
df['diff_carrier_employer'] = df['diff_carrier_employer'].apply(lambda x: x if x>0 else 0)
df['diff_employer_injury'] = df['diff_employer_injury'].apply(lambda x: x if x>0 else 0)


# Dependent Drop Outliers - There are 196 outliers
dataset=df
avg = dataset['Dependent'].mean()
std = dataset['Dependent'].std()
upper_outlier = avg + 5*std
lower_outlier = avg - 5*std

df[df.Dependent > upper_outlier]['Dependent'].count() 

#round up to drop outliers, til reasonable
df=dataset[dataset.Dependent > lower_outlier ]
df=dataset[dataset.Dependent < upper_outlier ]


######MORE FEATURE ENGINEERING

''' need to fix
#Top lawfirms
#Editing text
df['atty_firm_name'].value_counts()
df.atty_firm_name.astype(str)

def atty_simplifier(title):
    if 'TLEVY, STERN & FORD' :
        return 'TLEVY, STERN & FORD'
    elif 'TJ. LEEDS BARROL, IV ATTORNEY AT LAW' :
        return 'TJ. LEEDS BARROL'
    elif 'TLEVY, FORD & WALLACH':
        return 'TLEVY, FORD & WALLACH'
    elif 'TLEVY, STERN, & FORD' :
        return 'TLEVY, STERN & FORD'
    elif 'nan' :
        return 'nan'
    else:
        return 'other_law_firms'


df['atty_firm_name_simp'] = df['atty_firm_name'].apply(atty_simplifier)
df['atty_firm_name_simp'].value_counts()
'''

# Time of Day of injury
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
df['shift'].value_counts()



#SPILT CLAIM YEARS





# Last minute columns check
df['body_part'].value_counts()
df['cause'].value_counts()
df['claimant_age'].value_counts()
df['claim_st'].value_counts()
df['employ_status'].value_counts()
df['injury_postal'].value_counts()
df['lost_time_or_medicalonly'].value_counts()
df['osha_injury_type'].value_counts()
df['policy_yr'].value_counts()
df['diff_carrier_employer'].value_counts()
df[ 'atty_firm_name'].value_counts()
df[ 'depart_code'].value_counts()
df[ 'handling_office'].value_counts()
df[ 'injury_state'].value_counts()
df[ 'nature_injury'].value_counts()
df[ 'severity_index'].value_counts()
df[ 'reforms_dummy'].value_counts()
df[ 'diff_employer_injury'].value_counts()
df[ 'gender'].value_counts()
df[ 'detail_cause'].value_counts()
df[ 'how_injury_occur'].value_counts()
df[ 'jurisdiction'].value_counts()
df[ '#dependents'].value_counts()
df[ 'time_injury'].value_counts()
df[ 'length_employed'].value_counts()
df[ 'length_how_injury'].value_counts()
df[ 'marital_status'].value_counts()
df[ 'domestic_foreign'].value_counts()
df[ 'injury_city'].value_counts()
df[ 'type_loss'].value_counts()



df.isnull().sum()

df.to_csv('data_clean2.csv')
df.to_csv('data_clean_test.csv')



##################################################################################################################
########### DATA CLEANING - PART 3: After Feature Importance       ###############################################
##################################################################################################################


'''
TOP RANKED FEATURE BY ORDER OF IMPORTANCE
1. Lost time or Medical Only
2. Employ_status_unknown/Other
3. ave_wkly_wage
4. handling_office_head Offic
5. employ_status_Full-Time
6. reforms_dummy_California Reforms 2
7. #dependents
8. body_part_Shoulders
9. jurisdiction_California
10. handling_office_Sacremento
11. nature_injury_Rupture
12. injury_city_SAN FRANCISCO
13. length_employed
14. body_part_Head
15. reforms_dummy_ California Reform 1
16. handling_office_WC_Southwest
17. claimant_age
18. handling_office_Illinois
19. handling_office_Dallas WC
20. martial_status_Unmarried, Single, Widowed

'''

'''
QUESTIONS:
    do you need to take the log of dependents if running tweedie?
    
'''

df.columns
df['lost_time_or_medicalonly'].value_counts()