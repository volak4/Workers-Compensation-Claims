# -*- coding: utf-8 -*-
"""
Created on Wed May 20 10:31:20 2020

@author: volak
"""


#IMPORT LIBRARY
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#LOAD DATA
path = "C:/Users/volak/Google Drive/01 Sharpest Mind/Volak Sin/00 Works Comp/Data/data_clean1.csv"
df = pd.read_csv(path)
df.columns
#Continous Features
'Dependent', 'ave_wkly_wage', 'claimant_age', '#dependents', 'time_injury', 'policy_yr', 'length_employed', 'diff_carrier_employer', 'diff_employer_injury'


#Categorical Features
'body_part', 'cause', 'atty_firm_name', 'gender', 'marital_status', 'claim_st', 'depart_code', 'detail_cause', 'domestic_foreign', 'employ_status', 'handling_office', 'how_injury_occur', 'injury_city', 'injury_postal', 'injury_state', 'jurisdiction', 'lost_time_or_medicalonly', 'nature_injury', 'osha_injury_type', 'severity_index', 'type_loss', 'reforms_dummy'



missing=df.isnull().sum() / df.shape[0]
missing = pd.DataFrame(missing)
missing  

''' Missing Values
injury_city			       0.01%
osha_injury_type		       0.01%
lost_time_or_medicalonly	0.07%
injury_state			       0.08%
type_loss			          0.34%
dt_report_employer		   0.95%
severity_index			   2.23%
hire_date			          5.85%
claimant_age			      14.04%
injury_postal			      23.91%
reforms_dummy			      40.97%
depart_code			      52.72%
ave_wkly_wage			      61.89%
atty_firm_name			  80.55%
marital_status			  82.38%
#dependents			     96.11%
'''

################### UNIVARIATE ANALYSIS####################
#Plot Categorical Data
#Categorical
df_cat = df[['body_part', 'cause', 'atty_firm_name', 'gender', 'marital_status', 'claim_st', 
             'depart_code', 'detail_cause', 'domestic_foreign', 'employ_status', 'handling_office',
             'how_injury_occur', 'injury_city', 'injury_postal', 'injury_state', 'jurisdiction', 'lost_time_or_medicalonly',
             'nature_injury', 'osha_injury_type', 'severity_index', 'type_loss', 'reforms_dummy']]

df_cat = df[['atty_firm_name', 'body_part']]

'''just viewing top 5 contributors'''    
for i in df_cat.columns:
    cat_num = df_cat[i].value_counts()[:15]
    plt.figure(figsize=(15,4))
    print("graph for %s: total = %d" % (i, len(cat_num)))
    chart = sns.barplot(x=cat_num.index, y=cat_num)
    chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
    plt.show()

df.atty_firm_name.value_counts()






######Plot Continous Data
#Histogram
df_con = df[['LogDependent', 'ave_wkly_wage', 'claimant_age', '#dependents',
             'policy_yr', 'length_employed', 'diff_carrier_employer', 'diff_employer_injury',
             'length_how_injury']]

'''just viewing top 5 contributors'''    
for i in df_con.columns:
    con_num = df_con[i].value_counts()
    plt.figure(figsize=(15,4))
    print("graph for %s: total = %d" % (i, len(con_num)))
    chart = sns.distplot(con_num.index)
    chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
    plt.show()





################### BIVARIATE ANALYSIS####################





df['length_how_injury'] =df['how_injury_occur'].apply(lambda x: len(x))
df.describe()

df[df.diff_carrier_employer <0]['diff_carrier_employer'].count()

#Remove Negative Days
df['diff_carrier_employer'] = df['diff_carrier_employer'].apply(lambda x: x if x>0 else 0)
df['diff_employer_injury'] = df['diff_employer_injury'].apply(lambda x: x if x>0 else 0)

df.columns






#Plot Categorical Data
#Categorical
df_cat = df[['body_part',
       'cause', 'claimant_age', 'atty_firm_name', 'gender', 'hire_date',
       'marital_status', 'claim_st', 'depart_code', 'detail_cause',
       'domestic_foreign', 'dt_report_carrier', 'dt_report_employer',
       'employ_status', 'injury_date', 'handling_office', 'how_injury_occur',
       'Injury/Illness City', 'injury_postal', 'injury_state', 'jurisdiction',
       'lost_time_or_medicalonly', 'nature_injury', '#dependents',
       'osha_injury_type', 'severity_index', 'time_injury', 'type_loss',
       'policy_yr', 'reforms_dummy']]

#Create Box Plot
'''just viewing top 5 contributors'''    
for i in df_cat.columns:
    cat_num = df_cat[i].value_counts()[:5]
    plt.figure(figsize=(15,4))
    print("graph for %s: total = %d" % (i, len(cat_num)))
    chart = sns.barplot(x=cat_num.index, y=cat_num)
    chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
    plt.show()
    
# Want to see dependents by state
pd.pivot_table(df, index='body_part', values='Dependent').sort_values('Dependent', ascending = False)


######EXPLORING EVERY VARIABLE
# Dependent
dataset=df
avg = dataset['Dependent'].mean()
std = dataset['Dependent'].std()
upper_outlier = avg + 3*std
lower_outlier = avg - 3*std
df[df.Dependent > upper_outlier]['Dependent'].count() '''There are 196 outliers'''
df[df.Dependent <= 0]['Dependent'].count()





##### FIND CORRELATED VARIABLES AND REMOVE FROM MODEL
colormap = plt.cm.magma
plt.figure(figsize=(16,12))
plt.title('Pearson correlation of continuous features', y=1.05, size=15)
sns.heatmap(df_con.corr(),linewidths=0.1,vmax=1.0, square=True, 
            cmap=colormap, linecolor='white', annot=True)

df.columns
#############################################################################
###########################FEATURE IMPORTANCE################################
#############################################################################
df=df.drop(['Unnamed: 0', 'Obs_ID'], axis=1)
df_dum = pd.get_dummies(df)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_leaf=4, max_features=0.2, n_jobs=-1, random_state=0)
rf.fit(df_dum.drop(['Dependent','LogDependent'],axis=1), df_dum.LogDependent)
features = df_dum.drop(['Dependent','LogDependent'],axis=1).columns.values
print("----- Training Done -----")

#PRINT GRAPH - Scatter plot 
trace = go.Scatter(
    y = rf.feature_importances_,
    x = features,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 13,
        #size= rf.feature_importances_,
        #color = np.random.randn(500), #set color equal to a variable
        color = rf.feature_importances_,
        colorscale='Portland',
        showscale=True
    ),
    text = features
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Random Forest Feature Importance',
    hovermode= 'closest',
     xaxis= dict(
         ticklen= 5,
         showgrid=False,
        zeroline=False,
        showline=False
     ),
    yaxis=dict(
        title= 'Feature Importance',
        showgrid=False,
        zeroline=False,
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')    


#######################BAR PLOT
x, y = (list(x) for x in zip(*sorted(zip(rf.feature_importances_, features), 
                                                            reverse = False)))
trace2 = go.Bar(
    x=x ,
    y=y,
    marker=dict(
        color=x,
        colorscale = 'Viridis',
        reversescale = True
    ),
    name='Random Forest Feature importance',
    orientation='h',
)

layout = dict(
    title='Barplot of Feature importances',
     width = 900, height = 2000,
    yaxis=dict(
        showgrid=False,
        showline=False,
        showticklabels=True,
#         domain=[0, 0.85],
    ))

fig1 = go.Figure(data=[trace2])
fig1['layout'].update(layout)
py.iplot(fig1, filename='plots')

##### DECISION TREE VISUALIZATION
from sklearn import tree
from IPython.display import Image as PImage
from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont
import re

decision_tree = tree.DecisionTreeClassifier(max_depth = 3)
decision_tree.fit(train.drop(['id', 'target'],axis=1), train.target)

# Export our trained model as a .dot file
with open("tree1.dot", 'w') as f:
     f = tree.export_graphviz(decision_tree,
                              out_file=f,
                              max_depth = 4,
                              impurity = False,
                              feature_names = train.drop(['id', 'target'],axis=1).columns.values,
                              class_names = ['No', 'Yes'],
                              rounded = True,
                              filled= True )
        
#Convert .dot to .png to allow display in web notebook
check_call(['dot','-Tpng','tree1.dot','-o','tree1.png'])

# Annotating chart with PIL
img = Image.open("tree1.png")
draw = ImageDraw.Draw(img)
img.save('sample-out.png')
PImage("sample-out.png",)

################################GRADIENT BOOSTED FEATURE IMPORTANCE
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, min_samples_leaf=4, max_features=0.2, random_state=0)
gb.fit(train.drop(['id', 'target'],axis=1), train.target)
features = train.drop(['id', 'target'],axis=1).columns.values
print("----- Training Done -----")


# Scatter plot 
trace = go.Scatter(
    y = gb.feature_importances_,
    x = features,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 13,
        #size= rf.feature_importances_,
        #color = np.random.randn(500), #set color equal to a variable
        color = gb.feature_importances_,
        colorscale='Portland',
        showscale=True
    ),
    text = features
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Gradient Boosting Machine Feature Importance',
    hovermode= 'closest',
     xaxis= dict(
         ticklen= 5,
         showgrid=False,
        zeroline=False,
        showline=False
     ),
    yaxis=dict(
        title= 'Feature Importance',
        showgrid=False,
        zeroline=False,
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')



x, y = (list(x) for x in zip(*sorted(zip(gb.feature_importances_, features), 
                                                            reverse = False)))
trace2 = go.Bar(
    x=x ,
    y=y,
    marker=dict(
        color=x,
        colorscale = 'Viridis',
        reversescale = True
    ),
    name='Gradient Boosting Classifer Feature importance',
    orientation='h',
)

layout = dict(
    title='Barplot of Feature importances',
     width = 900, height = 2000,
    yaxis=dict(
        showgrid=False,
        showline=False,
        showticklabels=True,
    ))

fig1 = go.Figure(data=[trace2])
fig1['layout'].update(layout)
py.iplot(fig1, filename='plots')




####Send to excel
df.to_csv('data_eda.csv')