# 1. Pre - Processing
#--------------------------------
#import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set_theme(color_codes=True)
pd.set_option('display.max_columns', None)

#import data
df = pd.read_csv('credit_customers.csv')

#Understand the number of unique categorical variables (Features)
df.select_dtypes(include='object').nunique()
#--------------------------------------------------------------------------------------------------------------------

# 2. Exploratory Data Analysis
#--------------------------------
# list of categorical variables to plot
cat_vars = ['checking_status', 'credit_history','purpose', 'savings_status', 'employment',
            'personal_status', 'other_parties', 'property_magnitude', 'other_payment_plans', 'housing', 
            'job', 'own_telephone', 'foreign_worker','existing_credits','num_dependents']

#--------------------------------------------------------------------------------------------------------------------
#Create Bar plots to analyze Categorical Variables Vs "Class" Feature
# create figure with subplots
fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(20, 20))
axs = axs.flatten()

# create barplot for each categorical variable
for i, var in enumerate(cat_vars):
    sns.countplot(x=var, hue='class', data=df, ax=axs[i])
    axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation=90)

# adjust spacing between subplots
fig.tight_layout()

# show plot
plt.show()
#Img1.png
#--------------------------------------------------------------------------------------------------------------------
#Create Hist plots to analyze Categorical Variables Vs "Class" Feature
import warnings
warnings.filterwarnings("ignore")
# get list of categorical variables
cat_vars = ['checking_status', 'credit_history','purpose', 'savings_status', 'employment',
            'personal_status', 'other_parties', 'property_magnitude', 'other_payment_plans', 'housing', 
            'job', 'own_telephone', 'foreign_worker','existing_credits','num_dependents']

# create figure with subplots
fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(20, 20))
axs = axs.flatten()

# create histplot for each categorical variable
for i, var in enumerate(cat_vars):
    sns.histplot(x=var, hue='class', data=df, ax=axs[i], multiple="fill", kde=False, element="bars", fill=True, stat='density')
    axs[i].set_xticklabels(df[var].unique(), rotation=90)
    axs[i].set_xlabel(var)

# adjust spacing between subplots
fig.tight_layout()

# show plot
plt.show()
#Img2.png
#--------------------------------------------------------------------------------------------------------------------
#Create Box plots to analyze Numeric Variables
num_vars = ['duration', 'credit_amount', 'installment_commitment', 'residence_since', 'age']

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))
axs = axs.flatten()

for i, var in enumerate(num_vars):
    sns.boxplot(x=var, data=df, ax=axs[i])

fig.tight_layout()

plt.show()
#Img3.png
#--------------------------------------------------------------------------------------------------------------------
#Create Violin plots to analyze Numeric Variables Vs "Class" Feature
num_vars = ['duration', 'credit_amount', 'installment_commitment', 'residence_since', 'age']

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))
axs = axs.flatten()

for i, var in enumerate(num_vars):
    sns.violinplot(x=var, data=df, ax=axs[i])

fig.tight_layout()

plt.show()
#Img4.png
#--------------------------------------------------------------------------------------------------------------------
#Create Violin plots to analyze Numeric Variables Vs "Class" Feature
num_vars = ['duration', 'credit_amount', 'installment_commitment', 'residence_since', 'age']

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))
axs = axs.flatten()

for i, var in enumerate(num_vars):
    sns.violinplot(x=var, y='class', data=df, ax=axs[i])

fig.tight_layout()

plt.show()
#Img5.png
#--------------------------------------------------------------------------------------------------------------------
#Create Histogram to analyze Categorical Variables
num_vars = ['duration', 'credit_amount', 'installment_commitment', 'residence_since', 'age']

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))
axs = axs.flatten()

for i, var in enumerate(num_vars):
    sns.histplot(x=var, data=df, ax=axs[i])

fig.tight_layout()

plt.show()
#Img6.png
#--------------------------------------------------------------------------------------------------------------------
#Create Histogram to analyze Categorical Variables Vs "Class" Feature
num_vars = ['duration', 'credit_amount', 'installment_commitment', 'residence_since', 'age']

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))
axs = axs.flatten()

for i, var in enumerate(num_vars):
    sns.histplot(x=var, hue='class', data=df, ax=axs[i], multiple="stack")

fig.tight_layout()

plt.show()
#Img7.png
#--------------------------------------------------------------------------------------------------------------------
