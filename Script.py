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

# 3. Data Processing
#--------------------------------
from sklearn import preprocessing

# Loop over each column in the DataFrame where dtype is 'object'
for col in df.select_dtypes(include=['object']).columns:
    
    # Initialize a LabelEncoder object
    label_encoder = preprocessing.LabelEncoder()
    
    # Fit the encoder to the unique values in the column
    label_encoder.fit(df[col].unique())
    
    # Transform the column using the encoder
    df[col] = label_encoder.transform(df[col])
    
    # Print the column name and the unique encoded values
    print(f"{col}: {df[col].unique()}")
#--------------------------------------------------------------------------------------------------------------------
            
            
# 4. Balance the Class Feature
#--------------------------------
sns.countplot(df['class'])
df['class'].value_counts()

#Class - 1  => 700 (Majority)
#Class - 0  => 300

from sklearn.utils import resample
#create two different dataframe of majority and minority class 
df_majority = df[(df['class']==1)] 
df_minority = df[(df['class']==0)] 
# upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,    # sample with replacement
                                 n_samples= 700, # to match majority class
                                 random_state=0)  # reproducible results
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_minority_upsampled, df_majority])
#--------------------------------------------------------------------------------------------------------------------
            
            
# 5. Remove the Outlier using Z-Score because the Outlier is not extreme
#-----------------------------------------------------------------------
from scipy import stats

# define a function to remove outliers using z-score for only selected numerical columns
def remove_outliers(df_upsampled, cols, threshold=3):
    # loop over each selected column
    for col in cols:
        # calculate z-score for each data point in selected column
        z = np.abs(stats.zscore(df_upsampled[col]))
        # remove rows with z-score greater than threshold in selected column
        df_upsampled = df_upsampled[(z < threshold) | (df_upsampled[col].isnull())]
    return df_upsampled

selected_cols = ['duration', 'credit_amount', 'age']
df_clean = remove_outliers(df_upsampled, selected_cols)
df_clean.shape

#Correlation Heatmap
plt.figure(figsize=(20, 16))
sns.heatmap(df_clean.corr(), fmt='.2g', annot=True)
#Img8.png
#--------------------------------------------------------------------------------------------------------------------
            
            
# 6. Build Machine Learning Model with Hyperparameter Tuning
#-----------------------------------------------------------
X = df_clean.drop('class', axis=1)
y = df_clean['class']

#test size 20% and train size 80%
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=0)
#--------------------------------------------------------------------------------------------------------------------
            
            
# 7. Decision Tree
#------------------
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
dtree = DecisionTreeClassifier()
param_grid = {
    'max_depth': [3, 4, 5, 6, 7, 8],
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': [1, 2, 3, 4]
}

# Perform a grid search with cross-validation to find the best hyperparameters
grid_search = GridSearchCV(dtree, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Print the best hyperparameters
print(grid_search.best_params_)
#=> {'max_depth': 8, 'min_samples_leaf': 1, 'min_samples_split': 2}

from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(random_state=0, max_depth=8, min_samples_leaf=1, min_samples_split=2)
dtree.fit(X_train, y_train)
#=> DecisionTreeClassifier(max_depth=8, random_state=0)

y_pred = dtree.predict(X_test)
print("Accuracy Score :", round(accuracy_score(y_test, y_pred)*100 ,2), "%")
#=> Accuracy Score : 74.63 %

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, jaccard_score, log_loss
print('F-1 Score : ',(f1_score(y_test, y_pred, average='micro')))
print('Precision Score : ',(precision_score(y_test, y_pred, average='micro')))
print('Recall Score : ',(recall_score(y_test, y_pred, average='micro')))
print('Jaccard Score : ',(jaccard_score(y_test, y_pred, average='micro')))
print('Log Loss : ',(log_loss(y_test, y_pred)))

#=> F-1 Score :  0.746268656716418
#=> Precision Score :  0.746268656716418
#=> Recall Score :  0.746268656716418
#=> Jaccard Score :  0.5952380952380952
#=> Log Loss :  8.763656653654484
#--------------------------------------------------------------------------------------------------------------------
            
            
# 8. Random Forest
#-----------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
rfc = RandomForestClassifier()
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 5, 10],
    'max_features': ['sqrt', 'log2', None]
}

# Perform a grid search with cross-validation to find the best hyperparameters
grid_search = GridSearchCV(rfc, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Print the best hyperparameters
print(grid_search.best_params_)
#=> {'max_depth': None, 'max_features': 'sqrt', 'n_estimators': 200}

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=0, max_features='log2', n_estimators=100)
rfc.fit(X_train, y_train)
#=> RandomForestClassifier(max_features='log2', random_state=0)

y_pred = rfc.predict(X_test)
print("Accuracy Score :", round(accuracy_score(y_test, y_pred)*100 ,2), "%")
#=> Accuracy Score : 88.43 %

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, jaccard_score, log_loss
print('F-1 Score : ',(f1_score(y_test, y_pred, average='micro')))
print('Precision Score : ',(precision_score(y_test, y_pred, average='micro')))
print('Recall Score : ',(recall_score(y_test, y_pred, average='micro')))
print('Jaccard Score : ',(jaccard_score(y_test, y_pred, average='micro')))
print('Log Loss : ',(log_loss(y_test, y_pred)))

#=> F-1 Score :  0.8843283582089553
#=> Precision Score :  0.8843283582089553
#=> Recall Score :  0.8843283582089553
#=> Jaccard Score :  0.7926421404682275
#=> Log Loss :  3.9951838232056085
#--------------------------------------------------------------------------------------------------------------------
            
            
# 9. Logistic Regression
#------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
lr = LogisticRegression(random_state=0)
param_grid = {
    'penalty': ['l1', 'l2', 'elasticnet'],
    'solver': ['liblinear', 'saga']
}

# Perform a grid search with cross-validation to find the best hyperparameters
grid_search = GridSearchCV(lr, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Print the best hyperparameters
print(grid_search.best_params_)
#=> {'penalty': 'l1', 'solver': 'liblinear'}

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=0, solver='liblinear', penalty='l1')
lr.fit(X_train, y_train)	
#=> LogisticRegression(penalty='l1', random_state=0, solver='liblinear')

y_pred = lr.predict(X_test)
print("Accuracy Score :", round(accuracy_score(y_test, y_pred)*100 ,2), "%")
#=> Accuracy Score : 73.88 %

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, jaccard_score, log_loss
print('F-1 Score : ',(f1_score(y_test, y_pred, average='micro')))
print('Precision Score : ',(precision_score(y_test, y_pred, average='micro')))
print('Recall Score : ',(recall_score(y_test, y_pred, average='micro')))
print('Jaccard Score : ',(jaccard_score(y_test, y_pred, average='micro')))
print('Log Loss : ',(log_loss(y_test, y_pred)))

#=> F-1 Score :  0.7388059701492538
#=> Precision Score :  0.7388059701492538
#=> Recall Score :  0.7388059701492538
#=> Jaccard Score :  0.5857988165680473
#=> Log Loss :  9.02142661773808
#--------------------------------------------------------------------------------------------------------------------
            
            
# 10. Confusion Matrix
#---------------------

#Matrix for Decision Tree Accuracy
dtree = DecisionTreeClassifier(random_state=0, max_depth=8, min_samples_leaf=1, min_samples_split=2)
dtree.fit(X_train, y_train)
y_pred = dtree.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,5))
sns.heatmap(data=cm,linewidths=.5, annot=True,  cmap = 'Blues')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score for Decision Tree: {0}'.format(dtree.score(X_test, y_test))
plt.title(all_sample_title, size = 15)
#=> Text(0.5, 1.0, 'Accuracy Score for Decision Tree: 0.746268656716418')
#Img9.png

#Matrix for Random Forest Accuracy
rfc = RandomForestClassifier(random_state=0, max_features='log2', n_estimators=100)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,5))
sns.heatmap(data=cm,linewidths=.5, annot=True,  cmap = 'Blues')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score for Random Forest: {0}'.format(rfc.score(X_test, y_test))
plt.title(all_sample_title, size = 15)
#=> Text(0.5, 1.0, 'Accuracy Score for Random Forest: 0.8843283582089553')
#Img10.png

#Matrix for Logistic Regression Accuracy
lr = LogisticRegression(random_state=0, solver='liblinear', penalty='l1')
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,5))
sns.heatmap(data=cm,linewidths=.5, annot=True,  cmap = 'Blues')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score for Logistic Regression: {0}'.format(lr.score(X_test, y_test))
plt.title(all_sample_title, size = 15)
#=> Text(0.5, 1.0, 'Accuracy Score for Logistic Regression: 0.7388059701492538')
#Img11.png
