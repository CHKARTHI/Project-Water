# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 12:25:48 2023

@author: DELL
"""
#ignoring all warnings
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
%matplotlib inline
import seaborn as sns
sns.set_style('darkgrid')
from IPython.display import display, HTML

from scipy import stats
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.weightstats import ztest
from scipy import interp

from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
#pip install Boruta
from boruta import BorutaPy
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold,cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.tree import DecisionTreeClassifier as DT 
from sklearn.ensemble import RandomForestClassifier as RF 
from sklearn.ensemble import ExtraTreesClassifier as ET
from sklearn.ensemble import AdaBoostClassifier as AB
from xgboost import XGBClassifier 
from sklearn.naive_bayes import GaussianNB as GNB  
from sklearn.svm import SVC 
from sklearn.ensemble import VotingClassifier 
from sklearn.ensemble import StackingClassifier

#importing the dataset
wq_df = pd.read_csv('C:/Users/DELL/Desktop/Projects/water_potability.csv')
wq_df.head()

#Understanding Dataset Features

wq_df.info()
wq_df.describe()
wq_df.columns = wq_df.columns.str.lower()
wq_df.columns
wq_df.isnull().sum()
# wq_df = wq_df.dropna();
wq_df['ph'] = wq_df['ph'].fillna(wq_df['ph'].median())
wq_df['sulfate'] = wq_df['sulfate'].fillna(wq_df['sulfate'].median())
wq_df['trihalomethanes'] = wq_df['trihalomethanes'].fillna(wq_df['trihalomethanes'].median())
wq_df['ph_category'] = wq_df['ph'].apply(lambda value: 'acidic level' if value < 6.5 else 'alkaline level' if value > 8.5 else 'normal pH_surface water')
wq_df['ph_category'] = pd.Categorical(wq_df['ph_category'], 
                                      categories=['acidic level', 'alkaline level', 'normal pH_surface water'])
wq_df.sample(5)
wq_df.shape

#Descriptive Statistics
not_safe = round(wq_df.iloc[:,:-2][wq_df['potability']==0].describe(),2)
safe = round(wq_df.iloc[:,:-2][wq_df['potability']==1].describe(),2)
pd.concat([safe,not_safe],
          axis=1,
          keys=['Potable','Not Potable'])
          
ph_acidic = round(wq_df.iloc[:,1:-2][wq_df['ph_category']=='acidic level'].describe(),2)
ph_permissable = round(wq_df.iloc[:,1:-2][wq_df['ph_category']=='normal pH_surface water'].describe(),2)
ph_alkaline = round(wq_df.iloc[:,1:-2][wq_df['ph_category']=='alkaline level'].describe(),2)
df_ph_cat = pd.concat([ph_acidic,ph_permissable,ph_alkaline],axis=1,keys=['Acidic Level','Normal pH_surface water','Alkaline Level'])
with pd.option_context('display.max_columns',None):
    display(HTML(df_ph_cat.to_html()))    
    
#Inferential Statistics
qqplot(wq_df['ph'],line='s')
plt.show()

#Shapiro
s, q = stats.shapiro(wq_df['ph']) 
if q < 0.05:
    print(f'p-value = {round(q,2)} < 0.05, so the pH values in the dataset are normally distributed.')
else:
    print(f'p-value = {round(q,2)} > 0.05, so the pH values in the dataset are not normally distributed.')      
    
#Ztest    
y_safe = wq_df[wq_df['potability']==1]
n_safe = wq_df[wq_df['potability']==0] 
z, v = ztest(y_safe['ph'], n_safe['ph'])
if v < 0.05:
    print(f'p-value = {round(v,2)} < 0.05, so the null hypothesis is rejected.There is significant difference in the mean of ph values between potable water and non-potable water.')
else:
    print(f'p-value = {round(v,2)} > 0.05, so the null hypothesis is not rejected. There is no significant difference in the mean of ph values between potable water and non-potable water.')

#Annova
a, b = stats.f_oneway(wq_df[wq_df['ph_category'] == 'acidic level']['hardness'],
                      wq_df[wq_df['ph_category'] == 'normal pH_surface water']['hardness'],
                      wq_df[wq_df['ph_category'] == 'alkaline level']['hardness'])

print('ANOVA test for mean hardness levels across water samples with different ph category')
print('F Statistic:', a, '\tp-value:', b )

if b < 0.05:
    print(f'p-value = {b} < 0.05; reject the null hypothesis in favor of the alternative. There is a statistically significant difference in hardness for at least two groups out of the three of pH categories.')
else:
    print(f'p-value = {b} > 0.05; do not reject the null hypothesis. Hardness across the three pH categories are not statistically significantly different.')

#comparison
fig, ax = plt.subplots(1,2,figsize=(16,6), facecolor='#F2F4F4')
h = sns.boxplot(x="ph_category", y="hardness", data=wq_df, ax=ax[0],palette='spring')
h.set_title('Water pH Categories vs Hardness',fontsize=14)
median_hardness = round(wq_df.groupby(['ph_category'])['hardness'].median(),2)
vertical_offset = wq_df['hardness'].median() * 0.02 # offset from median for display
for xtick in h.get_xticks():
    h.text(xtick,
           median_hardness[xtick] + vertical_offset,
           median_hardness[xtick], 
           horizontalalignment='center',
           size='large',
           color='w',
           weight='semibold')

t = sns.boxplot(x='ph_category', y='trihalomethanes', data=wq_df, ax=ax[1],palette='summer')
t.set_title('Water pH Categories vs Trihalomethanes',fontsize=14)
median_th = round(wq_df.groupby(['ph_category'])['trihalomethanes'].median(),2)
vertical_offset = wq_df['hardness'].median() * 0.02 # offset from median for display
for xtick in t.get_xticks():
    t.text(xtick,
           median_th[xtick] + vertical_offset,
           median_th[xtick], 
           horizontalalignment='center',
           size='large',
           color='w',
           weight='semibold')

plt.suptitle('Water pH Categories - Hardness/ Trihalomethanes',fontsize=18);


#EDA
fig, ax = plt.subplots(figsize=(16,12), facecolor='#F2F4F4')
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.5)
count=1
for i in wq_df.columns[1:-2]:
    plt.subplot(4,2,count)
    h = sns.histplot(x=i, kde=True,data=wq_df,hue='potability',palette='coolwarm')
    h.set_title(('water quality by ' + i).title(), fontsize=13)
    count+=1
fig.suptitle(('Univariate plots depicting feature distributions for the water potability').title(),  x=0.5, y=0.92, fontsize=16);


fig, ax = plt.subplots(figsize=(16,12), facecolor='#F2F4F4')
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.5)
var = wq_df.loc[:,(wq_df.columns!='potability')&(wq_df.columns!='ph')].columns.to_list()
count=1
for i in var[:-1]:
    plt.subplot(4,2,count)
    h = sns.histplot(x=i, kde=True,data=wq_df,hue='ph_category',palette='Accent_r')
    h.set_title(('water ph categories by ' + i).title(), fontsize=13)
    count+=1
fig.suptitle(('Univariate plots depicting feature distributions for the water pH categories').title(), 
             x=0.5, y=0.92, fontsize=16);
            
plt.figure(figsize=(12,6))
sns.heatmap(wq_df.corr(),fmt='.2g',annot=True, cmap="PuOr")
plt.title('Correlation Between Variables', fontsize=18);

plt.figure(figsize=(12,6))
pairplot1 = sns.pairplot(wq_df.iloc[:,1:-1], hue='potability')
pairplot1.fig.suptitle("Water Potability Pairwise Plots",fontsize=26, y=1.01); 

pairplot2 = sns.pairplot(wq_df[var], hue='ph_category')
pairplot2.fig.suptitle('Water pH Categories Pairwise Plots',fontsize=26, y=1.02); 

#Model building

X = wq_df[wq_df.columns[:-2]]
Y = wq_df['potability']
# Create training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
print('Target_train: ', Counter(Y_train))
print('Target_test:',Counter(Y_test))
print('Features:', list(wq_df.columns[0:-2]))

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#feature selection using Boruta
# define baseline model
rf_clf = RF(random_state=42, class_weight='balanced',n_jobs=-1)
rf_clf.fit(X_train,Y_train)

# Apply The Full Featured Classifier To The Test Data
y_pred = rf_clf.predict(X_test)
ac = round(accuracy_score(Y_test, y_pred)*100,2)

# View The Accuracy Of Our Full Feature (8 Features) Model
print("Baseline Model Accuracy:", ac,"%")

# define Boruta feature selection method
boruta_selector = BorutaPy(rf_clf, n_estimators='auto', verbose=2, random_state=42)

# find all relevant features
boruta_selector.fit(np.array(X_train), np.array(Y_train))

# check selected features
print("Selected Features: ", boruta_selector.support_)
 
# check ranking of features
print("Ranking: ",boruta_selector.ranking_)

print("No. of significant features: ", boruta_selector.n_features_)

# Let's visualise it better in the form of a table
selected_rfe_features = pd.DataFrame({'Feature':list(X.columns),
                                      'Ranking':boruta_selector.ranking_})
selected_rfe_features.sort_values(by='Ranking')


X_important_train = boruta_selector.transform(np.array(X_train))
X_important_test = boruta_selector.transform(np.array(X_test))

classifiers = []
log_reg = LogisticRegression(max_iter=10000, class_weight='balanced')
classifiers.append(log_reg)
ridge_clf =  RidgeClassifier(random_state=42, class_weight='balanced')
classifiers.append(ridge_clf)
knn_neg = KNeighborsClassifier(metric='euclidean') 
classifiers.append(knn_neg)
dt_clf = DT(random_state=42, class_weight='balanced')
classifiers.append(dt_clf)
rf_clf = RF(random_state=42, class_weight='balanced',n_estimators=500)
classifiers.append(rf_clf)
etree_clf = ET(random_state=42, class_weight='balanced')
classifiers.append(etree_clf)
ab_clf = AB(random_state=42)
classifiers.append(ab_clf)
xgb_clf = XGBClassifier(objective="binary:logistic", random_state=42)
classifiers.append(xgb_clf)
GNB_model = GNB() 
classifiers.append(GNB_model)
SVM_clf = SVC(random_state=42, probability=True, class_weight='balanced')
classifiers.append(SVM_clf)
classifiers

accuracy_train_pot = []
accuracy_test_pot = []
for clf in classifiers:
    clf.fit(X_important_train, Y_train)
    pred_train_pot = clf.predict(X_important_train)
    pred_test_pot = clf.predict(X_important_test)
    an = round(accuracy_score(Y_train, pred_train_pot)*100,2)
    at = round(accuracy_score(Y_test, pred_test_pot)*100,2)
    accuracy_train_pot.append(an)
    accuracy_test_pot.append(at)
original_result = pd.DataFrame(data={'Model':['LR','KNN','Ridge','DT','RF','ET','AdaBoost','XGB','Gaussian Bayes','SVC'],
                                     'Accuracy_Training (%)':accuracy_train_pot,
                                     'Accuracy_Test (%)': accuracy_test_pot})
original_result.sort_values('Accuracy_Test (%)',ascending=False)

mean_cv = []
std_cv = []
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for i in classifiers:
    score = cross_val_score(i,X_important_train,Y_train,scoring='accuracy', cv=kf)
    mean_cv.append(round(score.mean()*100,2))
    std_cv.append(round(score.std()*100,2))
cv_result = pd.DataFrame(data={'Model':['LR','KNN','Ridge','DT','RF','ET','AdaBoost','XGB','Gaussian Bayes','SVC'],    
                               'Accuracy_Mean(%)': mean_cv,
                               'Accuracy_SDev(%)': std_cv})
cv_result.sort_values(['Accuracy_Mean(%)'],ascending=False)    

rf_clf.get_params().keys()


%%time
tuned_classifiers = []
params = []
score = []
param_rf = {'max_depth': [i for i in range(20, 35, 5)], 
            'min_samples_leaf': [3,7,11], 
            'max_features': [1,3,6]}
RF(random_state=42, class_weight='balanced',n_estimators=500)
grid_rf = GridSearchCV(rf_clf, param_rf, cv=5, scoring='accuracy', return_train_score=True)
grid_rf.fit(X_important_train, Y_train)
final_rf = grid_rf.best_estimator_
tuned_classifiers.append(final_rf)
params.append(grid_rf.best_params_)
best_score_rf = round(grid_rf.best_score_*100,2)
score.append(best_score_rf)

etree_clf.get_params().keys()

%%time
param_etree = {'max_depth': [i for i in range(20, 35, 5)], 
               'min_samples_leaf': [3,7,11], 
               'max_features': [1,3,6],
               'n_estimators': [50,100,150]}
etree_clf = ET(random_state=42, class_weight='balanced')
grid_etree = GridSearchCV(etree_clf, param_etree, cv=5, scoring='accuracy', return_train_score=True)
grid_etree.fit(X_important_train, Y_train)
final_etree = grid_etree.best_estimator_
tuned_classifiers.append(final_etree)
params.append(grid_etree.best_params_)
best_score_etree = round(grid_etree.best_score_*100,2)
score.append(best_score_etree)

ridge_clf.get_params().keys()

%%time
param_ridge = {'alpha' : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 
               'solver' : ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs']}

ridge_clf =  RidgeClassifier(random_state=42, class_weight='balanced')
grid_ridge = GridSearchCV(ridge_clf, param_ridge, cv=5, scoring='accuracy', return_train_score=True)
grid_ridge.fit(X_important_train, Y_train)
final_ridge = grid_ridge.best_estimator_
tuned_classifiers.append(final_ridge)
params.append(grid_ridge.best_params_)
best_score_ridge = round(grid_ridge.best_score_*100,2)
score.append(best_score_ridge)

SVM_clf.get_params().keys()

%%time
param_svc = {'C': [0.1, 1, 10, 100, 1000],
             'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
             'kernel': ['rbf']}
SVM_clf = SVC(random_state=42, probability=True, class_weight='balanced')
grid_svc = GridSearchCV(SVM_clf, param_svc, cv=5, scoring='accuracy', return_train_score=True)
grid_svc.fit(X_important_train, Y_train)
final_svc = grid_svc.best_estimator_
tuned_classifiers.append(final_svc)
params.append(grid_svc.best_params_)
best_score_svc = round(grid_svc.best_score_*100,2)
score.append(best_score_svc)

xgb_clf.get_params().keys()

%%time
param_xgb = {'max_depth': [3,6,18],
             'gamma': [3,5,7,9],
             'min_child_weight' : [3,7,10],
             'n_estimators': [100,130,180]}
xgb_clf = XGBClassifier(objective="binary:logistic", random_state=42)
grid_xgb = GridSearchCV(xgb_clf, param_xgb, cv=5, scoring='accuracy', return_train_score=True)
grid_xgb.fit(X_important_train, Y_train)
final_xgb = grid_xgb.best_estimator_
tuned_classifiers.append(final_xgb)
params.append(grid_xgb.best_params_)
best_score_xgb = round(grid_xgb.best_score_*100,2)
score.append(best_score_xgb)

tuned_result = pd.DataFrame(data={'Model':['RF','E.TREE','Ridge','SVC','XGB'], 
                                  'Best Parameters':params, 'Score (%)': score})
tuned_result.sort_values('Score (%)', ascending =False)

tuned_accuracy_train = []
tuned_accuracy_test = []

for clf in tuned_classifiers:
    clf.fit(X_important_train, Y_train)
    tuned_pred_train = clf.predict(X_important_train)
    tuned_pred_test = clf.predict(X_important_test)
    accuracy_train = round(accuracy_score(Y_train, tuned_pred_train)*100,2)
    accuracy_test = round(accuracy_score(Y_test, tuned_pred_test)*100,2)
    tuned_accuracy_train.append(accuracy_train)
    tuned_accuracy_test.append(accuracy_test)
    
tuned_result_pot = pd.DataFrame(data={'Model':['RF','E.TREE','Ridge','SVC','XGB'], 
                                      'Training Score (%)': tuned_accuracy_train,
                                      'Testing Score (%)': tuned_accuracy_test})
tuned_result_pot.sort_values('Testing Score (%)', ascending = False)

acctrain = []
acctest = []
def print_result(model): 
    model.fit(X_important_train, Y_train)
    predicted_train = model.predict(X_important_train)
    predicted_test = model.predict(X_important_test)
    accuracy_model_train = round(accuracy_score(Y_train, predicted_train)*100,2)
    accuracy_model_test = round(accuracy_score(Y_test, predicted_test)*100,2)
    acctrain.append(accuracy_model_train)
    acctest.append(accuracy_model_test)
    print(f'Accuracy on training data: {accuracy_model_train}%')
    print(f'Accuracy on testing data: {accuracy_model_test}%')
    
# form a voting classifier using the models that returned accuracy of more than 60% on the testing data
voting_clf = VotingClassifier(estimators = [('RF',final_rf), ('ET',final_etree), ('SVC',final_svc)],
                              voting = 'soft')
print('Result of Voting Classifier')
print_result(voting_clf)


print(voting_clf)



# Use tuned SVC, extra-trees & random forest models as base estimators and tuned XGB model as the final estimator
# Create Base Learners
base_learners = [('svc',final_svc), 
                 ('etree', ET(class_weight='balanced', max_depth=30, max_features=3,
                              min_samples_leaf=3, random_state=42)), 
                 ('rf', RF(class_weight='balanced', max_depth=25, max_features=1,
                           min_samples_leaf=3, n_estimators=500, random_state=42)),
                 ('ridge', final_ridge)]
# Initialize Stacking Classifier 
stacking_clf = StackingClassifier(estimators=base_learners, final_estimator=final_xgb)
print('Result of Stacking Classifier')
print_result(stacking_clf)

# Multi-layer Stacking
# Create learners per layer
layer_one = [('ridge_2', final_ridge), 
             ('etree_2', ET(class_weight='balanced', max_depth=30, max_features=3,
                            min_samples_leaf=3, random_state=42))]
layer_two_est = [('svc_2', final_svc),
                 ('rf_2', RF(class_weight='balanced', max_depth=25, max_features=1,
                             min_samples_leaf=3, n_estimators=500, random_state=42))]
layer_two = StackingClassifier(estimators=layer_two_est, final_estimator=final_xgb)
# Create multi-layer stacking model 
stack_clf = StackingClassifier(estimators=layer_one, final_estimator=layer_two)
print('Result of Multi-Layer Stacking Classifier')
print_result(stack_clf)

ensemble_learners = ['Voting Classifier','Stacking Classifer','Multi-Layer Stacking Classifer']
ensemble_result = pd.DataFrame(data={'Model':ensemble_learners, 
                                     'Training Score (%)': acctrain,
                                     'Testing Score (%)': acctest})
ensemble_result.sort_values('Testing Score (%)', ascending = False)

voting_clf.fit(X_important_train, Y_train)
voting_clf_train = voting_clf.predict(X_important_train)
voting_clf_test = voting_clf.predict(X_important_test)
accuracy_final_voting_train = round(accuracy_score(Y_train, voting_clf_train)*100,2)
accuracy_final_voting_test = round(accuracy_score(Y_test, voting_clf_test)*100,2)
print('Accuracy for final voting classifier on training data', accuracy_final_voting_train,'%')
print('Accuracy for final voting classifier on testing data', accuracy_final_voting_test,'%')

group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
train_cnf_matrix = confusion_matrix(Y_train,voting_clf_train)
train_counts = ["{0:0.0f}".format(value) for value in train_cnf_matrix.flatten()]
train_percentage = ["{0:.2%}".format(value) for value in train_cnf_matrix .flatten()/np.sum(train_cnf_matrix)]
train_labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,train_counts,train_percentage)]
train_labels = np.asarray(train_labels).reshape(2,2)
plt.figure(figsize = (16,5))
sns.heatmap(train_cnf_matrix, annot=train_labels, fmt='', cmap='coolwarm')
plt.title('Confusion Matrix for Training Data (Voting Classifier)',fontsize=16);

print("Classification report (Training): \n")
print(f"{classification_report(Y_train,voting_clf_train)}")

group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
test_cnf_matrix = confusion_matrix(Y_test,voting_clf_test)
test_counts = ["{0:0.0f}".format(value) for value in test_cnf_matrix.flatten()]
test_percentage = ["{0:.2%}".format(value) for value in test_cnf_matrix .flatten()/np.sum(test_cnf_matrix)]
test_labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,test_counts,test_percentage)]
test_labels = np.asarray(test_labels).reshape(2,2)
plt.figure(figsize = (16,5))
sns.heatmap(test_cnf_matrix, annot=test_labels, fmt='', cmap='RdBu')
plt.title('Confusion Matrix for Testing Data (Voting Classifier)',fontsize=16);

print("Classification report (Testing): \n")
print(f"{classification_report(Y_test,voting_clf_test)}")


#making new preddiction

def predict(model, inputs):
    input_df = pd.DataFrame([inputs])
    pred = model.predict(input_df)[0]
    return pred
first_input = {'ph':6.8,
               'hardness': 129.932,
               'solids': 19440.861,
               'chloramines': 9.143,
               'sulfate': 295.514}
predict(voting_clf, first_input)


