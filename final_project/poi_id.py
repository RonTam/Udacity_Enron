#!/usr/bin/python

# general
import sys
import pickle
import numpy as np
import pandas as pd
from pandas import DataFrame, Series

# sklearn - general
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectFromModel, SelectKBest

# sklearn - models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.svm import SVC
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list = ['poi',
#               'salary',
                'to_messages',
#                'deferral_payments',
#                'total_payments',
#                'exercised_stock_options',
#                'bonus',
#                'restricted_stock',
                'shared_receipt_with_poi',
#                'restricted_stock_deferred',
#                'total_stock_value',
#                'expenses',
#                'loan_advances',
                'from_messages',
#                'other',
                'from_this_person_to_poi',
#                'director_fees',
#                'deferred_income',
#		'deferred_ratio',
#                'long_term_incentive']
                'from_poi_to_this_person']
#                'message_ratio',
#                'poi_from_ratio',
#                'poi_to_ratio']
	
### Load the dictionary containing the dataset

with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers

bad_keys = ['TOTAL','LOCKHART EUGENE E','THE TRAVEL AGENCY IN THE PARK']
for i in bad_keys:
    if i in data_dict:
        del data_dict[i]

df = pd.DataFrame.from_dict(data_dict,orient='index')
df = df.replace('NaN',np.nan)
df = df.fillna(0)
df['message_ratio'] = df['to_messages']/df['from_messages']
df['poi_from_ratio'] = df['from_poi_to_this_person']/df['from_messages']
df['poi_to_ratio'] = df['from_this_person_to_poi']/df['to_messages']
df['deferred_ratio'] = df['deferral_payments']/df['total_payments']
df = df.replace('NaN',np.nan)
df = df.fillna(0)
df = df.replace('inf',0)

data_dict = df.T.to_dict()

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

my_dataset = data_dict

### Extract features and labels from dataset for local testing

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
	
clf = LogisticRegression(class_weight='balanced', n_jobs=-1, C=100000000000L, penalty ='l2', random_state=42)

####################################################################
####################### Logistic Regression ########################
####################################################################
##
## param_grid = {'C': [.1,1,100,10000,100000000000L],
##		'penalty': ['l1','l2']}
## clf = LogisticRegression(class_weight='balanced',n_jobs=-1)
## clf = GridSearchCV(clf, param_grid=param_grid)
##
##
####################################################################
##################          Random Forest       ####################
####################################################################
##
## param_grid = {'max_depth': [3, None],
##		'min_samples_split': [5, 10],
##		'min_samples_leaf': [5, 10],
##		'criterion' :['gini', 'entropy']}
##
## clf = RandomForestClassifier()
## clf = GridSearchCV(clf, param_grid=param_grid)
##
####################################################################
##########################    Naive Bayes    #######################
####################################################################
##
## clf = GaussianNB()
##
####################################################################
#########################      AdaBoost   ##########################
####################################################################
##
## clf = AdaBoostClassifier()
##
####################################################################
####################################################################

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

#best_parameters = clf.best_estimator_.get_params()
#for param_name in sorted(parameters.keys()):
#    print("\t%s: %r" % (param_name, best_parameters[param_name]))

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
