# -*- coding: utf-8 -*-
"""

"""
import os
import sys
import pandas as pd

from sklearn import tree
from sklearn import linear_model

from sklearn.cross_validation import train_test_split
from sklearn import cross_validation

from sklearn import metrics
import matplotlib.pyplot as plt


folder = sys.argv[1]
print("Loading files...")
outcomes_df = pd.read_csv(os.path.join(folder, "outcomes.csv"))
topics_df = pd.read_csv(os.path.join(folder, "topics.csv"))
   
df = pd.merge(outcomes_df, topics_df, how = 'left', on = 'projectid')

df["y"] = df["is_exciting"].apply(lambda x: 1 if x == "t" else 0)
df = df.fillna(0)

target = df.y
cols = [ 'topic' + str(j) for j in range(100)]
predictors = df[cols]

x_train, x_test, y_train, y_test = train_test_split(
    predictors, target, test_size=0.33, random_state=52)

logr_model = linear_model.LogisticRegression()
logr_model.fit(x_train, y_train)
logr_predictions = logr_model.predict(x_test)
                                 
dt_model = tree.DecisionTreeClassifier('entropy')
dt_model.fit(x_train, y_train)
dt_predictions = dt_model.predict(x_test)

def cross_score(model, predictors, target):
    scores = cross_validation.cross_val_score(model, predictors, 
                                 target, cv=5, scoring='roc_auc')
    print("ROC AUC : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    


def print_metrics(y_test, predictions):
    print(metrics.accuracy_score(y_test, predictions))


def plot_roc(fpr, tpr, roc_auc, label):
    plt.title('Receiver Operating Characteristic ' + label)
    plt.plot(fpr, tpr, 'b',
    label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def run_model_and_plot_roc(model, x_train, y_train, x_test, label):
    model.fit(x_train, y_train)
    predict = model.predict_proba(x_test)
    
    fpr, tpr, thresholds = metrics.roc_curve(y_test, predict[:,1])
    roc_auc = metrics.auc(fpr, tpr)
        
    plot_roc(fpr, tpr, roc_auc, label)
    return predict

logr_model = linear_model.LogisticRegression()
logr_predict = run_model_and_plot_roc(logr_model, x_train, y_train, x_test, 'Linear Regression')

tree_model = tree.DecisionTreeClassifier('entropy')
tree_predict = run_model_and_plot_roc(tree_model, x_train, y_train, x_test, 'Decision Tree')

sdg_model = linear_model.SGDClassifier(penalty="l2",loss="log",fit_intercept=True,
                                       shuffle=True,n_iter=20, n_jobs=-1,alpha=0.000005)
sdg_predict = run_model_and_plot_roc(sdg_model, x_train, y_train, x_test, 'SGD')

cross_score(logr_model, predictors, target)
cross_score(dt_model, predictors, target)
cross_score(sdg_model, predictors, target)

#print_metrics(y_test, logr_predictions)
#print_metrics(y_test, dt_predictions)