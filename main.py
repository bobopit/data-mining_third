import sys
import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
import pandas as pd
import pydotplus
from class_vis import prettyPicture, output_image
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from IPython.display import Image
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import stochastic_gradient
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans,Birch
from sklearn import  metrics
from sklearn.metrics import homogeneity_completeness_v_measure

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot( thresholds, precisions[:-1], "b--", label="Precision" )
    plt.plot( thresholds, recalls[:-1], "g-", label="Recall" )
    plt.xlabel( "Threshold" )
    plt.legend(loc="upper left")
    plt.ylim( [0, 1] )
    plt.plot(recalls, precisions)
    plt.show()


def plot_roc_curve( fpr, tpr, label=None):
    plt.plot( fpr, tpr, linewidth=2, label=label )
    plt.plot( [0,1], [0,1], "k--" )
    plt.axis([0,1,0,1])
    plt.xlabel( "False Positive Rate" )
    plt.ylabel( "True Positive Rate" )
    plt.show()


titanic_trian=pd.read_csv("train.csv")
titanic_test= pd.read_csv("test.csv")
x_train = titanic_trian[['Pclass','Age','Sex','SibSp','Parch','Embarked','Fare']]
x_test = titanic_test[['Pclass','Age','Sex','SibSp','Parch','Embarked','Fare']]
y_lable = titanic_trian['Survived']
y_test = pd.read_csv("gender_submission.csv")['Survived']


x_train['Age'].fillna(x_train['Age'].mean(),inplace = True)
x_test['Age'].fillna(x_test['Age'].mean(),inplace = True)
x_test['Fare'].fillna(x_test['Fare'].mean(),inplace = True)

x_train=pd.get_dummies(x_train)
x_test=pd.get_dummies(x_test)
x_train.info()
x_test.info()
##############################################################################################
dtc=DecisionTreeClassifier()
dtc.fit(x_train,y_lable)
dtc_y_pred=dtc.predict(x_test)
print "**************************************************************"
print('The accuracy of decision tree is',dtc.score(x_test,y_test))
print(classification_report(dtc_y_pred,y_test))

dtc_accu = cross_val_score( dtc, x_train, y_lable, cv=3, scoring="accuracy" )
print('The cross_val_score of decision tree is',dtc_accu)
print "**************************************************************"


# y_dtc_scores = dtc.decision_function( x_train )
# y_some_digit_pred = (y_dtc_scores > 0)
# sgd_score = roc_auc_score(y_lable, y_dtc_scores)
# fpr, tpr, thresholds = roc_curve( y_lable, y_dtc_scores )
# plot_roc_curve( fpr, tpr, "DTC" )

# y_re_scores = cross_val_predict( dtc, x_train, y_lable, cv=3, method="decision_function" )
# precisions, recalls, thresholds = precision_recall_curve( y_lable, y_re_scores )
# plot_precision_recall_vs_threshold( precisions, recalls, thresholds )
############################################################################################

rfc=RandomForestClassifier()
rfc.fit(x_train,y_lable)
rfc_y_pred=rfc.predict(x_test)
print "**************************************************************"
print('The accuracy of random forest classifier is',rfc.score(x_test,y_test))
print(classification_report(rfc_y_pred,y_test))

rfc_accu = cross_val_score( dtc, x_train, y_lable, cv=3, scoring="accuracy" )
print('The cross_val_score of decision tree is',rfc_accu)
print "**************************************************************"


# y_rfc_Roc = rfc.decision_function( x_train )
# y_some_digit_pred = (y_rfc_Roc > 0)
# sgd_score = roc_auc_score(y_lable, y_rfc_Roc)
# fpr, tpr, thresholds = roc_curve( y_lable, y_rfc_Roc )
# plot_roc_curve( fpr, tpr, "RFC" )
#
# y_r_scores = cross_val_predict( dtc, x_train, y_lable, cv=3, method="decision_function" )
# precisions, recalls, thresholds = precision_recall_curve( y_lable, y_r_scores )
# plot_precision_recall_vs_threshold( precisions, recalls, thresholds )

################################################################################################

sgd_clf = stochastic_gradient.SGDClassifier()
sgd_clf.fit( x_train, y_lable )
sdg_y_pred = sgd_clf.predict(x_test)
print "**************************************************************"
print('The accuracy of Stochastic Gradient Descent classifier is',sgd_clf.score(x_test,y_test))
print(classification_report(sdg_y_pred,y_test))

sgd_accu = cross_val_score( dtc, x_train, y_lable, cv=3, scoring="accuracy" )
print('The cross_val_score of decision tree is',sgd_accu)
print "**************************************************************"


y_scores = sgd_clf.decision_function( x_train )
y_some_digit_pred = (y_scores > 0)
sgd_score = roc_auc_score(y_lable, y_scores)
fpr, tpr, thresholds = roc_curve( y_lable, y_scores )
# plot_roc_curve( fpr, tpr, "SGD" )

y_sgd_scores = cross_val_predict( sgd_clf, x_train, y_lable, cv=3, method="decision_function" )
precisions, recalls, thresholds = precision_recall_curve( y_lable, y_sgd_scores )
# plot_precision_recall_vs_threshold( precisions, recalls, thresholds )


##################################################################################################



clf_km = KMeans(n_clusters=2, max_iter=300, n_init=10)
clf_km.fit(x_train,y_lable)
ypred = clf_km.predict(x_test)
labels= clf_km.labels_
print metrics.calinski_harabaz_score(x_train,labels=labels)

print("kmeans:", homogeneity_completeness_v_measure(y_test, ypred))



clf_Birch = Birch(n_clusters=2)
clf_Birch.fit(x_train,y_lable)
ypred = clf_Birch.predict(x_test)
labels= clf_Birch.labels_
print metrics.calinski_harabaz_score(x_train,labels=labels)












