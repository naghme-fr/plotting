# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 13:36:48 2023

@author: Admin
"""
import numpy as np
from sklearn import metrics

#from numpy import genfromtxt
#my_data = genfromtxt('C:/Users/Admin/Desktop/my_data.csv', delimiter=',')
#my_data=my_data.astype(int)

#prediction_test = genfromtxt('C:/Users/Admin/Desktop/prediction_test.csv', delimiter=',')
#prediction_test=prediction_test.astype(int)

my_data = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
       0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0])

prediction_test=np.array([1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           0,           0,           0,
                 0,           0,           0,           0,           0,
                 0,           0,           0,           0,           0,
                 0,           0,           0,           0,           0,
                 0,           0,           0,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1])



fpr1, tpr1, threshold = metrics.roc_curve(my_data, prediction_test)
roc_auc1 = metrics.auc(fpr1, tpr1)


x=np.array([1,1,1,1,0])
y=np.array([1,1,1,1,1])

fpr2, tpr2, threshold = metrics.roc_curve(x, y)
roc_auc2 = metrics.auc(fpr2, tpr2)

# method I: plt two roc in one curve
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr1, tpr1, 'b', label = 'AUC = %0.2f' % roc_auc1)
plt.plot(fpr2, tpr2, 'r', label = 'AUC = %0.2f' % roc_auc2)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

from sklearn.metrics import classification_report, confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
print(confusion_matrix(my_data, prediction_test, labels=[1,0]))
# Compute confusion matrix
cnf_matrix = confusion_matrix(my_data, prediction_test, labels=[1,0])
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['EZ=1','EZ=0'],normalize= False,  title='Confusion matrix')