from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture, output_image
from ClassifyNB import classify
from ClassifyNB import classify_svm
from ClassifyNB import *
from sklearn.metrics import accuracy_score
import numpy as np
import pylab as pl


features_train, labels_train, features_test, labels_test = makeTerrainData()

### the training data (features_train, labels_train) have both "fast" and "slow" points mixed
### in together--separate them so we can give them different colors in the scatterplot,
### and visually identify them
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


# You will need to complete this function imported from the ClassifyNB script.
# Be sure to change to that code tab to complete this quiz.
clf = classify(features_train, labels_train)
clf_svm = classify_svm(features_train, labels_train)
clf_tree_min_samples_split_2 = classify_tree_min_samples_split_2(features_train, labels_train)
clf_tree_min_samples_split_50 = classify_tree_min_samples_split_50(features_train, labels_train)
print type(clf)



### draw the decision boundary with the text points overlaid
prettyPicture(clf_tree_min_samples_split_50, features_test, labels_test)
output_image("test.png", "png", open("test.png", "rb").read())


#prettyPicture(clf_svm, features_test, labels_test)
#output_image("test_svm.png", "png", open("test_svm.png", "rb").read())
accuracy1 = accuracy_score(labels_test,clf_tree_min_samples_split_2.predict(features_test))
accuracy2 = accuracy_score(labels_test,clf_tree_min_samples_split_50.predict(features_test))
print accuracy1
print accuracy2
#print clf_tree.score(features_test, labels_test)