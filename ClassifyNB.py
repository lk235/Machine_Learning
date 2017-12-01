def classify(features_train, labels_train):
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    clf.fit(features_train,labels_train)
    return clf

def classify_svm(features_train, labels_train):
    from sklearn.svm import SVC
    clf = SVC(kernel="rbf",C=1000)
    clf.fit(features_train, labels_train)
    return clf
### import the sklearn module for GaussianNB
### create classifier
### fit the classifier on the training features and labels
### return the fit classifier


### your code goes here!