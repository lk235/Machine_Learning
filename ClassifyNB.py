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

def classify_tree_min_samples_split_2(features_train, labels_train):
    from sklearn import tree
    clf = tree.DecisionTreeClassifier(min_samples_split=2)
    clf.fit(features_train, labels_train)
    return clf

def classify_tree_min_samples_split_50(features_train, labels_train):
    from sklearn import tree
    clf = tree.DecisionTreeClassifier(min_samples_split=50)
    clf.fit(features_train, labels_train)
    return clf

def classify_knn(features_train, labels_train):
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(features_train,labels_train)
    return clf

def classify_RF(features_train, labels_train):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=10)
    clf.fit(features_train,labels_train)
    return clf

def classify_ADA(features_train, labels_train):
    from sklearn.ensemble import AdaBoostClassifier
    clf = AdaBoostClassifier(n_estimators=100)
    clf.fit(features_train, labels_train)
    return clf


### your code goes here!