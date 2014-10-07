import extractor as ex
from sklearn.svm import LinearSVC
from sklearn import linear_model
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.qda import QDA
from sklearn.lda import LDA
from sklearn.svm import SVC

"""
L1- Based Feature Selection
"""
def extract_linear_features_indexes(features, labels):
    """
    Perform Linear festure selection.
    """

    clf = LinearSVC(C=0.01, penalty="l1", dual=False)
    clf.fit(features, labels)

    return [i for i, e in enumerate(clf.coef_[0]) if e != 0 and abs(e) > 1e-6]
    
def extract_lasso_features_indexes(features, labels):
    """
    Perform Lasso feature selection.
    """

    clf = linear_model.Lasso(alpha=0.022, fit_intercept=False,
                             max_iter=2000,normalize=False, positive=False,
                             tol=0.001, warm_start=True)
    clf.fit(features, labels)

    return [i for i, e in enumerate(clf.coef_) if e != 0 and abs(e) > 1e-6]

def extract_features(included_index ,features, labels):
    """
    Return the only features that must be included in the classification
    process.
    """
    return features[:, included_index], labels    

def scaled_features(features,labels):
    max_features = features.max(axis = 0)
    max_features = (max_features + (max_features == 0))
    scaled_features = features / max_features
    return scaled_features, labels
    
def main():
    input_filename = 'data/input00.txt'
    output_filename = 'data/output00.txt'
    (train_features, train_labels,test_features, test_labels) = ex.extract(input_filename, output_filename)
    classifiers = {
        "NB Multinomial" : MultinomialNB(),
        "NB Gaussian": GaussianNB(),
        "Logistic Regression" : LogisticRegression(C=1e5, tol=0.001, fit_intercept=True),
        "Decision Tree" : DecisionTreeClassifier(min_samples_split=1, random_state=0),
        "KNN" : KNeighborsClassifier(n_neighbors=3),
        "SVM" : SVC(gamma=2, C=1),
        "LDA" : LDA(),
        "QDA" : QDA(reg_param=0.5),
        "Random Forest" : RandomForestClassifier(n_estimators=200),
        "AdaBoost" : AdaBoostClassifier(n_estimators=200),
    }
    
    print "-"*80, "\n", "Raw Dataset", "\n", "-"*80
    for name, classifier in classifiers.iteritems():
        clf = classifier.fit(train_features,train_labels)
        print name, clf.score(test_features,test_labels)
    
    print "-"*80, "\n", "Scaled Feature Dataset", "\n", "-"*80
    for name, classifier in classifiers.iteritems():
        (new_features,new_lables) = scaled_features(train_features, train_labels)
        clf = classifier.fit(new_features,new_lables)
        (new_test_features,new_test_lables) = scaled_features(train_features, train_labels)
        print name, clf.score(new_test_features,new_test_lables)
    
    print "-"*80, "\n", "Lasso Feature Selection", "\n", "-"*80
    for name, classifier in classifiers.iteritems():
        (new_features,new_lables) = extract_features(extract_lasso_features_indexes(train_features, train_labels),train_features, train_labels)
        clf = classifier.fit(new_features,new_lables)
        (new_test_features,new_test_lables) = extract_features(extract_lasso_features_indexes(train_features, train_labels),test_features,test_labels)
        print name, clf.score(new_test_features,new_test_lables)
    
    print "-"*80, "\n", "Linear Feature Selection", "\n", "-"*80
    for name, classifier in classifiers.iteritems():
        (new_features,new_lables) = extract_features(extract_linear_features_indexes(train_features, train_labels),train_features, train_labels)
        clf = classifier.fit(new_features,new_lables)
        (new_test_features,new_test_lables) = extract_features(extract_linear_features_indexes(train_features, train_labels),test_features,test_labels)
        print name, clf.score(new_test_features,new_test_lables)
    
if __name__ == '__main__':
    main()      
    
