import os

from utils import evaluate, compare
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.combine import SMOTETomek, SMOTEENN
from algorithm import CCRSelection


if __name__ == '__main__':
    classifiers = {
        'cart': DecisionTreeClassifier(),
        'cart-bag': BaggingClassifier(DecisionTreeClassifier()),
        'knn': KNeighborsClassifier(),
        'knn-bag': BaggingClassifier(KNeighborsClassifier()),
        'svm': LinearSVC(),
        'nb': GaussianNB()
    }

    energies = [0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0]

    results_path = os.path.join(os.path.dirname(__file__), 'results')

    if not os.path.exists(results_path):
        os.mkdir(results_path)

    for name, classifier in classifiers.iteritems():
        evaluate(None, classifier, '%s_base.csv' % name, type='final')
        evaluate(ADASYN(), classifier, '%s_adasyn.csv' % name, type='final')
        evaluate(SMOTE(), classifier, '%s_smote.csv' % name, type='final')
        evaluate(SMOTE(kind='borderline1'), classifier, '%s_borderline.csv' % name, type='final')
        evaluate(NeighbourhoodCleaningRule(), classifier, '%s_ncr.csv' % name, type='final')
        evaluate(SMOTETomek(), classifier, '%s_t-link.csv' % name, type='final')
        evaluate(SMOTEENN(), classifier, '%s_enn.csv' % name, type='final')
        evaluate(CCRSelection(energies=energies, classifier=classifier, measure=roc_auc_score),
                 classifier, '%s_ccr.csv' % name, type='final')

        summary, tables = compare(['%s_base.csv' % name, '%s_adasyn.csv' % name, '%s_smote.csv' % name,
                                   '%s_borderline.csv' % name, '%s_ncr.csv' % name, '%s_t-link.csv' % name,
                                   '%s_enn.csv' % name, '%s_ccr.csv' % name])

        print(summary)

        for measure, table in tables.iteritems():
            table.to_csv(os.path.join(results_path, 'table_%s_%s.csv' % (measure, name)), index=False)
