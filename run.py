from utils import evaluate, compare
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.combine import SMOTETomek, SMOTEENN
from algorithm import CCR


if __name__ == '__main__':
    evaluate(None, DecisionTreeClassifier(), 'base.csv')
    evaluate(ADASYN(), DecisionTreeClassifier(), 'adasyn.csv')
    evaluate(SMOTE(), DecisionTreeClassifier(), 'smote.csv')
    evaluate(SMOTE(kind='borderline1'), DecisionTreeClassifier(), 'borderline.csv')
    evaluate(NeighbourhoodCleaningRule(), DecisionTreeClassifier(), 'ncr.csv')
    evaluate(SMOTETomek(), DecisionTreeClassifier(), 't-link.csv')
    evaluate(SMOTEENN(), DecisionTreeClassifier(), 'enn.csv')
    evaluate(CCR(), DecisionTreeClassifier(), 'ccr.csv')

    summary, tables = compare(['base.csv', 'adasyn.csv', 'smote.csv', 'borderline.csv', 'ncr.csv', 't-link.csv',
                               'enn.csv', 'ccr.csv'])

    print(summary)
    print(tables)
