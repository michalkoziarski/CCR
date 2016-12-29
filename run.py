from utils import evaluate, compare
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from algorithm import CCR


if __name__ == '__main__':
    evaluate(None, DecisionTreeClassifier(), 'base.csv')
    evaluate(SMOTE(), DecisionTreeClassifier(), 'smote.csv')
    evaluate(ADASYN(), DecisionTreeClassifier(), 'adasyn.csv')
    evaluate(CCR(), DecisionTreeClassifier(), 'ccr.csv')

    summary, tables = compare(['base.csv', 'adasyn.csv', 'smote.csv', 'ccr.csv'])

    print(summary)
    print(tables)
