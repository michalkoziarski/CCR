import os
import datasets
import numpy as np
import pandas as pd

from sklearn import metrics


def evaluate(method, classifier, output_file):
    names = []
    partitions = []
    accuracies = []
    precisions = []
    recalls = []
    f_measures = []
    aucs = []
    g_means = []

    for name, folds in datasets.load_all().items():
        for i in range(len(folds)):
            (X_train, y_train), (X_test, y_test) = folds[i]
            labels = np.unique(y_test)
            counts = [len(y_test[y_test == label]) for label in labels]
            minority_class = labels[np.argmin(counts)]

            assert len(np.unique(y_train)) == len(np.unique(y_test)) == 2

            if method is not None:
                X_train, y_train = method.fit_sample(X_train, y_train)

            clf = classifier.fit(X_train, y_train)
            predictions = clf.predict(X_test)

            names.append(name)
            partitions.append(i)
            accuracies.append(metrics.accuracy_score(y_test, predictions))
            precisions.append(metrics.precision_score(y_test, predictions, pos_label=minority_class))
            recalls.append(metrics.recall_score(y_test, predictions, pos_label=minority_class))
            f_measures.append(metrics.f1_score(y_test, predictions))
            aucs.append(metrics.roc_auc_score(y_test, predictions))

            g_mean = 1.0

            for label in np.unique(y_test):
                idx = (y_test == label)
                g_mean *= metrics.accuracy_score(y_test[idx], predictions[idx])

            g_mean = np.sqrt(g_mean)
            g_means.append(g_mean)

    results_path = os.path.join(os.path.dirname(__file__), 'results')

    if not os.path.exists(results_path):
        os.mkdir(results_path)

    output_path = os.path.join(os.path.dirname(__file__), 'results', output_file)
    df = pd.DataFrame({'dataset': names, 'partition': partitions, 'accuracy': accuracies, 'precision': precisions,
                       'recall': recalls, 'f-measure': f_measures, 'auc': aucs, 'g-mean': g_means})
    df = df[['dataset', 'partition', 'accuracy', 'precision', 'recall', 'f-measure', 'auc', 'g-mean']]
    df.to_csv(output_path, index=False)


def compare(output_files):
    dfs = {}
    results = {}
    summary = {}
    tables = {}

    for f in output_files:
        path = os.path.join(os.path.dirname(__file__), 'results', f)
        dfs[f] = pd.read_csv(path)

    datasets = list(dfs.values())[0]['dataset'].unique()
    measures = ['accuracy', 'precision', 'recall', 'f-measure', 'auc', 'g-mean']

    for measure in measures:
        results[measure] = {}
        summary[measure] = {}
        tables[measure] = []

        for dataset in datasets:
            results[measure][dataset] = {}
            row = [dataset]

            for method in output_files:
                df = dfs[method]
                result = df[df['dataset'] == dataset][measure].mean()
                results[measure][dataset][method] = result
                row.append(result)

            tables[measure].append(row)

        for method in output_files:
            summary[measure][method] = 0

        tables[measure] = pd.DataFrame(tables[measure], columns=['Data sets'] + output_files)

    for measure in measures:
        for dataset in datasets:
            best_method = max(results[measure][dataset], key=results[measure][dataset].get)
            summary[measure][best_method] += 1

    return summary, tables
