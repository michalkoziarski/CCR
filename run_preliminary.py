import os
import pandas as pd
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns

from utils import evaluate, compare
from sklearn.tree import DecisionTreeClassifier
from algorithm import CCR


if __name__ == '__main__':
    results_path = os.path.join(os.path.dirname(__file__), 'results')

    if not os.path.exists(results_path):
        os.mkdir(results_path)

    comparable = []
    energies = [0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0]

    for energy in energies:
        file_name = 'preliminary_cart_energy_%s.csv' % energy
        evaluate(CCR(energy=energy), DecisionTreeClassifier(), file_name, type='preliminary')
        comparable.append(file_name)

    summary, tables = compare(comparable)

    for measure in ['auc', 'g-mean', 'f-measure']:
        table = tables[measure]
        data = []

        for dataset in table['dataset'].unique():
            for energy in energies:
                value = float(table[table['dataset'] == dataset]['preliminary_cart_energy_%s.csv' % energy])
                data.append([dataset.replace('-', '').replace('_', ''), energy, value])

        df = pd.DataFrame(data, columns=['dataset', 'energy', 'value'])

        grid = sns.FacetGrid(df, col='dataset', col_wrap=5)
        grid.set(ylim=(0.0, 1.0), xticks=range(len(energies)))
        grid.set_xticklabels(energies, rotation=90)
        grid.map(plt.plot, 'value')
        grid.savefig(os.path.join(results_path, 'preliminary_%s.pdf' % measure))
