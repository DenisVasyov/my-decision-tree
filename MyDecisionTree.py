import numpy as np
import pandas as pd
from collections import Counter
from tabulate import tabulate


class DecisionTree():

    def __init__(self,
                 max_depth,
                 min_samples_split,
                 criterion):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion

    global tree

    global criterion_task_corr
    criterion_task_corr = {'entropy': 'classification',
                           'gini': 'classification',
                           'variance': 'regression',
                           'median': 'regression'}

    global branch_idx_res_corr
    branch_idx_res_corr = {0: True,
                           1: False}

    global y_pred

    #   Метод для поиска значений для разбиения в случае задачи классификации
    def find_splitting_values_classification(self, data, feature):

        #         Берём средние между значениями, когда таргет в отсортированном массиве меняет значение
        data_temp = data[[feature, 'target']]
        feature = data_temp.drop(columns='target').columns.tolist()[0]
        data_temp = data_temp.sort_values(feature).reset_index(drop=True)

        #     Сохраняем все индексы, где текущее значение таргета отличается от предыдущего
        target_val_change_idxs = data_temp[
            (data_temp['target'].diff() != 0) & (data_temp['target'].diff().notna())].index.tolist()

        #   В список splitting_values будем сохранять значения для разбиения
        splitting_values = list()
        #     Проходимся по всем индексам
        for target_val_change_idx in target_val_change_idxs:
            splitting_value = np.mean(
                [data_temp.loc[target_val_change_idx, feature], data_temp.loc[target_val_change_idx - 1, feature]])
            splitting_values.append(splitting_value)

        return splitting_values

    #   Метод для поиска значений для разбиения в случае задачи регрессии
    def find_splitting_values_regression(self, data, feature):

        #         Берём все значения
        return data[feature].sort_values().values.tolist()

    #     Метод для вычисления вероятностей классов
    def calc_classes_probas(self, data):

        #     Сохраняем значения классов
        classes = data['target'].unique()
        #     Сохраняем кол-во объектов
        n_samples = data.shape[0]
        #     В словарь dct_classes_probas будут сохраняться вероятности каждого из классов (состояний)
        dct_classes_probas = dict(zip(classes, np.zeros(len(classes))))
        for class_ in classes:
            dct_classes_probas[class_] = data[data['target'] == class_].shape[0] / n_samples
        return dct_classes_probas

    def entropy(self, data):

        #     Вычисляем вероятности классов
        dct_classes_probas = self.calc_classes_probas(data)
        #     Рассчитываем значение энтропии
        entropy_val = - np.sum(
            [dct_classes_proba * np.log2(dct_classes_proba) for dct_classes_proba in dct_classes_probas.values()])
        return entropy_val

    def gini(self, data):

        #     Вычисляем вероятности классов
        dct_classes_probas = self.calc_classes_probas(data)
        #         Рассчитываем значение неопределённости Джини
        gini_val = 1 - np.sum([dct_classes_proba ** 2 for dct_classes_proba in dct_classes_probas.values()])
        return gini_val

    def variance(self, data):

        target_mean = data['target'].mean()
        return ((data['target'] - target_mean) ** 2).mean()

    def median(self, data):

        target_median = data['target'].median()
        return (data['target'] - target_median).abs().mean()

    #     Метод для расчёта прироста информации
    def information_gain(self, left_branch, right_branch):

        #         Функция критерия выбирается исходя из значения атрибута criterion
        calc_criterion = getattr(self, self.criterion)

        #     Значение критерия до разбиения
        s_splitting_node = calc_criterion(pd.concat([left_branch, right_branch]))
        #     Значение критерия в левой ветке
        s_left_branch = calc_criterion(left_branch)
        #     Значение критерия в правой ветке
        s_right_branch = calc_criterion(right_branch)

        #     Количество объектов разделяемой совокупности
        n_samples = pd.concat([left_branch, right_branch]).shape[0]
        #     Количество объектов левой ветки
        left_branch_n_samples = left_branch.shape[0]
        #     Количество объектов правой ветки
        right_branch_n_samples = right_branch.shape[0]

        information_gain = s_splitting_node - \
                           np.sum([(left_branch_n_samples / n_samples) * s_left_branch, \
                                   (right_branch_n_samples / n_samples) * s_right_branch])

        return information_gain

    #     Метод для разбиения совокупности по значению split_val признака feature
    def split(self, data, feature, split_val):

        left_branch = data[data[feature] < split_val]
        right_branch = data[data[feature] >= split_val]

        return left_branch, right_branch

    #     Метод для формирования листа в случае задачи классификации
    def to_terminal_classification(self, data):

        #         Берём самое часто встречающееся значение таргета
        return Counter(data['target']).most_common(n=1)[0][0]

    #     Метод для формирования листа в случае задачи регрессии
    def to_terminal_regression(self, data):

        #         Берём среднее арифметическое значение таргета
        return data['target'].mean()

    #     Метод для нахождения наилучшего разбиения набора данных
    def get_best_split(self, dataset):

        global criterion_task_corr

        data = dataset.copy()
        #     Сохраняем признаки
        features = data.drop(columns='target').columns.tolist()
        dct_features_split_vals = dict(zip(features, np.zeros(len(features))))
        #         Метод для поиска лучшего разбиения выбирается исходя из задачи
        task = criterion_task_corr[self.criterion]
        find_splitting_values_ = getattr(self, f'find_splitting_values_{task}')
        for feature in features:
            dct_features_split_vals[feature] = find_splitting_values_(data, feature)
        max_information_gain = 0
        for feature in features:
            for split_val in dct_features_split_vals[feature]:
                left_branch, right_branch = self.split(data, feature, split_val)
                information_gain_val = self.information_gain(left_branch, right_branch)
                if information_gain_val > max_information_gain:
                    max_information_gain = information_gain_val
                    best_feature = feature
                    best_split_val = split_val
        left_branch, right_branch = self.split(data, best_feature, best_split_val)

        return {'feature': best_feature, 'split_val': best_split_val, 'branches': [left_branch, right_branch]}

    def build(self, branch, current_depth=0, compare_res=None):

        global tree_
        global criterion_task_corr
        global branch_idx_res_corr

        node = self.get_best_split(branch)
        feature, split_val = node['feature'], node['split_val']
        left_branch, right_branch = node['branches'][0], node['branches'][1]

        comes_from = compare_res

        for idx, branch in enumerate(node['branches']):

            compare_res = branch_idx_res_corr[idx]

            #             Сохраняем данные нового узла
            new_node = pd.DataFrame({'feature': [feature],
                                     'split_val': [split_val],
                                     'compare_res': [compare_res],
                                     'comes_from': [comes_from],
                                     'current_depth': [current_depth],
                                     'left_branch': [left_branch],
                                     'right_branch': [right_branch]})

            #             Печатаем процесс построения дерева
            print(f'Текущая глубина: {current_depth}')
            print(f'Признак: {feature}')
            print(f'Значение: {split_val}')
            print(f'Результат сравнения: {compare_res}')
            print(f'Пришли из ветки: {comes_from}')
            print('Данные:')
            print((tabulate(branch, headers='keys', tablefmt='psql')))

            #             Добавляем новый узел к дереву
            tree_ = tree_.append(new_node)

            #             Если выполняется один из критериев останова, то из данных ветки формируем лист
            if (branch['target'].nunique() == 1) or \
                    (current_depth >= self.max_depth) or \
                    (branch.shape[0] <= self.min_samples_split):

                #                 Метод для задания листа определяется исходя из задачи
                task = criterion_task_corr[self.criterion]
                to_terminal_ = getattr(self, f'to_terminal_{task}')

                if compare_res:
                    #                     Если результат сравнения в узле True, т.е. мы в левой ветке =>
                    #                     => изменяем в последней строке (последний добавленный узел) значение left_branch
                    tree_.iloc[-1, tree_.columns.get_loc('left_branch')] = to_terminal_(branch)
                else:
                    #                     Если результат сравнения в узле False, т.е. мы в правой ветке =>
                    #                     => изменяем в последней строке (последний добавленный узел) значение right_branch
                    tree_.iloc[-1, tree_.columns.get_loc('right_branch')] = to_terminal_(branch)

            else:
                #             Если ни один из критериев останова не выполняется, то для данной ветки рекурсивно вызываем метод build
                self.build(branch, current_depth=current_depth + 1, compare_res=compare_res)

    def fit(self, X, y):

        data = pd.concat([pd.DataFrame(X), pd.Series(y, name='target')], axis=1)

        global tree_

        tree_ = pd.DataFrame(columns=['feature',
                                      'split_val',
                                      'compare_res',
                                      'comes_from',
                                      'current_depth',
                                      'left_branch',
                                      'right_branch'])
        self.build(data)
        tree_.index = range(1, tree_.shape[0] + 1)
        tree_.sort_values('current_depth', inplace=True)
        return tree_

    #     Метод, возвращающий решение, принимаемое в узле дерева
    def get_node_decision(self, sample, node):

        feature = node['feature'].unique()[0]
        split_val = node['split_val'].unique()[0]
        level = node['current_depth'].unique()[0]

        compare_res = sample[feature] < split_val
        if compare_res:
            decision = node.loc[node['compare_res'] == compare_res, ['left_branch']].values[0][0]
        else:
            decision = node.loc[node['compare_res'] == compare_res, ['right_branch']].values[0][0]

        #         Возвращаем решение, результат сравнения в узле и уровень узла в дереве
        return decision, compare_res, level

    #     Метод для получения предсказаний по одному объекту
    def get_prediction(self, sample, node, tree_):

        global y_pred

        prediction, compare_res, level = self.get_node_decision(sample, node)

        if type(prediction) in [int, float]:
            #             Если предсказание – число, значит, дошли до листа
            y_pred = np.append(y_pred, prediction)
        else:
            #             Если не число, то рекурсивно вызываем процедуру get_prediction для следующего узла
            next_node = tree_[(tree_['current_depth'] == level + 1) & (tree_['comes_from'] == compare_res)]
            self.get_prediction(sample, next_node, tree_)

    #   Метод для получения предсказаний по массиву
    def predict(self, X, tree_):

        global y_pred
        y_pred = np.array([])

        data = pd.DataFrame(X)

        # Начинаем построение дерева с корневого узла
        root = tree_[tree_['current_depth'] == 0]

        data.apply(self.get_prediction, axis=1, args=(root, tree_))

        return y_pred