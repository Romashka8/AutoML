import warnings
warnings.filterwarnings('ignore')

import tqdm
import pickle
import os

import pandas as pd
import numpy as np

from catboost import CatBoostClassifier, Pool

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score

import hyperopt

import matplotlib.pyplot as plt
import seaborn as sns


class CatBoostParameters:
    def __init__(self, params_dict=None):
        
        if params_dict is None:
            self.params = {
                'boosting_type': hyperopt.hp.choice('boosting_type', ['Ordered', 'Plain']),
                'iterations': hyperopt.hp.uniformint('iterations', 100, 150),
                'learning_rate': hyperopt.hp.uniform('learning_rate', 0.01, 0.9),
                'depth': hyperopt.hp.quniform('depth', 1, 2, 1),
                'l2_leaf_reg': hyperopt.hp.uniform('l2_leaf_reg', 1e-8, 10),
                'min_data_in_leaf': hyperopt.hp.quniform('min_data_in_leaf', 30, 250, 10)
            }
        else:
            self.params = {
                'boosting_type': hyperopt.hp.choice('boosting_type', params_dict['boosting_type']),
                'iterations': hyperopt.hp.uniformint('iterations', *params_dict['iterations']),
                'learning_rate': hyperopt.hp.uniform('learning_rate', *params_dict['learning_rate']),
                'depth': hyperopt.hp.quniform('depth', *params_dict['depth']),
                'l2_leaf_reg': hyperopt.hp.uniform('l2_leaf_reg', *params_dict['l2_leaf_reg']),
                'min_data_in_leaf': hyperopt.hp.quniform('min_data_in_leaf', *params_dict['min_data_in_leaf'])
            }

    def param_transformer(self, params):
        
        transformed = {
                'boosting_type': params['boosting_type'],
                'iterations': int(params['iterations']),
                'learning_rate': round(params['learning_rate'], 3),
                'depth': int(params['depth']),
                'l2_leaf_reg': round(params['l2_leaf_reg'], 3),
                'min_data_in_leaf': int(params['min_data_in_leaf'])
            }
        
        return transformed


class TrainCatBoostModel:
    def __init__(self, x_train, y_train, x_test, y_test,
                 random_seed=42, early_stop=50, max_evals=10,
                 params_dict=None, cat_features=None):

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.parameters = CatBoostParameters(params_dict)
        self.early_stop = early_stop
        self.model = None
        self.scoring = None
        self.type_model = None
        self.max_evals = max_evals
        self.cat_features = cat_features
        self.random_seed = random_seed

    def __plot_feature_importance(self, model, model_name):
        
        if isinstance(self.x_train, pd.DataFrame):
            col_names = self.x_train.columns.values
        else:
            col_names = [f'col_{i}' for i in range(self.x_train.shape[1])]
            
            feature_imp = pd.DataFrame(
                sorted(zip(model.feature_importances_, col_names)),
                columns=['Value', 'Feature']
            )

            plt.figure(figsize=(10, 10))
            sns.barplot(x='Value', y='Feature',
                        data=feature_imp.sort_values(by='Value', ascending=False),
                        palette='magma'
            )
            plt.title(f'{model_name} importance of features')

    def __cat(self, params):
        
        params = self.parameters.param_transformer(params)
        model = CatBoostClassifier(
            loss_function='Logloss',
            eval_metric='AUC',
            random_seed=self.random_seed,
            max_ctr_complexity=1,
            cat_features=self.cat_features,
            **params
        )

        early_stopping = {
            'early_stopping_rounds': self.early_stop,
            'eval_set': (self.x_test, self.y_test),
            'verbose': False
        }

        score = cross_val_score(
            model,
            self.x_train,
            self.y_train,
            scoring=self.scoring,
            cv=StratifiedKFold(5),
            fit_params=early_stopping
        ).mean()

        return {'loss': -score, 'status': hyperopt.STATUS_OK}

    def __cat_oos(self, params):
        
        params = self.parameters.param_transformer(params)
        model = CatBoostClassifier(
            loss_function='Logloss',
            eval_metric='AUC',
            random_seed=self.random_seed,
            max_ctr_complexity=1,
            cat_features=self.cat_features,
            **params
        )

        early_stopping = {
            'early_stopping_rounds': self.early_stop,
            'eval_set': (self.x_test, self.y_test),
            'verbose': False
        }

        score = model.fit(
            self.x_train,
            self.y_train,
            early_stopping_rounds=self.early_stop,
            eval_set=(self.x_test, self.y_test),
            verbose=False
        ).best_score_['validation']['AUC']

        return {'loss': -score, 'status': hyperopt.STATUS_OK}

    def __train_model(self, f, params):

        trials = hyperopt.Trials()
        best_model = hyperopt.fmin(
            fn=f,
            space=params,
            algo=hyperopt.tpe.suggest,
            max_evals=self.max_evals,
            trials=trials,
            rstate=np.random.default_rng(self.random_seed)
        )
        best_score = -trials.best_trial['result']['loss']
        
        return best_model, best_score

    def train(self, scoring='roc_auc', eval_metric='AUC', oos=False):
        
        self.scoring = scoring.lower()

        if oos:
            best_params, cv_score = self.__train_model(
                self.__cat_oos,
                self.parameters.params
            )
        else:
            best_params, cv_score = self.__train_model(
                self.__cat,
                self.parameters.params
            )

        best_params = hyperopt.space_eval(
            self.parameters.params,
            best_params
        )

        best_params = self.parameters.param_transformer(best_params)

        best_model = CatBoostClassifier(
            eval_metric=eval_metric,
            cat_features=self.cat_features,
            random_seed=self.random_seed,
            max_ctr_complexity=1,
            **best_params
        ).fit(
            self.x_train,
            self.y_train,
            early_stopping_rounds=self.early_stop,
            eval_set=(self.x_test, self.y_test),
            verbose=False
        )
        
        self.__plot_feature_importance(best_model, 'CatBoost')

        return best_model, cv_score, best_model.best_score_['validation'][eval_metric]


class FeatureSelector:

    def __init__(self, x_train, y_train,
                 x_test, y_test, percent_drop=10,
                 random_state=42, cat_features=None):

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.percent_drop = percent_drop
        self.columns = x_train.columns.values
        self.step = max(2, len(self.columns) // self.percent_drop)
        self.model = None
        self.feature_imp = None
        self.removed_columns = None
        self.left_bound = -np.Inf
        self.right_bound = -np.Inf
        self.random_state = random_state
        self.cat_features = cat_features
        self.features = []

    def _shuffle_data(self):

        X = pd.concat([self.x_train, self.x_test])
        y = np.concatenate((self.y_train, self.y_test))

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=0.3,
            random_state=self.random_state
        )

        return self

    def get_feature_importance(self):

        self.model = CatBoostClassifier(
            loss_function='Logloss',
            eval_metric='AUC',
            iterations=250,
            depth=5,
            min_data_in_leaf=50,
            random_seed=self.random_state,
            max_ctr_complexity=1,
            cat_features=list(set(self.cat_features)&set(self.columns))
        ).fit(
            self.x_train[self.columns],
            self.y_train,
            early_stopping_rounds=30,
            eval_set=(self.x_test[self.columns], self.y_test),
            cat_features=list(set(self.cat_features)&set(self.columns)),
            verbose=False
        )

        self.feature_imp = sorted(zip(self.columns, self.model.feature_importances_), key=lambda x: x[1], reverse=True)

        return self

    def get_scores(self):

        early_stopping = {
            'early_stopping_rounds': 30,
            'eval_set': (self.x_test[self.columns], self.y_test),
            'verbose': False
        }

        score = cross_val_score(
            self.model,
            self.x_train[self.columns],
            self.y_train,
            scoring='roc_auc',
            cv=StratifiedKFold(5),
            fit_params=early_stopping
        )

        left_bound = round(score.mean() - 2 * score.std(), 2)
        right_bound = round(score.mean() + 2 * score.std(), 2)

        return left_bound, right_bound

    def _remove_columns(self):

        self.removed_columns = [
            col[0] for col in self.feature_imp[(len(self.feature_imp) - self.step):]
        ]

        self.columns = [
            col[0] for col in self.feature_imp[:(len(self.feature_imp) - self.step)]
        ]

        return self

    def backward_selector(self, allowed_decrease=0):

        self.get_feature_importance()
        
        left_bound, right_bound = self.get_scores()
        self.step = max(2, len(self.columns) // self.percent_drop)
        left_new = left_bound * (1 + allowed_decrease)
        right_new = right_bound * (1 + allowed_decrease)
        print(left_bound, right_bound, left_new, right_new, self.left_bound, self.right_bound)

        if ((left_new >= self.left_bound) & (right_new >= self.right_bound)):

            self.left_bound = left_bound
            self.right_bound = right_bound
            self.features.append(self.columns)
            self._remove_columns()
            print(f'Columns remaining {len(self.columns)}')
            return self.backward_selector(allowed_decrease)

        else:
            self.columns = np.concatenate((self.columns, self.removed_columns))
            return self.feature_stability(allowed_decrease)

    def feature_stability(self, allowed_decrease=0):
        # стабильность по тому, как меняется номер фичи при вычислении feature importance
        print('Starting feature stability testing')

        feature_importance_stability = {}

        for rs in tqdm.tqdm(range(10), total=10):
            
            self.random_state = rs
            self._shuffle_data()
            self.get_feature_importance()

            for i, value in enumerate(self.feature_imp, start=1):

                if value[0] in feature_importance_stability:
                    feature_importance_stability[value[0]].append(i)
                else:
                    feature_importance_stability[value[0]] = [i]

        bad_cols = []

        for col in feature_importance_stability:

            if (max(feature_importance_stability[col]) - min(feature_importance_stability[col])) / len(feature_importance_stability) >= 0.4:
               bad_cols.append(col)

        print(f'{len(bad_cols)} potential columns to drop')

        if len(bad_cols) > 0:

            self.columns = [
                col for col in self.columns
                if col not in bad_cols
            ]

            left_bound, right_bound = self.get_scores()
            left_new = left_bound * (1 + allowed_decrease)
            right_new = right_bound * (1 + allowed_decrease)
            print(left_bound, right_bound, left_new, right_new, self.left_bound, self.right_bound)

            if ((left_new >= self.left_bound) & (right_new >= self.right_bound)):
                print('Bad columns was drop')
                return self.columns, self.features
            else:
                print('Not stable columns is not bad')
                return np.concatenate((self.columns, np.array(bad_cols))), self.features

        else:
            return self.columns, self.features
