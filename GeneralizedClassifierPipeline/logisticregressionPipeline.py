# --------------------------------------------------------------------------------------------------------------

from interface import *

import tqdm
import pickle
import os

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score

import hyperopt

import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------------------------------------------------------------------

class LogisticRegressionParameters(Parameters):

    """
    Setup logistic regression parameters.
    """

    def __init__(self, params_dict=None):
        
        if params_dict is None:

            self.params = {
                'penalty': hyperopt.hp.choice('penalty', ['l1', 'l2', 'elasticnet', 'none']),
                'C': hyperopt.hp.uniform('C', 0.01, 10),
                'solver': hyperopt.hp.choice('solver', ['newton-cg', 'lbfgs', 'liblinear', 'saga']),
                'max_iter': hyperopt.hp.uniformint('max_iter', 100, 500),
                'l1_ratio': hyperopt.hp.uniform('l1_ratio', 0, 1)  # Only for elasticnet
            }

        else:

            self.params = {
                'penalty': hyperopt.hp.choice('penalty', params_dict['penalty']),
                'C': hyperopt.hp.uniform('C', *params_dict['C']),
                'solver': hyperopt.hp.choice('solver', params_dict['solver']),
                'max_iter': hyperopt.hp.uniformint('max_iter', *params_dict['max_iter']),
                'l1_ratio': hyperopt.hp.uniform('l1_ratio', *params_dict['l1_ratio'])
            }

    def param_transformer(self, params):
        
        transformed = {
            'penalty': params['penalty'],
            'C': round(params['C'], 3),
            'solver': params['solver'],
            'max_iter': int(params['max_iter']),
            'l1_ratio': round(params.get('l1_ratio', None), 3) if 'l1_ratio' in params else None
        }
        
        # Remove None values
        return {k: v for k, v in transformed.items() if v is not None}

# --------------------------------------------------------------------------------------------------------------

class TrainLogisticRegressionModel(TrainModel):

    """
    Implemented logreg training pipeline.
    """

    def __init__(self, x_train, y_train, x_test, y_test,
                 random_seed=42, max_evals=10,
                 params_dict=None, scale_data=True):

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.parameters = LogisticRegressionParameters(params_dict)
        self.model = None
        self.scoring = None
        self.max_evals = max_evals
        self.random_seed = random_seed
        self.scale_data = scale_data
        self.scaler = StandardScaler() if scale_data else None

    def __scale_data(self, X_train, X_test=None):

        if self.scale_data:
            X_train_scaled = self.scaler.fit_transform(X_train)
            if X_test is not None:
                X_test_scaled = self.scaler.transform(X_test)
                return X_train_scaled, X_test_scaled
            return X_train_scaled
        return X_train if X_test is None else (X_train, X_test)

    def __plot_feature_importance(self, model, model_name):

        if isinstance(self.x_train, pd.DataFrame):
            col_names = self.x_train.columns.values
        else:
            col_names = [f'col_{i}' for i in range(self.x_train.shape[1])]
            
        # For logistic regression, we use absolute coefficients as importance
        coef = np.abs(model.coef_[0])
        feature_imp = pd.DataFrame(
            sorted(zip(coef, col_names)),
            columns=['Value', 'Feature']
        )

        plt.figure(figsize=(10, 10))
        sns.barplot(x='Value', y='Feature',
                    data=feature_imp.sort_values(by='Value', ascending=False),
                    palette='magma'
        )

        plt.title(f'{model_name} importance of features')

    def __lr(self, params):

        params = self.parameters.param_transformer(params)
        
        # Handle incompatible parameter combinations
        if params['penalty'] == 'elasticnet' and params['solver'] != 'saga':
            params['solver'] = 'saga'
        elif params['penalty'] == 'l1' and params['solver'] not in ['liblinear', 'saga']:
            params['solver'] = 'liblinear'
        elif params['penalty'] == 'none':
            params.pop('penalty', None)
        
        model = LogisticRegression(
            random_state=self.random_seed,
            **params
        )

        X_train_scaled = self.__scale_data(self.x_train)
        
        score = cross_val_score(
            model,
            X_train_scaled,
            self.y_train,
            scoring=self.scoring,
            cv=StratifiedKFold(5)
        ).mean()

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

    def train(self, scoring='roc_auc'):

        self.scoring = scoring.lower()
        
        best_params, cv_score = self.__train_model(
            self.__lr,
            self.parameters.params
        )

        best_params = hyperopt.space_eval(
            self.parameters.params,
            best_params
        )

        best_params = self.parameters.param_transformer(best_params)
        
        # Final model training
        X_train_scaled, X_test_scaled = self.__scale_data(self.x_train, self.x_test)
        
        best_model = LogisticRegression(
            random_state=self.random_seed,
            **best_params
        ).fit(X_train_scaled, self.y_train)
        
        test_score = roc_auc_score(
            self.y_test,
            best_model.predict_proba(X_test_scaled)[:, 1]
        )
        
        self.__plot_feature_importance(best_model, 'Logistic Regression')

        return best_model, cv_score, test_score

# --------------------------------------------------------------------------------------------------------------

class FeatureSelectorLR(FeatureSelector):

    def __init__(self, x_train, y_train,
                 x_test, y_test, percent_drop=10,
                 random_state=42, scale_data=True):

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.percent_drop = percent_drop
        self.columns = x_train.columns.values if hasattr(x_train, 'columns') else [f'col_{i}' for i in range(x_train.shape[1])]
        self.step = max(2, len(self.columns) // self.percent_drop)
        self.model = None
        self.feature_imp = None
        self.removed_columns = None
        self.left_bound = -np.Inf
        self.right_bound = -np.Inf
        self.random_state = random_state
        self.scale_data = scale_data
        self.scaler = StandardScaler() if scale_data else None
        self.features = []

    def _shuffle_data(self):

        X = np.concatenate([self.x_train, self.x_test])
        y = np.concatenate((self.y_train, self.y_test))

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=0.3,
            random_state=self.random_state
        )

        return self

    def __scale_data(self, X):

        if self.scale_data:
            return self.scaler.fit_transform(X)
        return X

    def get_feature_importance(self):

        X_train_scaled = self.__scale_data(self.x_train)
        
        self.model = LogisticRegression(
            penalty='l1',
            solver='liblinear',
            random_state=self.random_state
        ).fit(X_train_scaled, self.y_train)

        # For logistic regression, we use absolute coefficients as importance
        coef = np.abs(self.model.coef_[0])
        self.feature_imp = sorted(zip(self.columns, coef), key=lambda x: x[1], reverse=True)

        return self

    def get_scores(self):

        X_train_scaled = self.__scale_data(self.x_train)
        X_test_scaled = self.__scale_data(self.x_test)
        
        score = cross_val_score(
            self.model,
            X_train_scaled,
            self.y_train,
            scoring='roc_auc',
            cv=StratifiedKFold(5)
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

        # Update data
        if isinstance(self.x_train, pd.DataFrame):
            self.x_train = self.x_train[self.columns]
            self.x_test = self.x_test[self.columns]
        else:
            col_indices = [i for i, col in enumerate(self.columns) if col in self.columns]
            self.x_train = self.x_train[:, col_indices]
            self.x_test = self.x_test[:, col_indices]

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
            return self.feature_stability(allowed_decrease)

    def feature_stability(self, allowed_decrease=0):

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

    def forward_selector(self, target, params, val1_set=None, test_set=None, n_stop=50, 
                        percent_view=False, decimals=2):

        res_features = []
        res_features_with_metrics = []
        
        x_train = pd.DataFrame(self.x_train, columns=self.columns) if not isinstance(self.x_train, pd.DataFrame) else self.x_train
        x_test = pd.DataFrame(self.x_test, columns=self.columns) if not isinstance(self.x_test, pd.DataFrame) else self.x_test
        
        train_set = x_train.copy()
        train_set[target] = self.y_train
        val_set = x_test.copy()
        val_set[target] = self.y_test
        
        features = self.columns
        
        for i in tqdm.tqdm(range(min(n_stop, len(features)))):
            feat_roc_auc_score = []
            
            for f in features:
                if f not in res_features:
                    current_features = res_features.copy()
                    current_features.append(f)
                    
                    model = LogisticRegression(
                        **params,
                        random_state=self.random_state
                    )
                    
                    # Scale data
                    X_train = self.__scale_data(train_set[current_features])
                    model.fit(X_train, train_set[target])
                    
                    metrics = []
                    for dataset, dataset_name in zip(
                        [train_set, val_set, val1_set, test_set],
                        ['Train', 'Val', 'Val1', 'Test']
                    ):
                        if dataset is not None:
                            X = self.__scale_data(dataset[current_features])
                            predictions = model.predict_proba(X)[:, 1]
                            auc = roc_auc_score(dataset[target].values, predictions)
                            metrics.append(auc)
                        else:
                            metrics.append(np.nan)
                    
                    feat_roc_auc_score.append((f, *metrics))
            
            if feat_roc_auc_score:
                feat_roc_auc_score.sort(key=lambda x: -abs(x[2]))
                res_features.append(feat_roc_auc_score[0][0])
                res_features_with_metrics.append(feat_roc_auc_score[0])
        
        columns = ['Feature name', 'AUC Train', 'AUC Val', 'AUC Val1', 'AUC Test']
        res = pd.DataFrame(res_features_with_metrics, columns=columns)
        
        for s in ['Train', 'Val', 'Val1', 'Test']:
            res[f'Delta AUC {s}'] = [np.nan] + list(res[f'AUC {s}'].values[1:] - res[f'AUC {s}'].values[:-1])
        
        if percent_view:
            res.iloc[:, 1:] = (100 * res.iloc[:, 1:]).round(decimals)

        keep_res = [col for col in res.columns if not res[col].isna().all()]

        return res[keep_res]

# --------------------------------------------------------------------------------------------------------------
