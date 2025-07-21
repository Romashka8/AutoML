# --------------------------------------------------------------------------------------------------------------

from interface import *

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

# --------------------------------------------------------------------------------------------------------------

class CatBoostParameters(Parameters):

	"""
	Setup catboost training parameters.
	"""

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

# --------------------------------------------------------------------------------------------------------------



# --------------------------------------------------------------------------------------------------------------
