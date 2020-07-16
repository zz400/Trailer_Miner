#!/usr/bin/env python

"""
Utils related with model development, e.g. model wrappers. 

Author: Zhengyang Zhao
Date: Jun. 10 2020.
"""

from functools import partial
import copy

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib 
import pickle

from sklearn.metrics import confusion_matrix, cohen_kappa_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, PredefinedSplit, KFold
from sklearn.ensemble import RandomForestRegressor

from xgboost.sklearn import XGBRegressor 
import lightgbm as lgb


def plot_scatter_preds(preds, truth, error_tolerance, ):
    fig = plt.figure(figsize=(6, 6))
    sns.regplot(truth, preds, marker="+")
    
    min_ = 0
    max_ = 150 #max(preds.max(), truth.max())
    
    x = np.linspace(min_, max_, 100)
    y130 = x * (1 + error_tolerance)
    y70 = x * (1 - error_tolerance)
    
    plt.plot(x, x, 'g:', linewidth=2)
    plt.plot(x, y130, 'y:', linewidth=2)
    plt.plot(x, y70, 'y:', linewidth=2)
    plt.xlim(min_, max_)
    plt.ylim(min_, max_)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel("Box office Truth ($M)", fontsize=18, labelpad=10)
    plt.ylabel("Box office Predicted ($M)", fontsize=18, labelpad=10)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title("Model w/o Trailer Data", fontsize=22, y=1.04)
    plt.show()
#     return fig


def calc_in_range_percent(preds, truth, error_tolerance):
    truth = np.array(truth)
    num = len(preds)
    correct_num = 0
    for i in range(num):
        if (preds[i] <= (1 + error_tolerance) * truth[i]) and (preds[i] >= (1 - error_tolerance) * truth[i]):
            correct_num += 1
    correct_perc = correct_num / num
    return correct_perc
    


#################################################################
# Model Wrappers.
#################################################################


class RegressorModel(object):
    """
    A wrapper class for regression models.
    It can be used for training and prediction.
    Can plot feature importance and training progress (if relevant for model).
    """

    def __init__(self, model_wrapper=None):
        """
        Args: 
            columns (list): 
            model_wrapper: 
        """
        self.model_wrapper = model_wrapper
        
        
    def save_model(self, model_path):
        joblib.dump(self, model_path, compress = 1)
        
        
    def fit(self, X, y,
            n_splits,
            params=None,
            eval_metric='rmse',
            plot=True,
            plot_title=None,
            verbose=1):
        
        """
        Training the model.

        Args:
            X (pd.DataFrame), y: training data. 
            n_splits: cross-validation splits the data. 
            params (dict): training parameters. Including hyperparameters and:
                params['objective'] (str): 'regression' or 'classification',
                params['verbose'] (bool),
                params['cat_cols'] (list): categorical_columns, only used in LGB and CatBoost wrappers.
                params['early_stopping_rounds'] (int).
            eval_metric (str): metric for validataion.
            plot (bool): if true, plot 'feature importance', 'training curve', 'distribution of prediction', 'distribution of error'.
        """
        
        self.eval_metric = eval_metric
        self.verbose = verbose
        folds = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        self.columns = X.columns.to_list()
        
        self.models = []  # if n_splits=5, save 5 models.
        self.scores = []  # if n_splits=5, save 5 score items. Each is like: {'validation_0': {'rmse': 396.888855}, 'validation_1': {'rmse': 417.889496}}
        self.feature_importances = pd.DataFrame(columns=['feature', 'gain'])  #  if n_splits=5, then self.feature_importances is the stack of the 5 models.
        self.oof = np.empty(X.shape[0])   # Predicted results using cross-validation. OOF: "Out-of-fold".
        self.oof[:] = np.NaN
        
        for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y)):
            X_train, X_valid = X.iloc[train_index].copy(), X.iloc[valid_index].copy()
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
            model = copy.deepcopy(self.model_wrapper)
            model.fit(X_train, y_train, X_valid, y_valid, params=params)
            
            self.models.append(model)
            self.scores.append(model.best_score_)
            self.oof[valid_index] = model.predict(X_valid).reshape(-1,)
            
            fold_importance = pd.DataFrame({
                                    'feature': X_train.columns,
                                    'gain': model.feature_importances_
                                })
            self.feature_importances = self.feature_importances.append(fold_importance)
            
            if self.verbose > 1:
                print(f'\nFold {fold_n} started.')
                for val in model.best_score_.keys():
                    print(f"{self.eval_metric} score on {val}: {model.best_score_[val][self.eval_metric]:.3f}.")

        self.calc_scores_()
        
        if plot:
            # print(classification_report(y, self.oof.argmax(1)))
            fig, ax = plt.subplots(figsize=(20, 8))
            plt.subplot(1, 4, 1)
            self.plot_feature_importance(top_n=30)
            plt.subplot(1, 4, 2)
            self.plot_learning_curve()
            plt.subplot(1, 4, 3)
            errors = y.values.reshape(-1, 1) - self.oof
            errors = errors[~np.isnan(errors)]
            plt.hist(errors)
            plt.title('Distribution of errors')
            plt.subplot(1, 4, 4)
            plt.hist(self.oof[~np.isnan(self.oof)])
            plt.title('Distribution of oof predictions');
            plt.suptitle(plot_title)
    
    
    def predict(self, X_test, averaging='usual'):
        """
        Make prediction

        Args:
            X_test (pd.DataFrame): test data
            averaging: method of averaging
            
        Return:
            list: prediction of X_test
        """
        
        full_prediction = np.zeros(X_test.shape[0])
        for i in range(len(self.models)):
            y_pred = self.models[i].predict(X_test).reshape(-1)
            if averaging == 'usual':
                full_prediction += y_pred
            elif averaging == 'rank':
                full_prediction += pd.Series(y_pred).rank().values
        return full_prediction / len(self.models)
        
    
    def predict_range(self, X_test):
        """
        Make prediction, return the prediction from each CV model.

        Args:
            X_test (pd.DataFrame): test data
            
        Return:
            np array with shape: X_test.shape[0] x CV models
        """
        
        full_prediction = np.zeros((X_test.shape[0], len(self.models)))
        for i in range(len(self.models)):
            y_pred = self.models[i].predict(X_test).reshape(-1)
            full_prediction[:, i] = y_pred
        return full_prediction
        

    def calc_scores_(self):
        """
        Average the scores from the n_splits cross validation.
        """
        self.ave_scores = {}
        sets = [k for k in self.scores[0]]  # sets = ['validation_0', 'validation_1']
        print(f'\nFinished cross-validation training.')
        for val in sets:
            scores = [score[val][self.eval_metric] for score in self.scores]
            if self.verbose:
                print(f"CV mean {self.eval_metric} score on {val}: {np.mean(scores):.3f} +/- {np.std(scores):.3f} std.")
            self.ave_scores[val] = np.mean(scores)  # self.ave_scores: {'validation_0': 398.9524596, 'validation_1': 408.9034486}


    def plot_feature_importance(self, drop_null_importance=True, top_n=20):
        """
        Plot feature importance.

        Args:
            drop_null_importance (bool): drop columns with null feature importance
            top_n (int): show top n features.
        """
        
        fig = plt.figure(figsize=(4, 8))
        top_feats = self.get_top_features(drop_null_importance, top_n)
        feature_importances = self.feature_importances.loc[self.feature_importances.loc[:, 'feature'].isin(top_feats)]
        feature_importances.loc[:, 'feature'] = feature_importances.loc[:, 'feature'].astype(str)
        top_feats = [str(i) for i in top_feats]
        sns.barplot(data=feature_importances, x='gain', y='feature', orient='h', order=top_feats)
        plt.title("Model with Trailer Data", fontsize=22, y=1.04)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.ylabel("")
    

    
    def get_top_features(self, drop_null_importance=True, top_n=20):
        """
        Get top features by importance.
        
        Args:
            drop_null_importance (bool): drop columns with null feature importance
            top_n (int): show top n features.
        """
        
        grouped_feats = self.feature_importances.groupby(['feature'])['gain'].mean()  # average over folds.
        if drop_null_importance:
            grouped_feats = grouped_feats[grouped_feats != 0]
        return list(grouped_feats.sort_values(ascending=False).index)[:top_n]

    
    def plot_learning_curve(self):
        """
        Plot training learning curve.
        Inspired by `plot_metric` from https://lightgbm.readthedocs.io/en/latest/_modules/lightgbm/plotting.html
        
        An example of model.evals_result_: 
            {
                'validation_0': {'rmse': [0.259843, 0.26378, 0.26378, ...]},
                'validation_1': {'rmse': [0.22179, 0.202335, 0.196498, ...]}
            }
            
            'validation_0' represent train set;
            'validation_1' represent validation set;
        """
        
        fig = plt.figure(figsize=(4, 8))
        full_evals_results = pd.DataFrame()
        for model in self.models:
            evals_result = pd.DataFrame()
            for k in model.model.evals_result_.keys():  # iterate through different sets.
                evals_result[k] = model.model.evals_result_[k][self.eval_metric]
            evals_result = evals_result.reset_index().rename(columns={'index': 'iteration'})
            full_evals_results = full_evals_results.append(evals_result)

        full_evals_results = full_evals_results.melt(id_vars=['iteration']).rename(columns={'value': self.eval_metric,
                                                                                            'variable': 'dataset'})
        sns.lineplot(data=full_evals_results, x='iteration', y=self.eval_metric, hue='dataset')
        plt.title('Train Learning-Curve')
        
        
#################################################################
# Model Wrappers.
#################################################################

class RandForest_regr(object):
    """
    A wrapper for sklearn RandomForestRegressor model so that we will have a single api for various models.
    
    Example of params:
    params = { 
        'n_estimators': 100,
        'criterion': 'mse',
        'max_depth': 7,
        'min_samples_split': 2,
        'n_jobs': -1,
        'random_state': 123,
        'verbose': 0,
    }
    """

    def __init__(self):
        self.model = RandomForestRegressor()
        
    def fit(self, X_train, y_train, X_valid=None, y_valid=None, params=None):
        self.model.set_params(**params)
        self.model.fit(X=X_train, y=np.array(y_train).reshape(-1))
        score = mean_squared_error(y_train, self.model.predict(X_train))
        self.best_score_ = score
        self.feature_importances_ = self.model.feature_importances_
        
    def predict(self, X_test):
        return self.model.predict(X_test)
        
        

class XGBWrapper_regr(object):
    """
    A wrapper for xgboost model so that we will have a single api for various models.
    
    Example of params:
    params = { 
        'n_estimators': 50,  #################
        'max_depth':  3,  #################
        'learning_rate': 0.01, 
    #     'min_child_weight': np.arange(1, 4, 1),
    #     'gamma': np.arange(0, 0.03, 0.01),
    #     'reg_alpha': np.arange(0, 0.01, 0.003),
        'objective': 'reg:squarederror', #['reg:squaredlogerror']#, # squared loss.
        'verbosity': 0,
        'early_stopping_rounds': None,
        'n_jobs': -1,
        'random_state': 123
    }
    """

    def __init__(self):
        self.model = XGBRegressor()

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, params=None):

        self.model = self.model.set_params(**params)
        
        eval_set = [(X_train, y_train)]
        if X_valid is not None:
            eval_set.append((X_valid, y_valid))
        
        self.model.fit(X=X_train, y=y_train,
                       eval_set=eval_set, eval_metric='rmse',
                       verbose=params['verbosity'], 
                       early_stopping_rounds=params['early_stopping_rounds'])

        scores = self.model.evals_result()
        self.best_score_ = {k: {m: m_v[-1] for m, m_v in v.items()} for k, v in scores.items()}
#         self.best_score_ = {k: {m: n if m != 'cappa' else -n for m, n in v.items()} for k, v in self.best_score_.items()}

        self.feature_importances_ = self.model.feature_importances_
    
    def predict(self, X_test):
        return self.model.predict(X_test)

#     def predict_proba(self, X_test):
#         if self.model.objective == 'binary':
#             return self.model.predict_proba(X_test, ntree_limit=self.model.best_iteration)[:, 1]
#         else:
#             return self.model.predict_proba(X_test, ntree_limit=self.model.best_iteration)


class LGBWrapper_regr(object):
    """
    A wrapper for lightgbm model so that we will have a single api for various models.
    
    Example of params:
    params = {
         'n_estimators': 200,
         'num_leaves': 30,
         'min_data_in_leaf': 10,
         'objective': 'regression',
         'max_depth': 7,
         'learning_rate': 0.01,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.2,
         'early_stopping_rounds': 50,
         'verbose': 0,
         'n_jobs': -1,
    }
    """

    def __init__(self):
        self.model = lgb.LGBMRegressor()

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, params=None):
        if params['objective'] == 'regression':
            eval_metric = 'rmse'
        else:
            eval_metric = 'auc'

        eval_set = [(X_train, y_train)]
        eval_names = ['train']
        self.model = self.model.set_params(**params)

        if X_valid is not None:
            eval_set.append((X_valid, y_valid))
            eval_names.append('valid')

        if 'cat_cols' in params.keys():
            cat_cols = [col for col in params['cat_cols'] if col in X_train.columns]
            if len(cat_cols) > 0:
                categorical_columns = params['cat_cols']
            else:
                categorical_columns = 'auto'
        else:
            categorical_columns = 'auto'

        self.model.fit(X=X_train, y=y_train,
                       eval_set=eval_set, eval_names=eval_names, eval_metric=eval_metric,
                       verbose=params['verbose'], early_stopping_rounds=params['early_stopping_rounds'],
                       categorical_feature=categorical_columns)

        self.best_score_ = self.model.best_score_
        self.feature_importances_ = self.model.feature_importances_.astype(float)

    def predict(self, X_test):
        return self.model.predict(X_test, num_iteration=self.model.best_iteration_)

    
class LGBWrapper(object):
    """
    A wrapper for lightgbm model so that we will have a single api for various models.
    """

    def __init__(self):
        self.model = lgb.LGBMClassifier()

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, X_holdout=None, y_holdout=None, params=None):

        eval_set = [(X_train, y_train)]
        eval_names = ['train']
        self.model = self.model.set_params(**params)

        if X_valid is not None:
            eval_set.append((X_valid, y_valid))
            eval_names.append('valid')

        if X_holdout is not None:
            eval_set.append((X_holdout, y_holdout))
            eval_names.append('holdout')

        if 'cat_cols' in params.keys():
            cat_cols = [col for col in params['cat_cols'] if col in X_train.columns]
            if len(cat_cols) > 0:
                categorical_columns = params['cat_cols']
            else:
                categorical_columns = 'auto'
        else:
            categorical_columns = 'auto'

        self.model.fit(X=X_train, y=y_train,
                       eval_set=eval_set, eval_names=eval_names, eval_metric=eval_qwk_lgb,
                       verbose=params['verbose'], early_stopping_rounds=params['early_stopping_rounds'],
                       categorical_feature=categorical_columns)

        self.best_score_ = self.model.best_score_
        self.feature_importances_ = self.model.feature_importances_

    def predict_proba(self, X_test):
        if self.model.objective == 'binary':
            return self.model.predict_proba(X_test, num_iteration=self.model.best_iteration_)[:, 1]
        else:
            return self.model.predict_proba(X_test, num_iteration=self.model.best_iteration_)


class CatWrapper(object):
    """
    A wrapper for catboost model so that we will have a single api for various models.
    """

    def __init__(self):
        self.model = cat.CatBoostClassifier()

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, X_holdout=None, y_holdout=None, params=None):

        eval_set = [(X_train, y_train)]
        self.model = self.model.set_params(**{k: v for k, v in params.items() if k != 'cat_cols'})

        if X_valid is not None:
            eval_set.append((X_valid, y_valid))

        if X_holdout is not None:
            eval_set.append((X_holdout, y_holdout))

        if 'cat_cols' in params.keys():
            cat_cols = [col for col in params['cat_cols'] if col in X_train.columns]
            if len(cat_cols) > 0:
                categorical_columns = params['cat_cols']
            else:
                categorical_columns = None
        else:
            categorical_columns = None
        
        self.model.fit(X=X_train, y=y_train,
                       eval_set=eval_set,
                       verbose=params['verbose'], early_stopping_rounds=params['early_stopping_rounds'],
                       cat_features=categorical_columns)

        self.best_score_ = self.model.best_score_
        self.feature_importances_ = self.model.feature_importances_

    def predict_proba(self, X_test):
        if 'MultiClass' not in self.model.get_param('loss_function'):
            return self.model.predict_proba(X_test, ntree_end=self.model.best_iteration_)[:, 1]
        else:
            return self.model.predict_proba(X_test, ntree_end=self.model.best_iteration_)

