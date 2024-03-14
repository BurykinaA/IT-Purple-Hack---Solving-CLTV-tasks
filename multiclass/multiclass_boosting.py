import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import math
import pickle
import random
from sklearn.metrics import roc_auc_score
import optuna
import catboost as cb
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
from tqdm.auto import tqdm



random.seed(42)
np.random.seed(42)

class Boosting:

    def __init__(self, X_train, X_val, y_train, y_val, cat_features, weits, params = None):
        print("Init...")
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.weits = weits
        self.cat_features = cat_features
        self.params = params
        self.model = None
        self.best_params = None
        self.top_features = None
        self.train_pool = cb.Pool(data = X_train, label = y_train, cat_features = cat_features)
        self.val_pool = cb.Pool(data = X_val, label = y_val, cat_features = cat_features)
        print("Init Finished!")

    def train(self):
        if self.params is None:
            model = cb.CatBoostClassifier(
                learning_rate = 0.2,
                depth = 6,
                # l2_leaf_reg = 2.437,
                # random_seed = 42,
                # min_data_in_leaf = 30,
                # one_hot_max_size = 40,
                # colsample_bylevel = 0.079,
                loss_function = 'MultiClass',
                task_type = 'CPU',
                iterations = 300,
                use_best_model = True,
                verbose = 100,
                thread_count = -1,
                early_stopping_rounds = 100,
                eval_metric = 'AUC',
                class_weights=self.weits,
                # boosting_type = 'Plain',
                # bootstrap_type = 'MVS'
            )
        else:
            self.params["verbose"] = 100
            self.params["iterations"] = 800
            model = cb.CatBoostClassifier(**self.params)

        model.fit(
            self.train_pool,
            eval_set = self.val_pool
        )
        self.model = model
        # y_train_pred = model.predict_proba(self.X_train)[:, 1]
        # y_val_pred = model.predict_proba(self.X_val)[:, 1]

        # roc_auc_tr = roc_auc_score(self.y_train, y_train_pred)
        # roc_auc_val = roc_auc_score(self.y_val, y_val_pred)

        # print("ROC AUC на обучающей выборке:", roc_auc_tr)
        # print("ROC AUC на валидационной выборке:", roc_auc_val)


    def optimize_hyperparams(self):

        def objective(trial):
            params = {
                "objective" : trial.suggest_categorical("objective", ["MultiClass"]),
                "learning_rate" : trial.suggest_loguniform("learning_rate", 1e-5, 1e0),
                "l2_leaf_reg" : trial.suggest_loguniform("l2_leaf_reg", 1e-2, 3e0),
                "colsample_bylevel" : trial.suggest_float("colsample_bylevel", 0.01, 0.1, log = True),
                "depth" : trial.suggest_int("depth", 2, 5),
                "boosting_type" : trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
                "bootstrap_type" : trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]),
                "min_data_in_leaf" : trial.suggest_int("min_data_in_leaf", 2, 50),
                "one_hot_max_size" : trial.suggest_int("one_hot_max_size", 2, 50),
                "iterations" : trial.suggest_int("iterations", 500, 3500),
                "eval_metric" : "AUC"
            }

            if params["bootstrap_type"] == "Bayesian":
                params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
            elif params["bootstrap_type"] == "Bernoulli":
                params["subsample"] = trial.suggest_float("subsample", 0.1, 1, log = True)

            model = cb.CatBoostClassifier(
                loss_function = 'Logloss',
                random_seed = 42,
                task_type = 'CPU',
                use_best_model = True,
                verbose = False,
                **params
            )

            model.fit(
                self.train_pool,
                eval_set = self.val_pool
            )

            y_pred = model.predict_proba(self.X_val)[:, 1]

            roc_auc = roc_auc_score(self.y_val, y_pred)

            return roc_auc

        study = optuna.create_study(pruner=optuna.pruners.MedianPruner(n_warmup_steps = 5), direction = "maximize")
        study.optimize(objective, n_trials = 10, timeout = 60)

        self.best_params = study.best_params

        print("Best params:", self.best_params)

    def load_model(self, file_path):
        with open(file_path, "rb") as f:
            self.model = pickle.load(f)

    def save_model(self, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(self.model, f)

    def show_feats_imp(self):
        if self.model is None:
            raise ValueError("Model not found!")

        feature_importance = self.model.feature_importances_
        sorted_idx = np.argsort(feature_importance)

        plt.figure(figsize=(15, 10))
        plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
        plt.yticks(range(len(sorted_idx)), np.array(self.model.feature_names_)[sorted_idx])
        plt.title("Feature Importance")
        plt.show()

        self.top_features = np.flip(np.array(self.model.feature_names_)[sorted_idx])
        print(self.top_features)

    def top_feats_selection(self):

        top = []
        roc_tr = []
        roc_val = []

        for col in tqdm(self.top_features):

            top.append(col)
            top_cat = list(set(self.cat_features) & set(top))

            train_pool = cb.Pool(data = self.X_train[top], label = self.y_train, cat_features = top_cat)
            val_pool = cb.Pool(data = self.X_val[top], label = self.y_val, cat_features = top_cat)

            if self.params is None:
                model = cb.CatBoostClassifier(
                    learning_rate = 0.303,
                    depth = 6,
                    l2_leaf_reg = 2.437,
                    random_seed = 42,
                    min_data_in_leaf = 30,
                    one_hot_max_size = 40,
                    colsample_bylevel = 0.079,
                    loss_function = 'MultiClass',
                    task_type = 'CPU',
                    iterations = 1000,
                    use_best_model = True,
                    verbose = 100,
                    thread_count = -1,
                    early_stopping_rounds = 100,
                    eval_metric = 'AUC',
                    class_weights=weits,
                    boosting_type = 'Plain',
                    bootstrap_type = 'MVS'
                )
                path = "no_optuna_top_features.xlsx"
            else:
                self.params["verbose"] = 0
                self.params["iterations"] = 500
                path = "optuna_top_features.xlsx"
                model = cb.CatBoostClassifier(**self.params)

            model.fit(
                train_pool,
                eval_set = val_pool
            )

            y_train_pred = model.predict_proba(self.X_train[top])[:, 1]
            y_val_pred = model.predict_proba(self.X_val[top])[:, 1]

            roc_auc_tr = roc_auc_score(self.y_train, y_train_pred)
            roc_auc_val = roc_auc_score(self.y_val, y_val_pred)

            roc_tr.append(roc_auc_tr)
            roc_val.append(roc_auc_val)

        plt.figure(figsize=(15, 10))
        plt.plot(range(len(self.top_features)), roc_tr, marker = 'o', label = 'Train')
        plt.plot(range(len(self.top_features)), roc_val, marker = 'o', label = 'Valid')
        plt.xlabel("Number of Top Features")
        plt.ylabel("ROC AUC")
        plt.title("ROC AUC on Top-K Features")
        plt.legend()
        plt.show()

        stats = pd.DataFrame({
            "TRAIN" : roc_tr,
            "VALID" : roc_val
        })

        stats.to_excel(path, index = False)

    def one_factor_roc(self):
        story = pd.DataFrame()

        for feature in tqdm(self.X_train.columns):
            if self.params is None:
                model = cb.CatBoostClassifier(
                    learning_rate = 0.303,
                    depth = 6,
                    l2_leaf_reg = 2.437,
                    random_seed = 42,
                    min_data_in_leaf = 30,
                    one_hot_max_size = 40,
                    colsample_bylevel = 0.079,
                    loss_function = 'MultiClass',
                    task_type = 'CPU',
                    iterations = 1000,
                    use_best_model = True,
                    verbose = 100,
                    thread_count = -1,
                    early_stopping_rounds = 100,
                    eval_metric = 'AUC',
                    class_weights=weits,
                    boosting_type = 'Plain',
                    bootstrap_type = 'MVS'
                )
                path = "no_optuna_one_factor_roc.xlsx"
            else:
                self.params["verbose"] = False
                self.params["iterations"] = 500
                path = "optuna_one_factor_roc.xlsx"
                model = cb.CatBoostClassifier(**self.params)

            if feature in self.cat_features:
                train_pool = cb.Pool(data = self.X_train[[feature]], label = self.y_train, cat_features = [feature])
                val_pool = cb.Pool(data = self.X_val[[feature]], label = self.y_val, cat_features = [feature])
            else:
                train_pool = cb.Pool(data = self.X_train[[feature]], label = self.y_train)
                val_pool = cb.Pool(data = self.X_val[[feature]], label = self.y_val)

            model.fit(
                train_pool,
                eval_set = val_pool
            )

            y_train_pred = model.predict_proba(self.X_train[[feature]])[:, 1]
            y_val_pred = model.predict_proba(self.X_val[[feature]])[:, 1]

            roc_auc_tr = roc_auc_score(self.y_train, y_train_pred)
            roc_auc_val = roc_auc_score(self.y_val, y_val_pred)

            story = story.append(pd.DataFrame({
                'features' : [feature],
                'train' : [roc_auc_tr],
                'valid' : [roc_auc_val]
            }), ignore_index = True)

        plt.figure(figsize=(10, 7))
        plt.bar(range(len(story['features'])), story['train'], align = 'center', label = 'Train')
        plt.bar(range(len(story['features'])), story['valid'], align = 'edge', label = 'Valid')
        plt.xlabel("Features")
        plt.ylabel("ROC-AUC")
        plt.title("One-Factor ROC-AUC")
        plt.xticks(range(len(story['features'])), story['features'], rotation = 45)
        plt.legend()
        plt.tight_layout()
        story.to_excel(path, index = False)