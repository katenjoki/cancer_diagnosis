import time
import numpy as np 
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

#mlops
import mlflow
import logging
from mlflow.models.signature import infer_signature
from mlflow import sklearn, log_metric, log_param, log_artifacts

from collections import Counter
from imblearn.over_sampling import RandomOverSampler
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from catboost.utils import eval_metric
from sklearn.metrics import accuracy_score,roc_curve,auc,classification_report, confusion_matrix

from sklearn.model_selection import train_test_split, GridSearchCV

train = pd.read_csv('data/clean_train.csv')
test = pd.read_csv('data/clean_test.csv')

cat_cols = train.select_dtypes(include='object').columns.tolist()
#reorder the cat columns
train_reordered = train[cat_cols + [col for col in train.columns if col not in cat_cols]]
#get cat cols indices
cat_features = [train_reordered.columns.get_loc(col) for col in cat_cols]
train_reordered[cat_cols] = train_reordered[cat_cols].astype("category")

X = train_reordered.drop('DiagPeriodL90D',axis=1)
y = train_reordered['DiagPeriodL90D']

#resample X
ros = RandomOverSampler(random_state=37)
X_resampled, y_resampled = ros.fit_resample(X, y)
X_resampled_train,X_resampled_val,y_resampled_train,y_resampled_val = train_test_split(X_resampled,y_resampled,test_size=0.2)

def get_metrics(y_actual,y_preds):
    auc = eval_metric(y_actual, y_preds, 'AUC')
    accuracy = accuracy_score(y_actual,y_preds)
    metrics = {}
    metrics['AUC score'] = auc[0]
    metrics['Accuracy score'] = accuracy
    return metrics

mlflow.set_experiment('Prediction of Cancer Diagnosis')
with mlflow.start_run(run_name='Baseline Catboost'):
    #Catboost model
    
    clf = CatBoostClassifier(random_state=50)
    clf.fit(X_resampled_train, y_resampled_train, 
        cat_features=cat_features, 
        eval_set=(X_resampled_val, y_resampled_val), 
        verbose=False
        )
    y_pred = clf.predict(X_resampled_val)
    metrics = get_metrics(y_resampled_val.values,y_pred)

    #log the models
    mlflow.log_param("model","CatBoost model: baseline")
    mlflow.log_metrics(metrics)
    mlflow.catboost.log_model(clf, "catboost_model")
    print('Baseline CatBoost')
    print('--------------------------------------')
with mlflow.start_run(run_name='Tuned Catboost'):
    clf_params = {'depth': 9, 'learning_rate': 0.3}
    grid_clf = CatBoostClassifier(**clf_params)
    grid_clf.fit(X_resampled_train, y_resampled_train, 
        cat_features=cat_features, 
        eval_set=(X_resampled_val, y_resampled_val), 
        verbose=False
        )
    y_pred = grid_clf.predict(X_resampled_val)
    metrics = get_metrics(y_resampled_val.values,y_pred)

    #log the models
    mlflow.log_param("model","CatBoost model: hyperparameter tuned")
    mlflow.log_metrics(metrics)
    mlflow.catboost.log_model(grid_clf, "catboost_model")
    print('Tuned CatBoost')
    print('--------------------------------------')
with mlflow.start_run(run_name='Baseline LightGBM'):
    #LightGBM model
    lgb_model = lgb.LGBMClassifier(random_state=42)
    lgb_model.fit(X_resampled_train,y_resampled_train,categorical_feature=cat_features)
    y_pred = lgb_model.predict(X_resampled_val)
    metrics = get_metrics(y_resampled_val.values,y_pred)

    #log the models
    mlflow.log_param("model","LightGBM model: baseline")
    mlflow.log_metrics(metrics)
    mlflow.lightgbm.log_model(lgb_model, "lightgbm_model")
    print('Baseline LightGBM')
    print('--------------------------------------')
with mlflow.start_run(run_name='Tuned LightGBM'):
    lgb_params = {'learning_rate': 0.3, 'max_depth': None, 'n_estimators': 300}
    grid_lgb = lgb.LGBMClassifier(**lgb_params,random_state=42,verbose=0)
    grid_lgb.fit(X_resampled_train,y_resampled_train)
    y_pred = grid_lgb.predict(X_resampled_val)
    metrics = get_metrics(y_resampled_val.values,y_pred)

    #log the models
    mlflow.log_param("model","LightGBM model: hyperparameter tuned")
    mlflow.log_metrics(metrics)
    mlflow.lightgbm.log_model(grid_lgb, "lightgbm_model")
    print('Tuned LightGBM')
    print('--------------------------------------')
with mlflow.start_run(run_name='Baseline XGBoost'):
    #XGBoost model
    xgb_model = xgb.XGBClassifier(tree_method = 'hist',random_state=42,enable_categorical=True)
    xgb_model.fit(X_resampled_train,y_resampled_train)
    y_pred = xgb_model.predict(X_resampled_val)
    metrics = get_metrics(y_resampled_val.values,y_pred)
    
    #log the models
    mlflow.log_param("model","XGBoost model: baseline")
    mlflow.log_metrics(metrics)
    #mlflow.xgboost.log_model(xgb_model,'xgboost_model')
    # Log the XGBoost model with MLflow
    #mlflow.xgboost.log_model(artifact_path="xgboost_model.json", xgb_model=xgb_model)
    print('Baseline XGBoost')
    print('--------------------------------------')
with mlflow.start_run(run_name='Tuned XGBoost'):
    #XGBoost model
    xgb_params = {'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 300}
    grid_xgb = xgb.XGBClassifier(**xgb_params,tree_method = 'hist',random_state=42,enable_categorical=True)
    grid_xgb.fit(X_resampled_train,y_resampled_train)
    y_pred = grid_xgb.predict(X_resampled_val)
    metrics = get_metrics(y_resampled_val.values,y_pred)

    #log the models
    mlflow.log_param("model","XGBoost model: hyperparameter tuned")
    mlflow.log_metrics(metrics)
    #mlflow.log_artifacts('tuned_xgb_model.json')
    print('Tuned XGBoost')