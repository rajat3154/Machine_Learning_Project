import os
import sys
import dill
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        best_model = None
        best_score = -np.inf

        for model_name, model in models.items():
            try:
                param_grid = params.get(model_name, {})

                if param_grid:  
                    gs = GridSearchCV(model, param_grid, cv=3, scoring='r2', n_jobs=-1)
                    gs.fit(X_train, y_train)
                    best_model_instance = gs.best_estimator_
                else:  
                    best_model_instance = model
                    best_model_instance.fit(X_train, y_train)

                y_test_pred = best_model_instance.predict(X_test)
                test_score = r2_score(y_test, y_test_pred)

                report[model_name] = test_score

                if test_score > best_score:
                    best_score = test_score
                    best_model = best_model_instance

            except Exception as model_exception:
                print(f"Skipping {model_name} due to error: {model_exception}")
        return report, best_model

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)