from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import catboost as cb
import xgboost as xgb
import optuna
from typing import Callable


from data import (
    get_collision_data,
    convert_to_geodata,
    load_external_data,
    download_hospital_data,
    download_streets_data,
    download_neighbourhood_data
)
from config import (
    HEALTH_SERVICES_PATH,
    STREETS_PATH,
    NEIGHBOURHOODS_PATH,
    DATA_DIR,
    MODEL_DIR,
    CBModelConfig,
    XGBModelConfig
)
from constants import FEATURES_TO_DROP, CAT_FEATURES, Algorithm
from preprocessing import *

from custom_losses.utils import *


class MVCFatClassPipeline:

    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        cb_model_config: CBModelConfig = CBModelConfig(),
        xgb_model_config: XGBModelConfig = XGBModelConfig(),
        model_dir: Path = MODEL_DIR,
        cache_dir: Path = DATA_DIR / 'processed_data.csv'
    ):
        self.data_dir = data_dir
        self.cb_model_config = cb_model_config
        self.xgb_model_config = xgb_model_config
        self.model_dir = model_dir
        self.cache_dir = cache_dir

        self.algorithms: dict[Algorithm, Callable] = {
            Algorithm.CATBOOST: self._train_model_with_catboost,
            Algorithm.XGBOOST: self._train_model_with_xgboost
        }

        model_dir.mkdir(exist_ok=True)

    def _fetch_data(self):
        # Collision Data
        raw_collision_data = get_collision_data()
        self.raw_collisions = convert_to_geodata(raw_collision_data)

        # Hospital Data
        download_hospital_data(self.data_dir)
        self.hospitals = (
            load_external_data(HEALTH_SERVICES_PATH)
            .pipe(filter_toronto_hospitals_with_er)
        )

        # Neighbourhood Data
        download_neighbourhood_data(self.data_dir)
        self.neighbourhood = load_external_data(NEIGHBOURHOODS_PATH)

        # Street Data
        download_streets_data(self.data_dir / 'canada_streets')
        self.streets = (
            load_external_data(STREETS_PATH)
            .query('CSDNAME_L == "Toronto"')
        )

    def _prep_data(self, drop_features=True):

        collisions = self.raw_collisions

        collisions = collisions.query('NEIGHBOURHOOD_158!="NSA"')
        collisions = remove_whitespace(collisions)
        collisions = encode_datetime(collisions)
        collisions['INVAGE'] = collisions['INVAGE'].replace(
            ['0 to 4', '5 to 9'], ['00 to 04', '05 to 09']
        )

        collisions = fill_missing_road_classes(collisions, self.streets)
        collisions = fill_rare_classes(collisions)
        collisions = fill_acclass(collisions)
        collisions = fill_nearest_spatial(collisions, ['DISTRICT', 'TRAFFCTL'])
        collisions = fill_nearest_temporal(collisions, ['VISIBILITY', 'LIGHT', 'RDSFCOND'])

        collisions = remove_whitespace(
            collisions,
            columns=['INVAGE', 'MANOEUVER', 'DRIVACT'],
            remove_all=True
        )

        grouped_collisions = group_collisions(collisions).first()

        # InvAge
        grouped_collisions = long_to_wide(collisions, grouped_collisions, 'INVAGE')
        grouped_collisions = num_persons_feature(collisions, grouped_collisions)

        # Manoeuver
        grouped_collisions = long_to_wide(collisions, grouped_collisions, 'MANOEUVER', as_count=False, dropna=True)

        # Driver Action
        grouped_collisions['DRIVACT'] = grouped_collisions['DRIVACT'].fillna('Unknown')
        grouped_collisions = long_to_wide(collisions, grouped_collisions, 'DRIVACT', as_count=False, dropna=True)

        # Hospitals
        grouped_collisions = dist_to_nearest_hospital(grouped_collisions, self.hospitals)

        if drop_features:
            grouped_collisions = grouped_collisions.drop(columns=FEATURES_TO_DROP)

        change_dtypes(grouped_collisions)

        self.collisions = grouped_collisions

    def _to_disk(self, save_path: Path, overwrite: bool):
        if (save_path / 'processed_data.csv').exists() and not overwrite:
            print('processed_data.csv already exists and overwrite is set to False.')
            return
        self.collisions.to_csv(save_path / 'processed_data.csv', index=False)

    def _split_data(self, test_size=0.2, seed=42):
        X = self.collisions.drop(columns='ACCLASS')
        y = self.collisions['ACCLASS']

        split = train_test_split(X, y, test_size=test_size, random_state=seed,
                                 stratify=y)
        self.X_train, self.X_test, self.y_train, self.y_test = split

    def _train_model_with_catboost(self, n_trials: int):

        train_pool = cb.Pool(
            data=self.X_train,
            label=self.y_train,
            cat_features=CAT_FEATURES
        )

        if is_max_optimal(self.cb_model_config.params.eval_metric):
            self.direction = 'maximize'
        else:
            self.direction = 'minimize'

        # Tune hyperparameters
        study = optuna.create_study(direction='maximize')
        study.optimize(
            lambda trial: self._cb_objective(trial, train_pool),
            n_trials=n_trials
        )

        # Fit model with best hyperparameters
        best_params = study.best_trial.user_attrs['full_params']
        best_model = cb.CatBoostClassifier(**best_params, verbose=False, class_names=[0, 1])
        best_model.fit(train_pool)
        self.model = best_model
        self.feature_importance = pd.DataFrame(
            data=best_model.get_feature_importance(),
            index=self.X_train.columns
        )

    def _cb_objective(self, trial: optuna.Trial, train_pool: cb.Pool):

        params, config = unpack_params(trial, self.cb_model_config, 'CatBoost', self.y_train.to_numpy())

        cv_results = cb.cv(
            train_pool,
            params,
            **config.model_dump(exclude='params')
        )

        cv_scores = cv_results[f'test-{params["eval_metric"]}-mean']

        # Add optimal iterations to params
        params['iterations'] = cv_results['iterations'].max()
        trial.set_user_attr('full_params', params)

        if self.direction == 'maximize':
            return np.max(cv_scores)
        return np.min(cv_scores)

    def _train_model_with_xgboost(self, n_trials: int):

        dtrain = xgb.DMatrix(
            data=self.X_train,
            label=self.y_train.to_numpy(),
            enable_categorical=True
        )

        if is_max_optimal(self.xgb_model_config.params.eval_metric):
            self.direction = 'maximize'
        else:
            self.direction = 'minimize'

        # Tune hyperparameters
        study = optuna.create_study(direction=self.direction)
        study.optimize(
            lambda trial: self._xgb_objective(trial, dtrain),
            n_trials=n_trials
        )

        # Fit model with best hyperparameters
        best_params = study.best_trial.user_attrs['full_params']
        best_model = xgb.XGBClassifier(**best_params)
        best_model.fit(dtrain.get_data(), dtrain.get_label())
        self.model = best_model
        self.feature_importance = pd.DataFrame(
            data=best_model.feature_importances_,
            index=self.X_train.columns
        )

        best_model.save_model(self.model_dir / 'xgboost_model.json')

    def _xgb_objective(self, trial, dtrain):

        param, config = unpack_params(trial, self.xgb_model_config, 'XGBoost', self.y_train.to_numpy())

        cv_results = xgb.cv(
            param,
            dtrain,
            **config.model_dump(exclude='params')
        )

        if self.xgb_model_config.params.eval_metric in ['LDAM', 'EQ']:
            cv_scores = cv_results['test-loss-mean']
        else:
            cv_scores = cv_results[f'test-{param["eval_metric"]}-mean']

        # Add optimal iterations to params
        param['n_estimators'] = cv_scores.idxmin() + 1
        trial.set_user_attr('full_params', param)

        if self.direction == 'maximize':
            return np.max(cv_scores)
        return np.min(cv_scores)

    def _train_model(self, algorithm: Algorithm, n_trials: int):

        if (train_func := self.algorithms.get(algorithm)):
            train_func(n_trials)
        else:
            raise ValueError(f'Unsupported algorithm: {algorithm}')

    def run_training(
            self,
            algorithm: Algorithm,
            n_trials: int = 30,
            save_data: bool = True,
            save_path: Path = DATA_DIR,
            overwrite: bool = False
    ):

        if self.cache_dir is None:
            self._fetch_data()
            self._prep_data()
            if save_data:
                self._to_disk(save_path, overwrite)
        else:
            data = pd.read_csv(self.cache_dir)
            change_dtypes(data)
            self.collisions = data
        self._split_data()
        self._train_model(algorithm, n_trials)
        
    def save_model(self, file_name: 'str', file_path: Path = MODEL_DIR, overwrite: bool = False):
        if self.model is None:
            print('No model to save.')
        elif (file_path / file_name).exists() and not overwrite:
            print('Model with file name already exists. Set overwrite to True or set new file name.')
        else:
            self.model.save_model(file_path / file_name, format='json')

    def _load_model(self, filepath: Path):
        model = xgb.XGBClassifier()
        model.load_model(filepath)
        self.model = model

    def predict(self):
        y_pred = self.model.predict(self.X_test)
        self.report = classification_report(self.y_test, y_pred)
        self.confusion_matrix = confusion_matrix(self.y_test, y_pred)
        accuracy = np.mean(y_pred == self.y_test)
        return np.around(accuracy, 3)


if __name__ == '__main__':
    pipeline = MVCFatClassPipeline()
