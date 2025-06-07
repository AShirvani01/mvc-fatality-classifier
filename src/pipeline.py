from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
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
    CBModelConfig
)
from constants import FEATURES_TO_DROP, CAT_FEATURES, Algorithm
from preprocessing import (
    filter_toronto_hospitals_with_er,
    remove_whitespace,
    encode_datetime,
    group_collisions,
    fill_nearest_spatial,
    fill_nearest_temporal,
    long_to_wide,
    num_persons_feature,
    fill_missing_road_classes,
    dist_to_nearest_hospital
)


class MVCFatClassPipeline:

    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        cb_model_config: CBModelConfig = CBModelConfig()
    ):
        self.data_dir = data_dir

        self.algorithms: dict[Algorithm, Callable] = {
            Algorithm.CATBOOST: self._train_model_with_catboost,
            Algorithm.XGBOOST: self._train_model_with_xgboost
        }

    def _fetch_data(self):
        # Collision Data
        raw_collision_data = get_collision_data()
        self.collisions = convert_to_geodata(raw_collision_data)

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

    def _prep_data(self):

        collisions = self.collisions

        collisions = remove_whitespace(collisions)
        collisions = encode_datetime(collisions)

        # Filling missing/dropping rare classes in features
        collisions['IMPACTYPE'] = collisions['IMPACTYPE'].fillna('Unknown')
        collisions['MANOEUVER'] = np.where(
            collisions['MANOEUVER'] == 'Disabled',
            np.nan,
            collisions['MANOEUVER']
        )
        collisions = fill_missing_road_classes(collisions, self.streets)

        # Response
        collisions['ACCLASS'] = np.where(
            collisions['ACCLASS'] == 'Property Damage O',
            'Non-Fatal Injury',
            collisions['ACCLASS']
        )

        # Fill ACCLASS
        grouped_collisions = group_collisions(collisions)
        na_acclass = grouped_collisions['ACCLASS'].transform(lambda x: x.isna())
        atleast_1_fatal_injury = grouped_collisions['INJURY'].transform(lambda x: x.eq('Fatal').any())
        no_na_injury = grouped_collisions['INJURY'].transform(lambda x: x.notna().all())

        collisions['ACCLASS'] = np.select(
            condlist=[na_acclass & atleast_1_fatal_injury,
                      na_acclass & no_na_injury],
            choicelist=['Fatal', 'Non-Fatal Injury'],
            default=collisions['ACCLASS']
        )
        
        collisions = collisions.query('~ACCLASS.isna()')
        

        collisions = fill_nearest_spatial(
            collisions,
            ['DISTRICT', 'TRAFFCTL']
        )

        collisions = fill_nearest_temporal(
            collisions,
            ['VISIBILITY', 'LIGHT', 'RDSFCOND']
        )

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

        # Feature engineering
        grouped_collisions = dist_to_nearest_hospital(grouped_collisions, self.hospitals)

        grouped_collisions = grouped_collisions.drop(columns=FEATURES_TO_DROP)

        # One value columns to bool
        cols_with_one_value = grouped_collisions.nunique().eq(1)
        bool_cols = cols_with_one_value[cols_with_one_value].index
        grouped_collisions[bool_cols] = grouped_collisions[bool_cols].astype(bool)

        # Objects to category
        grouped_collisions[grouped_collisions.select_dtypes(['object']).columns] = (
            grouped_collisions
            .select_dtypes(['object'])
            .apply(lambda x: x.astype('category'))
        )

        self.collisions = grouped_collisions

    def _split_data(self, test_size=0.2, seed=42):
        X = self.collisions.drop(columns='ACCLASS')
        y = self.collisions['ACCLASS'].map({'Fatal': 1, 'Non-Fatal Injury': 0})

        split = train_test_split(X, y, test_size=test_size, random_state=seed)
        self.X_train, self.X_test, self.y_train, self.y_test = split

    def _train_model_with_catboost(self):

        train_pool = cb.Pool(
            data=self.X_train,
            label=self.y_train,
            cat_features=CAT_FEATURES
        )

        study = optuna.create_study(direction='minimize')
        study.optimize(
            lambda trial: self._cb_objective(trial, train_pool),
            n_trials=1
        )
        self.df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
        self.best_params = study.best_params
        self.best_value = study.best_value

    def _train_model_with_xgboost(self):

        dtrain = xgb.DMatrix(
            data=self.X_train,
            label=self.y_train.to_numpy(),
            enable_categorical=True
        )

        study = optuna.create_study(direction='minimize')
        study.optimize(
            lambda trial: self._xgb_objective(trial, dtrain),
            n_trials=1
        )
        self.df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
        self.best_params = study.best_params
        self.best_value = study.best_value

    def _xgb_objective(self, trial, dtrain):

        param = {
            'eval_metric': trial.suggest_categorical('eval_metric', ['logloss']),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1)
        }

        model_results = xgb.cv(
            param,
            dtrain,
            early_stopping_rounds=100,
            nfold=5,
            seed=42,
            verbose_eval=0
        )

        return np.min(model_results[f'test-{param["eval_metric"]}-mean'])

    def _cb_objective(self, trial, train_pool):

        params = {
            'objective': trial.suggest_categorical('objective', ['Logloss']),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1)
        }

        model_results = cb.cv(
            train_pool,
            params,
            early_stopping_rounds=100,
            fold_count=5,
            seed=42,
            verbose=0)

        return np.min(model_results[f'test-{params["objective"]}-mean'])

    def _train_model(self, algorithm):

        if (train_func := self.algorithms.get(algorithm)):
            train_func()
        else:
            raise ValueError(f'Unsupported algorithm: {algorithm}')

    def run_training(self, algorithm: Algorithm):

        self._fetch_data()
        self._prep_data()
        self._split_data()
        self._train_model(algorithm)


pipeline = MVCFatClassPipeline()
pipeline.run_training(Algorithm.XGBOOST)
