from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.model_selection import train_test_split
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
    CBModelConfig,
    XGBModelConfig
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
        cb_model_config: CBModelConfig = CBModelConfig(),
        xgb_model_config: XGBModelConfig = XGBModelConfig()
    ):
        self.data_dir = data_dir
        self.cb_model_config = cb_model_config
        self.xgb_model_config = xgb_model_config

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
        collisions['ROAD_CLASS'] = np.where(
            collisions['ROAD_CLASS'].isin(['Laneway', 'Major Shoreline']),
            'Other',
            collisions['ROAD_CLASS'])

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

    def _to_disk(self, save_path: Path, overwrite: bool):
        if (save_path / 'processed_data.csv').exists() and not overwrite:
            print('processed_data.csv already exists and overwrite is set to False.')
            return
        self.collisions.to_csv(save_path / 'processed_data.csv', index=False)

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

        # Tune hyperparameters
        study = optuna.create_study(direction='minimize')
        study.optimize(
            lambda trial: self._cb_objective(trial, train_pool),
            n_trials=1
        )

        # Fit model with best hyperparameters
        best_params = study.best_trial.user_attrs['full_params']
        best_model = cb.CatBoostClassifier(**best_params)
        best_model.fit(train_pool)
        self.best_model = best_model
        self.feature_importance = pd.DataFrame(
            data=best_model.get_feature_importance(),
            index=self.X_train.columns
        )

        # Save model
        best_model.save_model()

    def _cb_objective(self, trial: optuna.Trial, train_pool: cb.Pool):

        params = {}

        for param, value in self.cb_model_config.params.model_dump().items():
            if type(value) is list:
                distribution = trial.suggest_categorical(param, value)
            elif type(value) is tuple:
                if type(value[0]) is float:
                    distribution = trial.suggest_float(param, *value)
                else:
                    distribution = trial.suggest_int(param, *value)
            else:
                distribution = value

            params[param] = distribution

        cv_results = cb.cv(
            train_pool,
            params,
            **self.cb_model_config.model_dump(exclude='params')
        )

        cv_scores = cv_results[f'test-{params["eval_metric"]}-mean']

        # Add optimal iterations to params
        params['iterations'] = cv_results['iterations'].max()
        trial.set_user_attr('full_params', params)

        return np.min(cv_scores)

    def _train_model_with_xgboost(self):

        dtrain = xgb.DMatrix(
            data=self.X_train,
            label=self.y_train.to_numpy(),
            enable_categorical=True
        )

        # Tune hyperparameters
        study = optuna.create_study(direction='minimize')
        study.optimize(
            lambda trial: self._xgb_objective(trial, dtrain),
            n_trials=1
        )

        # Fit model with best hyperparameters
        best_params = study.best_trial.user_attrs['full_params']
        best_model = xgb.XGBClassifier(**best_params)
        best_model.fit(dtrain.get_data(), dtrain.get_label())
        self.best_model = best_model
        self.feature_importance = pd.DataFrame(
            data=best_model.feature_importances_,
            index=self.X_train.columns
        )

    def _xgb_objective(self, trial, dtrain):

        param = {}

        for key, value in self.xgb_model_config.params.model_dump().items():
            if type(value) is list:
                distribution = trial.suggest_categorical(key, value)
            elif type(value) is tuple:
                if type(value[0]) is float:
                    distribution = trial.suggest_float(key, *value)
                else:
                    distribution = trial.suggest_int(key, *value)
            else:
                distribution = value

            param[key] = distribution

        cv_results = xgb.cv(
            param,
            dtrain,
            **self.xgb_model_config.model_dump(exclude='params')
        )

        cv_scores = cv_results[f'test-{param["eval_metric"]}-mean']
        
        # Add optimal iterations to params
        param['num_boost_round'] = cv_scores.idxmin() + 1
        trial.set_user_attr('full_params', param)

        return np.min(cv_scores)

    def _train_model(self, algorithm):

        if (train_func := self.algorithms.get(algorithm)):
            train_func()
        else:
            raise ValueError(f'Unsupported algorithm: {algorithm}')

    def run_training(
            self,
            algorithm: Algorithm,
            save_data: bool = True,
            save_path: Path = DATA_DIR,
            overwrite: bool = False
    ):

        self._fetch_data()
        self._prep_data()
        if save_data:
            self._to_disk(save_path, overwrite)
        self._split_data()
        self._train_model(algorithm)


pipeline = MVCFatClassPipeline()
pipeline.run_training(Algorithm.XGBOOST)
