from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.model_selection import train_test_split
import catboost as cb
import xgboost as xgb

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
    DATA_DIR
)
from constants import FEATURES_TO_DROP, CAT_FEATURES
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

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir
        
        self.collisions = None
        self.hospitals = None
        self.neighbourhoods = None
        self.streets = None
        

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
        
    def _split_data(self, xgb=False, test_size=0.2, seed=42):
        X = self.collisions.drop(columns='ACCLASS')
        y = self.collisions['ACCLASS']
        if xgb:
            y = y.map({'Fatal': 1, 'Non-Fatal Injury': 0})
        return train_test_split(X, y, test_size=test_size, random_state=seed)

    def _train_model_with_catboost(self):
        X_train, X_test, y_train, y_test = self._split_data()

        train_pool = cb.Pool(
            data=X_train,
            label=y_train,
            cat_features=CAT_FEATURES
        )
        params = {
            'iterations': 200,
            'depth': 6,
            'learning_rate': 0.01,
            'loss_function': 'Logloss',
            'verbose': 100
        }
        cb_scores = cb.cv(
            train_pool,
            params,
            fold_count=5,
            early_stopping_rounds=100
        )

        self.X_test = X_test
        self.y_test = y_test
        self.cb_scores = cb_scores
    
