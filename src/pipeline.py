from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd

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
        collisions = collisions.query('~ACCLASS.isna()')

        # Group collisions
        grouped_collisions = group_collisions(collisions)

        condlist = [
            grouped_collisions['ACCLASS'].transform(lambda x: x.isna())
            & grouped_collisions['INJURY'].transform(lambda x: x.eq('Fatal').any()),
            grouped_collisions['ACCLASS'].transform(lambda x: x.isna())
            & grouped_collisions['INJURY'].transform(lambda x: x.notna().all())
        ]

        collisions['ACCLASS'] = np.select(
            condlist,
            choicelist=['Fatal', 'Non-Fatal Injury'],
            default=None
        )

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
        
        self.collisions = grouped_collisions
