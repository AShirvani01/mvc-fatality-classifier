import pandas as pd
import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree
from constants import HOSPITAL_LIST


def encode_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Encodes date and time into multiple features.

    Args:
        df: A Pandas DataFrame with DATE and TIME columns.

    Returns:
        The input dataframe with added DATETIME, YEAR, MONTH, Day of the Week,
        and HOUR features.
    """
    # if 'TIME' or 'DATE' not in df.columns:
    #     raise ValueError("TIME and/or DATE columns are not in the given Dataframe.")
    try:
        df['TIME'] = (
                    df['TIME']
                    .astype(str)
                    .apply(lambda t: '0' * (4-len(t)) + t)  # Format to 'HHMM'
        )

        df['DATETIME'] = (
                        (df['DATE'] + df['TIME'])
                        .pipe(pd.to_datetime, format='%Y-%m-%d%H%M')
                        .dt.round(freq='1h')  # Round to nearest hour
        )

        df['YEAR'] = df['DATETIME'].dt.year
        df['MONTH'] = df['DATETIME'].dt.month_name()
        df['DOW'] = df['DATETIME'].dt.day_name()
        df['HOUR'] = df['DATETIME'].dt.hour

        df = df.drop(columns=['TIME', 'DATE'], axis=1)

        return df

    except KeyError:
        print("TIME and/or DATE columns are not in the given Dataframe.")


def dist_to_nearest_hospital(
    collisions: gpd.GeoDataFrame,
    hospitals: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Finds distance to nearest hospital from location of collisions.

    Args:
        collisions: Geodataframe of the collision data.
        hospitals: Geodataframe of the hospital data.

    Returns:
        The input collisions Geodataframe with added DIST_TO_HOS feature.
    """
    nA = np.array(list(collisions['geometry'].apply(lambda x: (x.x, x.y))))
    nB = np.array(list(hospitals['geometry'].apply(lambda x: (x.x, x.y))))
    btree = cKDTree(nB)
    dist, idx = btree.query(nA, k=1, p=1)  # Manhattan distance
    hospitals_nearest = (
                    hospitals.iloc[idx]
                    .drop(columns='geometry')
                    .reset_index(drop=True)
    )
    gdf = pd.concat(
        [
            collisions.reset_index(drop=True),
            hospitals_nearest['ENGLISH_NAME'],
            pd.Series(dist, name='DIST_TO_HOS')
        ],
        axis=1
    )

    return gdf


def fill_missing_neighbourhoods(
    collisions: gpd.GeoDataFrame,
    neighbourhoods: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Fill neighbourhoods in collision Geodataframe labelled 'NSA'

    Args:
        collisions: Geodataframe of the collision data
        neighbourhoods: Geodataframe of the neighbourhood data

    Returns:
        The input collisions Geodataframe with filled in neighbourhoods.
    """
    gdf = collisions.sjoin_nearest(neighbourhoods, how='left')

    condition = collisions['NEIGHBOURHOOD_158'] == 'NSA'
    collisions.loc[condition, ['HOOD_158', 'NEIGHBOURHOOD_158']] = \
        gdf.loc[condition, ['Local_ID', 'NHName']].to_numpy()

    return collisions


def filter_toronto_hospitals_with_er(health_services: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Filter health services Geodataframe down to Toronto hospitals with ER."""
    toronto_hospitals_with_er = health_services[
        health_services['ENGLISH_NAME'].isin(HOSPITAL_LIST)
    ]

    return toronto_hospitals_with_er


def fill_missing_road_classes(
        collisions: gpd.GeoDataFrame,
        streets: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Fill road classes in collision Geodataframe labelled 'Pending' or NA

    Args:
        collisions: Geoedataframe of the collision data
        streets: Geodataframe of the street data

    Returns:
        The input collisions Geodataframe with filled in road classes.
    """
    missing_road_class = collisions.query('ROAD_CLASS.isna() | ROAD_CLASS == "Pending"')
    filled_road_class = missing_road_class.sjoin_nearest(streets, how='left')

    street_class_code = {
        '11': 'Expressway',
        '12': 'Expressway',  # Instead of Primary Highway
        '20': 'Local',  # Instead of Road
        '21': 'Major Arterial',  # Instead of Arterial
        '22': 'Collector',
        '23': 'Local',
        '25': 'Expressway Ramp'
    }

    filled_road_class['CLASS'] = filled_road_class['CLASS'].apply(lambda x: street_class_code[x])

    # Manually correct arterial road class edge cases
    arterials = filled_road_class.query('CLASS == "Major Arterial"')

    conditions = [
        arterials['STREET2'] == '27 S 427 C S RAMP',
        arterials['TYPE'].isin(['EXPY', 'PKY', 'HWY']),
        arterials['_id'].isin([17374, 17375]),
        arterials['_id'].isin([18555, 18556, 17203, 17204, 17205])
    ]

    classes = [
        'Expressway Ramp',
        'Expressway',
        'Laneway',
        'Minor Arterial'
    ]

    arterials['CLASS'] = np.select(conditions, classes, default=arterials['CLASS'])

    # Merge back into collisions dataframe
    filled_road_class.loc[
        filled_road_class['_id'].isin(arterials['_id']),
        'CLASS'
    ] = arterials['CLASS']

    collisions.loc[
        collisions['_id'].isin(filled_road_class['_id']),
        'ROAD_CLASS'
    ] = filled_road_class['CLASS']

    return collisions
