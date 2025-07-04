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

        cyclical_features = {
            'MONTH': [df['DATETIME'].dt.month - 1, 12],
            'DOW': [df['DATETIME'].dt.day_of_week, 7],
            'HOUR': [df['DATETIME'].dt.hour, 24]
        }

        for feature, (values, period) in cyclical_features.items():
            df[f'{feature}_sin'] = np.sin(2 * np.pi * values / period)
            df[f'{feature}_cos'] = np.cos(2 * np.pi * values / period)

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
                    .rename(columns={'ENGLISH_NAME': 'NEAREST_HOSPITAL'})
    )
    gdf = pd.concat(
        [
            collisions.reset_index(drop=True),
            hospitals_nearest['NEAREST_HOSPITAL'],
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
        gdf.loc[condition, ['Local_ID', 'NHName']].values

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
    ] = arterials['CLASS'].values

    collisions.loc[
        collisions['_id'].isin(filled_road_class['_id']),
        'ROAD_CLASS'
    ] = filled_road_class['CLASS'].values

    return collisions


def remove_whitespace(
    collisions: pd.DataFrame,
    columns: list[str] = None,
    remove_all: bool = False
) -> pd.DataFrame:
    """Remove whitespace in specified columns of collisions Dataframe.

    Args:
        collisions: Dataframe of the collision data.
        columns: columns from collisions to apply the transformation. If no
            columns are specified, apply transformation to all columns.
        remove_all: Remove all whitespace if true, else, remove leading and
            trailing whitespace only.

    Returns:
        The input collisions Dataframe with the removed whitespace.
    """

    if columns is None:
        columns = collisions.columns

    for feature in columns:
        if remove_all:
            collisions[feature] = collisions[feature].apply(
                lambda x: x.replace(' ', '') if isinstance(x, str) else x
            )
        else:
            collisions[feature] = collisions[feature].apply(
                lambda x: x.strip() if isinstance(x, str) else x
            )

    return collisions


def fill_nearest_spatial(
    collisions: gpd.GeoDataFrame,
    features: list[str]
) -> gpd.GeoDataFrame:
    """Fill all missing values in each given feature with the (spatially)
    closest rows with labelled (non-missing) values.

    Args:
        collisions: GeoDataframe of the collision data.
        features: list of features to apply transformation.
    """
    for feature in features:
        missing_rows = collisions.query(f'{feature}.isna()')
        labelled_rows = collisions.query(f'~{feature}.isna()')

        filled_rows = (
            missing_rows
            .sjoin_nearest(labelled_rows, how='left')
            .drop_duplicates(subset=['_id_left'], keep='first')
        )

        collisions.loc[
            collisions['_id'].isin(filled_rows['_id_left']), feature
        ] = filled_rows[f'{feature}_right'].values

    return collisions


def fill_nearest_temporal(
    collisions: pd.DataFrame,
    features: list[str],
    tolerance: pd.Timedelta = pd.Timedelta(2, 'days')
) -> pd.DataFrame:
    """Fill all missing values in each given feature with the (temporally)
    closest rows with labelled (non-missing) values.

    Args:
        collisions: Dataframe of the collision data with DATETIME feature.
        features: list of features to apply transformation.
        tolerance (pd.Timedelta): max time away from missing point that will
            be accepted. If greater than tolerance, returns None.

    Returns:
        pd.Dataframe: Input collision Dataframe with the features filled.
    """
    for feature in features:
        missing_rows = collisions.query(f'{feature}.isna()')
        labelled_rows = collisions.query(f'~{feature}.isna()')

        filled_rows = (
            pd.merge_asof(
                missing_rows,
                labelled_rows,
                on='DATETIME',
                direction='nearest',
                tolerance=tolerance
            )
        )

        collisions.loc[
            collisions['_id'].isin(filled_rows['_id_x']), feature
        ] = filled_rows[f'{feature}_y'].values

    return collisions


def group_collisions(
        collisions: pd.DataFrame,
        by: list[str] = ['DATETIME', 'geometry', 'ACCNUM'],
        as_index: bool = False
) -> pd.api.typing.DataFrameGroupBy:
    """Group collisions by time, place, and accident number.

    Args:
        collisions: Dataframe of the collision data with DATETIME feature.
        by (list[str]): features to group by.
        as_index (bool): return object with group labels as index.

    Returns:
        pd.api.typing.DataFrameGroupBy: groupby object that contains
            information about the group.
    """
    grouped_collisions = collisions.copy().groupby(
        by=by,
        dropna=False,
        as_index=as_index
    )

    return grouped_collisions


def num_persons_feature(
    collisions: pd.DataFrame,
    grouped_collisions: pd.DataFrame
) -> pd.DataFrame:
    """Create feature for number of persons involved in collision"""
    count = group_collisions(collisions).size()

    grouped_collisions['NUMPERSONS'] = count['size']

    return grouped_collisions


def long_to_wide(
    collisions: pd.DataFrame,
    grouped_collisions: pd.DataFrame,
    feature: str,
    as_count: bool = True,
    dropna: bool = False
) -> pd.DataFrame:
    """Convert feature from long to wide format.

    Args:
        collisions: Dataframe of the collision data.
        grouped_collisions: Dataframe of the collision data grouped by
            Datetime, geometry, and Accident Number.

    Returns:
        pd.DataFrame: Input grouped Dataframe with feature in wide format.
    """
    copy = collisions.copy()
    copy[feature] = copy[feature].apply(lambda x: f'{feature}_{x}')

    wide_counts = (
        group_collisions(
            copy,
            by=['DATETIME', 'geometry', 'ACCNUM', feature],
            as_index=True)
        .size()
        .unstack(fill_value=0)
    )

    if not as_count:
        wide_counts = wide_counts > 0

    if dropna & (f'{feature}_nan' in wide_counts.columns):
        wide_counts = wide_counts.drop(columns=[f'{feature}_nan'])

    # Merge counts with main dataframe
    merged_collisions = pd.merge(
        grouped_collisions.drop(columns=[feature]),
        wide_counts,
        on=['DATETIME', 'geometry', 'ACCNUM']
    )

    return merged_collisions

