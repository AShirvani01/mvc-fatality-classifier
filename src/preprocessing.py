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
            hospitals_nearest['ENGLISH_NA'],
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
