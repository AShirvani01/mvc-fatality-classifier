import pandas as pd
import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree


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
