# General libraries
import pandas as pd
import numpy as np

# For geodata
import geopandas as gpd

from data import (
    get_collision_data,
    convert_to_geodata,
    load_external_data,
    download_hospital_data,
    download_neighbourhood_data,
    download_streets_data
)
from config import (
    HEALTH_SERVICES_PATH,
    NEIGHBOURHOODS_PATH,
    STREETS_PATH
)
from preprocessing import (
    encode_datetime,
    dist_to_nearest_hospital,
    fill_missing_neighbourhoods,
    filter_toronto_hospitals_with_er,
    fill_missing_road_classes,
    remove_whitespace,
    fill_nearest_temporal,
    fill_nearest_spatial,
    group_collisions,
    long_to_wide,
    num_persons_feature
)
from visualize import plot_map


raw_collision_data = get_collision_data()
gdf = convert_to_geodata(raw_collision_data)
download_hospital_data()
download_neighbourhood_data()
download_streets_data()


# DATA PREPROCESSING
gdf = gdf.query('~(ACCLASS.isna() | ACCLASS == "Property Damage O")')
gdf = remove_whitespace(gdf)

gdf = encode_datetime(gdf)

# Encoding Geo Data & Filling in missing neighbourhoods
neighbourhood_gdf = load_external_data(NEIGHBOURHOODS_PATH)
gdf = fill_missing_neighbourhoods(gdf, neighbourhood_gdf)

# Adding nearest hospital feature
health_services = load_external_data(HEALTH_SERVICES_PATH)
toronto_hospitals_with_er = filter_toronto_hospitals_with_er(health_services)
gdf = dist_to_nearest_hospital(gdf, toronto_hospitals_with_er)


# Filling in missing Road Classes
streets = load_external_data(STREETS_PATH).query('CSDNAME_L == "Toronto"')
gdf = fill_missing_road_classes(gdf, streets)


# Fill in missing Districts, Traffic Control
gdf = fill_nearest_spatial(gdf, ['DISTRICT', 'TRAFFCTL'])

# Fill in missing VISIBILITY
gdf = fill_nearest_temporal(gdf, ['VISIBILITY', 'LIGHT', 'RDSFCOND'])


gdf = remove_whitespace(
    gdf,
    columns=['INVAGE', 'MANOEUVER', 'DRIVACT'],
    remove_all=True
)

grouped_gdf = group_collisions(gdf).first()

# INVAGE
grouped_gdf = long_to_wide(gdf, grouped_gdf, 'INVAGE')
grouped_gdf = num_persons_feature(gdf, grouped_gdf)

# IMPACTYPE
gdf['IMPACTYPE'] = gdf['IMPACTYPE'].fillna('Unknown')

# Manoeuver
gdf.loc[gdf['MANOEUVER'] == 'Disabled', 'MANOEUVER'] = np.nan
grouped_gdf = long_to_wide(gdf, grouped_gdf, 'MANOEUVER', as_count=False, dropna=True)

# Driver Action
grouped_gdf['DRIVACT'] = grouped_gdf['DRIVACT'].fillna('Unknown')
grouped_gdf = long_to_wide(gdf, grouped_gdf, 'DRIVACT', as_count=False, dropna=True)

