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
    remove_whitespace
)
from visualize import plot_map


raw_collision_data = get_collision_data()
gdf = convert_to_geodata(raw_collision_data)
download_hospital_data()
download_neighbourhood_data()
download_streets_data()


# DATA PREPROCESSING
gdf = remove_whitespace(gdf)

gdf = encode_datetime(gdf)

# Encoding Geo Data & Filling in missing neighbourhoods

neighbourhood_gdf = load_external_data(NEIGHBOURHOODS_PATH)
gdf = fill_missing_neighbourhoods(gdf, neighbourhood_gdf)

# Adding nearest hospital feature
health_services = load_external_data(HEALTH_SERVICES_PATH)
toronto_hospitals_with_er = filter_toronto_hospitals_with_er(health_services)
df = dist_to_nearest_hospital(gdf, toronto_hospitals_with_er)


# Filling in missing Road Classes
streets = load_external_data(STREETS_PATH).query('CSDNAME_L == "Toronto"')
gdf = fill_missing_road_classes(df, streets)



# Fill in missing Districts
missing_district = df.query('DISTRICT.isna()')
plot_map([missing_district])

filled_district = (
                    gpd.sjoin_nearest(missing_district,
                                      df.query('~DISTRICT.isna()'),
                                      how='left')
                    .drop_duplicates(subset=['_id_left'], keep='first')
                )
df.loc[df['_id'].isin(filled_district['_id_left']), 'DISTRICT'] = filled_district['DISTRICT_right']


# Fill in missing TRAFFCTL
# df['TRAFFCTL'] = np.where(df['ACCLOC'] == 'Non Intersection',
#                           'No Control',
#                           df['TRAFFCTL'])

missing_traffctl = df.query('TRAFFCTL.isna()')
plot_map([missing_traffctl])

filled_traffctl = (
                    gpd.sjoin_nearest(missing_traffctl,
                                      df.query('~TRAFFCTL.isna()'),
                                      how='left',
                                      distance_col='DIST')
                    .drop_duplicates(subset=['_id_left'], keep='first')
                )

plot_map([df.loc[df['_id'].isin(filled_traffctl['_id_right'])], missing_traffctl], ['green','red'])

df.loc[df['_id'].isin(filled_traffctl['_id_left']), 'TRAFFCTL'] = filled_traffctl['TRAFFCTL_right']


# Fill in missing VISIBILITY
missing_visibility = df.query('VISIBILITY.isna()')
plot_map([missing_visibility])

filled_visibility = (
                        pd.merge_asof(missing_visibility,
                                      df.query('~VISIBILITY.isna()'),
                                      on='DATETIME',
                                      direction='nearest')
    
    )

plot_map([df.loc[df['_id'].isin(filled_visibility['_id_y'])], missing_visibility], ['green','red'])

df.loc[df['_id'].isin(filled_visibility['_id_x']), 'VISIBILITY'] = filled_visibility['VISIBILITY_y']

