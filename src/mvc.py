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
from constants import (
    MUNICIPALITIES,
    HOSPITAL_TYPES,
    HOSPITAL_LIST
)
from preprocessing import (
    encode_datetime,
    dist_to_nearest_hospital,
    fill_missing_neighbourhoods
)
from visualize import plot_map


raw_collision_data = get_collision_data()
gdf = convert_to_geodata(raw_collision_data)
download_hospital_data()
download_neighbourhood_data()
download_streets_data()


# DATA PREPROCESSING

gdf = encode_datetime(gdf)

# Encoding Geo Data & Filling in missing neighbourhoods

neighbourhood_gdf = load_external_data(NEIGHBOURHOODS_PATH)
gdf = fill_missing_neighbourhoods(gdf, neighbourhood_gdf)

# Adding nearest hospital feature
health_services = load_external_data(HEALTH_SERVICES_PATH)
health_services_type_count = health_services['SERVICE_TY'].value_counts()

# Filter for Toronto hospitals
toronto_hospitals = health_services[
                    (health_services['SERVICE_TY'].isin(HOSPITAL_TYPES))
                    & (health_services['COMMUNITY'].isin(MUNICIPALITIES))
    ]

toronto_hospitals_with_er = health_services[
                        health_services['ENGLISH_NA'].isin(HOSPITAL_LIST)
    ]


df = dist_to_nearest_hospital(gdf, toronto_hospitals_with_er)
# df = df.drop(columns=['_id',
#                       'ACCNUM',
#                       'DATE',
#                       'TIME',
#                       'DATETIME',
#                       'DISTRICT',
#                       'NEIGHBOURHOOD_158',
#                       'HOOD_140',
#                       'NEIGHBOURHOOD_140',
#                       'DIVISION',
#                       'geometry',
#                       'STREET1',
#                       'STREET2',
#                       'OFFSET',
#                       'INJURY',
#                       'FATAL_NO',
#                       'index_right',
#                       'ID',
#                       'NAME',
#                       'ENGLISH_NA'], axis=1)

df = df.drop(columns='index_right', axis=1)

# Filling in missing Road Classes
missing_road_class = df.query('ROAD_CLASS.isna() | ROAD_CLASS == "Pending"')

streets = load_external_data(STREETS_PATH).query('CSDNAME_L == "Toronto"')


plot_map([missing_road_class])

temp = df.query('ROAD_CLASS == "Pending"')


filled_road_class = (
                    gpd.sjoin_nearest(missing_road_class, streets, how='left')
                    #.loc[:, ['STREET1', 'STREET2', 'NAME_right', 'TYPE', 'CLASS']]
                )

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

# Manually correct arterial road classes
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

# Merge back into main dataframe
filled_road_class.loc[filled_road_class['_id'].isin(arterials['_id']), 'CLASS'] = arterials['CLASS']
df.loc[df['_id'].isin(filled_road_class['_id']), 'ROAD_CLASS'] = filled_road_class['CLASS']

# Remove extra space from major arterial road class
df['ROAD_CLASS'] = np.where(
                            df['ROAD_CLASS'] == 'Major Arterial ',
                            'Major Arterial',
                            df['ROAD_CLASS']
                        )


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

