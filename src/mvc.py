# General libraries
import pandas as pd
import numpy as np

# To import dataset
import requests
from io import StringIO

# For visuals
import matplotlib.pyplot as plt

# For geodata
import geopandas as gpd
from shapely.geometry import shape
import json
from scipy.spatial import cKDTree


def get_data():
    """Returns the MVC dataset from open data Toronto."""
    base_url = "https://ckan0.cf.opendata.inter.prod-toronto.ca"
    url = f"{base_url}/api/3/action/package_show"
    params = {
        "id": ("motor-vehicle-collisions-involving-killed-or-"
               "seriously-injured-persons")
        }
    package = requests.get(url, params=params).json()

    resource = package['result']['resources'][0]
    url = f"{base_url}/datastore/dump/{resource['id']}"
    resource_dump_data = requests.get(url).text
    return StringIO(resource_dump_data)


df = pd.read_csv(get_data())

# EDA
df.head(10)
df.info()
df.isnull().sum()
df.duplicated().sum()

accident_class_count = df['ACCLASS'].value_counts()
road_class_count = df['ROAD_CLASS'].value_counts()
hood_count = df['NEIGHBOURHOOD_158'].value_counts()
missing_hoods = df.query('NEIGHBOURHOOD_158 == "NSA"')

# plt.hist(df.query('ROAD_CLASS == "Major Arterial"')['HOUR'])
# plt.show()

# DATA PREPROCESSING

# Response

# df.query('ACCLASS == ["Property Damage O", "None"]')
df = df.query('ACCLASS != ["Property Damage O", "None"]')  # Remove rare classes


# Features

# Encoding Date/Time
def encode_datetime(df):
    """Encodes date and time into multiple features.

    Args:
        df: A Pandas DataFrame with DATE and TIME columns.

    Returns:
        The input dataframe with added DATETIME, YEAR, MONTH, Day of the Week,
        and HOUR features.
    """
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

    return df


df = encode_datetime(df)

# Encoding Geo Data & Filling in missing neighbourhoods

shapefile = (
            gpd.read_file('data/toronto_neighbourhoods/neighbourhoods.shp')
            .to_crs(epsg=4326)  # Set coordinate system
    )
shapefile.plot()

df['geometry'] = (
                df['geometry']
                .apply(lambda x: shape(json.loads(x)).wkt)
                .pipe(gpd.GeoSeries.from_wkt)
    )

gdf = (
       gpd.GeoDataFrame(df, crs="EPSG:4326", geometry="geometry")
       .sjoin(shapefile, how="left")
    )

# Filling in neighbourhood missing in shapefile
temp = gdf.query('NEIGHBOURHOOD_158 == "NSA" & NAME.isna()')
gdf['NAME'] = gdf['NAME'].apply(lambda x: 'Morningside Heights' if x != x else x)
gdf['ID'] = gdf['ID'].apply(lambda x: 144 if x != x else x)

cond = (gdf['NEIGHBOURHOOD_158'] == "NSA") & (gdf['NAME'].isna())
gdf.loc[cond, ['HOOD_158', 'NEIGHBOURHOOD_158']] = gdf.loc[cond, ['ID', 'NAME']]


# Adding nearest hospital feature
health_services = gpd.read_file('data/ontario_health_services.geojson')
health_services_type_count = health_services['SERVICE_TY'].value_counts()

# Filter for Toronto hospitals
municipalities = [
                'Scarborough',
                'Toronto',
                'North York',
                'Etobicoke',
                'East York',
                'Mississauga',
                'Brampton',
                'Markham',
                'Thornhill',
                'Vaughan'
    ]

hospital_types = [
                "Hospital - Site",
                "Hospital - Corporation"
    ]

toronto_hospitals = health_services[
                    (health_services['SERVICE_TY'].isin(hospital_types))
                    & (health_services['COMMUNITY'].isin(municipalities))
    ]

hospital_list = [
                    'Humber River Hospital - Wilson',
                    'MacKenzie Health - Cortellucci Vaughan Hospital',
                    'Oak Valley Health - Markham',
                    'North York General Hospital - General Site',
                    'Scarborough Health Network - Birchmount',
                    'Scarborough Health Network - Scarborough General',
                    'Scarborough Health Network - Centenary',
                    'Sinai Health System - Mount Sinai',
                    'Sunnybrook Health Sciences Centre - Bayview Campus',
                    'Toronto East Health Network - Michael Garron Hospital',
                    'Trillium Health Partners- Mississauga',
                    'Trillium Health Partners - Credit Valley',
                    "Unity Health Toronto - St. Joseph's",
                    "Unity Health Toronto - St. Michael's",
                    'University Health Network - Toronto General',
                    'University Health Network - Toronto Western',
                    'William Osler Health System - Etobicoke',
                    'William Osler Health System - Civic'
    ]

toronto_hospitals_with_er = health_services[
                        health_services['ENGLISH_NA'].isin(hospital_list)
    ]


def dist_to_nearest_hospital(gdf_A, gdf_B):
    """Finds distance to nearest hospital from location of collisions.

    Args:
        gdf_A: GeoDataFrame of the collisions.
        gdf_B: GeoDataFrame of the hospitals.

    Returns:
        The input GeoDataFrame A with added DIST_TO_HOS feature.
    """
    nA = np.array(list(gdf_A['geometry'].apply(lambda x: (x.x, x.y))))
    nB = np.array(list(gdf_B['geometry'].apply(lambda x: (x.x, x.y))))
    btree = cKDTree(nB)
    dist, idx = btree.query(nA, k=1, p=1)  # Manhattan distance
    gdf_B_nearest = (
                    gdf_B.iloc[idx]
                    .drop(columns='geometry')
                    .reset_index(drop=True)
        )
    gdf = pd.concat([gdf_A.reset_index(drop=True),
                     gdf_B_nearest['ENGLISH_NA'],
                     pd.Series(dist, name='DIST_TO_HOS')],
                    axis=1)

    return gdf


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


streets = (
            gpd.read_file('data/canada_roads/canada_roads.shp')
            .query('CSDNAME_L == "Toronto"')
            .to_crs(epsg=4326)  # Set coordinate system
    )

streets.plot()


def plot_map(gdf: list[gpd.GeoDataFrame], colours=['red'], plot_all=False):
    fig, ax = plt.subplots(figsize=(100, 100))
    shapefile.plot(ax=ax, color='darkblue')
    streets.plot(ax=ax, alpha=0.7, color='orange')
    if plot_all:
        df.plot(ax=ax, alpha=0.2, color='red', markersize=100)
    for i,x in enumerate(gdf):
        x.plot(ax=ax, color=colours[i], markersize=100)
    plt.axis('off')
    plt.show()


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

