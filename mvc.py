# To import dataset
import requests
from io import StringIO
import pandas as pd

# For geodata
import geopandas as gpd
from shapely.geometry import shape
import json


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
                    .dt.round(freq='1H')  # Round to nearest hour
        )

    df['YEAR'] = df['DATETIME'].dt.year
    df['MONTH'] = df['DATETIME'].dt.month_name()
    df['DOW'] = df['DATETIME'].dt.day_name()
    df['HOUR'] = df['DATETIME'].dt.hour

    return df


# Encoding Geo Data & Filling in missing neighbourhoods

shapefile = (
            gpd.read_file('toneighshape/Neighbourhoods_v2_region.shp')
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



health_services = gpd.read_file('Ministry_of_Health_service_provider_locations.geojson')
health_services_type_count = health_services['SERVICE_TY'].value_counts()

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
