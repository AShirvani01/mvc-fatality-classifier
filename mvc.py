import requests
from io import StringIO
import pandas as pd


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
road_class_count = df['ROAD_CLASS'].value_counts()
hood_count = df['NEIGHBOURHOOD_158'].value_counts()
missing_neighborhoods = df.query('NEIGHBOURHOOD_158 == "NSA"')

# DATA PREPROCESSING

# Response
df['ACCLASS'].value_counts()
# df.query('ACCLASS == ["Property Damage O", "None"]')
df = df.query('ACCLASS != ["Property Damage O", "None"]')


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
