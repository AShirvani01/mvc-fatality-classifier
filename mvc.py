import requests
from io import StringIO
import pandas as pd


def get_data():
    """Returns the MVC dataset from open data Toronto"""
    base_url = "https://ckan0.cf.opendata.inter.prod-toronto.ca"
    url = base_url + "/api/3/action/package_show"
    params = {
        "id": ("motor-vehicle-collisions-involving-killed-or-"
               "seriously-injured-persons")
        }
    package = requests.get(url, params=params).json()

    resource = package['result']['resources'][0]
    url = base_url + "/datastore/dump/" + resource["id"]
    resource_dump_data = requests.get(url).text
    return StringIO(resource_dump_data)


df = pd.read_csv(get_data())

# EDA
df.head(10)
df.info()
df.isnull().sum()
df.duplicated().sum()
