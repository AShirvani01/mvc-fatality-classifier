import pandas as pd
import geopandas as gpd
import requests
from io import StringIO
from shapely.geometry import shape
import json

from pathlib import Path



def get_collision_data() -> pd.DataFrame:
    """Pulls the collision dataset from the Open Data Toronto API.
    
    Returns:
        pd.DataFrame: Dataframe where each row is a unique collision and each
        column corresponds to a feature.
    """
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
    collision_data = pd.read_csv(StringIO(resource_dump_data))
    
    return collision_data


def convert_to_geodata(df: pd.DataFrame, crs="EPSG:4326") -> gpd.GeoDataFrame:
    """Convert Dataframe to Geodataframe.
    
    Args:
        df (pd.DataFrame): Dataframe of collision data with 'geometry' column
            in GeoJSON format.
            e.g. {"type": "Point", "coordinates": [-79.3187970, 43.6995952]}
        crs: Coordinate Reference System of the data.
    
    Returns:
        gpd.GeoDataFrame: Geodataframe with geometry column in WKT format.
    """
    # Convert geometry column from GeoJSON -> WKT -> GeoSeries
    df['geometry'] = (
        df['geometry']
        .apply(lambda x: shape(json.loads(x)).wkt)
        .pipe(gpd.GeoSeries.from_wkt)
    )

    gdf = gpd.GeoDataFrame(df, crs=crs, geometry="geometry")
    
    return gdf


def load_external_data(file_path: Path, crs="EPSG:4326") -> gpd.GeoDataFrame:
    """Load external data.
    
    Args:
        file_path (Path): File path to external data
        crs: Coordinate Reference System of the data.
    
    Returns:
        gpd.GeoDataFrame: Geodataframe with geometry column in WKT format.
    """
    
    gdf = gpd.read_file(file_path).to_crs(crs=crs)
    
    return gdf
    
    
    
    
    
    
