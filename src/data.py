import pandas as pd
import geopandas as gpd
import requests
from io import StringIO, BytesIO
from zipfile import ZipFile
from shapely.geometry import shape
import json
from pathlib import Path

from config import DATA_DIR


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


def download_streets_data(output_dir: Path = DATA_DIR / "canada_streets"):
    base_url = "https://www12.statcan.gc.ca/census-recensement/2011"
    url = f"{base_url}/geo/RNF-FRR/files-fichiers/lrnf000r24a_e.zip"
    response = requests.get(url, stream=True)

    output_dir.mkdir(exist_ok=True)

    with response as r:
        with ZipFile(BytesIO(r.content)) as zip_file:
            zip_file.extractall(output_dir)

    rename_files(output_dir, 'canada_streets')


def rename_files(path: Path, new_file_name):

    if not path.exists():
        raise FileNotFoundError("File or subdirectory doesn't exist. Check for "
                                "typos and if files have been downloaded.")

    if path.is_file():
        path.rename(path.parent / new_file_name)
    
    if path.is_dir():
        for file in path.iterdir():
            if file.is_file():
                file.rename(path / f"{new_file_name}{file.suffix}")



