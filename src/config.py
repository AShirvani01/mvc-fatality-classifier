from pathlib import Path


def get_project_root():
    return Path(__file__).parent.parent.resolve()


data_dir = get_project_root() / 'data'

HEALTH_SERVICES_PATH = data_dir / 'ontario_health_services.geojson'
NEIGHBOURHOODS_PATH = data_dir / 'toronto_neighbourhoods' / 'neighbourhoods.shp'
ROADS_PATH = data_dir / 'canada_roads' / 'canada_roads.shp'
