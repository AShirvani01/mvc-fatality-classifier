from pathlib import Path


def get_project_root():
    return Path(__file__).parent.parent.resolve()


DATA_DIR = get_project_root() / 'data'

HEALTH_SERVICES_PATH = DATA_DIR / 'ontario_health_services.geojson'
NEIGHBOURHOODS_PATH = DATA_DIR / 'toronto_neighbourhoods.geojson'
STREETS_PATH = DATA_DIR / 'canada_streets' / 'canada_streets.shp'
