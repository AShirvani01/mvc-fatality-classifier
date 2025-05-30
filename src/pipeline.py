from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd

from data import (
    get_collision_data,
    convert_to_geodata,
    load_external_data,
    download_hospital_data,
    download_streets_data,
    download_neighbourhood_data
)
from config import (
    HEALTH_SERVICES_PATH,
    STREETS_PATH,
    NEIGHBOURHOODS_PATH,
    DATA_DIR
)



class MVCFatClassPipeline:

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir
        self.collisions = None
        self.hospitals = None
        self.neighbourhoods = None
        self.streets = None

    def _fetch_data(self):
        # Collision Data
        raw_collision_data = get_collision_data()
        self.collisions = convert_to_geodata(raw_collision_data)

        # Hospital Data
        download_hospital_data(self.data_dir)
        self.hospitals = (
            load_external_data(HEALTH_SERVICES_PATH)
            .pipe(filter_toronto_hospitals_with_er)
        )

        # Neighbourhood Data
        download_neighbourhood_data(self.data_dir)
        self.neighbourhood = load_external_data(NEIGHBOURHOODS_PATH)

        # Street Data
        download_streets_data(self.data_dir / 'canada_streets')
        self.streets = (
            load_external_data(STREETS_PATH)
            .query('CSDNAME_L == "Toronto"')
        )


        
        