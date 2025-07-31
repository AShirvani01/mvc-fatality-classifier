from enum import Enum

MUNICIPALITIES = [
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

HOSPITAL_LIST = [
    'Humber River Health - Wilson',
    'MacKenzie Health - Cortellucci Vaughan Hospital',
    'North York General Hospital - General Site',
    'Oak Valley Health - Markham',
    'Scarborough Health Network - Birchmount',
    'Scarborough Health Network - Scarborough General',
    'Scarborough Health Network - Centenary',
    'Sunnybrook Health Sciences Centre - Bayview Campus',
    'Michael Garron Hospital',
    'Trillium Health Partners- Mississauga',
    'Trillium Health Partners - Credit Valley',
    "Unity Health Toronto - St. Joseph's",
    "Unity Health Toronto - St. Michael's",
    'University Health Network - Toronto General',
    'University Health Network - Toronto Western',
    'William Osler Health System - Etobicoke',
    'William Osler Health System - Civic'
]

FEATURES_TO_DROP = [
    'DATETIME',
    'geometry',
    'ACCNUM',
    '_id',
    'STREET1',
    'STREET2',
    'OFFSET',
    'ACCLOC',
    'INJURY',
    'FATAL_NO',
    'INITDIR',
    'VEHTYPE',
    'DRIVCOND',
    'INVTYPE',
    'PEDTYPE',
    'PEDACT',
    'PEDCOND',
    'CYCLISTYPE',
    'CYCACT',
    'CYCCOND',
    'HOOD_158',
    'HOOD_140',
    'NEIGHBOURHOOD_140',
    'DIVISION'
]

CAT_FEATURES = [
    'ROAD_CLASS',
    'DISTRICT',
    'TRAFFCTL',
    'VISIBILITY',
    'LIGHT',
    'RDSFCOND',
    'IMPACTYPE',
    'NEIGHBOURHOOD_158',
    'NEAREST_HOSPITAL'
]


class Algorithm(Enum):
    CATBOOST = 'CatBoost'
    XGBOOST = 'XGBoost'
    LIGHTGBM = 'LightGBM'


MAX_OPTIMAL_METRICS = {
    'accuracy',
    'f1',
    'recall',
    'auc',
    'aucpr',
    'prauc',
    'mcc',
    'wkappa',
    'average_precision'
}

MIN_OPTIMAL_METRICS = {
    'error',
    'logloss',
    'brierscore',
    'focal'
}

CUSTOM_OBJECTIVES = {
    'LDAM',
    'Focal',
    'LA'
}

CUSTOM_PARAMS = {
    'LDAM_max_m',
    'Focal_gamma',
    'LA_tau'
}