from loss_functions import *
from loss_wrappers import loss_wrapper, cat_wrapper
import optuna
from constants import min_optimal_metrics, max_optimal_metrics


def is_max_optimal(metric: str) -> bool:
    metric = metric.lower().strip()

    if metric in max_optimal_metrics:
        return True
    elif metric in min_optimal_metrics:
        return False
    else:
        ValueError(f'Unknown metric: {metric}')


def create_objective(objective_name, y_true, params, algorithm):

    if objective_name == 'LDAM':
        loss = LDAM_loss(y_true, params['LDAM_max_m'])
        loss.get_init_score()

    # Wrappers
    if algorithm == 'XGBoost':
        objective = loss_wrapper(loss)

    elif algorithm == 'CatBoost':
        objective = cat_wrapper(loss)

    return objective


def unpack_params(trial: optuna.Trial, config, algorithm, y_true):
    params = {}

    for param, value in config.params.model_dump().items():
        if type(value) is list:
            distribution = trial.suggest_categorical(param, value)
        elif type(value) is tuple:
            if type(value[0]) is float:
                distribution = trial.suggest_float(param, *value)
            else:
                distribution = trial.suggest_int(param, *value)
        else:
            distribution = value

        params[param] = distribution

    # Custom Loss
    custom_names = ['LDAM', 'EQ']

    if params['objective'] in custom_names:
        params.pop('scale_pos_weight')
        objective = create_objective(params['objective'], y_true, params, algorithm)

        if algorithm == 'XGBoost':
            config.obj = objective.get_gradient
            params.pop('objective')

            if params['eval_metric'] in custom_names:
                config.custom_metric = objective.get_metric
                params.pop('eval_metric')

        if algorithm == 'CatBoost':
            params['objective'] = objective
            if params['eval_metric'] in custom_names:
                params['eval_metric'] = objective

    # Remove unnecessary custom params
    custom_params = ['LDAM_max_m', 'EQ_gamma', 'EQ_mu']

    for param in custom_params:
        params.pop(param)

    return params, config
