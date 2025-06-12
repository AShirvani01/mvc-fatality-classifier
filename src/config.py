from pathlib import Path
from typing import Union, Literal, Annotated, ClassVar
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    NonNegativeFloat
)


def get_project_root():
    return Path(__file__).parent.parent.resolve()


DATA_DIR = get_project_root() / 'data'

HEALTH_SERVICES_PATH = DATA_DIR / 'ontario_health_services.geojson'
NEIGHBOURHOODS_PATH = DATA_DIR / 'toronto_neighbourhoods.geojson'
STREETS_PATH = DATA_DIR / 'canada_streets' / 'canada_streets.shp'


class CBParams(BaseModel):
    """CatBoost model training parameters. For details, see
    [CatBoost's documentation](
    https://catboost.ai/docs/en/references/training-parameters/).

    Args:
        objective (Literal): Loss function in training. Defaults to Logloss.
        eval_metric (Literal): Metric to compare models. Defaults to Accuracy.
        auto_class_weights (Literal): Class weights.
        depth (int): Limit the max depth for tree model.
        l2_leaf_reg (float): Coefficient of the L2 regularization term.
        learning_rate (float): Learning rate.
        rsm (float): Random subspace method. The percentage of features to use
            at each split selection.
    """
    # Type alias
    Objective: ClassVar[type] = Literal['Logloss']
    EvalMetric: ClassVar[type] = Literal['Logloss', 'Accuracy', 'F1', 'Recall', 'AUC']
    AutoClassWeights: ClassVar[type] = Literal['None', 'Balanced', 'SqrtBalanced']
    Depth: ClassVar[type] = Annotated[int, Field(ge=1, le=16)]
    L2LeafReg: ClassVar[type] = NonNegativeFloat
    LearningRate: ClassVar[type] = Annotated[float, Field(gt=0, le=1)]
    RSM: ClassVar[type] = Annotated[float, Field(gt=0, le=1)]

    # Params/Defining defaults
    objective: Objective | list[Objective] = 'Logloss'
    eval_metric: EvalMetric | list[EvalMetric] = 'Accuracy'
    auto_class_weights: AutoClassWeights | list[AutoClassWeights] = 'Balanced'
    depth: Depth | tuple[Depth, Depth] | list[Depth] = (1, 16)
    l2_leaf_reg: L2LeafReg | tuple[L2LeafReg, L2LeafReg] | list[L2LeafReg] = (0., 100.)
    learning_rate: LearningRate | tuple[LearningRate, LearningRate] | list[LearningRate] = (0.01, 0.1)
    rsm: RSM | tuple[RSM, RSM] | list[RSM] = (0.1, 1)

    model_config = ConfigDict(extra='forbid')


class CBModelConfig(BaseModel):
    """CatBoost Model configuration.

    Args:
        early_stopping_rounds (int): If provided, stop training if one
            metric of one validation data doesn't improve in the last
            `early_stopping_round` rounds. Defaults to 100.
        fold_count (int): Number of different cross validation folds to train.
            Defaults to 5.
        num_boost_round (int): Number of boosting iterations. Defaults to 10000.
        seed (int): The random seed used for training. Defaults to 42.
        verbose (Union[bool, int]): show log for training. Defaults to 100.
    """

    params: CBParams = CBParams()
    early_stopping_rounds: int = 100
    fold_count: int = 5
    num_boost_round: int = 10_000
    seed: int = 42
    verbose: Union[bool, int] = False

    model_config = ConfigDict(extra='forbid')


class XGBParams(BaseModel):
    """XGBoost model training parameters. For details, see
    [XGBoost's documentation](
    https://xgboost.readthedocs.io/en/stable/parameter.html).

    Args:
        objective (Literal): Loss function in training. Defaults to Logloss.
        eval_metric (Literal): Metric to compare models. Defaults to Accuracy.
        max_depth (int): Limit the max depth for tree model.
        reg_lambda (float): Coefficient of the L2 regularization term.
        learning_rate (float): Learning rate.
        subsample (float): Random subspace method. The percentage of features
            to use at each split selection.
    """
    # Type alias
    Objective: ClassVar[type] = Literal['binary:logistic']
    EvalMetric: ClassVar[type] = Literal['logloss', 'error', 'auc']
    MaxDepth: ClassVar[type] = Annotated[int, Field(ge=1, le=16)]
    RegLambda: ClassVar[type] = NonNegativeFloat
    LearningRate: ClassVar[type] = Annotated[float, Field(gt=0, le=1)]
    SubSample: ClassVar[type] = Annotated[float, Field(gt=0, le=1)]

    # Params/Defining defaults
    objective: Objective | list[Objective] = 'binary:logistic'
    eval_metric: EvalMetric | list[EvalMetric] = 'error'
    max_depth: MaxDepth | tuple[MaxDepth, MaxDepth] | list[MaxDepth] = (1, 16)
    reg_lambda: RegLambda | tuple[RegLambda, RegLambda] | list[RegLambda] = (0., 100.)
    learning_rate: LearningRate | tuple[LearningRate, LearningRate] | list[LearningRate] = (0.01, 0.1)
    subsample: SubSample | tuple[SubSample, SubSample] | list[SubSample] = (0.1, 1)

    model_config = ConfigDict(extra='forbid')


class XGBModelConfig(BaseModel):
    """XGBoost Model configuration.

    Args:
        early_stopping_rounds (int): If provided, stop training if one
            metric of one validation data doesn't improve in the last
            `early_stopping_round` rounds. Defaults to 100.
        nfold (int): Number of different cross validation folds to train.
            Defaults to 5.
        num_boost_round (int): Number of boosting iterations. Defaults to 10000.
        seed (int): The random seed used for training. Defaults to 42.
        verbosity (Union[bool, int]): show log for training. Defaults to 100.
    """

    params: XGBParams = XGBParams()
    early_stopping_rounds: int = 100
    nfold: int = 5
    num_boost_round: int = 10_000
    seed: int = 42
    verbose_eval: Union[bool, int] = False

    model_config = ConfigDict(extra='forbid')