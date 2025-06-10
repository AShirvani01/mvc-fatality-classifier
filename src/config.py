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
    depth: Depth | tuple[Depth, Depth] = (1, 16)
    l2_leaf_reg: L2LeafReg | tuple[L2LeafReg, L2LeafReg] = (0., 15.)
    learning_rate: LearningRate | tuple[LearningRate, LearningRate] = (0.01, 0.1)
    rsm: RSM | tuple[RSM, RSM] = (0.1, 1)

    model_config = ConfigDict(extra='forbid')

    # Validation
    @field_validator('depth', 'l2_leaf_reg', 'learning_rate', 'rsm')
    @classmethod
    def validate_range_order(cls, field):
        if type(field) is tuple:
            if field[0] > field[1]:
                raise ValueError(f'Min {field} must be <= Max {field}.')
        return field



class CBModelConfig(BaseModel):
    """CatBoost Model configuration.

    Args:
        early_stopping_round (int): If provided, stop training if one
            metric of one validation data doesn't improve in the last 
            `early_stopping_round` rounds. Defaults to 100.
        fold_count (int): Number of different cross validation folds to train.
            Defaults to 5.
        num_boost_round (int): Number of boosting iterations. Defaults to 10000.
        seed (int): The random seed used for training. Defaults to 42.
        verbose (Union[bool, int]): show log for training. Defaults to 100.
    """

    early_stopping_rounds: int = 100
    fold_count: int = 5
    num_boost_round: int = 10_000
    seed: int = 42
    verbose: Union[bool, int] = 100

    model_config = ConfigDict(extra='forbid')
