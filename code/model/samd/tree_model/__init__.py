from .tree import TreeModel
from .token_recycle import TokenRecycle
from .eagle import Eagle
from .eagle2 import Eagle2
from typing import Dict, Union
from .eagle3 import Eagle3

tree_model_cls: Dict[
    str, 
    Union[TokenRecycle, Eagle]
] = {
    "token_recycle": TokenRecycle,
    "eagle": Eagle,
    "eagle2": Eagle2,
    "eagle3": Eagle3
}
