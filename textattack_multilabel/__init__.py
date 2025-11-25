# TextAttack-Multilabel Package
"""Multi-label adversarial attack extension for TextAttack."""

__version__ = "0.1.0"

# Core exports
from .multilabel_model_wrapper import MultilabelModelWrapper
from .goal_function import (
    MultilabelClassificationGoalFunction,
    MultilabelClassificationGoalFunctionResult,
)
from .attack_components import (
    AttackList,
    AttackResult_new_diff_color,
    AlzantotGeneticAlgorithm,
    GeneticAlgorithm,
    GreedyWordSwapWIRTruncated,
    PartOfSpeechTry,
)

# Attack recipes
from .multilabel_target_attack_recipe import MultilabelACL23_recipe
from .multilabel_transform_attack_recipe import MultilabelACL23Transform

# Alias for backward compatibility
MultilabelACL23 = MultilabelACL23_recipe

__all__ = [
    "AttackList",
    "AttackResult_new_diff_color",
    "AlzantotGeneticAlgorithm",
    "GeneticAlgorithm",
    "GreedyWordSwapWIRTruncated",
    "MultilabelACL23",
    "MultilabelACL23_recipe",
    "MultilabelACL23Transform",
    "MultilabelClassificationGoalFunction",
    "MultilabelClassificationGoalFunctionResult",
    "MultilabelModelWrapper",
    "PartOfSpeechTry",
]

