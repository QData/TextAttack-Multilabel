import pytest
from unittest.mock import Mock

from textattack_multilabel.attack_components import GreedyWordSwapWIRTruncated
from textattack_multilabel.goal_function import MultilabelClassificationGoalFunction


def test_multilabel_goal_function():
    """Test MultilabelClassificationGoalFunction initializes correctly."""
    mock_model_wrapper = Mock()

    goal_func = MultilabelClassificationGoalFunction(
        model_wrapper=mock_model_wrapper,
        labels_to_maximize=[0, 1],
        labels_to_minimize=[2],
        maximize_target_score=0.8,
        minimize_target_score=0.2
    )

    assert goal_func.labels_to_maximize == [0, 1]
    assert goal_func.labels_to_minimize == [2]


def test_greedy_word_swap_wir_truncated():
    """Test GreedyWordSwapWIRTruncated initialization."""
    search_method = GreedyWordSwapWIRTruncated(wir_method="delete", truncate_words_to=50)

    assert search_method.wir_method == "delete"
