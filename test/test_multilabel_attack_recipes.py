import pytest
from unittest.mock import Mock, MagicMock
from example_toxic_adv_examples.multilabel_acl2023 import MultilabelACL23


def test_build_multilabel_acl23():
    """Test building MultilabelACL23 attack recipe."""
    # Mock model_wrapper
    mock_model_wrapper = Mock()
    mock_model_wrapper.model_max_length = 128

    # Test build
    attack = MultilabelACL23.build(
        model_wrapper=mock_model_wrapper,
        labels_to_maximize=[1],  # maximize toxic label
        labels_to_minimize=[],   # no minimization
        maximize_target_score=0.8,
        minimize_target_score=0.2,
        wir_method="weighted-saliency"
    )

    assert attack is not None
    assert hasattr(attack, 'goal')
    assert hasattr(attack, 'constraints')
    assert hasattr(attack, 'transformation')
    assert hasattr(attack, 'search_method')


def test_build_multilabel_acl23_transform():
    """Test building MultilabelACL23Transform attack recipe."""
    mock_model_wrapper = Mock()
    mock_model_wrapper.model_max_length = 128

    attack = MultilabelACL23.MultilabelACL23Transform.build(
        model_wrapper=mock_model_wrapper,
        labels_to_maximize=[0],  # maximize first label
        labels_to_minimize=[1],  # minimize second label
        transform_method="wordnet",
        wir_method="delete"
    )

