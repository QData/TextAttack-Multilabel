import pytest
from unittest.mock import Mock
import torch
from textattack_multilabel.model import MultilabelModelWrapper


def test_multilabel_model_wrapper():
    """Test MultilabelModelWrapper initialization and forward pass."""
    # Mock model, tokenizer
    mock_model = Mock()
    mock_model.eval.return_value = None
    mock_model.to.return_value = mock_model
    mock_model.__call__ = Mock(return_value=Mock(logits=torch.randn(1, 6)))

    mock_tokenizer = Mock()
    mock_tokenizer.return_value = {'input_ids': torch.tensor([[1, 2, 3]]), 'attention_mask': torch.tensor([[1, 1, 1]])}

    # Init wrapper
    wrapper = MultilabelModelWrapper(mock_model, mock_tokenizer, multilabel=True)

    assert wrapper.model == mock_model
    assert wrapper.tokenizer == mock_tokenizer
    assert wrapper.multilabel == True

    # Test prediction
    text = "sample text"
    predictions = wrapper.predict([text])
    assert predictions.shape == (1, 6)  # 6 toxicity labels


def test_model_wrapper_single_label():
    """Test with single-label mode."""
    mock_model = Mock()
    mock_tokenizer = Mock()

    wrapper = MultilabelModelWrapper(mock_model, mock_tokenizer, multilabel=False)
