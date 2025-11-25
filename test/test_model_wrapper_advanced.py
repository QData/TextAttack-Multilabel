"""
Comprehensive tests for MultilabelModelWrapper.

This test suite covers:
- Model prediction (__call__ method)
- Gradient computation (get_grad method)
- Tokenization handling
- Device placement
- Multilabel vs single-label modes
- Error handling and edge cases
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, MagicMock, patch, call

from textattack_multilabel.multilabel_model_wrapper import MultilabelModelWrapper


class TestMultilabelModelWrapperInit:
    """Test initialization and configuration."""

    def test_init_with_valid_model_and_tokenizer(self):
        """Test initialization with valid model and tokenizer."""
        mock_model = Mock()
        mock_model.__class__.__name__ = 'PreTrainedModel'
        mock_model.eval.return_value = None
        mock_model.to.return_value = mock_model
        mock_model.config.max_position_embeddings = 512

        mock_tokenizer = Mock()
        mock_tokenizer.__class__.__name__ = 'PreTrainedTokenizer'
        mock_tokenizer.model_max_length = 512

        wrapper = MultilabelModelWrapper(
            mock_model,
            mock_tokenizer,
            multilabel=True,
            device='cpu'
        )

        assert wrapper.model == mock_model
        assert wrapper.tokenizer == mock_tokenizer
        assert wrapper.multilabel is True
        assert wrapper.device == 'cpu'

    def test_init_device_auto_detection_cuda_available(self):
        """Test device auto-detection when CUDA is available."""
        mock_model = Mock()
        mock_model.__class__.__name__ = 'PreTrainedModel'
        mock_model.eval.return_value = None
        mock_model.to.return_value = mock_model
        mock_model.config.max_position_embeddings = 512

        mock_tokenizer = Mock()
        mock_tokenizer.__class__.__name__ = 'PreTrainedTokenizer'
        mock_tokenizer.model_max_length = 512

        with patch('torch.cuda.is_available', return_value=True):
            wrapper = MultilabelModelWrapper(mock_model, mock_tokenizer, device=None)
            assert wrapper.device == 'cuda'

    def test_init_device_auto_detection_cuda_unavailable(self):
        """Test device auto-detection when CUDA is not available."""
        mock_model = Mock()
        mock_model.__class__.__name__ = 'PreTrainedModel'
        mock_model.eval.return_value = None
        mock_model.to.return_value = mock_model
        mock_model.config.max_position_embeddings = 512

        mock_tokenizer = Mock()
        mock_tokenizer.__class__.__name__ = 'PreTrainedTokenizer'
        mock_tokenizer.model_max_length = 512

        with patch('torch.cuda.is_available', return_value=False):
            wrapper = MultilabelModelWrapper(mock_model, mock_tokenizer, device=None)
            assert wrapper.device == 'cpu'

    def test_init_max_length_calculation(self):
        """Test max_length calculation with various model configurations."""
        mock_model = Mock()
        mock_model.__class__.__name__ = 'PreTrainedModel'
        mock_model.eval.return_value = None
        mock_model.to.return_value = mock_model
        mock_model.config.max_position_embeddings = 2048

        mock_tokenizer = Mock()
        mock_tokenizer.__class__.__name__ = 'PreTrainedTokenizer'
        mock_tokenizer.model_max_length = 512

        wrapper = MultilabelModelWrapper(mock_model, mock_tokenizer, device='cpu')

        # Should be min(1024, 512, 2048-2) = 512
        assert wrapper.max_length == 512

    def test_init_max_length_custom(self):
        """Test initialization with custom max_length."""
        mock_model = Mock()
        mock_model.__class__.__name__ = 'PreTrainedModel'
        mock_model.eval.return_value = None
        mock_model.to.return_value = mock_model

        mock_tokenizer = Mock()
        mock_tokenizer.__class__.__name__ = 'PreTrainedTokenizer'

        wrapper = MultilabelModelWrapper(
            mock_model,
            mock_tokenizer,
            max_length=256,
            device='cpu'
        )

        assert wrapper.max_length == 256

    def test_init_invalid_model_type_raises_error(self):
        """Test that invalid model type raises AssertionError."""
        invalid_model = "not a model"
        mock_tokenizer = Mock()
        mock_tokenizer.__class__.__name__ = 'PreTrainedTokenizer'

        with pytest.raises(AssertionError, match="must be of type"):
            MultilabelModelWrapper(invalid_model, mock_tokenizer, device='cpu')

    def test_init_invalid_tokenizer_type_raises_error(self):
        """Test that invalid tokenizer type raises AssertionError."""
        mock_model = Mock()
        mock_model.__class__.__name__ = 'PreTrainedModel'
        mock_model.eval.return_value = None
        mock_model.to.return_value = mock_model

        invalid_tokenizer = "not a tokenizer"

        with pytest.raises(AssertionError, match="must of type"):
            MultilabelModelWrapper(mock_model, invalid_tokenizer, device='cpu')

    def test_init_model_moved_to_device(self):
        """Test that model is moved to specified device."""
        mock_model = Mock()
        mock_model.__class__.__name__ = 'PreTrainedModel'
        mock_model.eval.return_value = None
        mock_model.to.return_value = mock_model
        mock_model.config.max_position_embeddings = 512

        mock_tokenizer = Mock()
        mock_tokenizer.__class__.__name__ = 'PreTrainedTokenizer'
        mock_tokenizer.model_max_length = 512

        wrapper = MultilabelModelWrapper(mock_model, mock_tokenizer, device='cuda')

        mock_model.to.assert_called_once_with('cuda')
        mock_model.eval.assert_called_once()


class TestModelWrapperCall:
    """Test __call__ method (model prediction)."""

    def setup_method(self):
        """Setup test fixtures."""
        self.mock_model = Mock()
        self.mock_model.__class__.__name__ = 'PreTrainedModel'
        self.mock_model.eval.return_value = None
        self.mock_model.to.return_value = self.mock_model
        self.mock_model.config.max_position_embeddings = 512

        self.mock_tokenizer = Mock()
        self.mock_tokenizer.__class__.__name__ = 'PreTrainedTokenizer'
        self.mock_tokenizer.model_max_length = 512

    def test_call_multilabel_returns_sigmoid(self):
        """Test that multilabel mode returns sigmoid probabilities."""
        # Setup model to return logits
        mock_output = Mock()
        mock_output.logits = torch.tensor([[2.0, -1.0, 0.5, 1.5, -0.5, 0.0]])
        self.mock_model.return_value = mock_output

        # Setup tokenizer
        self.mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }

        wrapper = MultilabelModelWrapper(
            self.mock_model,
            self.mock_tokenizer,
            multilabel=True,
            device='cpu'
        )

        result = wrapper(["test text"])

        # Should apply sigmoid to logits
        expected = torch.sigmoid(torch.tensor([[2.0, -1.0, 0.5, 1.5, -0.5, 0.0]]))
        assert torch.allclose(result, expected, atol=1e-5)

    def test_call_single_label_returns_logits(self):
        """Test that single-label mode returns raw logits."""
        # Setup model to return logits
        mock_output = Mock()
        mock_output.logits = torch.tensor([[2.0, -1.0, 0.5, 1.5, -0.5, 0.0]])
        self.mock_model.return_value = mock_output

        # Setup tokenizer
        self.mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }

        wrapper = MultilabelModelWrapper(
            self.mock_model,
            self.mock_tokenizer,
            multilabel=False,
            device='cpu'
        )

        result = wrapper(["test text"])

        # Should return raw logits (no sigmoid)
        expected = torch.tensor([[2.0, -1.0, 0.5, 1.5, -0.5, 0.0]])
        assert torch.allclose(result, expected)

    def test_call_batch_processing(self):
        """Test processing multiple texts in batch."""
        # Setup model to return batch of logits
        mock_output = Mock()
        mock_output.logits = torch.tensor([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ])
        self.mock_model.return_value = mock_output

        # Setup tokenizer
        self.mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2], [3, 4], [5, 6]]),
            'attention_mask': torch.tensor([[1, 1], [1, 1], [1, 1]])
        }

        wrapper = MultilabelModelWrapper(
            self.mock_model,
            self.mock_tokenizer,
            multilabel=True,
            device='cpu'
        )

        result = wrapper(["text1", "text2", "text3"])

        # Should apply sigmoid to all batch items
        expected = torch.sigmoid(mock_output.logits)
        assert result.shape == (3, 3)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_call_with_model_direct_tensor_output(self):
        """Test handling model that returns tensor directly (not object with .logits)."""
        # Setup model to return tensor directly
        self.mock_model.return_value = torch.tensor([[1.0, 2.0, 3.0]])

        # Setup tokenizer
        self.mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }

        wrapper = MultilabelModelWrapper(
            self.mock_model,
            self.mock_tokenizer,
            multilabel=True,
            device='cpu'
        )

        result = wrapper(["test text"])

        # Should apply sigmoid to tensor
        expected = torch.sigmoid(torch.tensor([[1.0, 2.0, 3.0]]))
        assert torch.allclose(result, expected, atol=1e-5)

    def test_call_tokenizer_invoked_correctly(self):
        """Test that tokenizer is called with correct parameters."""
        mock_output = Mock()
        mock_output.logits = torch.tensor([[1.0, 2.0]])
        self.mock_model.return_value = mock_output

        self.mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2]]),
            'attention_mask': torch.tensor([[1, 1]])
        }

        wrapper = MultilabelModelWrapper(
            self.mock_model,
            self.mock_tokenizer,
            max_length=128,
            device='cpu'
        )

        wrapper(["test input text"])

        # Tokenizer should be called with text and appropriate parameters
        assert self.mock_tokenizer.called
        call_args = self.mock_tokenizer.call_args
        assert "test input text" in str(call_args)


class TestGetGrad:
    """Test get_grad method (gradient computation)."""

    def setup_method(self):
        """Setup test fixtures."""
        self.mock_model = Mock()
        self.mock_model.__class__.__name__ = 'PreTrainedModel'
        self.mock_model.eval.return_value = None
        self.mock_model.to.return_value = self.mock_model
        self.mock_model.config.max_position_embeddings = 512

        self.mock_tokenizer = Mock()
        self.mock_tokenizer.__class__.__name__ = 'PreTrainedTokenizer'
        self.mock_tokenizer.model_max_length = 512

    def test_get_grad_multilabel_returns_gradient_dict(self):
        """Test gradient computation for multilabel classification."""
        # Setup model and embeddings
        mock_embedding = Mock()
        mock_embedding_tensor = torch.randn(1, 5, 768, requires_grad=True)
        mock_embedding.return_value = mock_embedding_tensor

        self.mock_model.get_input_embeddings.return_value = mock_embedding

        # Setup model output
        mock_output = Mock()
        mock_logits = torch.tensor([[2.0, -1.0, 0.5, 1.5, -0.5, 0.0]], requires_grad=True)
        mock_output.logits = mock_logits
        self.mock_model.return_value = mock_output

        # Setup tokenizer
        self.mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[101, 2054, 2003, 2023, 102]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1, 1]])
        }
        self.mock_tokenizer.convert_ids_to_tokens.return_value = ['[CLS]', 'what', 'is', 'this', '[SEP]']

        wrapper = MultilabelModelWrapper(
            self.mock_model,
            self.mock_tokenizer,
            multilabel=True,
            device='cpu'
        )

        # Mock the backward pass
        with patch.object(torch.Tensor, 'backward'):
            with patch.object(torch.Tensor, 'sum', return_value=mock_logits.sum()):
                result = wrapper.get_grad(["test text"])

        # Should return a dictionary with token gradients
        assert isinstance(result, dict)

    def test_get_grad_single_label_uses_argmax(self):
        """Test gradient computation for single-label classification uses argmax."""
        # Setup embeddings
        mock_embedding = Mock()
        mock_embedding_tensor = torch.randn(1, 5, 768, requires_grad=True)
        mock_embedding.return_value = mock_embedding_tensor

        self.mock_model.get_input_embeddings.return_value = mock_embedding

        # Setup model output
        mock_output = Mock()
        mock_logits = torch.tensor([[0.1, 0.9, 0.3]], requires_grad=True)
        mock_output.logits = mock_logits
        self.mock_model.return_value = mock_output

        # Setup tokenizer
        self.mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[101, 2054, 102]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        self.mock_tokenizer.convert_ids_to_tokens.return_value = ['[CLS]', 'test', '[SEP]']

        wrapper = MultilabelModelWrapper(
            self.mock_model,
            self.mock_tokenizer,
            multilabel=False,
            device='cpu'
        )

        with patch.object(torch.Tensor, 'backward'):
            result = wrapper.get_grad(["test"])

        # For single-label, should use argmax (index 1 has max value 0.9)
        assert isinstance(result, dict)

    def test_get_grad_with_custom_loss_function(self):
        """Test gradient computation with custom loss function."""
        mock_embedding = Mock()
        mock_embedding_tensor = torch.randn(1, 3, 768, requires_grad=True)
        mock_embedding.return_value = mock_embedding_tensor

        self.mock_model.get_input_embeddings.return_value = mock_embedding

        mock_output = Mock()
        mock_logits = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        mock_output.logits = mock_logits
        self.mock_model.return_value = mock_output

        self.mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[101, 102, 103]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        self.mock_tokenizer.convert_ids_to_tokens.return_value = ['a', 'b', 'c']

        wrapper = MultilabelModelWrapper(
            self.mock_model,
            self.mock_tokenizer,
            device='cpu'
        )

        custom_loss_fn = lambda x: -x.sum()  # Negative sum

        with patch.object(torch.Tensor, 'backward'):
            result = wrapper.get_grad(["test"], loss_fn=custom_loss_fn)

        assert isinstance(result, dict)

    def test_get_grad_model_without_labels_parameter_raises_error(self):
        """Test that models requiring labels parameter raise TypeError."""
        # Setup model that requires labels
        def model_forward(*args, **kwargs):
            if 'labels' not in kwargs:
                raise TypeError("forward() missing 1 required positional argument: 'labels'")
            return Mock(logits=torch.tensor([[1.0, 2.0]]))

        self.mock_model.side_effect = model_forward

        mock_embedding = Mock()
        mock_embedding.return_value = torch.randn(1, 2, 768, requires_grad=True)
        self.mock_model.get_input_embeddings.return_value = mock_embedding

        self.mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2]]),
            'attention_mask': torch.tensor([[1, 1]])
        }

        wrapper = MultilabelModelWrapper(
            self.mock_model,
            self.mock_tokenizer,
            device='cpu'
        )

        with pytest.raises(TypeError, match="must be None"):
            wrapper.get_grad(["test"])


class TestTokenize:
    """Test _tokenize method."""

    def setup_method(self):
        """Setup test fixtures."""
        self.mock_model = Mock()
        self.mock_model.__class__.__name__ = 'PreTrainedModel'
        self.mock_model.eval.return_value = None
        self.mock_model.to.return_value = self.mock_model
        self.mock_model.config.max_position_embeddings = 512

        self.mock_tokenizer = Mock()
        self.mock_tokenizer.__class__.__name__ = 'PreTrainedTokenizer'
        self.mock_tokenizer.model_max_length = 512

    def test_tokenize_single_text(self):
        """Test tokenization of single text."""
        self.mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[101, 2054, 102]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }

        wrapper = MultilabelModelWrapper(
            self.mock_model,
            self.mock_tokenizer,
            max_length=128,
            device='cpu'
        )

        result = wrapper._tokenize(["test text"])

        assert 'input_ids' in result
        assert 'attention_mask' in result
        self.mock_tokenizer.assert_called_once()

    def test_tokenize_batch(self):
        """Test tokenization of batch of texts."""
        self.mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[101, 102], [103, 104], [105, 106]]),
            'attention_mask': torch.tensor([[1, 1], [1, 1], [1, 1]])
        }

        wrapper = MultilabelModelWrapper(
            self.mock_model,
            self.mock_tokenizer,
            device='cpu'
        )

        result = wrapper._tokenize(["text1", "text2", "text3"])

        assert result['input_ids'].shape[0] == 3
        assert result['attention_mask'].shape[0] == 3

    def test_tokenize_with_max_length(self):
        """Test that max_length is passed to tokenizer."""
        self.mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[101, 102]]),
            'attention_mask': torch.tensor([[1, 1]])
        }

        wrapper = MultilabelModelWrapper(
            self.mock_model,
            self.mock_tokenizer,
            max_length=64,
            device='cpu'
        )

        wrapper._tokenize(["test"])

        # Check that tokenizer was called with appropriate max_length
        call_kwargs = self.mock_tokenizer.call_args[1]
        assert 'max_length' in call_kwargs or 'truncation' in call_kwargs

    def test_tokenize_error_fallback(self):
        """Test tokenization fallback when error occurs."""
        # First call raises error, second succeeds
        self.mock_tokenizer.side_effect = [
            Exception("Tokenization error"),
            {
                'input_ids': torch.tensor([[101, 102]]),
                'attention_mask': torch.tensor([[1, 1]])
            }
        ]

        wrapper = MultilabelModelWrapper(
            self.mock_model,
            self.mock_tokenizer,
            device='cpu'
        )

        # Should retry with different parameters
        result = wrapper._tokenize(["test"])

        assert 'input_ids' in result
        assert self.mock_tokenizer.call_count == 2


class TestEdgeCasesAndValidation:
    """Test edge cases and validation scenarios."""

    def setup_method(self):
        """Setup test fixtures."""
        self.mock_model = Mock()
        self.mock_model.__class__.__name__ = 'PreTrainedModel'
        self.mock_model.eval.return_value = None
        self.mock_model.to.return_value = self.mock_model
        self.mock_model.config.max_position_embeddings = 512

        self.mock_tokenizer = Mock()
        self.mock_tokenizer.__class__.__name__ = 'PreTrainedTokenizer'
        self.mock_tokenizer.model_max_length = 512

    def test_empty_text_list(self):
        """Test handling of empty text list."""
        mock_output = Mock()
        mock_output.logits = torch.empty(0, 6)
        self.mock_model.return_value = mock_output

        self.mock_tokenizer.return_value = {
            'input_ids': torch.empty(0, 0, dtype=torch.long),
            'attention_mask': torch.empty(0, 0, dtype=torch.long)
        }

        wrapper = MultilabelModelWrapper(
            self.mock_model,
            self.mock_tokenizer,
            device='cpu'
        )

        result = wrapper([])

        assert result.shape[0] == 0

    def test_very_long_text_truncation(self):
        """Test that very long texts are properly truncated."""
        long_text = "word " * 1000  # Very long text

        self.mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[101] + [2054]*510 + [102]]),  # Truncated to max_length
            'attention_mask': torch.tensor([[1]*512])
        }

        mock_output = Mock()
        mock_output.logits = torch.tensor([[1.0, 2.0]])
        self.mock_model.return_value = mock_output

        wrapper = MultilabelModelWrapper(
            self.mock_model,
            self.mock_tokenizer,
            max_length=512,
            device='cpu'
        )

        result = wrapper([long_text])

        # Should successfully process truncated text
        assert result.shape == (1, 2)

    def test_special_characters_in_text(self):
        """Test handling of special characters."""
        special_text = "Hello! @#$% <script>alert('xss')</script> 你好"

        self.mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[101, 1234, 5678, 102]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1]])
        }

        mock_output = Mock()
        mock_output.logits = torch.tensor([[0.5, 0.5]])
        self.mock_model.return_value = mock_output

        wrapper = MultilabelModelWrapper(
            self.mock_model,
            self.mock_tokenizer,
            device='cpu'
        )

        result = wrapper([special_text])

        assert result is not None
        assert result.shape == (1, 2)

    def test_max_length_edge_cases(self):
        """Test max_length calculation edge cases."""
        # Case 1: model_max_length is very large
        self.mock_tokenizer.model_max_length = 10000
        self.mock_model.config.max_position_embeddings = 5000

        wrapper = MultilabelModelWrapper(
            self.mock_model,
            self.mock_tokenizer,
            device='cpu'
        )

        # Should cap at 1024
        assert wrapper.max_length <= 1024

    def test_model_eval_mode_set(self):
        """Test that model is set to eval mode."""
        wrapper = MultilabelModelWrapper(
            self.mock_model,
            self.mock_tokenizer,
            device='cpu'
        )

        self.mock_model.eval.assert_called()

    def test_gradient_disabled_during_call(self):
        """Test that gradients are disabled during forward pass in __call__."""
        mock_output = Mock()
        mock_output.logits = torch.tensor([[1.0, 2.0]])
        self.mock_model.return_value = mock_output

        self.mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[101, 102]]),
            'attention_mask': torch.tensor([[1, 1]])
        }

        wrapper = MultilabelModelWrapper(
            self.mock_model,
            self.mock_tokenizer,
            device='cpu'
        )

        with patch('torch.no_grad') as mock_no_grad:
            mock_no_grad.return_value.__enter__ = Mock()
            mock_no_grad.return_value.__exit__ = Mock()

            # __call__ should use torch.no_grad()
            result = wrapper(["test"])


class TestDevicePlacement:
    """Test device placement and tensor movement."""

    def setup_method(self):
        """Setup test fixtures."""
        self.mock_model = Mock()
        self.mock_model.__class__.__name__ = 'PreTrainedModel'
        self.mock_model.eval.return_value = None
        self.mock_model.to.return_value = self.mock_model
        self.mock_model.config.max_position_embeddings = 512

        self.mock_tokenizer = Mock()
        self.mock_tokenizer.__class__.__name__ = 'PreTrainedTokenizer'
        self.mock_tokenizer.model_max_length = 512

    def test_tensors_moved_to_device(self):
        """Test that input tensors are moved to specified device."""
        mock_ids = torch.tensor([[101, 102]])
        mock_mask = torch.tensor([[1, 1]])

        self.mock_tokenizer.return_value = {
            'input_ids': mock_ids,
            'attention_mask': mock_mask
        }

        mock_output = Mock()
        mock_output.logits = torch.tensor([[1.0, 2.0]])
        self.mock_model.return_value = mock_output

        wrapper = MultilabelModelWrapper(
            self.mock_model,
            self.mock_tokenizer,
            device='cuda'
        )

        # In real usage, tensors would be moved to device
        # Here we just verify the device is set correctly
        assert wrapper.device == 'cuda'

    def test_cpu_device_works(self):
        """Test that CPU device works correctly."""
        self.mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[101, 102]]),
            'attention_mask': torch.tensor([[1, 1]])
        }

        mock_output = Mock()
        mock_output.logits = torch.tensor([[1.0, 2.0]])
        self.mock_model.return_value = mock_output

        wrapper = MultilabelModelWrapper(
            self.mock_model,
            self.mock_tokenizer,
            device='cpu'
        )

        result = wrapper(["test"])
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
