"""
Comprehensive tests for MultilabelClassificationGoalFunction.

This test suite covers:
- Output processing and validation
- Score calculation logic
- Goal completion detection
- Edge cases and error handling
- Result formatting
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, MagicMock

from textattack_multilabel.goal_function import (
    MultilabelClassificationGoalFunction,
    MultilabelClassificationGoalFunctionResult,
)


class TestMultilabelClassificationGoalFunctionInit:
    """Test initialization and configuration."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        mock_model = Mock()
        goal_func = MultilabelClassificationGoalFunction(mock_model)

        assert goal_func.model == mock_model
        assert goal_func.labels_to_maximize == [0]
        assert goal_func.labels_to_minimize == []
        assert goal_func.maximize_target_score == 0.5
        assert goal_func.minimize_target_score == 0.5

    def test_init_with_single_int_labels(self):
        """Test initialization with integer labels (converted to list)."""
        mock_model = Mock()
        goal_func = MultilabelClassificationGoalFunction(
            mock_model,
            labels_to_maximize=2,
            labels_to_minimize=3
        )

        assert goal_func.labels_to_maximize == [2]
        assert goal_func.labels_to_minimize == [3]

    def test_init_with_list_labels(self):
        """Test initialization with list of labels."""
        mock_model = Mock()
        goal_func = MultilabelClassificationGoalFunction(
            mock_model,
            labels_to_maximize=[0, 1, 2],
            labels_to_minimize=[3, 4, 5]
        )

        assert goal_func.labels_to_maximize == [0, 1, 2]
        assert goal_func.labels_to_minimize == [3, 4, 5]

    def test_init_with_custom_thresholds(self):
        """Test initialization with custom target scores."""
        mock_model = Mock()
        goal_func = MultilabelClassificationGoalFunction(
            mock_model,
            maximize_target_score=0.8,
            minimize_target_score=0.2
        )

        assert goal_func.maximize_target_score == 0.8
        assert goal_func.minimize_target_score == 0.2

    def test_init_with_none_minimize(self):
        """Test initialization with None for minimize labels."""
        mock_model = Mock()
        goal_func = MultilabelClassificationGoalFunction(
            mock_model,
            labels_to_maximize=[0, 1],
            labels_to_minimize=None
        )

        assert goal_func.labels_to_maximize == [0, 1]
        assert goal_func.labels_to_minimize == []

    def test_init_validates_labels_provided(self):
        """Test that initialization fails when neither maximize nor minimize labels provided."""
        mock_model = Mock()

        with pytest.raises(AssertionError):
            MultilabelClassificationGoalFunction(
                mock_model,
                labels_to_maximize=[],
                labels_to_minimize=[]
            )


class TestProcessModelOutputs:
    """Test _process_model_outputs method."""

    def setup_method(self):
        """Setup test fixtures."""
        self.mock_model = Mock()
        self.goal_func = MultilabelClassificationGoalFunction(
            self.mock_model,
            labels_to_maximize=[0, 1],
            labels_to_minimize=[2, 3]
        )

    def test_process_tensor_2d(self):
        """Test processing 2D tensor (batch of predictions)."""
        outputs = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
        result = self.goal_func._process_model_outputs(outputs=outputs)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 6)
        assert torch.allclose(result, outputs)

    def test_process_tensor_1d(self):
        """Test processing 1D tensor (single prediction) - should unsqueeze."""
        outputs = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        result = self.goal_func._process_model_outputs(outputs=outputs)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 6)
        assert torch.allclose(result, outputs.unsqueeze(0))

    def test_process_numpy_array(self):
        """Test processing numpy array (converted to tensor)."""
        outputs = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
        result = self.goal_func._process_model_outputs(outputs=outputs)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 6)
        assert torch.allclose(result, torch.tensor(outputs, dtype=torch.float))

    def test_process_list(self):
        """Test processing list (converted to tensor)."""
        outputs = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]
        result = self.goal_func._process_model_outputs(outputs=outputs)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 6)

    def test_process_invalid_type_raises_error(self):
        """Test that invalid types raise TypeError."""
        with pytest.raises(TypeError, match="Must have"):
            self.goal_func._process_model_outputs(outputs="invalid")

        with pytest.raises(TypeError, match="Must have"):
            self.goal_func._process_model_outputs(outputs=123)

    def test_process_3d_tensor_raises_error(self):
        """Test that 3D tensors raise ValueError."""
        outputs = torch.randn(2, 3, 4)

        with pytest.raises(ValueError, match="must be 1D or 2D"):
            self.goal_func._process_model_outputs(outputs=outputs)

    def test_process_wrong_batch_size_raises_error(self):
        """Test that wrong batch size raises ValueError."""
        outputs = torch.randn(3, 6)  # batch_size=3, but we expect 1

        with pytest.raises(ValueError, match=r"Model return score of shape .* for 1 inputs\."):
            self.goal_func._process_model_outputs(outputs=outputs)

    def test_process_values_out_of_range_applies_sigmoid(self):
        """Test that values outside [0,1] trigger sigmoid application."""
        # Raw logits (outside [0,1] range)
        outputs = torch.tensor([[2.5, -1.3, 0.8, -0.5, 1.2, 0.3]])
        result = self.goal_func._process_model_outputs(outputs=outputs)

        # Result should be sigmoid of input
        expected = torch.sigmoid(outputs)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_process_values_in_range_no_sigmoid(self):
        """Test that values in [0,1] are not modified."""
        outputs = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
        result = self.goal_func._process_model_outputs(outputs=outputs)

        # Should be unchanged (no sigmoid)
        assert torch.allclose(result, outputs)

    def test_process_edge_case_zeros(self):
        """Test processing all-zero outputs."""
        outputs = torch.zeros(1, 6)
        result = self.goal_func._process_model_outputs(outputs=outputs)

        assert torch.allclose(result, outputs)

    def test_process_edge_case_ones(self):
        """Test processing all-one outputs."""
        outputs = torch.ones(1, 6)
        result = self.goal_func._process_model_outputs(outputs=outputs)

        assert torch.allclose(result, outputs)


class TestGetScore:
    """Test _get_score method."""

    def setup_method(self):
        """Setup test fixtures."""
        self.mock_model = Mock()

    def test_score_maximize_only(self):
        """Test score calculation with only maximize labels."""
        goal_func = MultilabelClassificationGoalFunction(
            self.mock_model,
            labels_to_maximize=[0, 1, 2],
            labels_to_minimize=[]
        )

        model_output = torch.tensor([0.3, 0.5, 0.7, 0.1, 0.2, 0.4])
        score = goal_func._get_score(model_output, "sample text")

        # Score = sum of labels_to_maximize = 0.3 + 0.5 + 0.7 = 1.5
        expected_score = 0.3 + 0.5 + 0.7
        assert abs(score - expected_score) < 1e-5

    def test_score_minimize_only(self):
        """Test score calculation with only minimize labels."""
        goal_func = MultilabelClassificationGoalFunction(
            self.mock_model,
            labels_to_maximize=[],
            labels_to_minimize=[3, 4, 5]
        )

        model_output = torch.tensor([0.3, 0.5, 0.7, 0.2, 0.4, 0.6])
        score = goal_func._get_score(model_output, "sample text")

        # Score = (1 - sum of labels_to_minimize) = 1 - (0.2 + 0.4 + 0.6) = 1 - 1.2 = -0.2
        expected_score = 1 - (0.2 + 0.4 + 0.6)
        assert abs(score - expected_score) < 1e-5

    def test_score_both_maximize_and_minimize(self):
        """Test score calculation with both maximize and minimize labels."""
        goal_func = MultilabelClassificationGoalFunction(
            self.mock_model,
            labels_to_maximize=[0, 1],
            labels_to_minimize=[3, 4]
        )

        model_output = torch.tensor([0.8, 0.7, 0.5, 0.1, 0.2, 0.3])
        score = goal_func._get_score(model_output, "sample text")

        # Score = sum(maximize) + (1 - sum(minimize))
        # = (0.8 + 0.7) + (1 - (0.1 + 0.2))
        # = 1.5 + 0.7 = 2.2
        expected_score = (0.8 + 0.7) + (1 - (0.1 + 0.2))
        assert abs(score - expected_score) < 1e-5

    def test_score_single_label_maximize(self):
        """Test score with single maximize label."""
        goal_func = MultilabelClassificationGoalFunction(
            self.mock_model,
            labels_to_maximize=[2],
            labels_to_minimize=[]
        )

        model_output = torch.tensor([0.1, 0.2, 0.9, 0.1, 0.1, 0.1])
        score = goal_func._get_score(model_output, "sample text")

        assert abs(score - 0.9) < 1e-5

    def test_score_single_label_minimize(self):
        """Test score with single minimize label."""
        goal_func = MultilabelClassificationGoalFunction(
            self.mock_model,
            labels_to_maximize=[],
            labels_to_minimize=[2]
        )

        model_output = torch.tensor([0.1, 0.2, 0.3, 0.1, 0.1, 0.1])
        score = goal_func._get_score(model_output, "sample text")

        # Score = 1 - 0.3 = 0.7
        assert abs(score - 0.7) < 1e-5

    def test_score_all_labels_maximize(self):
        """Test maximizing all labels."""
        goal_func = MultilabelClassificationGoalFunction(
            self.mock_model,
            labels_to_maximize=[0, 1, 2, 3, 4, 5],
            labels_to_minimize=[]
        )

        model_output = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        score = goal_func._get_score(model_output, "sample text")

        expected_score = sum([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        assert abs(score - expected_score) < 1e-5

    def test_score_all_labels_minimize(self):
        """Test minimizing all labels."""
        goal_func = MultilabelClassificationGoalFunction(
            self.mock_model,
            labels_to_maximize=[],
            labels_to_minimize=[0, 1, 2, 3, 4, 5]
        )

        model_output = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        score = goal_func._get_score(model_output, "sample text")

        expected_score = 1 - sum([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        assert abs(score - expected_score) < 1e-5


class TestIsGoalComplete:
    """Test _is_goal_complete method."""

    def setup_method(self):
        """Setup test fixtures."""
        self.mock_model = Mock()

    def test_goal_complete_maximize_all_above_threshold(self):
        """Test goal completion when all maximize labels exceed threshold."""
        goal_func = MultilabelClassificationGoalFunction(
            self.mock_model,
            labels_to_maximize=[0, 1, 2],
            labels_to_minimize=[],
            maximize_target_score=0.7
        )

        # All maximize labels > 0.7
        model_output = torch.tensor([0.8, 0.9, 0.75, 0.1, 0.2, 0.3])
        result = goal_func._is_goal_complete(model_output, "sample text")

        assert result is True

    def test_goal_incomplete_maximize_one_below_threshold(self):
        """Test goal incomplete when one maximize label below threshold."""
        goal_func = MultilabelClassificationGoalFunction(
            self.mock_model,
            labels_to_maximize=[0, 1, 2],
            labels_to_minimize=[],
            maximize_target_score=0.7
        )

        # One maximize label (0.6) < 0.7
        model_output = torch.tensor([0.8, 0.9, 0.6, 0.1, 0.2, 0.3])
        result = goal_func._is_goal_complete(model_output, "sample text")

        assert result is False

    def test_goal_complete_minimize_all_below_threshold(self):
        """Test goal completion when all minimize labels below threshold."""
        goal_func = MultilabelClassificationGoalFunction(
            self.mock_model,
            labels_to_maximize=[],
            labels_to_minimize=[3, 4, 5],
            minimize_target_score=0.3
        )

        # All minimize labels < 0.3
        model_output = torch.tensor([0.8, 0.9, 0.75, 0.1, 0.2, 0.25])
        result = goal_func._is_goal_complete(model_output, "sample text")

        assert result is True

    def test_goal_incomplete_minimize_one_above_threshold(self):
        """Test goal incomplete when one minimize label above threshold."""
        goal_func = MultilabelClassificationGoalFunction(
            self.mock_model,
            labels_to_maximize=[],
            labels_to_minimize=[3, 4, 5],
            minimize_target_score=0.3
        )

        # One minimize label (0.4) > 0.3
        model_output = torch.tensor([0.8, 0.9, 0.75, 0.1, 0.2, 0.4])
        result = goal_func._is_goal_complete(model_output, "sample text")

        assert result is False

    def test_goal_complete_both_maximize_and_minimize(self):
        """Test goal completion with both maximize and minimize labels."""
        goal_func = MultilabelClassificationGoalFunction(
            self.mock_model,
            labels_to_maximize=[0, 1],
            labels_to_minimize=[4, 5],
            maximize_target_score=0.8,
            minimize_target_score=0.2
        )

        # Maximize labels > 0.8 AND minimize labels < 0.2
        model_output = torch.tensor([0.85, 0.9, 0.5, 0.5, 0.15, 0.1])
        result = goal_func._is_goal_complete(model_output, "sample text")

        assert result is True

    def test_goal_incomplete_maximize_success_minimize_fail(self):
        """Test goal incomplete when maximize succeeds but minimize fails."""
        goal_func = MultilabelClassificationGoalFunction(
            self.mock_model,
            labels_to_maximize=[0, 1],
            labels_to_minimize=[4, 5],
            maximize_target_score=0.8,
            minimize_target_score=0.2
        )

        # Maximize OK (> 0.8), but minimize fails (0.3 > 0.2)
        model_output = torch.tensor([0.85, 0.9, 0.5, 0.5, 0.15, 0.3])
        result = goal_func._is_goal_complete(model_output, "sample text")

        assert result is False

    def test_goal_incomplete_maximize_fail_minimize_success(self):
        """Test goal incomplete when maximize fails but minimize succeeds."""
        goal_func = MultilabelClassificationGoalFunction(
            self.mock_model,
            labels_to_maximize=[0, 1],
            labels_to_minimize=[4, 5],
            maximize_target_score=0.8,
            minimize_target_score=0.2
        )

        # Maximize fails (0.75 < 0.8), but minimize OK (< 0.2)
        model_output = torch.tensor([0.75, 0.9, 0.5, 0.5, 0.15, 0.1])
        result = goal_func._is_goal_complete(model_output, "sample text")

        assert result is False

    def test_goal_complete_empty_maximize_labels(self):
        """Test goal complete when no maximize labels (always True)."""
        goal_func = MultilabelClassificationGoalFunction(
            self.mock_model,
            labels_to_maximize=[],
            labels_to_minimize=[4, 5],
            minimize_target_score=0.2
        )

        # No maximize labels, so max_complete = True
        # Minimize labels < 0.2, so min_complete = True
        model_output = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.1, 0.15])
        result = goal_func._is_goal_complete(model_output, "sample text")

        assert result is True

    def test_goal_complete_empty_minimize_labels(self):
        """Test goal complete when no minimize labels (always True)."""
        goal_func = MultilabelClassificationGoalFunction(
            self.mock_model,
            labels_to_maximize=[0, 1],
            labels_to_minimize=[],
            maximize_target_score=0.8
        )

        # Maximize labels > 0.8, so max_complete = True
        # No minimize labels, so min_complete = True
        model_output = torch.tensor([0.85, 0.9, 0.5, 0.5, 0.5, 0.5])
        result = goal_func._is_goal_complete(model_output, "sample text")

        assert result is True

    def test_goal_edge_case_exactly_at_threshold(self):
        """Test goal completion when values exactly at threshold."""
        goal_func = MultilabelClassificationGoalFunction(
            self.mock_model,
            labels_to_maximize=[0],
            labels_to_minimize=[1],
            maximize_target_score=0.8,
            minimize_target_score=0.2
        )

        # Exactly at threshold - should NOT complete (needs to EXCEED for maximize)
        model_output = torch.tensor([0.8, 0.2, 0.5, 0.5, 0.5, 0.5])
        result = goal_func._is_goal_complete(model_output, "sample text")

        assert result is False  # 0.8 is NOT > 0.8, and 0.2 is NOT < 0.2


class TestGoalFunctionResult:
    """Test MultilabelClassificationGoalFunctionResult."""

    def test_result_type_returned(self):
        """Test that correct result type is returned."""
        mock_model = Mock()
        goal_func = MultilabelClassificationGoalFunction(mock_model)

        result_type = goal_func._goal_function_result_type()
        assert result_type == MultilabelClassificationGoalFunctionResult

    def test_result_get_text_color_input(self):
        """Test get_text_color_input returns 'blue'."""
        result = MultilabelClassificationGoalFunctionResult(
            attacked_text=Mock(),
            raw_output=None,
            output=0.5,
            goal_status=Mock(),
            score=1.0,
            num_queries=1,
            ground_truth_output=0.0
        )

        assert result.get_text_color_input() == "blue"

    def test_result_get_text_color_perturbed(self):
        """Test get_text_color_perturbed returns 'red'."""
        result = MultilabelClassificationGoalFunctionResult(
            attacked_text=Mock(),
            raw_output=None,
            output=0.5,
            goal_status=Mock(),
            score=1.0,
            num_queries=1,
            ground_truth_output=0.0
        )

        assert result.get_text_color_perturbed() == "red"

    def test_result_get_colored_output(self):
        """Test get_colored_output returns integer."""
        result = MultilabelClassificationGoalFunctionResult(
            attacked_text=Mock(),
            raw_output=None,
            output=5.7,
            goal_status=Mock(),
            score=1.0,
            num_queries=1,
            ground_truth_output=0.0
        )

        # Should return int(output)
        assert result.get_colored_output() == 5


class TestEdgeCasesAndValidation:
    """Test edge cases and validation scenarios."""

    def test_large_number_of_labels(self):
        """Test with many labels."""
        mock_model = Mock()
        goal_func = MultilabelClassificationGoalFunction(
            mock_model,
            labels_to_maximize=list(range(50)),
            labels_to_minimize=list(range(50, 100))
        )

        assert len(goal_func.labels_to_maximize) == 50
        assert len(goal_func.labels_to_minimize) == 50

    def test_very_high_threshold(self):
        """Test with very high target score."""
        mock_model = Mock()
        goal_func = MultilabelClassificationGoalFunction(
            mock_model,
            maximize_target_score=0.99
        )

        model_output = torch.tensor([0.98, 0.5, 0.5, 0.5, 0.5, 0.5])
        result = goal_func._is_goal_complete(model_output, "text")

        assert result is False  # 0.98 not > 0.99

    def test_very_low_threshold(self):
        """Test with very low target score."""
        mock_model = Mock()
        goal_func = MultilabelClassificationGoalFunction(
            mock_model,
            labels_to_minimize=[0],
            minimize_target_score=0.01
        )

        model_output = torch.tensor([0.02, 0.5, 0.5, 0.5, 0.5, 0.5])
        result = goal_func._is_goal_complete(model_output, "text")

        assert result is False  # 0.02 not < 0.01

    def test_score_with_zero_probabilities(self):
        """Test score calculation with zero probabilities."""
        mock_model = Mock()
        goal_func = MultilabelClassificationGoalFunction(
            mock_model,
            labels_to_maximize=[0, 1],
            labels_to_minimize=[2, 3]
        )

        model_output = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.5, 0.5])
        score = goal_func._get_score(model_output, "text")

        # (0.0 + 0.0) + (1 - (0.0 + 0.0)) = 0 + 1 = 1.0
        assert abs(score - 1.0) < 1e-5

    def test_score_with_one_probabilities(self):
        """Test score calculation with probability 1.0."""
        mock_model = Mock()
        goal_func = MultilabelClassificationGoalFunction(
            mock_model,
            labels_to_maximize=[0, 1],
            labels_to_minimize=[2, 3]
        )

        model_output = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.5, 0.5])
        score = goal_func._get_score(model_output, "text")

        # (1.0 + 1.0) + (1 - (0.0 + 0.0)) = 2.0 + 1.0 = 3.0
        assert abs(score - 3.0) < 1e-5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
