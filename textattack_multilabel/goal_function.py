from typing import List, Optional, Union
import time

import numpy as np
import torch

from textattack.goal_functions import GoalFunction
from textattack.goal_function_results import GoalFunctionResult


class MultilabelClassificationGoalFunctionResult(GoalFunctionResult):
    def get_text_color_input(self):
        return "blue"

    def get_text_color_perturbed(self):
        return "red"

    def get_colored_output(self, color_method=None):
        return str(self.output.cpu().tolist() if hasattr(self.output, 'cpu') else self.output)


class MultilabelClassificationGoalFunction(GoalFunction):
    """Goal Function for multilabel classification.

    Maximize predicted scores for some targeted labels and minimize scores for
    some other labels at the same time.

    Examples
    ---------
    Taking a multilabel toxicity detection problem for example, this goal
    function is set to minimize the scores for all toxicity classes and
    maximize the scores for "Benign" class.

    Input : (x = "original sentence", y = [0, 1, 0.5, 0]) for
    ["Benign", "Hate Speech", "Abusive Language", "Profanity"])

    A successful attack will generate an adversarial example:
    Output: (x = sentence_adversarial, y_pred = [1, 0, 0, 0]) where
    sentence_adversarial has same semantic meaning, but is slightly different
    from, the "original sentence". For human readers, sentence_adversarial
    should have the same y_pred as y, but model predicts flipped labels
    (y_pred).
    """

    def __init__(
        self,
        *args,
        labels_to_maximize: Optional[Union[int, List[int]]] = 0,
        labels_to_minimize: Optional[Union[int, List[int]]] = [],
        maximize_target_score: Optional[float] = 0.5,
        minimize_target_score: Optional[float] = 0.5,
        **kwargs,
    ):
        """Init the goal function.

        Parameters
        ----------
        labels_to_maximize : Optional[int, List[int]]
            default = 0
        labels_to_minimize : Optional[int, List[int]]
            if not provided, will use all labels not in labels_to_maximize
        """
        super().__init__(*args, **kwargs)
        self.labels_to_maximize = (
            [labels_to_maximize]
            if isinstance(labels_to_maximize, int)
            else labels_to_maximize or []
        )
        self.labels_to_minimize = (
            [labels_to_minimize]
            if isinstance(labels_to_minimize, int)
            else labels_to_minimize or []
        )
        self.maximize_target_score = maximize_target_score
        self.minimize_target_score = minimize_target_score

        assert (
            self.labels_to_minimize or self.labels_to_maximize
        ), f"{self.labels_to_maximize=} and {self.labels_to_minimize=}"

    def _goal_function_result_type(self):
        return MultilabelClassificationGoalFunctionResult

    def _process_model_outputs(self, inputs=None, outputs=None):
        """Processes and validates a list of model outputs."""
        if inputs is None:
            inputs = ["dummy"]

        if isinstance(inputs, list) and len(inputs) <= 3:  # Only log for small batches to avoid spam
            print(f"Making model prediction on {len(inputs)} inputs")
            # Log the first few words of first input if it's a string
            if inputs and isinstance(inputs[0], str):
                first_words = inputs[0].split()[:5]
                print(f"Sample input: {' '.join(first_words)}...")

        # Automatically cast a list or ndarray of predictions to a tensor.
        if isinstance(outputs, list) or isinstance(outputs, np.ndarray):
            outputs = torch.tensor(outputs, dtype=torch.float32)
        if not isinstance(outputs, torch.Tensor):
            raise TypeError(
                "Must have list, np.ndarray, or torch.Tensor of "
                f"scores. Got type {type(outputs)}"
            )

        # Validation check on model score dimensions
        expected_batch = len(inputs) if isinstance(inputs, (list, tuple)) else 1
        if outputs.ndim == 1:
            # Unsqueeze prediction, if it's been squeezed by the model.
            if expected_batch == 1:
                outputs = outputs.unsqueeze(dim=0)
            else:
                raise ValueError(
                    f"Model return score of shape {outputs.shape} for {expected_batch} inputs."
                )
        elif outputs.ndim != 2:
            # If model somehow returns too may dimensions, throw an error.
            raise ValueError(
                f"Output must be 1D or 2D tensor, got shape {outputs.shape}"
            )
        elif outputs.shape[0] != expected_batch:
            # If model returns an incorrect number of scores, throw an error.
            raise ValueError(
                f"Model return score of shape {outputs.shape} for {expected_batch} inputs."
            )
        elif not ((outputs >= 0) & (outputs <= 1)).all():
            # Values in each row should be within range [0,1]. The model should return a
            # set of probabilities corresponding to each label.
            outputs = outputs.sigmoid()
        return outputs.cpu()

    def _get_score(
        self, model_output: "torch.Tensor", attacked_text: str
    ) -> "torch.float":
        """Return a single-value score to maximize over transformations.

        MultilabelClassificationGoalFunction maximized all labels in
        self.labels_to_maximize and minimizes all labels in self.labels_to_minimize.
        """
        labels_to_minimize, labels_to_maximize = (
            self.labels_to_minimize,
            self.labels_to_maximize,
        )

        score = model_output[labels_to_maximize].sum()
        if self.labels_to_minimize:
            score += 1 - model_output[labels_to_minimize].sum()
        return score

    def _is_goal_complete(
        self, model_output: "torch.Tensor", attacked_text: str
    ) -> "torch.bool":
        """Return True if goal is completed.

        For maximize labels: ALL must exceed the maximize_target_score
        For minimize labels: ALL must fall below the minimize_target_score
        """
        max_complete = True if len(self.labels_to_maximize) == 0 else (
            model_output[self.labels_to_maximize] > self.maximize_target_score
        ).any()  ### .any() or .all() - any labels must exceed threshold due to multilabel nature

        min_complete = True if len(self.labels_to_minimize) == 0 else (
            model_output[self.labels_to_minimize] < self.minimize_target_score
        ).all()

        return bool(max_complete and min_complete)


__all__ = [
    "MultilabelClassificationGoalFunction",
    "MultilabelClassificationGoalFunctionResult",
]
