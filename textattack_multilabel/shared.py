
from typing import List, Optional, Union
from collections import OrderedDict
from abc import ABC, abstractmethod

import torch
import numpy as np
from torch.nn.functional import softmax
import transformers

import textattack
from textattack.models.wrappers import ModelWrapper, PyTorchModelWrapper
from textattack.models.helpers import T5ForTextToText
from textattack.models.tokenizers import T5Tokenizer
from textattack.goal_functions import GoalFunction
from textattack.goal_function_results import GoalFunctionResult, GoalFunctionResultStatus
from textattack.search_methods import (GreedyWordSwapWIR, BeamSearch, PopulationBasedSearch,
                                       PopulationMember, SearchMethod)
from textattack.constraints.grammaticality import PartOfSpeech
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.constraints import Constraint, PreTransformationConstraint
from textattack.transformations import CompositeTransformation, Transformation
from textattack.shared import AttackedText, utils
from textattack.shared.validators import transformation_consists_of_word_swaps
from textattack.attack_results import (
    FailedAttackResult, MaximizedAttackResult, SkippedAttackResult, SuccessfulAttackResult
)
from textattack.attack import Attack



torch.cuda.empty_cache()

def AttackResult_new_diff_color(self, color_method=None):
    """Highlights the difference between two texts using color.

    Has to account for deletions and insertions from original text to
    perturbed. Relies on the index map stored in
    ``self.original_result.attacked_text.attack_attrs["original_index_map"]``.
    """
    t1 = self.original_result.attacked_text
    t2 = self.perturbed_result.attacked_text

    # if detect(t1.text) == "zh-cn" or detect(t1.text) == "ko":
    #     return t1.printable_text(), t2.printable_text()

    if color_method is None:
        return t1.printable_text(), t2.printable_text()

    color_1 = self.original_result.get_text_color_input()
    color_2 = self.perturbed_result.get_text_color_perturbed()

    # iterate through and count equal/unequal words
    words_1_idxs = []
    t2_equal_idxs = set()
    original_index_map = t2.attack_attrs["original_index_map"]
    for t1_idx, t2_idx in enumerate(original_index_map):
        if t2_idx == -1:
            # add words in t1 that are not in t2
            words_1_idxs.append(t1_idx)
        else:
            w1 = t1.words[t1_idx]
            w2 = t2.words[t2_idx]
            if w1 == w2:
                t2_equal_idxs.add(t2_idx)
            else:
                words_1_idxs.append(t1_idx)

    # words to color in t2 are all the words that didn't have an equal,
    # mapped word in t1
    words_2_idxs = list(sorted(set(range(t2.num_words)) - t2_equal_idxs))

    # make lists of colored words
    words_1 = [t1.words[i] for i in words_1_idxs]
    words_1 = [utils.color_text(w, color_1, color_method) for w in words_1]
    words_2 = [t2.words[i] for i in words_2_idxs]
    words_2 = [utils.color_text(w, color_2, color_method) for w in words_2]

    t1 = self.original_result.attacked_text.replace_words_at_indices(
        words_1_idxs, words_1
    )
    t2 = self.perturbed_result.attacked_text.replace_words_at_indices(
        words_2_idxs, words_2
    )

    key_color = ("bold", "underline")
    return (
        t1.printable_text(key_color=key_color, key_color_method=color_method),
        t2.printable_text(key_color=key_color, key_color_method=color_method),
    )




class GreedyWordSwapWIRTruncated(GreedyWordSwapWIR):
    def __init__(self, wir_method="unk", unk_token="[UNK]", truncate_words_to=-1):
        self.wir_method = wir_method
        self.unk_token = unk_token
        self.truncate_words_to = truncate_words_to

    def _get_unk_scores(self, indices_to_order):
        """Compute importance scores using UNK token replacement."""
        leave_one_texts = [
            self.initial_text.replace_word_at_index(i, self.unk_token)
            for i in indices_to_order
        ]
        leave_one_results, search_over = self.get_goal_results(leave_one_texts)
        index_scores = np.array([result.score for result in leave_one_results])
        return index_scores, search_over

    def _get_saliency_scores(self, indices_to_order, len_text):
        """Compute weighted saliency scores."""
        leave_one_texts = [
            self.initial_text.replace_word_at_index(i, self.unk_token)
            for i in indices_to_order
        ]
        leave_one_results, search_over = self.get_goal_results(leave_one_texts)
        saliency_scores = np.array([result.score for result in leave_one_results])

        softmax_saliency_scores = softmax(
            torch.Tensor(saliency_scores), dim=0
        ).numpy()

        delta_ps = []
        for idx in indices_to_order:
            transformed_text_candidates = self.get_transformations(
                self.initial_text,
                original_text=self.initial_text,
                indices_to_modify=[idx],
            )
            if not transformed_text_candidates:
                delta_ps.append(0.0)
                continue
            swap_results, search_over = self.get_goal_results(
                transformed_text_candidates
            )
            score_change = [result.score for result in swap_results]
            if not score_change:
                delta_ps.append(0.0)
                continue
            max_score_change = np.max(score_change)
            delta_ps.append(max_score_change)

            if search_over:
                delta_ps.extend([0.0] * (len_text - len(delta_ps)))
                break

        index_scores = softmax_saliency_scores * np.array(delta_ps)
        return index_scores, search_over

    def _get_delete_scores(self, indices_to_order):
        """Compute importance scores using word deletion."""
        leave_one_texts = [
            self.initial_text.delete_word_at_index(i) for i in indices_to_order
        ]
        leave_one_results, search_over = self.get_goal_results(leave_one_texts)
        index_scores = np.array([result.score for result in leave_one_results])
        return index_scores, search_over

    def _get_gradient_scores(self, indices_to_order, len_text):
        """Compute gradient-based importance scores."""
        victim_model = self.get_victim_model()
        index_scores = np.zeros(len_text)
        grad_output = victim_model.get_grad(self.initial_text.tokenizer_input)
        gradient = grad_output["gradient"]
        word2token_mapping = self.initial_text.align_with_model_tokens(victim_model)
        for i, index in enumerate(indices_to_order):
            matched_tokens = word2token_mapping[index]
            if not matched_tokens:
                index_scores[i] = 0.0
            else:
                try:
                    agg_grad = np.mean(gradient[matched_tokens], axis=0)
                    index_scores[i] = np.linalg.norm(agg_grad, ord=1)
                except IndexError:
                    index_scores[i] = 0.0
        return index_scores, False

    def _get_index_order(self, initial_text, max_len=-1):
        """Returns word indices of ``initial_text`` in descending order of
        importance."""
        if self.wir_method == "gradient":
            len_text, indices_to_order = self.get_indices_to_order(
                initial_text,
                indices_to_modify = set(range(len(initial_text.words[:self.truncate_words_to])))
            )
        else:
            len_text, indices_to_order = self.get_indices_to_order(initial_text)

        if self.wir_method == "unk":
            index_scores, search_over = self._get_unk_scores(indices_to_order)

        elif self.wir_method == "weighted-saliency":
            index_scores, search_over = self._get_saliency_scores(indices_to_order, len_text)

        elif self.wir_method == "delete":
            index_scores, search_over = self._get_delete_scores(indices_to_order)

        elif self.wir_method == "gradient":
            index_scores, search_over = self._get_gradient_scores(indices_to_order, len_text)

        elif self.wir_method == "random":
            index_order = indices_to_order
            np.random.shuffle(index_order)
            search_over = False
        else:
            raise ValueError(f"Unsupported WIR method {self.wir_method}")

        if self.wir_method != "random":
            index_order = np.array(indices_to_order)[(-index_scores).argsort()]

        return index_order, search_over


class PartOfSpeechTry(PartOfSpeech):
    def _check_constraint(self, transformed_text, reference_text):
        try:
            indices = transformed_text.attack_attrs["newly_modified_indices"]
        except KeyError:
            raise KeyError(
                "Cannot apply part-of-speech constraint without `newly_modified_indices`"
            )

        try:
            for i in indices:
                reference_word = reference_text.words[i]
                transformed_word = transformed_text.words[i]
                before_ctx = reference_text.words[max(i - 4, 0) : i]
                after_ctx = reference_text.words[
                    i + 1 : min(i + 4, len(reference_text.words))
                ]
                ref_pos = self._get_pos(before_ctx, reference_word, after_ctx)
                replace_pos = self._get_pos(before_ctx, transformed_word, after_ctx)
                if not self._can_replace_pos(ref_pos, replace_pos):
                    return False
        except IndexError:
            return False

        return True


class MultilabelClassificationGoalFunctionResult(GoalFunctionResult):
    def get_text_color_input(self):
        return "red"

    def get_text_color_perturbed(self):
        return "blue"

    def get_colored_output(self, color_method=None):
        return str(self.output)



class MultilabelClassificationGoalFunction(GoalFunction):
    """Goal Function for multilabel classification

    Maximize predicted scores for some targeted labels and
    minimize scores for some other labels at the same time.

    Examples
    ---------
    Taking a multilabel toxicity detection problem for example,
    this goal function is set to minimize the scores for all toxicity classes
    and maximize the scores for "Benign" class

    Input : (x = "original sentence", y = [0, 1, 0.5, 0])
    for ["Benign", "Hate Speech", "Abusive Language", "Profanity"])

    A successful attack will generate an adversarial example:
    Output: (x = sentence_adversarial, y_pred = [1, 0, 0, 0])
    where sentence_adversarial has same semantic meaning, but is slightly different from,
    the "original sentence". For human readers, sentence_adversarial should have the same
    y_pred as y, but model predicts flipped labels (y_pred).
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
        """Init the goal function

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
            else labels_to_maximize
        )
        self.labels_to_minimize = (
            [labels_to_minimize]
            if isinstance(labels_to_minimize, int)
            else labels_to_minimize
        )
        self.maximize_target_score = maximize_target_score
        self.minimize_target_score = minimize_target_score

        assert (
            labels_to_minimize or labels_to_maximize
        ), f"{labels_to_maximize=} and {labels_to_minimize=}"



    def _goal_function_result_type(self):
        return MultilabelClassificationGoalFunctionResult

    def _process_model_outputs(self, inputs, outputs):
        """Processes and validates a list of model outputs."""
        # Automatically cast a list or ndarray of predictions to a tensor.
        if isinstance(outputs, list) or isinstance(outputs, np.ndarray):
            outputs = torch.tensor(outputs)
        if not isinstance(outputs, torch.Tensor):
            raise TypeError(
                "Must have list, np.ndarray, or torch.Tensor of "
                f"scores. Got type {type(outputs)}"
            )

        # Validation check on model score dimensions
        if outputs.ndim == 1:
            # Unsqueeze prediction, if it's been squeezed by the model.
            if len(inputs) == 1:
                outputs = outputs.unsqueeze(dim=0)
            else:
                raise ValueError(
                    f"Model return score of shape {outputs.shape} for {len(inputs)} inputs."
                )
        elif outputs.ndim != 2:
            # If model somehow returns too may dimensions, throw an error.
            raise ValueError(
                f"Model return score of shape {outputs.shape} for {len(inputs)} inputs."
            )
        elif outputs.shape[0] != len(inputs):
            # If model returns an incorrect number of scores, throw an error.
            raise ValueError(
                f"Model return score of shape {outputs.shape} for {len(inputs)} inputs."
            )
        elif not ((outputs >= 0) & (outputs <= 1)).all():
            # Values in each row should be within range [0,1]. The model should return a
            # set of probabilities corresponding to each label.
            outputs = outputs.sigmoid()
        return outputs.cpu()

    def _get_score(
        self, model_output: "torch.Tensor", attacked_text: str
    ) -> "torch.float":
        """Return a single-value score to maximize over transformations

        MultilabelClassificationGoalFunction maximized all labels in self.labels_to_maximize and
        minimizes all labels in self.labels_to_minimize
        """
        labels_to_minimize, labels_to_maximize = (
            self.labels_to_minimize,
            self.labels_to_maximize,
        )

        score = model_output[labels_to_maximize].sum()
        score += 1 - model_output[labels_to_minimize].sum()
        return score

    def _is_goal_complete(
        self, model_output: "torch.Tensor", attacked_text: str
    ) -> "torch.bool":
        """Return True is goal is completed"""

        max_complete=True if len(self.labels_to_maximize) == 0 else (
            model_output[self.labels_to_maximize] > self.maximize_target_score
        ).any()

        min_complete = True if len(self.labels_to_minimize) == 0 else (
            model_output[self.labels_to_minimize] < self.minimize_target_score
        ).all()

        return max_complete and min_complete


# class HuggingFaceMultilabelModelWrapper(PyTorchModelWrapper):
#     """Loads a HuggingFace ``transformers`` model and tokenizer."""

#     def __init__(self, model, tokenizer, max_length=None, device='cuda'):
#         assert isinstance(
#             model, (transformers.PreTrainedModel, T5ForTextToText)
#         ), f"`model` must be of type `transformers.PreTrainedModel`, but got type {type(model)}."
#         assert isinstance(
#             tokenizer,
#             (
#                 transformers.PreTrainedTokenizer,
#                 transformers.PreTrainedTokenizerFast,
#                 T5Tokenizer,
#             ),
#         ), f"`tokenizer` must of type `transformers.PreTrainedTokenizer` or `transformers.PreTrainedTokenizerFast`, but got type {type(tokenizer)}."
#         self.device = device
#         self.model = model.to(self.device)
#         self.tokenizer = tokenizer
#         # Default max length is set to be int(1e30), so we force 512 to enable batching.
#         self.max_length = max_length or min(
#             1024,
#             tokenizer.model_max_length,
#             model.config.max_position_embeddings - 2
#         )


#     def __call__(self, text_input_list):
#         """Passes inputs to HuggingFace models as keyword arguments.

#         (Regular PyTorch ``nn.Module`` models typically take inputs as
#         positional arguments.)
#         """
#         inputs_dict = self.tokenizer(
#             text_input_list,
#             add_special_tokens=True,
#             padding="max_length",
#             max_length=self.max_length,
#             truncation=True,
#             return_tensors="pt",
#         )
#         inputs_dict.to(self.device)

#         with torch.no_grad():
#             outputs = self.model(**inputs_dict)

#         mode = getattr(self.model.config, "problem_type", None)
#         # multi_label_classification
#         # single_label_classification
            

#         if isinstance(outputs[0], str):
#             # HuggingFace sequence-to-sequence models return a list of
#             # string predictions as output. In this case, return the full
#             # list of outputs.
#             return outputs
#         else:
#             # HuggingFace classification models return a tuple as output
#             # where the first item in the tuple corresponds to the list of
#             # scores for each input.
#             if mode =='multi_label_classification':

#                 return outputs.logits.sigmoid()
#             else:
#                 return outputs.logits

#     def get_grad(self, text_input):
#         """Get gradient of loss with respect to input tokens.

#         Args:
#             text_input (str): input string
#         Returns:
#             Dict of ids, tokens, and gradient as numpy array.
#         """
#         if isinstance(self.model, textattack.models.helpers.T5ForTextToText):
#             raise NotImplementedError(
#                 "`get_grads` for T5FotTextToText has not been implemented yet."
#             )

#         self.model.train()
#         embedding_layer = self.model.get_input_embeddings()
#         original_state = embedding_layer.weight.requires_grad
#         embedding_layer.weight.requires_grad = True

#         emb_grads = []

#         def grad_hook(module, grad_in, grad_out):
#             emb_grads.append(grad_out[0])

#         emb_hook = embedding_layer.register_full_backward_hook(grad_hook)

#         self.model.zero_grad()

#         input_dict = self.tokenizer(
#             [text_input],
#             add_special_tokens=True,
#             return_tensors="pt",
#             padding="max_length",
#             max_length=self.max_length,
#             truncation=True,
#         )
#         input_dict.to(self.device)
#         predictions = self.model(**input_dict).logits
#         mode = getattr(self.model.config, "problem_type", None)

#         try:
#             if mode =='multi_label_classification':
#                 labels = predictions.sigmoid()
#             else:
#                 labels = predictions.argmax(dim=1)

#             loss = self.model(**input_dict, labels=labels)[0]
#         except TypeError:
#             raise TypeError(
#                 f"{type(self.model)} class does not take in `labels` to calculate loss. "
#                 "One cause for this might be if you instantiatedyour model using `transformer.AutoModel` "
#                 "(instead of `transformers.AutoModelForSequenceClassification`)."
#             )

#         loss.backward()

#         # grad w.r.t to word embeddings
#         grad = emb_grads[0][0].cpu().numpy()

#         embedding_layer.weight.requires_grad = original_state
#         emb_hook.remove()
#         self.model.eval()

#         output = {"ids": input_dict["input_ids"], "gradient": grad}

#         return output

#     def _tokenize(self, inputs):
#         """Helper method that for `tokenize`
#         Args:
#             inputs (list[str]): list of input strings
#         Returns:
#             tokens (list[list[str]]): List of list of tokens as strings
#         """
#         try:
#             encodings = self.tokenizer(inputs, truncation=True)
#             num_texts = len(inputs)
#             return [encodings.tokens(i) for i in range(num_texts)]
#         except ValueError:
#             return [
#                 self.tokenizer.convert_ids_to_tokens(
#                     self.tokenizer([x], truncation=True)["input_ids"][0]
#                 )
#                 for x in inputs
#             ]





class GeneticAlgorithm(PopulationBasedSearch, ABC):
    """Base class for attacking a model with word substiutitions using a
    genetic algorithm.

    Args:
        pop_size (int): The population size. Defaults to 20.
        max_iters (int): The maximum number of iterations to use. Defaults to 50.
        temp (float): Temperature for softmax function used to normalize probability dist when sampling parents.
            Higher temperature increases the sensitivity to lower probability candidates.
        give_up_if_no_improvement (bool): If True, stop the search early if no candidate that improves the score is found.
        post_crossover_check (bool): If True, check if child produced from crossover step passes the constraints.
        max_crossover_retries (int): Maximum number of crossover retries if resulting child fails to pass the constraints.
            Applied only when `post_crossover_check` is set to `True`.
            Setting it to 0 means we immediately take one of the parents at random as the child upon failure.
    """

    def __init__(
        self,
        pop_size=60,
        max_iters=20,
        temp=0.3,
        give_up_if_no_improvement=False,
        post_crossover_check=True,
        max_crossover_retries=20,
    ):
        self.max_iters = max_iters
        self.pop_size = pop_size
        self.temp = temp
        self.give_up_if_no_improvement = give_up_if_no_improvement
        self.post_crossover_check = post_crossover_check
        self.max_crossover_retries = max_crossover_retries

        # internal flag to indicate if search should end immediately
        self._search_over = False

    @abstractmethod
    def _modify_population_member(self, pop_member, new_text, new_result, word_idx):
        """Modify `pop_member` by returning a new copy with `new_text`,
        `new_result`, and, `attributes` altered appropriately for given
        `word_idx`"""
        raise NotImplementedError()

    @abstractmethod
    def _get_word_select_prob_weights(self, pop_member):
        """Get the attribute of `pop_member` that is used for determining
        probability of each word being selected for perturbation."""
        raise NotImplementedError

    def _perturb(self, pop_member, original_result, index=None):
        """Perturb `pop_member` and return it. Replaces a word at a random
        (unless `index` is specified) in `pop_member`.

        Args:
            pop_member (PopulationMember): The population member being perturbed.
            original_result (GoalFunctionResult): Result of original sample being attacked
            index (int): Index of word to perturb.
        Returns:
            Perturbed `PopulationMember`
        """
        num_words = pop_member.attacked_text.num_words
        # `word_select_prob_weights` is a list of values used for sampling one word to transform
        word_select_prob_weights = np.copy(
            self._get_word_select_prob_weights(pop_member)
        )
        non_zero_indices = np.count_nonzero(word_select_prob_weights)
        if non_zero_indices == 0:
            return pop_member
        iterations = 0

        try:
            while iterations < non_zero_indices:
                if index:
                    idx = index
                else:
                    w_select_probs = word_select_prob_weights / np.sum(
                        word_select_prob_weights
                    )
                    idx = np.random.choice(num_words, 1, p=w_select_probs)[0]

                transformed_texts = self.get_transformations(
                    pop_member.attacked_text,
                    original_text=original_result.attacked_text,
                    indices_to_modify=[idx],
                )

                if not len(transformed_texts):
                    iterations += 1
                    continue

                new_results, self._search_over = self.get_goal_results(transformed_texts)

                diff_scores = (
                    torch.Tensor([r.score for r in new_results]) - pop_member.result.score
                )
                if len(diff_scores) and diff_scores.max() > 0:
                    idx_with_max_score = diff_scores.argmax()
                    pop_member = self._modify_population_member(
                        pop_member,
                        transformed_texts[idx_with_max_score],
                        new_results[idx_with_max_score],
                        idx,
                    )
                    return pop_member

                word_select_prob_weights[idx] = 0
                iterations += 1

                if self._search_over:
                    break
        except IndexError as e:
            print(f"Index error in perturbation: {e}")
            return pop_member
        except Exception as e:
            print(f"Unexpected error in perturbation: {e}")
            import traceback
            traceback.print_exc()

        return pop_member

    @abstractmethod
    def _crossover_operation(self, pop_member1, pop_member2):
        """Actual operation that takes `pop_member1` text and `pop_member2`
        text and mixes the two to generate crossover between `pop_member1` and
        `pop_member2`.

        Args:
            pop_member1 (PopulationMember): The first population member.
            pop_member2 (PopulationMember): The second population member.
        Returns:
            Tuple of `AttackedText` and a dictionary of attributes.
        """
        raise NotImplementedError()

    def _post_crossover_check(
        self, new_text, parent_text1, parent_text2, original_text
    ):
        """Check if `new_text` that has been produced by performing crossover
        between `parent_text1` and `parent_text2` aligns with the constraints.

        Args:
            new_text (AttackedText): Text produced by crossover operation
            parent_text1 (AttackedText): Parent text of `new_text`
            parent_text2 (AttackedText): Second parent text of `new_text`
            original_text (AttackedText): Original text
        Returns:
            `True` if `new_text` meets the constraints. If otherwise, return `False`.
        """
        if "last_transformation" in new_text.attack_attrs:
            previous_text = (
                parent_text1
                if "last_transformation" in parent_text1.attack_attrs
                else parent_text2
            )
            passed_constraints = self._check_constraints(
                new_text, previous_text, original_text=original_text
            )
            return passed_constraints
        else:
            # `new_text` has not been actually transformed, so return True
            return True

    def _crossover(self, pop_member1, pop_member2, original_text):
        """Generates a crossover between pop_member1 and pop_member2.

        If the child fails to satisfy the constraints, we re-try crossover for a fix number of times,
        before taking one of the parents at random as the resulting child.
        Args:
            pop_member1 (PopulationMember): The first population member.
            pop_member2 (PopulationMember): The second population member.
            original_text (AttackedText): Original text
        Returns:
            A population member containing the crossover.
        """
        x1_text = pop_member1.attacked_text
        x2_text = pop_member2.attacked_text

        num_tries = 0
        passed_constraints = False
        while num_tries < self.max_crossover_retries + 1:
            new_text, attributes = self._crossover_operation(pop_member1, pop_member2)

            replaced_indices = new_text.attack_attrs["newly_modified_indices"]
            new_text.attack_attrs["modified_indices"] = (
                x1_text.attack_attrs["modified_indices"] - replaced_indices
            ) | (x2_text.attack_attrs["modified_indices"] & replaced_indices)

            if "last_transformation" in x1_text.attack_attrs:
                new_text.attack_attrs["last_transformation"] = x1_text.attack_attrs[
                    "last_transformation"
                ]
            elif "last_transformation" in x2_text.attack_attrs:
                new_text.attack_attrs["last_transformation"] = x2_text.attack_attrs[
                    "last_transformation"
                ]

            if self.post_crossover_check:
                passed_constraints = self._post_crossover_check(
                    new_text, x1_text, x2_text, original_text
                )

            if not self.post_crossover_check or passed_constraints:
                break

            num_tries += 1

        if self.post_crossover_check and not passed_constraints:
            # If we cannot find a child that passes the constraints,
            # we just randomly pick one of the parents to be the child for the next iteration.
            pop_mem = pop_member1 if np.random.uniform() < 0.5 else pop_member2
            return pop_mem
        else:
            new_results, self._search_over = self.get_goal_results([new_text])
            return PopulationMember(
                new_text, result=new_results[0], attributes=attributes
            )

    @abstractmethod
    def _initialize_population(self, initial_result, pop_size):
        """
        Initialize a population of size `pop_size` with `initial_result`
        Args:
            initial_result (GoalFunctionResult): Original text
            pop_size (int): size of population
        Returns:
            population as `list[PopulationMember]`
        """
        raise NotImplementedError()

    def perform_search(self, initial_result):
        self._search_over = False
        population = self._initialize_population(initial_result, self.pop_size)
        pop_size = len(population)
        current_score = initial_result.score

        for i in range(self.max_iters):
            population = sorted(population, key=lambda x: x.result.score, reverse=True)

            if (
                self._search_over
                or population[0].result.goal_status
                == GoalFunctionResultStatus.SUCCEEDED
            ):
                break

            if population[0].result.score > current_score:
                current_score = population[0].result.score
            elif self.give_up_if_no_improvement:
                break

            pop_scores = torch.Tensor([pm.result.score for pm in population])
            logits = ((-pop_scores) / self.temp).exp()
            select_probs = (logits / logits.sum()).cpu().numpy()

            parent1_idx = np.random.choice(pop_size, size=pop_size - 1, p=select_probs)
            parent2_idx = np.random.choice(pop_size, size=pop_size - 1, p=select_probs)

            children = []
            for idx in range(pop_size - 1):
                child = self._crossover(
                    population[parent1_idx[idx]],
                    population[parent2_idx[idx]],
                    initial_result.attacked_text,
                )
                if self._search_over:
                    break

                child = self._perturb(child, initial_result)
                children.append(child)

                # We need two `search_over` checks b/c value might change both in
                # `crossover` method and `perturb` method.
                if self._search_over:
                    break

            population = [population[0]] + children

        return population[0].result

    def check_transformation_compatibility(self, transformation):
        """The genetic algorithm is specifically designed for word
        substitutions."""
        return transformation_consists_of_word_swaps(transformation)

    @property
    def is_black_box(self):
        return True

    def extra_repr_keys(self):
        return [
            "pop_size",
            "max_iters",
            "temp",
            "give_up_if_no_improvement",
            "post_crossover_check",
            "max_crossover_retries",
        ]



class AlzantotGeneticAlgorithm(GeneticAlgorithm):
    """Attacks a model with word substiutitions using a genetic algorithm.

    Args:
        pop_size (int): The population size. Defaults to 60.
        max_iters (int): The maximum number of iterations to use. Defaults to 20.
        temp (float): Temperature for softmax function used to normalize probability dist when sampling parents.
            Higher temperature increases the sensitivity to lower probability candidates.
        give_up_if_no_improvement (bool): If True, stop the search early if no candidate that improves the score is found.
        post_crossover_check (bool): If True, check if child produced from crossover step passes the constraints.
        max_crossover_retries (int): Maximum number of crossover retries if resulting child fails to pass the constraints.
            Applied only when `post_crossover_check` is set to `True`.
            Setting it to 0 means we immediately take one of the parents at random as the child upon failure.
    """

    def __init__(
        self,
        pop_size=60,
        max_iters=20,
        temp=0.3,
        give_up_if_no_improvement=False,
        post_crossover_check=True,
        max_crossover_retries=20,
    ):
        super().__init__(
            pop_size=pop_size,
            max_iters=max_iters,
            temp=temp,
            give_up_if_no_improvement=give_up_if_no_improvement,
            post_crossover_check=post_crossover_check,
            max_crossover_retries=max_crossover_retries,
        )

    def _modify_population_member(self, pop_member, new_text, new_result, word_idx):
        """Modify `pop_member` by returning a new copy with `new_text`,
        `new_result`, and `num_candidate_transformations` altered appropriately
        for given `word_idx`"""
        num_candidate_transformations = np.copy(
            pop_member.attributes["num_candidate_transformations"]
        )
        num_candidate_transformations[word_idx] = 0
        return PopulationMember(
            new_text,
            result=new_result,
            attributes={"num_candidate_transformations": num_candidate_transformations},
        )

    def _get_word_select_prob_weights(self, pop_member):
        """Get the attribute of `pop_member` that is used for determining
        probability of each word being selected for perturbation."""
        return pop_member.attributes["num_candidate_transformations"]

    def _crossover_operation(self, pop_member1, pop_member2):
        """Actual operation that takes `pop_member1` text and `pop_member2`
        text and mixes the two to generate crossover between `pop_member1` and
        `pop_member2`.

        Args:
            pop_member1 (PopulationMember): The first population member.
            pop_member2 (PopulationMember): The second population member.
        Returns:
            Tuple of `AttackedText` and a dictionary of attributes.
        """
        indices_to_replace = []
        words_to_replace = []
        num_candidate_transformations = np.copy(
            pop_member1.attributes["num_candidate_transformations"]
        )

        for i in range(pop_member1.num_words):
            if (np.random.uniform() < 0.5
                and i >= 0 and i < len(pop_member2.words)
                and i < len(pop_member2.attributes["num_candidate_transformations"])):
                indices_to_replace.append(i)
                words_to_replace.append(pop_member2.words[i])
                num_candidate_transformations[i] = pop_member2.attributes[
                    "num_candidate_transformations"
                ][i]

        new_text = pop_member1.attacked_text.replace_words_at_indices(
            indices_to_replace, words_to_replace
        )
        return (
            new_text,
            {"num_candidate_transformations": num_candidate_transformations},
        )

    def _initialize_population(self, initial_result, pop_size):
        """
        Initialize a population of size `pop_size` with `initial_result`
        Args:
            initial_result (GoalFunctionResult): Original text
            pop_size (int): size of population
        Returns:
            population as `list[PopulationMember]`
        """
        words = initial_result.attacked_text.words
        num_candidate_transformations = np.zeros(len(words))
        transformed_texts = self.get_transformations(
            initial_result.attacked_text, original_text=initial_result.attacked_text
        )
        for transformed_text in transformed_texts:
            diff_idx = next(
                iter(transformed_text.attack_attrs["newly_modified_indices"])
            )
            num_candidate_transformations[diff_idx] += 1

        # Just b/c there are no replacements now doesn't mean we never want to select the word for perturbation
        # Therefore, we give small non-zero probability for words with no replacements
        # Epsilon is some small number to approximately assign small probability
        min_num_candidates = np.amin(num_candidate_transformations)
        epsilon = max(1, int(min_num_candidates * 0.1))
        for i in range(len(num_candidate_transformations)):
            num_candidate_transformations[i] = max(
                num_candidate_transformations[i], epsilon
            )

        population = []
        for _ in range(pop_size):
            pop_member = PopulationMember(
                initial_result.attacked_text,
                initial_result,
                attributes={
                    "num_candidate_transformations": np.copy(
                        num_candidate_transformations
                    )
                },
            )
            # Perturb `pop_member` in-place
            pop_member = self._perturb(pop_member, initial_result)
            population.append(pop_member)

        return population


class AttackList(Attack):

    def attack(self, example, ground_truth_output):
        """Attack a single example.

        Args:
            example (:obj:`str`, :obj:`OrderedDict[str, str]` or :class:`~textattack.shared.AttackedText`):
                Example to attack. It can be a single string or an `OrderedDict` where
                keys represent the input fields (e.g. "premise", "hypothesis") and the values are the actual input textx.
                Also accepts :class:`~textattack.shared.AttackedText` that wraps around the input.
            ground_truth_output(:obj:`int`, :obj:`float`, :obj:`str` or :obj:`list[float]`):
                Ground truth output of `example`.
                For classification tasks, it should be an integer representing the ground truth label.
                For regression tasks (e.g. STS), it should be the target value.
                For seq2seq tasks (e.g. translation), it should be the target string.
        Returns:
            :class:`~textattack.attack_results.AttackResult` that represents the result of the attack.
        """
        assert isinstance(
            example, (str, OrderedDict, AttackedText)
        ), "`example` must either be `str`, `collections.OrderedDict`, `textattack.shared.AttackedText`."
        if isinstance(example, (str, OrderedDict)):
            example = AttackedText(example)

        assert isinstance(
            ground_truth_output, (int, str, list, np.integer)
        ), f"`ground_truth_output` must either be `str`, `int` or list of float but is {type(ground_truth_output)}"
        goal_function_result, _ = self.goal_function.init_attack_example(
            example, ground_truth_output
        )
        if goal_function_result.goal_status == GoalFunctionResultStatus.SKIPPED:
            return SkippedAttackResult(goal_function_result)
        else:
            result = self._attack(goal_function_result)
            return result
