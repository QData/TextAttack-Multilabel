from typing import List, Optional, Union
from collections import OrderedDict
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch.nn.functional import softmax

torch.cuda.empty_cache()

from textattack.attack_results import (
    FailedAttackResult,
    MaximizedAttackResult,
    SkippedAttackResult,
    SuccessfulAttackResult,
)
from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import (
    GreedyWordSwapWIR,
    PopulationBasedSearch,
    PopulationMember,
)
from textattack.shared import AttackedText, utils
from textattack.shared.validators import transformation_consists_of_word_swaps
from textattack.constraints.grammaticality import PartOfSpeech
from textattack.attack import Attack


def AttackResult_new_diff_color(self, color_method=None):
    """Highlights the difference between two texts using color."""
    t1 = self.original_result.attacked_text
    t2 = self.perturbed_result.attacked_text

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
        """Returns word indices of ``initial_text`` in descending order of importance."""
        if self.wir_method == "gradient":
            len_text, indices_to_order = self.get_indices_to_order(
                initial_text,
                indices_to_modify=set(range(len(initial_text.words[: self.truncate_words_to]))),
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


class GeneticAlgorithm(PopulationBasedSearch, ABC):
    """Base class for attacking a model with word substitutions using a genetic algorithm."""

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
        """Modify `pop_member` by returning a new copy with updates for `word_idx`."""
        raise NotImplementedError()

    @abstractmethod
    def _get_word_select_prob_weights(self, pop_member):
        """Get values used to sample one word to transform."""
        raise NotImplementedError

    def _perturb(self, pop_member, original_result, index=None):
        """Perturb `pop_member` and return it by replacing one word."""
        num_words = pop_member.attacked_text.num_words
        word_select_prob_weights = np.copy(self._get_word_select_prob_weights(pop_member))
        non_zero_indices = np.count_nonzero(word_select_prob_weights)
        if non_zero_indices == 0:
            return pop_member
        iterations = 0

        try:
            while iterations < non_zero_indices:
                if index:
                    idx = index
                else:
                    w_select_probs = word_select_prob_weights / np.sum(word_select_prob_weights)
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

                diff_scores = torch.Tensor([r.score for r in new_results]) - pop_member.result.score
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
        """Mix `pop_member1` and `pop_member2` to generate crossover."""
        raise NotImplementedError()

    def _post_crossover_check(self, new_text, parent_text1, parent_text2, original_text):
        """Check if `new_text` produced by crossover aligns with the constraints."""
        if "last_transformation" in new_text.attack_attrs:
            previous_text = (
                parent_text1 if "last_transformation" in parent_text1.attack_attrs else parent_text2
            )
            passed_constraints = self._check_constraints(
                new_text, previous_text, original_text=original_text
            )
            return passed_constraints
        else:
            return True

    def _crossover(self, pop_member1, pop_member2, original_text):
        """Generates a crossover between pop_member1 and pop_member2."""
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
                new_text.attack_attrs["last_transformation"] = x1_text.attack_attrs["last_transformation"]
            elif "last_transformation" in x2_text.attack_attrs:
                new_text.attack_attrs["last_transformation"] = x2_text.attack_attrs["last_transformation"]

            if self.post_crossover_check:
                passed_constraints = self._post_crossover_check(new_text, x1_text, x2_text, original_text)

            if not self.post_crossover_check or passed_constraints:
                break

            num_tries += 1

        if self.post_crossover_check and not passed_constraints:
            pop_mem = pop_member1 if np.random.uniform() < 0.5 else pop_member2
            return pop_mem
        else:
            new_results, self._search_over = self.get_goal_results([new_text])
            return PopulationMember(new_text, result=new_results[0], attributes=attributes)

    @abstractmethod
    def _initialize_population(self, initial_result, pop_size):
        """
        Initialize a population of size `pop_size` with `initial_result`.
        """
        raise NotImplementedError()

    def perform_search(self, initial_result):
        self._search_over = False
        population = self._initialize_population(initial_result, self.pop_size)
        pop_size = len(population)
        current_score = initial_result.score

        for _ in range(self.max_iters):
            population = sorted(population, key=lambda x: x.result.score, reverse=True)

            if self._search_over or population[0].result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
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

                if self._search_over:
                    break

            population = [population[0]] + children

        return population[0].result

    def check_transformation_compatibility(self, transformation):
        """The genetic algorithm is specifically designed for word substitutions."""
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
    """Attacks a model with word substitutions using a genetic algorithm."""

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
        """Return a new copy with `new_text`, `new_result`, and updated candidate counts."""
        num_candidate_transformations = np.copy(pop_member.attributes["num_candidate_transformations"])
        num_candidate_transformations[word_idx] = 0
        return PopulationMember(
            new_text,
            result=new_result,
            attributes={"num_candidate_transformations": num_candidate_transformations},
        )

    def _get_word_select_prob_weights(self, pop_member):
        """Get values used to sample one word to transform."""
        return pop_member.attributes["num_candidate_transformations"]

    def _crossover_operation(self, pop_member1, pop_member2):
        """Mix the two population members to generate crossover."""
        indices_to_replace = []
        words_to_replace = []
        num_candidate_transformations = np.copy(pop_member1.attributes["num_candidate_transformations"])

        for i in range(pop_member1.num_words):
            if (
                np.random.uniform() < 0.5
                and i >= 0
                and i < len(pop_member2.words)
                and i < len(pop_member2.attributes["num_candidate_transformations"])
            ):
                indices_to_replace.append(i)
                words_to_replace.append(pop_member2.words[i])
                num_candidate_transformations[i] = pop_member2.attributes["num_candidate_transformations"][i]

        new_text = pop_member1.attacked_text.replace_words_at_indices(indices_to_replace, words_to_replace)
        return new_text, {"num_candidate_transformations": num_candidate_transformations}

    def _initialize_population(self, initial_result, pop_size):
        """
        Initialize a population of size `pop_size` with `initial_result`.
        """
        words = initial_result.attacked_text.words
        num_candidate_transformations = np.zeros(len(words))
        transformed_texts = self.get_transformations(
            initial_result.attacked_text, original_text=initial_result.attacked_text
        )
        for transformed_text in transformed_texts:
            diff_idx = next(iter(transformed_text.attack_attrs["newly_modified_indices"]))
            num_candidate_transformations[diff_idx] += 1

        min_num_candidates = np.amin(num_candidate_transformations)
        epsilon = max(1, int(min_num_candidates * 0.1))
        for i in range(len(num_candidate_transformations)):
            num_candidate_transformations[i] = max(num_candidate_transformations[i], epsilon)

        population = []
        for _ in range(pop_size):
            pop_member = PopulationMember(
                initial_result.attacked_text,
                initial_result,
                attributes={"num_candidate_transformations": np.copy(num_candidate_transformations)},
            )
            pop_member = self._perturb(pop_member, initial_result)
            population.append(pop_member)

        return population


class AttackList(Attack):
    def attack(self, example, ground_truth_output):
        """Attack a single example."""
        assert isinstance(
            example, (str, OrderedDict, AttackedText)
        ), "`example` must either be `str`, `collections.OrderedDict`, `textattack.shared.AttackedText`."
        if isinstance(example, (str, OrderedDict)):
            example = AttackedText(example)

        assert isinstance(
            ground_truth_output, (int, str, list, np.integer)
        ), f"`ground_truth_output` must either be `str`, `int` or list of float but is {type(ground_truth_output)}"
        goal_function_result, _ = self.goal_function.init_attack_example(example, ground_truth_output)
        if goal_function_result.goal_status == GoalFunctionResultStatus.SKIPPED:
            return SkippedAttackResult(goal_function_result)
        else:
            result = self._attack(goal_function_result)
            return result


__all__ = [
    "AttackList",
    "AttackResult_new_diff_color",
    "AlzantotGeneticAlgorithm",
    "GeneticAlgorithm",
    "GreedyWordSwapWIRTruncated",
    "PartOfSpeechTry",
]
