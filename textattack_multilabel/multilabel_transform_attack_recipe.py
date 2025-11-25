"""
MultilabelACL23Transform Attack Recipe

Alternative attack recipe using single transformation methods (GloVe, MLM, WordNet)
instead of composite transformations.
"""

from typing import Optional, Union, List

from textattack.attack_recipes import AttackRecipe
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
    InputColumnModification,
    MaxModificationRate,
    MaxWordIndexModification
)
from textattack.constraints.semantics.sentence_encoders import SBERT, UniversalSentenceEncoder
from textattack.search_methods import BeamSearch
from textattack.transformations import (
    WordSwapMaskedLM,
    WordSwapEmbedding,
    WordSwapWordNet
)

from textattack_multilabel.attack_components import (
    PartOfSpeechTry,
    GreedyWordSwapWIRTruncated,
    AlzantotGeneticAlgorithm,
    AttackList,
)
from textattack_multilabel.goal_function import MultilabelClassificationGoalFunction


class MultilabelACL23Transform(AttackRecipe):
    """Attack recipe with single transformation methods.

    This recipe is similar to MultilabelACL23_recipe but allows choosing between
    different single transformation methods (glove, mlm, wordnet) instead of using
    composite transformations.

    Args:
        model_wrapper: The model to attack
        labels_to_maximize: List of label indices to maximize (make toxic)
        labels_to_minimize: List of label indices to minimize (make non-toxic)
        maximize_target_score: Target score threshold for maximization
        minimize_target_score: Target score threshold for minimization
        wir_method: Word importance ranking method ('gradient', 'delete', 'weighted-saliency', 'unk', 'beam', 'genetic')
        transform_method: Transformation method to use ('glove', 'mlm', 'wordnet')
        knn: Number of nearest neighbors for word embeddings
        pos_constraint: Whether to use part-of-speech constraint
        sbert_constraint: Whether to use SBERT semantic similarity constraint
    """

    @staticmethod
    def build(model_wrapper,
              labels_to_maximize: Optional[Union[int, List[int]]] = 0,
              labels_to_minimize: Optional[Union[int, List[int]]] = None,
              maximize_target_score: Optional[float] = 0.5,
              minimize_target_score: Optional[float] = 0.5,
              wir_method: str = "weighted-saliency",
              transform_method: str = "glove",
              knn: int = 20,
              pos_constraint: bool = True,
              sbert_constraint: bool = False
              ):
        """Build attack recipe.

        Returns:
            AttackList: Configured attack instance
        """

        # Basic constraints
        constraints = [RepeatModification(), StopwordModification()]

        # Part-of-speech constraint
        if pos_constraint:
            input_column_modification = InputColumnModification(["premise", "hypothesis"], {"premise"})
            constraints.append(input_column_modification)
            constraints.append(PartOfSpeechTry(allow_verb_noun_swap=False))

        constraints.append(MaxModificationRate(max_rate=0.1, min_threshold=4))

        # Semantic similarity constraint
        if sbert_constraint:
            use_constraint = SBERT(
                model_name='all-mpnet-base-v2',
                threshold=0.85,
                metric="cosine",
                compare_against_original=True,
                window_size=15,
                skip_text_shorter_than_window=False,
            )
        else:
            use_constraint = UniversalSentenceEncoder(
                threshold=0.840845057,
                metric="angular",
                compare_against_original=False,
                window_size=15,
                skip_text_shorter_than_window=True,
            )

        constraints.append(use_constraint)

        # Transformation method selection
        if transform_method == "glove":
            transformation = WordSwapEmbedding(max_candidates=knn)
        elif transform_method == "mlm":
            transformation = WordSwapMaskedLM(
                method="bae", max_candidates=knn, min_confidence=0.0, batch_size=16
            )
        elif transform_method == "wordnet":
            transformation = WordSwapWordNet()
        else:
            raise ValueError(f"Unknown transform_method={transform_method}")

        # Goal function
        goal_function = MultilabelClassificationGoalFunction(
            model_wrapper,
            labels_to_maximize=labels_to_maximize,
            labels_to_minimize=labels_to_minimize,
            maximize_target_score=maximize_target_score,
            minimize_target_score=minimize_target_score,
            model_batch_size=32,
        )

        # Search method selection
        if wir_method == 'gradient':
            # Gradient-based word importance ranking with truncation
            max_len = getattr(model_wrapper, "max_length", None) or min(
                1024, model_wrapper.tokenizer.model_max_length, model_wrapper.model.config.max_position_embeddings - 2
            )
            search_method = GreedyWordSwapWIRTruncated(wir_method="gradient", truncate_words_to=max_len)
            constraints.append(MaxWordIndexModification(max_len))

        elif wir_method == 'beam':
            search_method = BeamSearch(beam_width=3)

        elif wir_method == 'genetic':
            search_method = AlzantotGeneticAlgorithm(
                pop_size=25, max_iters=10, give_up_if_no_improvement=True, post_crossover_check=False
            )

        else:  # wir_method in ['delete', 'weighted-saliency', 'unk']
            search_method = GreedyWordSwapWIRTruncated(wir_method=wir_method)

        return AttackList(goal_function, constraints, transformation, search_method)


__all__ = ["MultilabelACL23Transform"]
