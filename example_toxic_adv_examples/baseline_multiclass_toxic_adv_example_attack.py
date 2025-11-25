from typing import Optional

from textattack import Attack
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
    CompositeTransformation,
    WordSwapHomoglyphSwap,
    WordSwapNeighboringCharacterSwap,
    WordSwapRandomCharacterDeletion,
    WordSwapRandomCharacterInsertion,
)

from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.goal_functions import UntargetedClassification, TargetedClassification

from textattack.attack_recipes import AttackRecipe
from textattack.transformations import WordSwapWordNet

from textattack_multilabel.attack_components import (
    PartOfSpeechTry,
    GreedyWordSwapWIRTruncated,
    AlzantotGeneticAlgorithm,
    AttackList,
)
from textattack_multilabel.goal_function import MultilabelClassificationGoalFunction

class MulticlassACL23(AttackRecipe):

    @staticmethod
    def build(model_wrapper,
              target_class: Optional[int] = None,
              wir_method: str = "delete",
              pos_constraint: bool = True,
              sbert_constraint: bool = False
              ):
        """Build attack recipe.

        Args:
            model_wrapper (:class:`~textattack.models.wrappers.ModelWrapper`):
                Model wrapper containing both the model and the tokenizer.
            mlm (:obj:`bool`, `optional`, defaults to :obj:`False`):
                If :obj:`True`, load `A2T-MLM` attack. Otherwise, load regular `A2T` attack.
            target_class (:obj:`int`, `optional`, defaults to :obj:`None`):
                if specified, will search perturbations to maximize the target class score

        Returns:
            :class:`~textattack.Attack`: A2T attack.
        """
        constraints = [RepeatModification(), StopwordModification()]

        if pos_constraint:
            input_column_modification = InputColumnModification(["premise", "hypothesis"], {"premise"})
            constraints.append(input_column_modification)
            constraints.append(PartOfSpeechTry(allow_verb_noun_swap=False))

        constraints.append(MaxModificationRate(max_rate=0.1, min_threshold=4))

        if sbert_constraint:
            print("Using SBERT constraint!")

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

        transformation = CompositeTransformation(
            [
                # (1) Insert: Insert a space into the word.
                # Generally, words are segmented by spaces in English. Therefore,
                # we can deceive classifiers by inserting spaces into words.
                WordSwapRandomCharacterInsertion(
                    random_one=True,
                    letters_to_insert=" ",
                    skip_first_char=True,
                    skip_last_char=True,
                ),
                # (2) Delete: Delete a random character of the word except for the first
                # and the last character.
                WordSwapRandomCharacterDeletion(
                    random_one=True, skip_first_char=True, skip_last_char=True
                ),
                # (3) Swap: Swap random two adjacent letters in the word but do not
                # alter the first or last letter. This is a common occurrence when
                # typing quickly and is easy to implement.
                WordSwapNeighboringCharacterSwap(
                    random_one=True, skip_first_char=True, skip_last_char=True
                ),
                # (4) Substitute-C (Sub-C): Replace characters with visually similar
                # characters (e.g., replacing “o” with “0”, “l” with “1”, “a” with “@”)
                # or adjacent characters in the keyboard (e.g., replacing “m” with “n”).
                WordSwapHomoglyphSwap(),
                # (5) Substitute-W
                # (Sub-W): Replace a word with its topk nearest neighbors in a
                # context-aware word vector space. Specifically, we use the pre-trained
                # GloVe model [30] provided by Stanford for word embedding and set
                # topk = 5 in the experiment.
                WordSwapEmbedding(max_candidates=20),
            ]
        )

        if target_class is None:
            # Goal is untargeted classification
            goal_function = UntargetedClassification(model_wrapper, model_batch_size=32)
        else:
            # Goal is targeted classification
            goal_function = TargetedClassification(model_wrapper, target_class=target_class, model_batch_size=32)

        if wir_method == 'gradient':
            #
            # Greedily swap words with "Word Importance Ranking".
            #
            max_len = getattr(model_wrapper, "max_length", None) or min(
                1024, model_wrapper.tokenizer.model_max_length, model_wrapper.model.config.max_position_embeddings - 2
            )
            search_method = GreedyWordSwapWIRTruncated(wir_method="gradient", truncate_words_to=max_len)

            constraints.append(MaxWordIndexModification(max_len))  # this constraint is necessary for gradient based search method

        elif wir_method == 'beam':
            search_method = BeamSearch(beam_width=3)

        elif wir_method == 'genetic':
            search_method = AlzantotGeneticAlgorithm(
                pop_size=25, max_iters=10, give_up_if_no_improvement=True, post_crossover_check=False
            )

        else: # wir_method in ['delete', 'weighted-saliency', 'unk']
            search_method = GreedyWordSwapWIRTruncated(wir_method=wir_method)

        return Attack(goal_function, constraints, transformation, search_method)

class MulticlassACL23Transform(AttackRecipe):

    @staticmethod
    def build(model_wrapper,
              target_class: Optional[int] = None,
              wir_method: str = "weighted-saliency",
              transform_method: str = "glove",
              knn: int = 20,
              pos_constraint: bool = True,
              sbert_constraint: bool = False
              ):
        """Build attack recipe.

        """

        constraints = [RepeatModification(), StopwordModification()]

        if pos_constraint:
            input_column_modification = InputColumnModification(["premise", "hypothesis"], {"premise"})
            constraints.append(input_column_modification)
            constraints.append(PartOfSpeechTry(allow_verb_noun_swap=False))

        constraints.append(MaxModificationRate(max_rate=0.1, min_threshold=4))

        if sbert_constraint:
            print("Using SBERT constraint!")
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

        # transformation = CompositeTransformation(
        #     [
        #         # (5) Substitute-W
        #         # (Sub-W): Replace a word with its topk nearest neighbors in a
        #         # context-aware word vector space. Specifically, we use the pre-trained
        #         # GloVe model [30] provided by Stanford for word embedding and set
        #         # topk = 5 in the experiment.
        #
        #         if transform_method == "glove":
        #             WordSwapEmbedding(max_candidates=20),
        #     ]
        # )

        if target_class is None:
            # Goal is untargeted classification
            goal_function = UntargetedClassification(model_wrapper, model_batch_size=32)
        else:
            # Goal is targeted classification
            goal_function = TargetedClassification(model_wrapper, target_class=target_class, model_batch_size=32)

        if wir_method == 'gradient':
            #
            # Greedily swap words with "Word Importance Ranking".
            #
            max_len = getattr(model_wrapper, "max_length", None) or min(
                1024, model_wrapper.tokenizer.model_max_length, model_wrapper.model.config.max_position_embeddings - 2
            )
            search_method = GreedyWordSwapWIRTruncated(wir_method="gradient", truncate_words_to=max_len)

            constraints.append(MaxWordIndexModification(max_len))  # this constraint is necessary for gradient based search method

        elif wir_method == 'beam':
            search_method = BeamSearch(beam_width=3)

        elif wir_method == 'genetic':
            search_method = AlzantotGeneticAlgorithm(
                pop_size=25, max_iters=10, give_up_if_no_improvement=True, post_crossover_check=False
            )

        else: # wir_method in ['delete', 'weighted-saliency', 'unk']
            search_method = GreedyWordSwapWIRTruncated(wir_method=wir_method)

        return Attack(goal_function, constraints, transformation, search_method)
