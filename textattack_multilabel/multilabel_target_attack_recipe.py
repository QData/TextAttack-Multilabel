"""
MultilabelACL23_recipe Attack Recipe

Main attack recipe using composite transformations for multilabel adversarial attacks.
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
from textattack.constraints.semantics.sentence_encoders import (
    SBERT,
    UniversalSentenceEncoder,
)
from textattack.constraints.semantics import WordEmbeddingDistance
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

from textattack_multilabel.attack_components import (
    PartOfSpeechTry,
    GreedyWordSwapWIRTruncated,
    AlzantotGeneticAlgorithm,
    AttackList,
)
from textattack_multilabel.goal_function import MultilabelClassificationGoalFunction

class MultilabelACL23_recipe(AttackRecipe):
    @staticmethod
    def build(
        model_wrapper,
        labels_to_maximize: Optional[Union[int, List[int]]] = 0,
        labels_to_minimize: Optional[Union[int, List[int]]] = None,
        maximize_target_score: Optional[float] = 0.5,
        minimize_target_score: Optional[float] = 0.5,
        wir_method: str = "delete",
        pos_constraint: bool = True,
        sbert_constraint: bool = False,
        sbert_model_name: str = "all-mpnet-base-v2",
        sbert_threshold: float = 0.75,
        universal_encoder_threshold: float = 0.840845057,
        **kwargs,
    ):
        constraints = [RepeatModification(), StopwordModification()]

        if pos_constraint:
            input_column_modification = InputColumnModification(["premise", "hypothesis"], {"premise"})
            constraints.append(input_column_modification)
            constraints.append(PartOfSpeechTry(allow_verb_noun_swap=False))

        constraints.append(MaxModificationRate(max_rate=0.1, min_threshold=4))

        # sent_encoder = BERT(
        #     model_name="sentence-transformers/all-MiniLM-L6-v2",
        #     threshold=0.9,
        #     metric="cosine",
        # )

        #
        # constraints.append(sent_encoder)

        if sbert_constraint:
            print("Using SBERT constraint!")

            use_constraint = SBERT(
                model_name=sbert_model_name,
                threshold=sbert_threshold,
                metric="cosine",
                compare_against_original=True,
                window_size=15,
                skip_text_shorter_than_window=True,
            )
            constraints.append(use_constraint)
        else:
            # Skip UniversalSentenceEncoder to avoid TensorFlow Hub issues
            # that can cause hanging during model loading
            print("Skipping UniversalSentenceEncoder constraint to avoid potential TensorFlow Hub loading issues")
            pass

        # constraints.append(UniversalSentenceEncoder(threshold=0.8))

        # constraints.append(WordEmbeddingDistance(min_cos_sim=0.5))

        # if mlm:
        #     transformation = WordSwapMaskedLM(method="bae", max_candidates=20, min_confidence=0.0, batch_size=16)
        # else:
        #     transformation = WordSwapEmbedding(max_candidates=20)
        #     constraints.append(WordEmbeddingDistance(min_cos_sim=0.8))

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

        # transformation = WordSwapEmbedding(max_candidates=20)

        #
        # Goal is untargeted classification
        #
        goal_function = MultilabelClassificationGoalFunction(
            model_wrapper,
            labels_to_maximize=labels_to_maximize,
            labels_to_minimize=labels_to_minimize,
            maximize_target_score=maximize_target_score,
            minimize_target_score=minimize_target_score,
            model_batch_size=32,
        )

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

        return AttackList(goal_function, constraints, transformation, search_method)


__all__ = ["MultilabelACL23_recipe"]
