from typing import List, Optional, Union

from textattack.goal_functions import GoalFunction
import torch
import numpy as np

from textattack.goal_function_results import GoalFunctionResult
from textattack.search_methods import GreedyWordSwapWIR, BeamSearch
from textattack.constraints.grammaticality import PartOfSpeech
from torch.nn.functional import softmax

import transformers

import textattack
from textattack.models.helpers import T5ForTextToText
from textattack.models.tokenizers import T5Tokenizer

from textattack.models.wrappers.pytorch_model_wrapper import PyTorchModelWrapper

from textattack.attack import Attack

from abc import ABC, abstractmethod

import numpy as np
import torch

from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import PopulationBasedSearch, PopulationMember
from textattack.shared.validators import transformation_consists_of_word_swaps


from collections import OrderedDict
from typing import List, Union

import lru
import numpy as np
import torch

import textattack
from textattack.attack_results import (
    FailedAttackResult,
    MaximizedAttackResult,
    SkippedAttackResult,
    SuccessfulAttackResult,
)

from textattack.constraints import Constraint, PreTransformationConstraint
from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.goal_functions import GoalFunction
from textattack.models.wrappers import ModelWrapper
from textattack.search_methods import SearchMethod
from textattack.shared import AttackedText, utils
from textattack.transformations import CompositeTransformation, Transformation



class MultilabelModelWrapper(PyTorchModelWrapper):
    """Loads a HuggingFace ``transformers`` model and tokenizer."""

    def __init__(self, model, tokenizer, multilabel=True, max_length=None, device='cuda'):
        assert isinstance(
            model, (transformers.PreTrainedModel, T5ForTextToText)
        ), f"`model` must be of type `transformers.PreTrainedModel`, but got type {type(model)}."
        assert isinstance(
            tokenizer,
            (
                transformers.PreTrainedTokenizer,
                transformers.PreTrainedTokenizerFast,
                T5Tokenizer,
            ),
        ), f"`tokenizer` must of type `transformers.PreTrainedTokenizer` or `transformers.PreTrainedTokenizerFast`, but got type {type(tokenizer)}."
        self.device = device
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.multilabel = multilabel

        # Default max length is set to be int(1e30), so we force 512 to enable batching.
        self.max_length = max_length or min(
            1024,
            tokenizer.model_max_length,
            model.config.max_position_embeddings - 2
        )


    def __call__(self, text_input_list):
        """Passes inputs to HuggingFace models as keyword arguments.

        (Regular PyTorch ``nn.Module`` models typically take inputs as
        positional arguments.)
        """
        inputs_dict = self.tokenizer(
            text_input_list,
            add_special_tokens=True,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        inputs_dict.to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs_dict)

        if isinstance(outputs[0], str):
            # HuggingFace sequence-to-sequence models return a list of
            # string predictions as output. In this case, return the full
            # list of outputs.
            return outputs
        else:
            # HuggingFace classification models return a tuple as output
            # where the first item in the tuple corresponds to the list of
            # scores for each input.

            if self.multilabel:
                return outputs.logits.sigmoid()
            else: 
                return outputs.logits

    def get_grad(self, text_input):
        """Get gradient of loss with respect to input tokens.

        Args:
            text_input (str): input string
        Returns:
            Dict of ids, tokens, and gradient as numpy array.
        """
        if isinstance(self.model, textattack.models.helpers.T5ForTextToText):
            raise NotImplementedError(
                "`get_grads` for T5FotTextToText has not been implemented yet."
            )

        self.model.train()
        embedding_layer = self.model.get_input_embeddings()
        original_state = embedding_layer.weight.requires_grad
        embedding_layer.weight.requires_grad = True

        emb_grads = []

        def grad_hook(module, grad_in, grad_out):
            emb_grads.append(grad_out[0])

        emb_hook = embedding_layer.register_full_backward_hook(grad_hook)

        self.model.zero_grad()

        input_dict = self.tokenizer(
            [text_input],
            add_special_tokens=True,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )
        input_dict.to(self.device)
        predictions = self.model(**input_dict).logits

        try:
            if self.multilabel:
                labels = predictions.sigmoid()
            else:
                labels = predictions.argmax(dim=1)

            loss = self.model(**input_dict, labels=labels)[0]
        except TypeError:
            raise TypeError(
                f"{type(self.model)} class does not take in `labels` to calculate loss. "
                "One cause for this might be if you instantiatedyour model using `transformer.AutoModel` "
                "(instead of `transformers.AutoModelForSequenceClassification`)."
            )

        loss.backward()

        # grad w.r.t to word embeddings
        grad = emb_grads[0][0].cpu().numpy()

        embedding_layer.weight.requires_grad = original_state
        emb_hook.remove()
        self.model.eval()

        output = {"ids": input_dict["input_ids"], "gradient": grad}

        return output

    def _tokenize(self, inputs):
        """Helper method that for `tokenize`
        Args:
            inputs (list[str]): list of input strings
        Returns:
            tokens (list[list[str]]): List of list of tokens as strings
        """
        try:
            encodings = self.tokenizer(inputs, truncation=True)
            num_texts = len(inputs)
            return [encodings.tokens(i) for i in range(num_texts)]
        except ValueError:
            return [
                self.tokenizer.convert_ids_to_tokens(
                    self.tokenizer([x], truncation=True)["input_ids"][0]
                )
                for x in inputs
            ]

