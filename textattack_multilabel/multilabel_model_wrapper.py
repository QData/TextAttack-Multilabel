from typing import List, Optional, Union

import torch
import transformers

from textattack.models.helpers import T5ForTextToText
from textattack.models.tokenizers import T5Tokenizer
from textattack.models.wrappers.pytorch_model_wrapper import PyTorchModelWrapper



class MultilabelModelWrapper(PyTorchModelWrapper):
    """Loads a HuggingFace ``transformers`` model and tokenizer."""

    def __init__(self, model, tokenizer, multilabel=True, max_length=None, device=None):
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

        # Auto-detect device if not specified
        if device is None:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

