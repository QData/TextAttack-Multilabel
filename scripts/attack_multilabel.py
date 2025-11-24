"""
Training script for attacking a dataset
Specifically attacking the DistilBERT Binary model

Using multi-label attack capable recipes

Tokenizer: Load using transformers.AutoTokenizer
Model    : Load using AutoModelForSequenceClassification
Attack recipe: Use the multi-label version of the attack recipes
"""

import os
import argparse
import random
import math
import datetime
import logging
import sys
import time
import torch
try:
    torch._C._initExtension(manager_path=r"")
except Exception:
    pass

from pathlib import Path

import textattack
import transformers
import datasets
import pandas as pd
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForSequenceClassification, DistilBertForSequenceClassification
from transformers import RobertaTokenizer, DistilBertTokenizer
from transformers import AutoTokenizer, EarlyStoppingCallback

# from configs import DATASET_CONFIGS
# from s3utility import download_s3_files
# from process_data import process_jigsaw, process_tweet
# from utility import int_or_float,remove_pattern_text, create_local_directory
# f
# import multiprocessing

from datasets import load_from_disk, Dataset, concatenate_datasets, DatasetDict
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import Trainer, TrainingArguments

# from robertox import RoBERToxForSequenceClassification, RoBERToxTokenizer
# from textattack.attack_recipes.multilabel_classification_attacks import MultilabelPruthi2019, MultilabelA2T, \
#     MultilabelTextFooler
# from textattack.attack_recipes.multilabel_classification_attacks import MultilabelDeepWordBugGao2018, \
#     MultilabelPWWSRen2019, MultilabelTextBuggerLi2018
import textattack

from textattack_multilabel.shared import AttackResult_new_diff_color
from textattack_multilabel.model import MultilabelModelWrapper
from textattack_multilabel.multilabel_acl2023 import MultilabelACL23

from textattack import Attacker
import nltk

nltk.download('omw-1.4')

import tensorflow as tf
import numpy as np
from scipy.special import expit

# from textattack.datasets import Dataset
from textattack.loggers import CSVLogger


textattack.attack_results.AttackResult.diff_color = AttackResult_new_diff_color

### needed to avoid this issue: https://github.com/tensorflow/tensorflow/issues/57359
tf.config.experimental.enable_tensor_float_32_execution(False)


from detoxify.detoxify import load_checkpoint


def attack_model(args):

    threshold1 = 0.5
    threshold2 = 0.5

    model, tokenizer, class_names = load_checkpoint(
            model_type="original", device='cuda'
    )

    # tokenizer = AutoTokenizer.from_pretrained('FacebookAI/roberta-base')
    # model = AutoModelForSequenceClassification.from_pretrained(model_name)

    tokenizer.model_max_length=128
    model_wrapper = MultilabelModelWrapper(model, tokenizer, multilabel=True, device='cuda', max_length=128)

    df = pd.read_csv(args.data)
    df=df.assign(labels=df[['toxic', 'severe_toxic','obscene','threat','insult','identity_hate']].values.tolist())
    df=df.rename(columns={'comment_text':'text'})[['text', 'labels']]

    df_benign=df[df.labels.apply(lambda x: np.all(np.asarray(x)<threshold1))]
    df_toxic=df[df.labels.apply(lambda x: np.any(np.asarray(x)>threshold2))]
    df_benign = df_benign.sample(500, random_state=22)
    df_toxic = df_toxic.sample(500, random_state=22)
    
    if args.attack == 'benign':

        dataset = Dataset.from_pandas(df_benign)
        df_ = df_benign

        attack = MultilabelACL23.build(model_wrapper,  # using huggingface model wrapper
                                        labels_to_maximize=list(range(6)), # maximize all labels because all labels are toxic
                                        labels_to_minimize=[], # no labels will be minimized
                                        maximize_target_score=threshold2,
                                        minimize_target_score=threshold1,
                                        wir_method='unk',
                                        pos_constraint=True,
                                        sbert_constraint=False
        )

    else:  # attack == 'harmful'

        dataset = Dataset.from_pandas(df_toxic)
        df_ = df_toxic

        attack = MultilabelACL23.build(model_wrapper,  # using huggingface model wrapper
                                        labels_to_maximize=[], # do not maximize any labels
                                        labels_to_minimize=list(range(6)), # minimize all (toxic) labels
                                        maximize_target_score=threshold2,
                                        minimize_target_score=threshold1,
                                        wir_method='unk',
                                        pos_constraint=True,
                                        sbert_constraint=False
                                        )

    dataset = textattack.datasets.Dataset(
        [(x, y) for x, y in zip(dataset["text"], dataset["labels"])])
    
    attack_args = textattack.AttackArgs(num_examples=-1)  ## check other options available in this args TODO
    attacker = Attacker(attack, dataset, attack_args)
    attack_results = attacker.attack_dataset()

    attack_logger = CSVLogger(color_method='html')
    for result in attack_results:
        attack_logger.log_attack_result(result)
    df_attacks = attack_logger.df

    # Adding the ground truth label back which is fractional in the case of jigsaw
    df_attacks.loc[:, 'ground_truth_label'] = df_['labels']
    # text column which will be used for retraining the base model
    print(df_attacks.columns)
    df_attacks.loc[:, 'text'] = df_attacks['perturbed_text'].replace('<font color = .{1,6}>|</font>', '', regex=True)
    df_attacks['text'] = df_attacks['text'].replace('<SPLIT>', '\n', regex=True)
    print(df_attacks.head())
    print(df_attacks.columns)

    df_attacks["original_output"] = df_attacks["original_output"].apply(lambda x: x.cpu().numpy())
    df_attacks["perturbed_output"] = df_attacks["perturbed_output"].apply(lambda x: x.cpu().numpy())
    df_attacks.to_parquet(args.output)

if __name__ == "__main__":

    ## Initiate the start time of code execution
    start_time = time.time()

    ##Allowing gpu growth for tf, might not be necessary anymore check -- TODO
    # configuration = tf.compat.v1.ConfigProto()
    # configuration.gpu_options.allow_growth = True
    # session = tf.compat.v1.Session(config=configuration)

    ## Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attack",
        type=str,
        required=True,
        choices=['harmful', 'benign'],
        help="Attack 'harmful' or 'benign' examples. Attacking harmful or benign examples will generate benign or harmful adversarial examples, respectively. For example, positive adversarial examples will be predicted as benign (all labels are off) by the target model. ",
    )


    parser.add_argument(
        "--data",
        type=str,
        default='data/jigsaw_toxic_comments/test.csv',
        help="Input data",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output parquet file with attack results",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )

    args = parser.parse_args()
    print(f"Arguments passed to the code : {args}")

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('attack_multilabel.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Arguments passed to the code : {args}")

    # Validate inputs
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Data file not found: {args.data}")
    if not args.output.endswith('.parquet'):
        raise ValueError("Output must be a .parquet file")

    try:
        attack_model(args)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        logger.error("CUDA out of memory. Try reducing batch size or free GPU memory.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
