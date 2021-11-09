from allennlp.models import model
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.data import DataIterator#
from allennlp.data.dataset_readers import DatasetReader
from allennlp.models import Model
from allennlp.training import Trainer
from allennlp.training.util import evaluate
from allennlp.common.util import prepare_global_logging, cleanup_global_logging, prepare_environment
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, PretrainedBertIndexer
from allennlp.data import vocabulary
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from models import NSDSlotTaggingModel
from predictors import SlotFillingPredictor
from dataset_readers import MultiFileDatasetReader
from metrics import NSDSpanBasedF1Measure
from utils import *

from typing import Any, Union, Dict, Iterable, List, Optional, Tuple
from time import *
import numpy as np
import pandas as pd
import argparse
import os
import logging

vocabulary.DEFAULT_OOV_TOKEN = "[UNK]"  # set for bert

VOCAB = ('<PAD>', '[CLS]', '[SEP]', 'O', 'B-ns', 'I-ns')

tag2idx = {tag: idx for idx, tag in enumerate(VOCAB)}
idx2tag = {idx: tag for idx, tag in enumerate(VOCAB)}

def parse_args():
    arg_parser = argparse.ArgumentParser()
    # arg_parser.add_argument("--mode",type=str,choices=["train", "test", "both"], default="test",
    #                     help="Specify running mode: only train, only test or both.")
    # arg_parser.add_argument("--dataset",type=str,choices=["SnipsNSD5%", "SnipsNSD15%", "SnipsNSD30%"], default=None,
    #                     help="The dataset to use.")
    arg_parser.add_argument("--output_dir",type=str, default="./output",
                        help="The path of trained model.")
    # arg_parser.add_argument("--cuda",type=int, default=1,
    #                     help="cuda device.")
    # arg_parser.add_argument("--threshold", type=float,default=None,
    #                     help="The specified threshold value.")
    # arg_parser.add_argument("--batch_size",type=int, default=200,
    #                     help="Batch size.")
    arg_parser.add_argument("--result",type=str, default="temp_10",
                        help="The path of result.")
    args = arg_parser.parse_args()
    return args

args = parse_args()

model_dir = args.output_dir




# predict
# archive = load_archive(model_dir,cuda_device=args.cuda)
# predictor = Predictor.from_archive(archive=archive, predictor_name="slot_filling_predictor")
# train_outputs = predictor.predict_multi(file_path = os.path.join("data",args.dataset,"train") ,batch_size = args.batch_size)
# test_outputs = predictor.predict_multi(file_path = os.path.join("data",args.dataset,"test") ,batch_size = args.batch_size)
# ns_labels = ["ns","B-ns","I-ns"]

# GDA
# gda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=None, store_covariance=True)
# gda.fit(np.array(train_outputs["encoder_outs"]), train_outputs["true_labels"])
# gda_means = gda.means_ 

# test_gda_result = confidence(np.array(test_outputs["encoder_outs"]), gda.means_, "euclidean", gda.covariance_)
# test_score = pd.Series(test_gda_result.min(axis=1))
# test_ns_idx = [idx_vo for idx_vo , _vo in enumerate(test_outputs["true_labels"]) if _vo in ns_labels]
# test_ind_idx = [idx_vi for idx_vi , _vi in enumerate(test_outputs["true_labels"]) if _vi not in ns_labels]
# test_ns_score = test_score[test_ns_idx]
# test_ind_score = test_score[test_ind_idx]

# threshold
# threshold = args.threshold

# override
# test_y_ns = pd.Series(test_outputs["predict_labels"])
# test_y_ns[test_score[test_score> threshold].index] = "ns"
# test_y_ns = list(test_y_ns)

# Metrics —— ROSE
# start_idx = 0
# end_idx = 0
test_pred_lines = []
test_true_lines = []
# seq_lines = pd.DataFrame(test_outputs["tokens"])
# for i,seq in enumerate(seq_lines["tokens"]):
#     start_idx = end_idx
#     end_idx = start_idx + len(seq)
#     adju_pred_line = parse_line(test_y_ns[start_idx:end_idx])
#     test_true_line = test_outputs["true_labels"][start_idx:end_idx]
#     test_pred_lines.append(adju_pred_line)
#     test_true_lines.append(test_true_line)
# rose_metric(test_true_lines,test_pred_lines)


result = open(args.result, "r", encoding='utf-8')

lines = result.read().splitlines()

entries = result.read().strip().split('\n\n')

for entry in entries:
    true = []
    pred = []
    for line in entry.splitlines():
        if tag2idx[line.split()[1]] != "[SEP]":
            true.append(line.split()[1])
            pred.append(line.split()[2])

    # true = [tag2idx[line.split()[1]] for line in entry.splitlines()]
    # pred = [tag2idx[line.split()[2]] for line in entry.splitlines()]
    
    test_true_lines.append(true)
    test_pred_lines.append(pred)

rose_metric(test_true_lines,test_pred_lines)

# Metrics —— Token
# test_pred_tokens = parse_token(test_y_ns)
# test_true_tokens = parse_token(test_outputs["true_labels"])

test_true_tokens = []
test_pred_tokens = []


for line in lines:
    if len(line) > 0 and tag2idx[line.split()[1]] != "[SEP]":
        true.append(line.split()[1])
        pred.append(line.split()[2])


# test_true_tokens =  [line.split()[1] for line in lines if len(line) > 0]
# test_pred_tokens =  [line.split()[2] for line in lines if len(line) > 0]


token_metric(test_true_tokens,test_pred_tokens)
