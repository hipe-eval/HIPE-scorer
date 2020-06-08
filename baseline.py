#!/usr/bin/env python
# coding: utf-8

"""
Train and evaluate a CRF baseline model for the HIPE Shared Task
"""

import csv
import os
import sklearn_crfsuite

import argparse


from ner_evaluation.utils import *
from clef_evaluation import *


def parse_args():
    """Parse the arguments given with program call"""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-t", "--train", required=True, action="store", dest="f_train", help="name of train file",
    )

    parser.add_argument(
        "-d", "--dev", required=True, action="store", dest="f_dev", help="name of development file",
    )

    parser.add_argument(
        "-p",
        "--pred",
        required=True,
        action="store",
        dest="f_pred",
        help="name of prediction file that the systems outputs",
    )

    parser.add_argument(
        "-c",
        "--cols",
        required=True,
        action="store",
        dest="cols",
        help="name of column for which the baseline is trained (separated by comma if multiple)",
    )

    parser.add_argument(
        "-e",
        "--eval",
        required=False,
        action="store",
        dest="eval",
        default=None,
        help="type of evaluation",
        choices={"nerc_fine", "nerc_coarse"},
    )

    return parser.parse_args()


def word2features(sent, i):
    word = sent[i].TOKEN
    # postag = sent[i][1]

    features = {
        "bias": 1.0,
        "word.lower()": word.lower(),
        "word[-3:]": word[-3:],
        "word[-2:]": word[-2:],
        "word.isupper()": word.isupper(),
        "word.istitle()": word.istitle(),
        "word.isdigit()": word.isdigit(),
        # 'postag': postag,
        # 'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i - 1].TOKEN
        # postag1 = sent[i-1][1]
        features.update(
            {
                "-1:word.lower()": word1.lower(),
                "-1:word.istitle()": word1.istitle(),
                "-1:word.isupper()": word1.isupper(),
                # '-1:postag': postag1,
                # '-1:postag[:2]': postag1[:2],
            }
        )
    else:
        features["BOS"] = True

    if i < len(sent) - 1:
        word1 = sent[i + 1].TOKEN
        # postag1 = sent[i+1][1]
        features.update(
            {
                "+1:word.lower()": word1.lower(),
                "+1:word.istitle()": word1.istitle(),
                "+1:word.isupper()": word1.isupper(),
                # '+1:postag': postag1,
                # '+1:postag[:2]': postag1[:2],
            }
        )
    else:
        features["EOS"] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent, column):
    return [getattr(token, column) for token in sent]


def prepare_data(data, column):
    X = [sent2features(s) for s in data]
    y = [sent2labels(s, column) for s in data]
    return X, y


def collect_predictions(pred, col, y_pred):
    for i_doc, doc in enumerate(pred):
        for i_tok, tok in enumerate(doc):

            label = y_pred[i_doc][i_tok]
            setattr(tok, col, label)

    return pred


def write_predictions(fname, pred, dev):

    header = dev[0][0][0].fieldnames

    with open(fname, "w") as csvfile:
        writer = csv.DictWriter(csvfile, delimiter="\t", fieldnames=header)
        writer.writeheader()
        # get segmentation structure from dev set
        for i_doc, doc in enumerate(dev):
            writer.writerow({"TOKEN": "# document_id"})
            tok_pos_start = 0
            for sent in doc:
                writer.writerow({"TOKEN": "# segment_iiif_link"})
                tok_pos_end = tok_pos_start + len(sent)
                for i_tok in range(tok_pos_start, tok_pos_end):
                    writer.writerow(pred[i_doc][i_tok].get_values())

                tok_pos_start += len(sent)
            # add empty line after each document according to gold standard
            writer.writerow({"TOKEN": ""})


def pipeline(f_train, f_pred, f_dev, cols, eval):

    # get data
    train = read_conll_annotations(f_train)
    dev = read_conll_annotations(f_dev, structure_only=True)
    pred = read_conll_annotations(f_dev, structure_only=True)

    # flatten documents to represent an entire document as a single sentence
    train = [[tok for sent in doc for tok in sent] for doc in train]
    pred = [[tok for sent in doc for tok in sent] for doc in pred]

    for col in cols.split(","):

        # preprocessing
        X_train, y_train = prepare_data(train, col)
        X_pred, y_pred = prepare_data(pred, col)

        # training
        crf = sklearn_crfsuite.CRF(
            algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True
        )
        crf.fit(X_train, y_train)

        # predicting
        y_pred = crf.predict(X_pred)
        pred = collect_predictions(pred, col, y_pred)

    write_predictions(f_pred, pred, dev)

    if eval:
        get_results(f_dev, f_pred, eval)


if __name__ == "__main__":
    args = parse_args()
    pipeline(args.f_train, args.f_pred, args.f_dev, args.cols, args.eval)
