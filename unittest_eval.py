#!/usr/bin/env python3
# coding: utf-8


"""
Unit testing to check the evaluation results for the HIPE Shared Task
"""

from ner_evaluation.ner_eval import Evaluator

import unittest
import filecmp
import os


class TestStringMethods(unittest.TestCase):
    def setUp(self):
        pass

    def test_eval_results(self):
        tst_path = "test/results.tsv"
        ref_path = "test/results_ref.tsv"
        self.assertTrue(
            filecmp.cmp(tst_path, ref_path, shallow=False),
            "Evaluation results are different than expected",
        )

    def test_segmentation(self):
        ev = Evaluator(
            "test/synthetic_testing_data_gold.tsv",
            "test/synthetic_testing_data_pred.tsv",
        )
        n_docs = 2
        n_segments = 21
        n_tokens = 132
        self.assertEqual(ev.n_docs_true, n_docs, "Not all documents were parsed")
        self.assertEqual(ev.n_lines_true, n_segments, "Not all lines were parsed")
        self.assertEqual(ev.n_toks_true, n_tokens, "Not all tokens were parsed")


if __name__ == "__main__":
    os.system(
        "python clef_evaluation.py -g test/synthetic_testing_data_gold.tsv -p test/synthetic_testing_data_pred.tsv -t nerc_fine"
    )
    unittest.main()
