#!/usr/bin/env python3
# coding: utf-8


"""
Unit test to check the evaluation results for the HIPE Shared Task

Each test case is responsible for reading
 - true gold predictions
 - system predictions
 - reference evaluation json data (expected output from evaluator)


Reference evaluation json data has the following format:
 - NER: for each ner type a bunch of evaluation metrics
 - NEL: for each QID a bunch of evaluation metrics

"""
import os

from hipe_evaluation.ner_eval import Evaluator
from typing import Dict, Optional, Any
import unittest
import logging
import json


class TestEvaluationResults(unittest.TestCase):
    def _test_hipe2020(self):

        evaluator: Evaluator = Evaluator(
            "hipe_evaluation/tests/data/unittest-true_bundle3_de_2020.tsv",
            "hipe_evaluation/tests/data/unittest-pred_bundle3_de_2020.tsv",
        )
        self.assertEqual(evaluator.n_docs_true, 2, "Not all documents were parsed")
        self.assertEqual(evaluator.n_lines_true, 21, "Not all lines were parsed")
        self.assertEqual(evaluator.n_toks_true, 129, "Not all tokens were parsed")
        nerc_fine_reference_data = "hipe_evaluation/tests/data/unittest-pred_bundle3_de_2020.ref_results_nerc_fine.json"
        self._do_evaluation(
            evaluator,
            nerc_fine_reference_data,
            column_name="NE-FINE-LIT",
            eval_type="nerc",
        )

    def test_ner_lit_1(self):
        """Test 1: 1 NER-COARSE-LIT entity in gold, 0 in system response."""

        true_path = "hipe_evaluation/tests/data/unittest-ner-lit-1-true.tsv"
        pred_path = true_path.replace("-true", "-pred")
        eval_reference_path = pred_path + ".ref_results_ner-coarse-lit.json"
        evaluator: Evaluator = Evaluator(
            true_path,
            pred_path,
        )
        self.assertEqual(evaluator.n_docs_true, 1, "Not all documents were parsed")
        self.assertEqual(evaluator.n_lines_true, 1, "Not all layout lines  were parsed")
        self.assertEqual(evaluator.n_toks_true, 16, "Not all tokens were parsed")

        self._do_evaluation(
            evaluator,
            eval_reference_path,
            column_name="NE-COARSE-LIT",
            eval_type="nerc",
        )

    def _do_evaluation(
        self,
        evaluator: Evaluator,
        eval_reference_path: str,
        column_name: str = "NE-COARSE-LIT",
        eval_type: str = "nerc",
        tags=None,
        merge_lines: bool = False,
    ):
        """Run evaluator and compare to reference data"""
        eval_global, eval_per_tag = evaluator.evaluate(
            column_name, eval_type=eval_type, tags=tags, merge_lines=merge_lines
        )
        eval_per_tag["ALL"] = eval_global

        self._compare_eval_results(eval_reference_path, eval_per_tag)

    def _test_eval_results_nel(self):
        ref_path = "hipe_evaluation/tests/results/ref_results_nel_all.json"

        eval_global, eval_per_tag = self.evaluator.evaluate(
            "NEL-LIT", eval_type="nel", tags=None, merge_lines=True, n_best=3
        )
        eval_per_tag["ALL"] = eval_global

        # with open("results_nel_all.json", "w") as jsonfile:
        #     json.dump(
        #         eval_per_tag, jsonfile, indent=4,
        #     )

        self._compare_eval_results(ref_path, eval_per_tag)

    def _test_eval_results_nel_union(self):
        ref_path = "hipe_evaluation/tests/results/ref_results_nel_all.json"

        eval_global, eval_per_tag = self.evaluator.evaluate(
            ["NEL-LIT", "NEL-METO"],
            eval_type="nel",
            tags=None,
            merge_lines=True,
            n_best=1,
        )
        eval_per_tag["ALL"] = eval_global

        with open("results_nel_all.json", "w") as jsonfile:
            json.dump(
                eval_per_tag,
                jsonfile,
                indent=4,
            )

        self._compare_eval_results(ref_path, eval_per_tag)

    def _compare_eval_results(self, ref_path: str, tst):
        # in case the ref_path does not exist already
        # we populate it with the tst data.
        # A manual check/selection of evaluations/fields is necessary
        if not os.path.exists(ref_path) or os.stat(ref_path).st_size == 0:
            with open(ref_path, "w") as f_ref:
                json.dump(tst, f_ref, indent=4)
                msg = (
                    f"Reference evaluation file {ref_path} didn't exist so far or was empty. "
                    "It was filled with the results from the current evaluation results."
                    "Please check and edit carefully."
                )
                logging.warning(msg)

        with open(ref_path) as f_ref:
            ref = json.load(f_ref)
        ref_path_sorted = ref_path + ".sorted.tmp"
        with open(ref_path_sorted, "w") as ref_sorted:
            json.dump(ref, ref_sorted, sort_keys=True, indent=4)

        tst_path = ref_path.replace(".ref_results", ".tst_results")
        tst_path_sorted = tst_path
        if ref_path != tst_path:
            tst_path_sorted += ".sorted.tmp"
            with open(tst_path + ".sorted.tmp", "w") as f_tst:
                json.dump(tst, f_tst, sort_keys=True, indent=4)
        else:
            logging.warning(
                f"Reference path filename MUST CONTAIN with '.ref_results' for diff output."
            )

        for eval_type in ref:
            for label in ref[eval_type]:
                for metric in ref[eval_type][label]:
                    self.assertAlmostEqual(
                        ref[eval_type][label][metric],
                        tst[eval_type][label][metric],
                        msg=f"Evaluation mismatch found: \ndiff '{ref_path_sorted}'  '{tst_path_sorted}'\n'"
                        + f"Evaluation type: '{eval_type}'; label:  '{label}'; metric: '{metric}'",
                    )


if __name__ == "__main__":
    unittest.main()