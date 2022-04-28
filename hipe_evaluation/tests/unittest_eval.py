#!/usr/bin/env python3
# coding: utf-8


"""
Unit test to check the evaluation results for the HIPE Shared Task
"""

from hipe_evaluation.ner_eval import Evaluator

import unittest
import json


class TestEvaluationResults(unittest.TestCase):
    def setUp(self):
        self.n_docs = 2
        self.n_segments = 21
        self.n_tokens = 129
        self.evaluator = Evaluator(
            "hipe_evaluation/tests/data/unittest-true_bundle3_de_1.tsv",
            "hipe_evaluation/tests/data/unittest-pred_bundle3_de_1.tsv",
        )
        self.assertEqual(self.evaluator.n_docs_true, self.n_docs, "Not all documents were parsed")
        self.assertEqual(self.evaluator.n_lines_true, self.n_segments, "Not all lines were parsed")
        self.assertEqual(self.evaluator.n_toks_true, self.n_tokens, "Not all tokens were parsed")

    def test_eval_results_nerc(self):
        ref_path = "hipe_evaluation/tests/data/ref_results_nerc_fine_all.json"

        eval_global, eval_per_tag = self.evaluator.evaluate(
            "NE-FINE-LIT", eval_type="nerc", tags=None, merge_lines=True
        )
        eval_per_tag["ALL"] = eval_global

        # with open("results_nerc_fine_all.json", "w") as jsonfile:
        #     json.dump(
        #         eval_per_tag, jsonfile, indent=4,
        #     )

        self._compare_eval_results(ref_path, eval_per_tag)

    def test_eval_results_nel(self):
        ref_path = "hipe_evaluation/tests/data/ref_results_nel_all.json"

        eval_global, eval_per_tag = self.evaluator.evaluate(
            "NEL-LIT", eval_type="nel", tags=None, merge_lines=True, n_best=3
        )
        eval_per_tag["ALL"] = eval_global

        # with open("results_nel_all.json", "w") as jsonfile:
        #     json.dump(
        #         eval_per_tag, jsonfile, indent=4,
        #     )

        self._compare_eval_results(ref_path, eval_per_tag)

    def test_eval_results_nel_union(self):
        ref_path = "hipe_evaluation/tests/data/ref_results_nel_all.json"

        eval_global, eval_per_tag = self.evaluator.evaluate(
            ["NEL-LIT", "NEL-METO"], eval_type="nel", tags=None, merge_lines=True, n_best=1
        )
        eval_per_tag["ALL"] = eval_global

        with open("results_nel_all.json", "w") as jsonfile:
            json.dump(
                eval_per_tag, jsonfile, indent=4,
            )

        self._compare_eval_results(ref_path, eval_per_tag)

    def test_eval2022_results_nel(self):
        ref_path = "hipe_evaluation/tests/data/ref_results_nel_all.json"

        eval_global, eval_per_tag = self.evaluator.evaluate(
            "NEL-LIT",
            eval_type="nel",
            tags=None,
            merge_lines=True,
            n_best=3,
            additional_columns=["NE-COARSE-LIT"]
        )
        eval_per_tag["ALL"] = eval_global

        # with open("results_nel_all.json", "w") as jsonfile:
        #     json.dump(
        #         eval_per_tag, jsonfile, indent=4,
        #     )

        self._compare_eval_results(ref_path, eval_per_tag)

    def test_eval2022_results_nel_union(self):
        ref_path = "hipe_evaluation/tests/data/ref_results_nel_all.json"

        eval_global, eval_per_tag = self.evaluator.evaluate(
            ["NEL-LIT", "NEL-METO"],
            eval_type="nel",
            tags=None,
            merge_lines=True,
            n_best=1,
            additional_columns=["NE-COARSE-LIT"]
        )
        eval_per_tag["ALL"] = eval_global

        with open("results_nel_all.json", "w") as jsonfile:
            json.dump(
                eval_per_tag, jsonfile, indent=4,
            )

        self._compare_eval_results(ref_path, eval_per_tag)

    def _compare_eval_results(self, ref_path, tst):
        with open(ref_path) as f_ref:
            ref = json.load(f_ref)

        for eval_type in ref:
            for label in ref[eval_type]:
                for metric in ref[eval_type][label]:
                    self.assertAlmostEqual(
                        ref[eval_type][label][metric],
                        tst[eval_type][label][metric],
                        msg=f"Evaluation results are not in line with reference. "
                        + f"Mismatch in literal column for {eval_type}, {label}, {metric}",
                    )


if __name__ == "__main__":
    unittest.main()
