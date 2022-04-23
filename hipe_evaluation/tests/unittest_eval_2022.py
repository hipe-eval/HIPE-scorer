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

Entity matching scenarios (as in compute_metrics):
    Scenario I  : exact match of both type and boundaries (TP).
    Scenario II : spurious entity (insertion, FP).
    Scenario III: missed entity (deletion, FN).
    Scenario IV : type substitution (counted as both FP and FN in strict and fuzzy regimes).
    Scenario V  : span substitution (overlap) (counted as both FP and FN in strict regime and as TP in fuzzy regime).
    Scenario VI : type and span substitution (overlap) (counted as FP in strict and fuzzy regimes).


"""
import os

from hipe_evaluation.ner_eval import Evaluator
from typing import Dict, Optional, Any, Set
import unittest
import logging
import json


def get_hipe_2022_tagset_all(file: str = "./tagset-hipe2022-all.txt") -> Set[str]:
    with open(file) as f_in:
        tagset = set(f_in.read().upper().splitlines())
    return tagset


class TestEvaluationResults(unittest.TestCase):
    """Class for 2022 HIPE evaluation unittests"""

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
        """Test data 1: 1 NER-COARSE-LIT entity in gold, 0 in system response.
        (cf. scenario III)
        """

        true_path = "hipe_evaluation/tests/data/unittest-ner-1-true.tsv"
        pred_path = true_path.replace("-true", "-pred")
        eval_reference_path = pred_path + ".ner-coarse-lit_ref_results.json"
        evaluator: Evaluator = Evaluator(
            true_path,
            pred_path,
        )
        self.assertEqual(evaluator.n_docs_true, 1, "Not all documents were parsed")
        self.assertEqual(
            evaluator.n_lines_true, 1, "Not all layout lines were parsed"
        )  # although there are 2 sent
        self.assertEqual(evaluator.n_toks_true, 16, "Not all tokens were parsed")

        self._do_evaluation(
            evaluator,
            eval_reference_path,
            column_name="NE-COARSE-LIT",
            eval_type="nerc",
            tags=get_hipe_2022_tagset_all(),
        )

    def test_ner_lit_2_coarse(self):
        """Test data 2: NE-COARSE-LIT: 2 entity in gold, 2 in system response.
        (cf. scenario I)
        """

        true_path = "hipe_evaluation/tests/data/unittest-ner-2-true.tsv"
        pred_path = true_path.replace("-true", "-pred")
        eval_reference_path = pred_path + ".coarse-lit_ref_results.json"
        evaluator: Evaluator = Evaluator(
            true_path,
            pred_path,
        )
        self.assertEqual(evaluator.n_docs_true, 1, "Not all documents were parsed")
        self.assertEqual(
            evaluator.n_lines_true, 1, "Not all layout lines were parsed"
        )  # although there are 2 sent
        self.assertEqual(evaluator.n_toks_true, 32, "Not all tokens were parsed")

        self._do_evaluation(
            evaluator,
            eval_reference_path,
            column_name="NE-COARSE-LIT",
            eval_type="nerc",
            macro=False,
        )

    def test_ner_lit_2_nested(self):
        """Test 2: NE-NESTED: 1 entity in gold (Hambourg as loc.adm.town), 0 in system response.
        (cf. scenario I)
        """

        true_path = "hipe_evaluation/tests/data/unittest-ner-2-true.tsv"
        pred_path = true_path.replace("-true", "-pred")
        eval_reference_path = pred_path + ".nested_ref_results.json"
        evaluator: Evaluator = Evaluator(
            true_path,
            pred_path,
        )
        self.assertEqual(evaluator.n_docs_true, 1, "Not all documents were parsed")
        self.assertEqual(
            evaluator.n_lines_true, 1, "Not all layout lines were parsed"
        )  # although there are 2 sent
        self.assertEqual(evaluator.n_toks_true, 32, "Not all tokens were parsed")

        self._do_evaluation(
            evaluator,
            eval_reference_path,
            column_name="NE-NESTED",
            eval_type="nerc",
            tags=get_hipe_2022_tagset_all(),
            macro=False,
        )

    def test_ner_lit_2_fine(self):
        """Test 2: NE-NESTED: 1 entity in gold (Hambourg as loc.adm.town), 0 in system response.
        (cf. scenario I)
        """

        true_path = "hipe_evaluation/tests/data/unittest-ner-2-true.tsv"
        pred_path = true_path.replace("-true", "-pred")
        eval_reference_path = pred_path + ".fine-lit_ref_results.json"
        evaluator: Evaluator = Evaluator(
            true_path,
            pred_path,
        )
        self.assertEqual(evaluator.n_docs_true, 1, "Not all documents were parsed")
        self.assertEqual(
            evaluator.n_lines_true, 1, "Not all layout lines were parsed"
        )  # although there are 2 sent
        self.assertEqual(evaluator.n_toks_true, 32, "Not all tokens were parsed")

        self._do_evaluation(
            evaluator,
            eval_reference_path,
            column_name="NE-FINE-LIT",
            eval_type="nerc",
            tags=get_hipe_2022_tagset_all(),
            macro=False,
        )

    def test_ner_lit_3(self):
        """Test 3:
        3 NER-COARSE-LIT entity in gold, 3 in system response, with 1 partial (boundary overlap).
        Details:
        - 1 ORG (Société Suisse des imprimeurs): scenario I
        - 1 LOC (Frauenfeld): scenario I
        - 1 LOC (ville de Berne): scenario V
        """

        true_path = "hipe_evaluation/tests/data/unittest-ner-lit-coarse-3-true.tsv"
        pred_path = true_path.replace("-true", "-pred")
        eval_reference_path = pred_path + ".ref_results.json"
        evaluator: Evaluator = Evaluator(
            true_path,
            pred_path,
        )
        self.assertEqual(evaluator.n_docs_true, 1, "Not all documents were parsed")
        self.assertEqual(
            evaluator.n_lines_true, 1, "Not all layout lines were parsed"
        )  # although there are 2 sent
        self.assertEqual(evaluator.n_toks_true, 37, "Not all tokens were parsed")

        self._do_evaluation(
            evaluator,
            eval_reference_path,
            column_name="NE-COARSE-LIT",
            eval_type="nerc",
            macro=False,
        )

    def test_ner_lit_4(self):
        """Test 4:
        3 NER-COARSE-LIT entity in gold, 3 in system response, with 1 partial (=exact boundaries but wrong type)
        Details:
        - 1 ORG (Société Suisse des imprimeurs): scenario IV
        - 1 LOC (Frauenfeld): scenario I
        """

        true_path = "hipe_evaluation/tests/data/unittest-ner-lit-coarse-4-true.tsv"
        pred_path = true_path.replace("-true", "-pred")
        eval_reference_path = pred_path + ".ref_results.json"
        evaluator: Evaluator = Evaluator(
            true_path,
            pred_path,
        )
        self.assertEqual(evaluator.n_docs_true, 1, "Not all documents were parsed")
        self.assertEqual(evaluator.n_lines_true, 1, "Not all layout lines were parsed")
        self.assertEqual(evaluator.n_toks_true, 24, "Not all tokens were parsed")

        self._do_evaluation(
            evaluator,
            eval_reference_path,
            column_name="NE-COARSE-LIT",
            eval_type="nerc",
            tags=get_hipe_2022_tagset_all(),
            macro=False,
        )

    def _do_evaluation(
        self,
        evaluator: Evaluator,
        eval_reference_path: str,
        column_name: str = "NE-COARSE-LIT",
        eval_type: str = "nerc",
        tags=None,
        merge_lines: bool = False,
        macro: bool = True,
    ):
        """Run evaluator and compare to reference data"""

        eval_global, eval_per_tag = evaluator.evaluate(
            column_name, eval_type=eval_type, tags=tags, merge_lines=merge_lines
        )
        eval_per_tag["ALL"] = eval_global

        self._compare_eval_results(eval_reference_path, eval_per_tag, incl_macro=macro)

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

        self._compare_eval_results(ref_path, eval_per_tag, True)

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

        self._compare_eval_results(ref_path, eval_per_tag, True)

    def _compare_eval_results(self, ref_path: str, tst, incl_macro: bool = True):
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

        # tst_path = ref_path.replace("ref_results.", "tst_results.")
        tst_path = ref_path.replace("ref_results.", "tst_results.")
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
                    if not incl_macro and "macro" in metric:
                        continue
                    else:
                        self.assertAlmostEqual(
                            ref[eval_type][label][metric],
                            tst[eval_type][label][metric],
                            msg=f"Evaluation mismatch found: \ndiff '{ref_path_sorted}'  '{tst_path_sorted}'\n'"
                            + f"Evaluation type: '{eval_type}'; label:  '{label}'; metric: '{metric}'",
                        )


if __name__ == "__main__":
    unittest.main()
