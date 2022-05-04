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

    def test_ner_lit_1(self):
        """ NER  Test 1:
        1 NER-COARSE-LIT entity in gold, 0 in system response.
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
        self.assertEqual(evaluator.n_lines_true, 1, "Not all layout lines were parsed") # lines = hipe2020 segment legacy
        self.assertEqual(evaluator.n_toks_true, 16, "Not all tokens were parsed")

        self._do_evaluation(
            evaluator,
            eval_reference_path,
            column_name="NE-COARSE-LIT",
            eval_type="nerc",
            tags=get_hipe_2022_tagset_all(),
        )

    def test_ner_lit_2_coarse(self):
        """ NER Test 2:
        NE-COARSE-LIT: 2 entity in gold, 2 in system response.
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
        self.assertEqual(evaluator.n_lines_true, 1, "Not all layout lines were parsed")
        self.assertEqual(evaluator.n_toks_true, 32, "Not all tokens were parsed")

        self._do_evaluation(
            evaluator,
            eval_reference_path,
            column_name="NE-COARSE-LIT",
            eval_type="nerc",
            macro=False,
        )

    def test_ner_lit_2_coarse_iobes(self):
        """ NER Test 2:
        NE-COARSE-LIT: 2 entity in gold, 2 in system response.
        (cf. scenario I)
        """
        true_path = "hipe_evaluation/tests/data/unittest-ner-2-IOBES-true.tsv"
        pred_path = true_path.replace("-true", "-pred")
        eval_reference_path = pred_path + ".coarse-lit_ref_results.json"
        evaluator: Evaluator = Evaluator(
            true_path,
            pred_path,
        )
        self.assertEqual(evaluator.n_docs_true, 1, "Not all documents were parsed")
        self.assertEqual(evaluator.n_lines_true, 1, "Not all layout lines were parsed")
        self.assertEqual(evaluator.n_toks_true, 32, "Not all tokens were parsed")

        self._do_evaluation(
            evaluator,
            eval_reference_path,
            column_name="NE-COARSE-LIT",
            eval_type="nerc",
            macro=False,
        )




    def test_ner_lit_2_nested(self):
        """ NER Test 2:
        NE-NESTED: 1 entity in gold (Hambourg as loc.adm.town), 0 in system response.
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
        self.assertEqual(evaluator.n_lines_true, 1, "Not all layout lines were parsed")
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
        """ NER Test 2:
        """
        true_path = "hipe_evaluation/tests/data/unittest-ner-2-true.tsv"
        pred_path = true_path.replace("-true", "-pred")
        eval_reference_path = pred_path + ".fine-lit_ref_results.json"
        evaluator: Evaluator = Evaluator(
            true_path,
            pred_path,
        )
        self.assertEqual(evaluator.n_docs_true, 1, "Not all documents were parsed")
        self.assertEqual(evaluator.n_lines_true, 1, "Not all layout lines were parsed")
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
        """ NER Test 3:
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
        self.assertEqual(evaluator.n_lines_true, 1, "Not all layout lines were parsed")
        self.assertEqual(evaluator.n_toks_true, 37, "Not all tokens were parsed")

        self._do_evaluation(
            evaluator,
            eval_reference_path,
            column_name="NE-COARSE-LIT",
            eval_type="nerc",
            macro=False,
        )

    def test_ner_lit_4(self):
        """ NER Test 4:
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

    def test_nel_1(self):
        """ NEL Test 1
        2 QIDs and 1 NIL entity links in gold, 3 in system response, with 1 partial
        Details:
        - 1 ORG Société Suisse des imprimeurs: NIL OK, scenario I
        - 1 LOC (Frauenfeld): QID OK, scenario I
        - 1 LOC (ville de Berne): QID OK, "partial" mention coveragescenario V
        """
        true_path = "hipe_evaluation/tests/data/unittest-nel-1-true.tsv"
        pred_path = true_path.replace("-true", "-pred")
        eval_reference_path = pred_path + ".ref_results.json"
        evaluator: Evaluator = Evaluator(
            true_path,
            pred_path,
        )
        self.assertEqual(evaluator.n_docs_true, 1, "Not all documents were parsed")
        self.assertEqual(evaluator.n_lines_true, 1, "Not all layout lines were parsed")
        self.assertEqual(evaluator.n_toks_true, 37, "Not all tokens were parsed")

        self._do_evaluation(
            evaluator,
            eval_reference_path,
            column_name="NEL-LIT",
            eval_type="nel",
            macro=False
        )

    def test_nel_2_consecutive_NIL_pred_concat(self):
        """ NEL Test 2: consecutive NIL in system response, incorrectly evaluated as one (concatenated).
        2 QIDs and 2 NIL links in gold, idem in system response, with consecutive NIL (highly improbable).
        Details:
        - Lasie/NIL: OK, isolated in gold and in system
        - Berteaux/Q3300415: is NIL in system, thus creates 2 consecutive NIL in pred (with following Lasie)
        - Lasie/NIL: OK
        - Reinach/Q172161 : OK

        With correctly divided NIL (hipe 2022):
        True: NIL, Q3300415, NIL, Q172161
        Pred: NIL, NIL, NIL, Q172161

        With incorrectly divided NIL (hipe 2020):
        True: NIL, Q3300415, NIL, Q172161
        Pred: NIL, NIL, Q172161
        """
        true_path = "hipe_evaluation/tests/data/unittest-nel-2-true.tsv"
        pred_path = true_path.replace("-true", "-pred")
        eval_reference_path = pred_path + ".concatNIL_ref_results.json"
        evaluator: Evaluator = Evaluator(
            true_path,
            pred_path,
        )
        self.assertEqual(evaluator.n_docs_true, 1, "Not all documents were parsed")
        self.assertEqual(evaluator.n_lines_true, 1, "Not all layout lines were parsed")
        self.assertEqual(evaluator.n_toks_true, 30, "Not all tokens were parsed")

        self._do_evaluation(
            evaluator,
            eval_reference_path,
            column_name="NEL-LIT",
            eval_type="nel",
            macro=False,
        )

    def test_nel_2_consecutive_NIL_pred_separated(self):
        """ NEL Test 2: consecutive NIL in system response, correctly evaluated as two (separated based on ner tags)
        2 QIDs and 2 NIL entity links in gold, idem in system response, with consecutive NIL (highly improbable)
        Details:
        - Lasie/NIL: OK, isolated in gold and in system
        - Berteaux/Q3300415: is NIL in system, thus creates 2 consecutive NIL in pred (with following Lasie)
        - Lasie/NIL: OK
        - Reinach/Q172161 : OK

        With correctly divided NIL (hipe 2022):
        True: NIL, Q3300415, NIL, Q172161
        Pred: NIL, NIL, NIL, Q172161

        With incorrectly divided NIL (hipe2020):
        True: NIL, Q3300415, NIL, Q172161
        Pred: NIL, NIL, Q172161

        """

        true_path = "hipe_evaluation/tests/data/unittest-nel-2-true.tsv"
        pred_path = true_path.replace("-true", "-pred")
        eval_reference_path = pred_path + ".separatedNIL_ref_results.json"
        evaluator: Evaluator = Evaluator(
            true_path,
            pred_path,
        )
        self.assertEqual(evaluator.n_docs_true, 1, "Not all documents were parsed")
        self.assertEqual(evaluator.n_lines_true, 1, "Not all layout lines were parsed")
        self.assertEqual(evaluator.n_toks_true, 30, "Not all tokens were parsed")

        self._do_evaluation(
            evaluator,
            eval_reference_path,
            column_name="NEL-LIT",
            eval_type="nel",
            macro=False,
            additional_cols=["NE-COARSE-LIT"]  # when this param is not None it triggers another collect_link_objects
        )

    def _do_evaluation(
        self,
        evaluator: Evaluator,
        eval_reference_path: str,
        column_name: str,
        eval_type: str,
        tags=None,
        merge_lines: bool = False,
        macro: bool = False,
        additional_cols: list = None  # for El link segmentation based on ner columns
    ):
        """Run evaluator and compare to reference data"""

        eval_global, eval_per_tag = evaluator.evaluate(
            column_name,
            eval_type=eval_type,
            tags=tags,
            merge_lines=merge_lines,
            additional_columns=additional_cols
        )
        eval_per_tag["ALL"] = eval_global

        self._compare_eval_results(eval_reference_path, eval_per_tag, incl_macro=macro)

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
                    if not incl_macro and "macro" in metric:  # does not compare macro figures
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
