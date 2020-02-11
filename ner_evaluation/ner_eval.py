#!/usr/bin/env python3
# coding: utf-8

import logging
from copy import deepcopy
import numpy as np

from .utils import (
    read_conll_annotations,
    collect_named_entities,
    collect_link_objects,
    get_all_tags,
    column_selector,
    check_tag_selection,
    check_spurious_tags,
)


logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level="DEBUG",
)


class Evaluator:
    def __init__(self, f_true, f_pred, glueinng_cols=None):
        """
        """

        self.true = read_conll_annotations(f_true, glueinng_cols)
        self.pred = read_conll_annotations(f_pred, glueinng_cols)

        self.n_docs_true = len(self.true)
        self.n_docs_pred = len(self.pred)

        self.n_lines_pred = sum([len(doc) for doc in self.pred])
        self.n_lines_true = sum([len(doc) for doc in self.true])

        self.n_toks_pred = sum([len(line) for doc in self.pred for line in doc])
        self.n_toks_true = sum([len(line) for doc in self.true for line in doc])

        self.check_segment_mismatch()

        # Setup dict into which metrics will be stored.
        self.metrics = {
            "correct": 0,
            "incorrect": 0,
            "partial": 0,
            "missed": 0,
            "spurious": 0,
            "possible": 0,
            "actual": 0,
            "TP": 0,
            "FP": 0,
            "FN": 0,
            "P_micro": 0,
            "recall_micro": 0,
            "F1_micro": 0,
            "P_macro_doc": [],
            "R_macro_doc": [],
            "F1_macro_doc": [],
        }

        #
        # self.slot_error_rate = {
        #     "deletion": 0,
        #     "insertion": 0,
        #     "substitution_both": 0,
        #     "substitution_type": 0,
        #     "substitution_span": 0,
        # }

        # Copy results dict to cover the four schemes.
        self.metric_schema = {
            "strict": deepcopy(self.metrics),
            "ent_type": deepcopy(self.metrics),
            "partial": deepcopy(self.metrics),
            "exact": deepcopy(self.metrics),
            # "slot_error_rate": deepcopy(self.slot_error_rate),
        }

    def check_segment_mismatch(self):
        logging.info("Datasets imported (Gold/Predictions).")
        logging.info(f"Number of docs: {self.n_docs_true}\t{self.n_docs_pred}")
        logging.info(f"Number of lines: {self.n_lines_true}\t{self.n_lines_pred}")
        logging.info(f"Number of tokens: {self.n_toks_true}\t{self.n_toks_pred}")

        data_format_true = [[len(line) for line in doc] for doc in self.true]
        data_format_pred = [[len(line) for line in doc] for doc in self.pred]

        if data_format_true != data_format_pred:
            raise ValueError("Data mismatch between true and prediction dataset")

    def evaluate(self, column, eval_type, tags, merge_lines=False):

        try:
            y_true = [column_selector(doc, column) for doc in self.true]
            y_pred = [column_selector(doc, column) for doc in self.pred]
        except AttributeError:
            raise AttributeError(
                "Provided annotation column is not available for both predicted and true file"
            )

        if tags:
            logging.info(f"Provided tags for the column {column}: {tags}")
            tags = check_tag_selection(y_true, tags)
        elif eval_type == "nerc":
            # For NERC, only tags which are covered by the gold standard are considered
            tags = get_all_tags(y_true)
            check_spurious_tags(y_true, y_pred)
        elif eval_type == "nel":
            # For NEL, any tag in gold standard or predictions are considered
            tags = get_all_tags(y_true) | get_all_tags(y_pred)

        logging.info(f"Evaluating on {column} for the following tags: {tags}")

        # Create an accumulator to store overall results
        results = deepcopy(self.metric_schema)
        results_per_type = {e: deepcopy(self.metric_schema) for e in tags}

        # Iterate document-wise
        for y_true_doc, y_pred_doc in zip(y_true, y_pred):

            # Create an accumulator to store document-level results
            doc_results = deepcopy(self.metric_schema)
            doc_results_per_type = {e: deepcopy(self.metric_schema) for e in tags}

            # merge lines within a doc as entities can stretch across two lines
            if merge_lines:
                y_true_doc = [[line for lines in y_true_doc for line in lines]]
                y_pred_doc = [[line for lines in y_pred_doc for line in lines]]

            # Iterate segment-wise (i.e. sentences or lines)
            for y_true_seg, y_pred_seg in zip(y_true_doc, y_pred_doc):

                # Compute result for one segment
                if eval_type == "nerc":
                    seg_results, seg_results_per_type = self.compute_metrics(
                        collect_named_entities(y_true_seg),
                        collect_named_entities(y_pred_seg),
                        tags,
                    )
                elif eval_type == "nel":
                    seg_results, seg_results_per_type = self.compute_metrics(
                        collect_link_objects(y_true_seg),
                        collect_link_objects(y_pred_seg),
                        tags,
                    )

                # accumulate overall stats
                results, results_per_type = self.accumulate_stats(
                    results, results_per_type, seg_results, seg_results_per_type
                )

                # accumulate stats within document
                doc_results, doc_results_per_type = self.accumulate_stats(
                    doc_results, doc_results_per_type, seg_results, seg_results_per_type
                )

            # Compute document-level metrics by entity type
            for e_type in tags:
                doc_results_per_type[e_type] = compute_precision_recall_wrapper(
                    doc_results_per_type[e_type]
                )
                results_per_type[e_type] = self.accumulate_doc_scores(
                    results_per_type[e_type], doc_results_per_type[e_type]
                )

            # Compute document-level metrics across entity types
            doc_results = compute_precision_recall_wrapper(doc_results)
            results = self.accumulate_doc_scores(results, doc_results)

        # Compute overall metrics by entity type
        for e_type in tags:
            results_per_type[e_type] = compute_precision_recall_wrapper(
                results_per_type[e_type]
            )
            results_per_type[e_type] = compute_macro_doc_scores(
                results_per_type[e_type]
            )

        # Compute overall metrics across entity types
        results = compute_precision_recall_wrapper(results)
        results = compute_macro_doc_scores(results)
        results = compute_macro_type_scores(results, results_per_type)

        return results, results_per_type

    def accumulate_doc_scores(self, results, doc_results):

        for eval_schema in results:
            actual = doc_results[eval_schema]["actual"]
            possible = doc_results[eval_schema]["possible"]

            # to compute precision dismiss documents for which no entities were predicted
            if actual != 0:
                results[eval_schema]["P_macro_doc"].append(
                    doc_results[eval_schema]["P_micro"]
                )
            # to compute recall dismiss documents for which no entities exists in gold standard
            if possible != 0:
                results[eval_schema]["R_macro_doc"].append(
                    doc_results[eval_schema]["R_micro"]
                )
            # to compute recall dismiss documents for which no entities exists in gold standard
            if possible != 0 and actual != 0:
                results[eval_schema]["F1_macro_doc"].append(
                    doc_results[eval_schema]["F1_micro"]
                )

        return results

    def accumulate_stats(
        self, results, results_per_type, tmp_results, tmp_results_per_type
    ):

        for eval_schema in results:
            # Aggregate metrics across entity types
            for metric in results[eval_schema]:
                results[eval_schema][metric] += tmp_results[eval_schema][metric]

                # Aggregate metrics by entity type
                for e_type in results_per_type:
                    results_per_type[e_type][eval_schema][
                        metric
                    ] += tmp_results_per_type[e_type][eval_schema][metric]

        return results, results_per_type

    def compute_metrics(self, true_named_entities, pred_named_entities, tags):

        # overall results
        evaluation = deepcopy(self.metric_schema)

        # results by entity type
        evaluation_agg_entities_type = {e: deepcopy(self.metric_schema) for e in tags}

        # keep track of entities that overlapped
        true_which_overlapped_with_pred = []

        # Subset into only the tags that we are interested in.
        # NOTE: we remove the tags we don't want from both the predicted and the
        # true entities. This covers the two cases where mismatches can occur:
        #
        # 1) Where the model predicts a tag that is not present in the true data
        # 2) Where there is a tag in the true data that the model is not capable of
        # predicting.

        true_named_entities = [ent for ent in true_named_entities if ent.e_type in tags]
        pred_named_entities = [ent for ent in pred_named_entities if ent.e_type in tags]

        # go through each predicted named-entity
        for pred in pred_named_entities:
            found_overlap = False

            # Check each of the potential scenarios in turn. See
            # http://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/
            # for scenario explanation.

            # Scenario I: Exact match between true and pred

            if pred in true_named_entities:
                true_which_overlapped_with_pred.append(pred)
                evaluation["strict"]["correct"] += 1
                evaluation["ent_type"]["correct"] += 1
                evaluation["exact"]["correct"] += 1
                evaluation["partial"]["correct"] += 1

                # for the agg. by e_type results
                evaluation_agg_entities_type[pred.e_type]["strict"]["correct"] += 1
                evaluation_agg_entities_type[pred.e_type]["ent_type"]["correct"] += 1
                evaluation_agg_entities_type[pred.e_type]["exact"]["correct"] += 1
                evaluation_agg_entities_type[pred.e_type]["partial"]["correct"] += 1

            else:

                # check for overlaps with any of the true entities
                for true in true_named_entities:

                    # NOTE: error in original code: missing + 1
                    # overlapping needs to take into account last token as well
                    pred_range = range(pred.start_offset, pred.end_offset + 1)
                    true_range = range(true.start_offset, true.end_offset + 1)

                    # Scenario IV: Offsets match, but entity type is wrong

                    if (
                        true.start_offset == pred.start_offset
                        and pred.end_offset == true.end_offset
                        and true.e_type != pred.e_type
                    ):

                        # overall results
                        evaluation["strict"]["incorrect"] += 1
                        evaluation["ent_type"]["incorrect"] += 1
                        evaluation["partial"]["correct"] += 1
                        evaluation["exact"]["correct"] += 1

                        # evaluation["slot_error_rate"]["substitution_type"] += 1

                        # aggregated by entity type results
                        evaluation_agg_entities_type[true.e_type]["strict"][
                            "incorrect"
                        ] += 1
                        evaluation_agg_entities_type[true.e_type]["ent_type"][
                            "incorrect"
                        ] += 1
                        evaluation_agg_entities_type[true.e_type]["partial"][
                            "correct"
                        ] += 1
                        evaluation_agg_entities_type[true.e_type]["exact"][
                            "correct"
                        ] += 1

                        true_which_overlapped_with_pred.append(true)
                        found_overlap = True

                        break

                    # check for an overlap i.e. not exact boundary match, with true entities
                    # NOTE: error in original code:
                    # overlaps with true entititie must only counted once
                    elif (
                        find_overlap(true_range, pred_range)
                        and true not in true_which_overlapped_with_pred
                    ):

                        true_which_overlapped_with_pred.append(true)
                        found_overlap = True

                        # Scenario V: There is an overlap (but offsets do not match
                        # exactly), and the entity type is the same.
                        # 2.1 overlaps with the same entity type

                        if pred.e_type == true.e_type:

                            # overall results
                            evaluation["strict"]["incorrect"] += 1
                            evaluation["ent_type"]["correct"] += 1
                            evaluation["partial"]["partial"] += 1
                            evaluation["exact"]["incorrect"] += 1

                            # evaluation["slot_error_rate"]["substitution_span"] += 1

                            # aggregated by entity type results
                            evaluation_agg_entities_type[true.e_type]["strict"][
                                "incorrect"
                            ] += 1
                            evaluation_agg_entities_type[true.e_type]["ent_type"][
                                "correct"
                            ] += 1
                            evaluation_agg_entities_type[true.e_type]["partial"][
                                "partial"
                            ] += 1
                            evaluation_agg_entities_type[true.e_type]["exact"][
                                "incorrect"
                            ] += 1

                            break

                        # Scenario VI: Entities overlap, but the entity type is
                        # different.

                        else:
                            # overall results
                            evaluation["strict"]["incorrect"] += 1
                            evaluation["ent_type"]["incorrect"] += 1
                            evaluation["partial"]["partial"] += 1
                            evaluation["exact"]["incorrect"] += 1

                            # evaluation["slot_error_rate"]["substitution_both"] += 1

                            # aggregated by entity type results
                            # Results against the true entity

                            evaluation_agg_entities_type[true.e_type]["strict"][
                                "incorrect"
                            ] += 1
                            evaluation_agg_entities_type[true.e_type]["partial"][
                                "partial"
                            ] += 1
                            evaluation_agg_entities_type[true.e_type]["ent_type"][
                                "incorrect"
                            ] += 1
                            evaluation_agg_entities_type[true.e_type]["exact"][
                                "incorrect"
                            ] += 1

                            break

                # Scenario II: Entities are spurious (i.e., over-generated).

                if not found_overlap:

                    # Overall results
                    evaluation["strict"]["spurious"] += 1
                    evaluation["ent_type"]["spurious"] += 1
                    evaluation["partial"]["spurious"] += 1
                    evaluation["exact"]["spurious"] += 1

                    # evaluation["slot_error_rate"]["insertion"] += 1

                    # Aggregated by entity type results

                    # NOTE: error in original code:
                    # a spurious entity for a particular tag should be only
                    # attributed to the respective tag

                    if pred.e_type in tags:
                        spurious_tags = [pred.e_type]

                    else:
                        # NOTE: when pred.e_type is not found in tags
                        # or when it simply does not appear in the test set, then it is
                        # spurious, but it is not clear where to assign it at the tag
                        # level. In this case, it is applied to all target_tags
                        # found in this example. This will mean that the sum of the
                        # evaluation_agg_entities will not equal evaluation.

                        spurious_tags = tags

                    for true in spurious_tags:
                        evaluation_agg_entities_type[true]["strict"]["spurious"] += 1
                        evaluation_agg_entities_type[true]["ent_type"]["spurious"] += 1
                        evaluation_agg_entities_type[true]["partial"]["spurious"] += 1
                        evaluation_agg_entities_type[true]["exact"]["spurious"] += 1

        # Scenario III: Entity was missed entirely.

        for true in true_named_entities:
            if true in true_which_overlapped_with_pred:
                continue
            else:
                # overall results
                evaluation["strict"]["missed"] += 1
                evaluation["ent_type"]["missed"] += 1
                evaluation["partial"]["missed"] += 1
                evaluation["exact"]["missed"] += 1

                # evaluation["slot_error_rate"]["deletion"] += 1

                # for the agg. by e_type
                evaluation_agg_entities_type[true.e_type]["strict"]["missed"] += 1
                evaluation_agg_entities_type[true.e_type]["ent_type"]["missed"] += 1
                evaluation_agg_entities_type[true.e_type]["partial"]["missed"] += 1
                evaluation_agg_entities_type[true.e_type]["exact"]["missed"] += 1

        # Compute 'possible', 'actual' according to SemEval-2013 Task 9.1 on the
        # overall results, and use these to calculate precision and recall.
        for eval_type in evaluation:
            # if eval_type != "slot_error_rate":
            evaluation[eval_type] = compute_actual_possible(evaluation[eval_type])

        # Compute 'possible', 'actual', and precision and recall on entity level
        # results. Start by cycling through the accumulated results.
        for entity_type, entity_level in evaluation_agg_entities_type.items():

            # Cycle through the evaluation types for each dict containing entity
            # level results.
            for eval_type in entity_level:
                # if eval_type != "slot_error_rate":
                evaluation_agg_entities_type[entity_type][
                    eval_type
                ] = compute_actual_possible(entity_level[eval_type])

        return evaluation, evaluation_agg_entities_type


def find_overlap(true_range, pred_range):
    """Find the overlap between two ranges

    Find the overlap between two ranges. Return the overlapping values if
    present, else return an empty set().

    Examples:

    >>> find_overlap((1, 2), (2, 3))
    2
    >>> find_overlap((1, 2), (3, 4))
    set()
    """

    true_set = set(true_range)
    pred_set = set(pred_range)

    overlaps = true_set.intersection(pred_set)

    return overlaps


def compute_actual_possible(results):
    """
    Takes a result dict that has been output by compute metrics.
    Returns the results dict with actual, possible populated.
    """

    correct = results["correct"]
    incorrect = results["incorrect"]
    partial = results["partial"]
    missed = results["missed"]
    spurious = results["spurious"]

    # Possible: number annotations in the gold-standard which contribute to the
    # final score
    possible = correct + incorrect + partial + missed

    # Actual: number of annotations produced by the NER system
    actual = correct + incorrect + partial + spurious

    results["actual"] = actual
    results["possible"] = possible

    results["TP"] = correct
    results["FP"] = actual - correct
    results["FN"] = possible - correct

    return results


def compute_precision_recall(results, partial=False):
    """
    Takes a result dict that has been output by compute metrics.
    Returns the results dict with precison and recall populated.

    When the results dicts is from partial or ent_type metrics, then
    partial_or_type=True to ensure the right calculation is used for
    calculating precision and recall.
    """

    actual = results["actual"]
    possible = results["possible"]
    partial = results["partial"]
    correct = results["correct"]

    # in the entity type matching scenario (fuzzy),
    # overlapping entities and entities with strict boundary matches are rewarded equally
    if partial:
        precision = (correct + 0.5 * partial) / actual if actual > 0 else 0
        recall = (correct + 0.5 * partial) / possible if possible > 0 else 0

    else:
        precision = correct / actual if actual > 0 else 0
        recall = correct / possible if possible > 0 else 0

    results["P_micro"] = precision
    results["R_micro"] = recall
    results["F1_micro"] = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return results


def compute_precision_recall_wrapper(results):
    """
    Wraps the compute_precision_recall function and runs on a dict of results
    """

    results_a = {
        key: compute_precision_recall(value, True)
        for key, value in results.items()
        if key in ["partial"]
    }
    results_b = {
        key: compute_precision_recall(value)
        for key, value in results.items()
        if key in ["strict", "exact", "ent_type"]
    }

    # TODO: compute SER
    # results_c = {
    #     key: value for key, value in results.items() if key in ["slot_error_rate"]
    # }

    results = {**results_a, **results_b}

    return results


def compute_macro_type_scores(results, results_per_type):

    """

    https://towardsdatascience.com/a-tale-of-two-macro-f1s-8811ddcf8f04
    """

    for eval_schema in results:
        precision_sum = 0
        recall_sum = 0
        f1_sum = 0

        n_tags = len(results_per_type)

        for tag in results_per_type:
            precision_sum += results_per_type[tag][eval_schema]["P_micro"]
            recall_sum += results_per_type[tag][eval_schema]["R_micro"]
            f1_sum += results_per_type[tag][eval_schema]["R_micro"]

        precision_macro = precision_sum / n_tags
        recall_macro = recall_sum / n_tags
        f1_macro_mean = f1_sum / n_tags
        f1_macro_recomp = (
            2 * (precision_macro * recall_macro) / (precision_macro + recall_macro)
            if (precision_macro + recall_macro) > 0
            else 0
        )

        results[eval_schema]["P_macro"] = precision_macro
        results[eval_schema]["R_macro"] = recall_macro
        results[eval_schema]["F1_macro"] = f1_macro_mean  # sklearn-style
        results[eval_schema]["F1_macro (recomputed from P & R)"] = f1_macro_recomp

    return results


def compute_macro_doc_scores(results):

    metrics = ("P_macro_doc", "R_macro_doc", "F1_macro_doc")

    for eval_schema in results:
        for metric in metrics:
            vals = results[eval_schema][metric]
            results[eval_schema][metric] = (
                float(np.mean(vals)) if len(vals) > 0 else None
            )
            results[eval_schema][metric + "_std"] = (
                float(np.std(vals)) if len(vals) > 0 else None
            )

    return results


def compute_slot_error_rate(results, results_per_type):

    """
    https://pdfs.semanticscholar.org/451b/61b390b86ae5629a21461d4c619ea34046e0.pdf
    https://www.aclweb.org/anthology/I11-1058.pdf
    """

    raise NotImplementedError

    # 'Q21': {'ent_type': {'correct': 5,
    #    'incorrect': 0,
    #    'partial': 0,
    #    'missed': 0,
    #    'spurious': 0,
    #    'possible': 5,
    #    'actual': 5,
    #    'precision': 1.0,
    #    'recall': 1.0,
    #    'f1': 1.0},
    #
    #
    #
    #    D+I+SST+ 0.5×(SS+ST)Ref
    #
    #
    #
    #
    # for eval_schema in results:
    #     for coref_chain in results_per_type:
    #
    #         deletions += results['ent_type']['missing']
    #         insertions += results['ent_type']['spurious']
    #
    #         substitutions_both += results['wrong_type_with_overlap']
    #         substitutions_type += results['ent_type']['correct']
    #
    #         substitutions_span += results['partial']['correct']
    #
    # return results
