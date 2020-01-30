#!/usr/bin/env python3
# coding: utf-8

import logging
from collections import namedtuple
from copy import deepcopy

from .utils import (
    read_conll_annotations,
    collect_named_entities,
    collect_link_objects,
    get_all_gold_tags,
    segment2labels,
    sanity_check_tags,
)


logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level="DEBUG",
)


class Evaluator:
    def __init__(self, f_true, f_pred):
        """
        """

        self.true = read_conll_annotations(f_true)
        self.pred = read_conll_annotations(f_pred)

        if len(self.true) != len(self.pred):
            raise ValueError("Number of predicted documents does not equal true")

        logging.info(
            "Imported %s predictions for %s true examples",
            len(self.pred),
            len(self.true),
        )

        # Setup dict into which metrics will be stored.
        self.metrics = {
            "correct": 0,
            "incorrect": 0,
            "partial": 0,
            "missed": 0,
            "spurious": 0,
            "possible": 0,
            "actual": 0,
            "precision": 0,
            "recall": 0,
            "f1": 0,
        }

        self.slot_error_rate = {
            "deletion": 0,
            "insertion": 0,
            "substitution_both": 0,
            "substitution_type": 0,
            "substitution_span": 0,
            "precision": 0,
            "recall": 0,
            "f1": 0,
        }

        # Copy results dict to cover the four schemes.
        self.metric_schema = {
            "strict": deepcopy(self.metrics),
            "ent_type": deepcopy(self.metrics),
            "partial": deepcopy(self.metrics),
            "exact": deepcopy(self.metrics),
            "slot_error_rate": deepcopy(self.slot_error_rate),
        }

    def evaluate_globally(self):
        raise NotImplementedError

    def evaluate_doc_wise(self):
        raise NotImplementedError

    def evaluate(self, anno_type, eval_type, tags):

        try:
            y_true_segments = [segment2labels(s, anno_type) for s in self.true]
            y_pred_segments = [segment2labels(s, anno_type) for s in self.pred]
        except AttributeError:
            raise AttributeError(
                "Provided annotation column is not available for both predicted and true file"
            )

        if tags:
            logging.info(f"Given tags for the column {anno_type}: {tags}")
            tags = sanity_check_tags(y_true_segments, tags)
        else:
            tags = get_all_gold_tags(y_true_segments)

        logging.info(f"Evaluating on {anno_type} for the following tags: {tags}")

        # Create an accumulator to store results
        results = deepcopy(self.metric_schema)
        results_per_type = {e: deepcopy(self.metric_schema) for e in tags}

        # Iterate segment-wise (i.e. sentences or lines)
        for y_true_seg, y_pred_seg in zip(y_true_segments, y_pred_segments):

            # Check that the length of the true and predicted examples are the
            # same. This must be checked here, because another error may not
            # be thrown if the lengths do not match.
            if len(y_true_seg) != len(y_pred_seg):
                raise ValueError(
                    f"Segment length of prediction ({len(y_pred_seg)}) does not match true ({len(y_true_seg)}) example length"
                )

            # Compute results for one message
            if eval_type == "nerc":
                tmp_results, tmp_agg_results = self.compute_metrics(
                    collect_named_entities(y_true_seg),
                    collect_named_entities(y_pred_seg),
                    tags,
                )
            elif eval_type == "nel":
                tmp_results, tmp_agg_results = self.compute_metrics(
                    collect_link_objects(y_true_seg),
                    collect_link_objects(y_pred_seg),
                    tags,
                )

            # Accumulate the results across segments
            for eval_schema in results:
                # Aggregate metrics globally
                for metric in results[eval_schema]:
                    results[eval_schema][metric] += tmp_results[eval_schema][metric]

                    # Aggregate metrics by entity type
                    for e_type in tags:
                        results_per_type[e_type][eval_schema][
                            metric
                        ] += tmp_agg_results[e_type][eval_schema][metric]

        # Calculate precision and recall at the individual entity level
        for e_type in tags:
            results_per_type[e_type] = compute_precision_recall_wrapper(
                results_per_type[e_type]
            )

        # Calculate global precision, recall, f1 including macro average
        results = compute_precision_recall_wrapper(results)
        results = compute_macro_scores(results, results_per_type)

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

                    pred_range = range(pred.start_offset, pred.end_offset)
                    true_range = range(true.start_offset, true.end_offset)

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

                        evaluation["slot_error_rate"]["substitution_type"] += 1

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
                    elif find_overlap(true_range, pred_range):

                        true_which_overlapped_with_pred.append(true)

                        # Scenario V: There is an overlap (but offsets do not match
                        # exactly), and the entity type is the same.
                        # 2.1 overlaps with the same entity type

                        if pred.e_type == true.e_type:

                            # overall results
                            evaluation["strict"]["incorrect"] += 1
                            evaluation["ent_type"]["correct"] += 1
                            evaluation["partial"]["partial"] += 1
                            evaluation["exact"]["incorrect"] += 1

                            evaluation["slot_error_rate"]["substitution_span"] += 1

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

                            found_overlap = True

                            break

                        # Scenario VI: Entities overlap, but the entity type is
                        # different.

                        else:
                            # overall results
                            evaluation["strict"]["incorrect"] += 1
                            evaluation["ent_type"]["incorrect"] += 1
                            evaluation["partial"]["partial"] += 1
                            evaluation["exact"]["incorrect"] += 1

                            evaluation["slot_error_rate"]["substitution_both"] += 1

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

                            # Results against the predicted entity

                            # evaluation_agg_entities_type[pred.e_type]['strict']['spurious'] += 1

                            found_overlap = True

                            break

                # Scenario II: Entities are spurious (i.e., over-generated).

                if not found_overlap:

                    # Overall results

                    evaluation["strict"]["spurious"] += 1
                    evaluation["ent_type"]["spurious"] += 1
                    evaluation["partial"]["spurious"] += 1
                    evaluation["exact"]["spurious"] += 1

                    evaluation["slot_error_rate"]["insertion"] += 1

                    # Aggregated by entity type results

                    # NOTE: when pred.e_type is not found in tags
                    # or when it simply does not appear in the test set, then it is
                    # spurious, but it is not clear where to assign it at the tag
                    # level. In this case, it is applied to all target_tags
                    # found in this example. This will mean that the sum of the
                    # evaluation_agg_entities will not equal evaluation.

                    for true in tags:

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

                evaluation["slot_error_rate"]["deletion"] += 1

                # for the agg. by e_type
                evaluation_agg_entities_type[true.e_type]["strict"]["missed"] += 1
                evaluation_agg_entities_type[true.e_type]["ent_type"]["missed"] += 1
                evaluation_agg_entities_type[true.e_type]["partial"]["missed"] += 1
                evaluation_agg_entities_type[true.e_type]["exact"]["missed"] += 1

        # Compute 'possible', 'actual' according to SemEval-2013 Task 9.1 on the
        # overall results, and use these to calculate precision and recall.
        for eval_type in evaluation:
            if eval_type != "slot_error_rate":
                evaluation[eval_type] = compute_actual_possible(evaluation[eval_type])

        # Compute 'possible', 'actual', and precision and recall on entity level
        # results. Start by cycling through the accumulated results.
        for entity_type, entity_level in evaluation_agg_entities_type.items():

            # Cycle through the evaluation types for each dict containing entity
            # level results.
            for eval_type in entity_level:
                if eval_type != "slot_error_rate":
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

    When the results dicts is from partial or ent_type metrics, then
    partial_or_type=True to ensure the right calculation is used for
    calculating precision and recall.
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

    return results


def compute_precision_recall(results, partial_or_type=False):
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

    if partial_or_type:  ## TODO: check as partial is always zero in case of type
        precision = (correct + 0.5 * partial) / actual if actual > 0 else 0
        recall = (correct + 0.5 * partial) / possible if possible > 0 else 0

    else:
        precision = correct / actual if actual > 0 else 0
        recall = correct / possible if possible > 0 else 0

    results["precision"] = precision
    results["recall"] = recall
    results["f1"] = (
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
        if key in ["partial", "ent_type"]
    }
    results_b = {
        key: compute_precision_recall(value)
        for key, value in results.items()
        if key in ["strict", "exact"]
    }

    # TODO: compute SER
    results_c = {
        key: value for key, value in results.items() if key in ["slot_error_rate"]
    }

    results = {**results_a, **results_b, **results_c}

    return results


def compute_macro_scores(results, results_per_type):

    """

    https://towardsdatascience.com/a-tale-of-two-macro-f1s-8811ddcf8f04
    """

    for eval_schema in results:
        precision_sum = 0
        recall_sum = 0
        f1_sum = 0

        n_tags = len(results_per_type)

        for tag in results_per_type:
            precision_sum += results_per_type[tag][eval_schema]["precision"]
            recall_sum += results_per_type[tag][eval_schema]["recall"]
            f1_sum += results_per_type[tag][eval_schema]["f1"]

        precision_macro = precision_sum / n_tags
        recall_macro = recall_sum / n_tags
        f1_macro_mean = f1_sum / n_tags
        f1_macro_recomp = (
            2 * (precision_macro * recall_macro) / (precision_macro + recall_macro)
            if (precision_macro + recall_macro) > 0
            else 0
        )

        results[eval_schema]["precision_macro"] = precision_macro
        results[eval_schema]["recall_macro"] = recall_macro
        results[eval_schema]["f1_macro"] = f1_macro_mean  # sklearn-style
        results[eval_schema]["f1_macro (recomputed from P & R)"] = f1_macro_recomp

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
