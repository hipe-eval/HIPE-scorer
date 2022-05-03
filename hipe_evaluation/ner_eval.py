#!/usr/bin/env python3
# coding: utf-8

"""
The official evaluation module for the CLEF-HIPE-2020 shared task.
"""


import logging
from collections import defaultdict, Counter
from copy import deepcopy
from typing import Union, List
import numpy as np

from hipe_evaluation.utils import (
    read_conll_annotations,
    collect_named_entities,
    collect_link_objects,
    get_all_tags,
    column_selector,
    filter_entities_by_noise,
    filter_entities_by_date,
)


logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, f_true, f_pred, glueing_cols=None):  # TODO: I would refactor as f_gold
        """
        An Evaluator evaluates the system response according to the
        gold standard.

        Both files (gold/system) need to be aligned on a token level.
        Any comment lines may be omitted, and, in this case, the segmentation
        into lines and documents gets reconstructed according to the gold standard.

        Per system, there is a single Evaluator that works as wrapper for
        various evaluation scenarios and for various columns via the method "evaluate".

        :param str f_true: file name of the gold standard (HIPE tsv-format).
        :param str f_pred: file name of the system response (HIPE tsv-format).
        :param list glueing_cols: concat the annotation of two columns (list with tuples).
        :return: Evaluator object.

        """

        logging.info(f"Reading system response file '{f_pred}' and gold standard '{f_true}'.")

        self.f_true = f_true
        self.f_pred = f_pred

        self.true = read_conll_annotations(f_true, glueing_cols)
        self.pred = read_conll_annotations(f_pred, glueing_cols)

        if len(self.true) != len(self.pred) and len(self.pred) == 1:
            # try to automatically reconstruct the segmentation of the predictions
            logging.info("Reconstructing the segmentation of the predictions.")
            self.reconstruct_segmentation()

        self.n_docs_true = len(self.true)
        self.n_docs_pred = len(self.pred)

        self.n_lines_pred = sum([len(doc) for doc in self.pred])
        self.n_lines_true = sum([len(doc) for doc in self.true])

        self.n_toks_pred = sum([len(line) for doc in self.pred for line in doc])
        self.n_toks_true = sum([len(line) for doc in self.true for line in doc])

        self.check_segment_mismatch()

        # metrics that will be collected for each evaluation scheme
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
            "R_micro": 0,
            "F1_micro": 0,
            "P_macro_doc": [],
            "R_macro_doc": [],
            "F1_macro_doc": [],
        }

        # TODO: remove (?)
        # self.slot_error_rate = {
        #     "deletion": 0,
        #     "insertion": 0,
        #     "substitution_both": 0,
        #     "substitution_type": 0,
        #     "substitution_span": 0,
        # }

        # four evaluation schemes
        self.metric_schema = {
            "strict": deepcopy(self.metrics),
            "ent_type": deepcopy(self.metrics),
            "partial": deepcopy(self.metrics),
            "exact": deepcopy(self.metrics),
            # "slot_error_rate": deepcopy(self.slot_error_rate), TODO: remove (?)
        }

    def check_segment_mismatch(self):
        """
        Assert the alignment between gold standard and the system response.
        """

        logging.info("Datasets imported (Gold/Predictions).")
        logging.info(f"Number of docs: {self.n_docs_true}\t{self.n_docs_pred}")
        logging.info(f"Number of lines: {self.n_lines_true}\t{self.n_lines_pred}")
        logging.info(f"Number of tokens: {self.n_toks_true}\t{self.n_toks_pred}")

        data_format_true = [[len(line) for line in doc] for doc in self.true]
        data_format_pred = [[len(line) for line in doc] for doc in self.pred]

        try:
            assert data_format_true == data_format_pred
        except AssertionError:
            msg = f"Data mismatch between system response '{self.f_pred}' and gold standard due to wrong segmentation or missing lines."
            logging.error(msg)
            raise AssertionError(msg)

    def reconstruct_segmentation(self):
        """Restructure the flat segmentation of the system annotations.

        While the content remains unchanged, the system response is properly
        restructured into documents and sentences according to the structure of the gold annotation.
        """

        sents_pred = []
        docs_pred = []
        tok_pos_start = 0
        max_mismatch_reports_per_doc = 20
        patched_pred_tokens_counter = Counter()

        for i_doc_true, docs_true in enumerate(self.true):
            for doc_true in docs_true:
                n_doc_sent_true = len(doc_true)
                tok_pos_end = tok_pos_start + n_doc_sent_true
                sent_pred = self.pred[0][0][tok_pos_start:tok_pos_end]
                toks_pred = [tok.TOKEN for tok in sent_pred]
                toks_true = [tok.TOKEN for tok in doc_true]
                if toks_true != toks_pred:
                    logging.warning(f"Different tokens in GS ({len(toks_true)} tokens in total) and system output ({len(toks_pred)} tokens in total): ")
                    if len(toks_true) == len(toks_pred):
                        logging.warning(
                            f"Given equal length documents, trying to patch system response tokens with GS tokens...")
                        for i, tok_true in enumerate(toks_true):
                            if tok_true != toks_pred[i]:
                                patched_pred_tokens_counter[(toks_pred[i], tok_true)] += 1
                                toks_pred[i] = tok_true
                                if len(patched_pred_tokens_counter) > max_mismatch_reports_per_doc:
                                    msg = (
                                        f"Giving up now... Patched more than {max_mismatch_reports_per_doc} {patched_pred_tokens_counter} confusion pairs. "
                                        f"The system response '{self.f_pred}' is not in line with the gold standard. \n"
                                    )
                                    logging.error(msg)
                                    msg = f"More then {patched_pred_tokens_counter} token mismatches found. Giving up..."
                                    assert False, msg

                        logging.warning(f"Patched {sum(patched_pred_tokens_counter.values())} tokens: {patched_pred_tokens_counter}")
                    else:
                        current_mismatch_reports_per_doc = 0
                        for i,tok_true in enumerate(toks_true):
                            if tok_true != toks_pred[i]:

                                msg = (
                                    f"The system response '{self.f_pred}' is not in line with the gold standard. \n"
                                    f"The mismatch occured in GS document {i_doc_true + 1} at token position {tok_pos_start + i}:\n"
                                    f"   GS: {toks_true[i - 3:i + 4]}\n"
                                    f"  SYS: {toks_pred[i - 3:i + 4]}\n"
                                )
                                logging.error(msg)
                                current_mismatch_reports_per_doc += 1
                                if current_mismatch_reports_per_doc > max_mismatch_reports_per_doc:
                                    msg = f"More then {max_mismatch_reports_per_doc} token mismatches found. Giving up..."
                                    assert False, msg

                sents_pred.append(sent_pred)
                tok_pos_start += n_doc_sent_true

            docs_pred.append(sents_pred)
            sents_pred = []

        self.pred = docs_pred

    def evaluate(
        self,
        columns: Union[List[str], str],
        eval_type: str,
        tags: set = None,  # TODO: could be renamed "expected_tags"
        merge_lines: bool = False,
        n_best: int = 1,
        noise_level: tuple = None,
        time_period: tuple = None,
        additional_columns: list = None,
    ):
        """Collect extensive statistics across labels and per entity type. TODO: correct "across columns"?

        For both, document-averaged and entity-type averaged
        macro scores are computed in addition to the global metrics.

        Alternative annotations via n-best or columns are only allowed for links,
        not for entities.

        :param list columns: name of column that contains the annotations.
        :param str eval_type: define evaluation type for either links (nel) or entities (nerc).
        :param set tags: limit evaluation to valid tag set.
        :param bool merge_lines: option to drop line segmentation to allow entity spans across lines. TODO: check
        :param int n_best: number of alternative links that should be considered.
        :param tuple noise_level: lower and upper Levenshtein distance to limit evaluation to noisy entities.
        :param tuple time_period: start and end date to limit evaluation to a particular period.
        :param list additional_columns: name of column that contains the additional annotations (nel).
        :return: Aggregated statistics across labels and per entity type.
        :rtype: Tuple(list, list)

        """
        if eval_type not in {"nerc", "nel"}:
            logging.error(f"Unrecognized eval_type '{eval_type}': Aborting evaluation...")
            exit(1)

        if isinstance(columns, str):
            columns = [columns]

        if isinstance(additional_columns, str):
            additional_columns = [additional_columns]

        logging.info(f"Evaluating column {columns} in system response file '{self.f_pred}'")

        if noise_level:
            noise_lower, noise_upper = noise_level

            logging.info(
                f"Limit evaluation to noisy entities with a Levenshtein distance from {noise_lower} to {noise_upper}."
            )

        if time_period:
            date_start, date_end = time_period
            logging.info(
                f"Limit evaluation to entities of the period between {date_start} and {date_end}."
            )

        tags = self.set_evaluation_tags(columns, tags, eval_type)

        # Create an accumulator to store overall results
        results = deepcopy(self.metric_schema)
        results_per_type = defaultdict(lambda: deepcopy(self.metric_schema))

        # Iterate document-wise
        for y_true_doc, y_pred_doc in zip(self.true, self.pred):

            # Create an accumulator to store document-level results
            doc_results = deepcopy(self.metric_schema)
            doc_results_per_type = defaultdict(lambda: deepcopy(self.metric_schema))

            # merge lines within a doc as entities can stretch across two lines
            if merge_lines:
                y_true_doc = [[line for lines in y_true_doc for line in lines]]
                y_pred_doc = [[line for lines in y_pred_doc for line in lines]]

            # Iterate segment-wise (i.e. sentences or lines)
            for y_true_seg, y_pred_seg in zip(y_true_doc, y_pred_doc):

                if noise_level:
                    y_true_seg, y_pred_seg = filter_entities_by_noise(
                        y_true_seg, y_pred_seg, noise_lower, noise_upper
                    )

                if time_period:
                    y_true_seg, y_pred_seg = filter_entities_by_date(
                        y_true_seg, y_pred_seg, date_start, date_end
                    )

                # Compute result for one segment
                if eval_type == "nerc":

                    seg_results, seg_results_per_type = self.compute_metrics(
                        collect_named_entities(y_true_seg, columns),
                        collect_named_entities(y_pred_seg, columns),
                        tags,
                    )

                elif eval_type == "nel":
                    seg_results, seg_results_per_type = self.compute_metrics(
                        collect_link_objects(y_true_seg, columns, additional_columns, gs=True),
                        collect_link_objects(y_pred_seg, columns, additional_columns, n_best),
                        tags,
                    )

                # accumulate overall stats
                results, results_per_type = self.accumulate_stats(
                    results, results_per_type, seg_results, seg_results_per_type
                )

                # accumulate stats across documents
                doc_results, doc_results_per_type = self.accumulate_stats(
                    doc_results, doc_results_per_type, seg_results, seg_results_per_type
                )

            # Compute document-level metrics by entity type
            for e_type in results_per_type:
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
        for e_type in results_per_type:
            results_per_type[e_type] = compute_precision_recall_wrapper(results_per_type[e_type])
            results_per_type[e_type] = compute_macro_doc_scores(results_per_type[e_type])

        # Compute overall metrics across entity types
        results = compute_precision_recall_wrapper(results)
        results = compute_macro_doc_scores(results)
        results = compute_macro_type_scores(results, results_per_type)

        return results, results_per_type

    def accumulate_doc_scores(self, results, doc_results):
        """Accumulate the scores (P, R, F1) across documents.

        When a entity does not occur in a particular document according to the gold standard,
        it is dismissed as it would artificially lower the final measure.

        :param dict results: nested accumulator of scores across document.
        :param dict doc_results: nested scores of current document.
        :return: accumulator updated with the scores of current document.
        :rtype: dict

        """

        for eval_schema in results:
            actual = doc_results[eval_schema]["actual"]
            possible = doc_results[eval_schema]["possible"]

            # to compute precision dismiss documents for which no entities were predicted
            if actual != 0:
                results[eval_schema]["P_macro_doc"].append(doc_results[eval_schema]["P_micro"])
            # to compute recall dismiss documents for which no entities exists in gold standard
            if possible != 0:
                results[eval_schema]["R_macro_doc"].append(doc_results[eval_schema]["R_micro"])
            # to compute recall dismiss documents for which no entities exists in gold standard
            if possible != 0 and actual != 0:
                results[eval_schema]["F1_macro_doc"].append(doc_results[eval_schema]["F1_micro"])

        return results

    def accumulate_stats(self, results, results_per_type, tmp_results, tmp_results_per_type):
        """Accumulate the scores across lines.

        :param dict results: nested accumulator of scores across lines.
        :param dict results_per_type: nested accumulator of scores per type across lines.
        :param dict tmp_results: scores of current line.
        :param dict tmp_results_per_type: scores of current line per type.
        :return: updated accumulator across labels and per entity type.
        :rtype: Tuple(dict, dict)

        """

        for eval_schema in results:
            # Aggregate metrics across entity types
            for metric in results[eval_schema]:
                results[eval_schema][metric] += tmp_results[eval_schema][metric]

                # Aggregate metrics by entity type
                for e_type in tmp_results_per_type:
                    results_per_type[e_type][eval_schema][metric] += tmp_results_per_type[e_type][
                        eval_schema
                    ][metric]

        return results, results_per_type

    def compute_metrics(self, true_named_entities: list, pred_named_entities: list, tags: set):
        """Compute the metrics of segment for all evaluation scenarios.
        Example of input:
        [
        [Entity(e_type='PERS', start_offset=6, end_offset=9, span_text='Sociétésuissedesimprimeurs')],
        [Entity(e_type='LOC', start_offset=13, end_offset=13, span_text='Frauenfeld')]
        ]

        Scenario I  : exact match of both type and boundaries (TP).
        Scenario II : spurious entity (insertion, FP).
        Scenario III: missed entity (deletion, FN).
        Scenario IV : type substitution (counted as both FP and FN in strict and fuzzy regimes).
        Scenario V  : span substitution (overlap) (counted as both FP and FN in strict regime and as TP in fuzzy regime).
        Scenario VI : type and span substitution (overlap) (counted as FP in strict and fuzzy regimes).

        NB: evaluation["ent_type"] corresponds to the fuzzy regime.

        :param list(Entity) true_named_entities: nested list with entity annotations of gold standard.
        :param list(Entity) pred_named_entities: nested list with entity annotations of system response.
        :param set tags: limit to provided tags.
        :return: nested results and results per entity type
        :rtype: Tuple(dict, dict)

        """

        # overall results
        evaluation = deepcopy(self.metric_schema)

        # results by entity type
        evaluation_agg_entities_type = defaultdict(lambda: deepcopy(self.metric_schema))

        # keep track of entities that overlapped
        true_which_overlapped_with_pred = []

        # Subset into only the tags that we are interested in.
        # NOTE: we remove the tags we don't want from both the predicted and the
        # true entities. This covers the two cases where mismatches can occur:
        #
        # 1) Where the model predicts a tag that is not present in the true data
        # 2) Where there is a tag in the true data that the model is not capable of
        # predicting.

        # only allow alternatives in prediction file, not in gold standard
        true_named_entities = [ent[0] for ent in true_named_entities if ent[0].e_type in tags]
        pred_named_entities = [
            ent for ent in pred_named_entities if any([e.e_type in tags for e in ent])
        ]

        # go through each predicted named-entity
        for pred in pred_named_entities:
            found_overlap = False

            # Check each of the potential scenarios in turn. See
            # http://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/
            # for scenario explanation.

            # Scenario I: Exact match between true and pred
            # type: equal
            # boundaries: equal
            for true in true_named_entities:
                if any(p == true for p in pred):
                    true_which_overlapped_with_pred.append(true)
                    evaluation["strict"]["correct"] += 1
                    evaluation["ent_type"]["correct"] += 1
                    evaluation["exact"]["correct"] += 1
                    evaluation["partial"]["correct"] += 1

                    # aggregated by entity type results
                    evaluation_agg_entities_type[true.e_type]["strict"]["correct"] += 1
                    evaluation_agg_entities_type[true.e_type]["ent_type"]["correct"] += 1
                    evaluation_agg_entities_type[true.e_type]["exact"]["correct"] += 1
                    evaluation_agg_entities_type[true.e_type]["partial"]["correct"] += 1

                    break

            else:

                # for the current pred, check for boundary overlaps with any of the true entities
                for true in true_named_entities:

                    # NOTE: error in original code: missing + 1
                    # overlapping needs to take into account last token as well
                    pred_range = range(pred[0].start_offset, pred[0].end_offset + 1)
                    true_range = range(true.start_offset, true.end_offset + 1)

                    # Scenario IV: Substitution type, offsets match, but entity type is wrong.
                    # type: different
                    # boundaries: equal
                    if (
                        true.start_offset == pred[0].start_offset
                        and pred[0].end_offset == true.end_offset
                        and true.e_type != pred[0].e_type
                    ):

                        # overall results
                        evaluation["strict"]["incorrect"] += 1
                        evaluation["ent_type"]["incorrect"] += 1
                        evaluation["partial"]["correct"] += 1
                        evaluation["exact"]["correct"] += 1

                        # evaluation["slot_error_rate"]["substitution_type"] += 1

                        # aggregated by entity type results
                        evaluation_agg_entities_type[true.e_type]["strict"]["incorrect"] += 1
                        evaluation_agg_entities_type[true.e_type]["ent_type"]["incorrect"] += 1
                        evaluation_agg_entities_type[true.e_type]["partial"]["correct"] += 1
                        evaluation_agg_entities_type[true.e_type]["exact"]["correct"] += 1

                        true_which_overlapped_with_pred.append(true)
                        found_overlap = True

                        break

                    # check for an overlap, i.e. not exact boundary match, with true entities
                    # NOTE: error in original code:
                    # overlaps with true entities must only counted once
                    elif (
                        find_overlap(true_range, pred_range)
                        and true not in true_which_overlapped_with_pred
                    ):

                        true_which_overlapped_with_pred.append(true)
                        found_overlap = True

                        # Scenario V: Substitution span, offsets do not match
                        # exactly and entity type is the same.
                        # type: equal
                        # boundaries: overlap

                        if any(p.e_type == true.e_type for p in pred):

                            # overall results
                            evaluation["strict"]["incorrect"] += 1
                            evaluation["ent_type"]["correct"] += 1
                            evaluation["partial"]["partial"] += 1
                            evaluation["exact"]["incorrect"] += 1

                            # evaluation["slot_error_rate"]["substitution_span"] += 1

                            # aggregated by entity type results
                            evaluation_agg_entities_type[true.e_type]["strict"]["incorrect"] += 1
                            evaluation_agg_entities_type[true.e_type]["ent_type"]["correct"] += 1
                            evaluation_agg_entities_type[true.e_type]["partial"]["partial"] += 1
                            evaluation_agg_entities_type[true.e_type]["exact"]["incorrect"] += 1

                            break

                        # Scenario VI: Substitution span and type, offsets do not match
                        # exactly and entity type is different.
                        # type: different
                        # boundaries: overlap

                        else:
                            # overall results
                            evaluation["strict"]["incorrect"] += 1
                            evaluation["ent_type"]["incorrect"] += 1
                            evaluation["partial"]["partial"] += 1
                            evaluation["exact"]["incorrect"] += 1

                            # evaluation["slot_error_rate"]["substitution_both"] += 1

                            # aggregated by entity type results
                            # Results against the true entity

                            evaluation_agg_entities_type[true.e_type]["strict"]["incorrect"] += 1
                            evaluation_agg_entities_type[true.e_type]["partial"]["partial"] += 1
                            evaluation_agg_entities_type[true.e_type]["ent_type"]["incorrect"] += 1
                            evaluation_agg_entities_type[true.e_type]["exact"]["incorrect"] += 1

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
                    if pred[0].e_type in tags:
                        spurious_tags = [pred[0].e_type]

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
            if true not in true_which_overlapped_with_pred:
                # overall results
                evaluation["strict"]["missed"] += 1
                evaluation["ent_type"]["missed"] += 1
                evaluation["partial"]["missed"] += 1
                evaluation["exact"]["missed"] += 1

                # evaluation["slot_error_rate"]["deletion"] += 1

                # aggregated by entity type results
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
                evaluation_agg_entities_type[entity_type][eval_type] = compute_actual_possible(
                    entity_level[eval_type]
                )

        return evaluation, evaluation_agg_entities_type

    def set_evaluation_tags(self, columns, tags, eval_type):

        try:
            y_true = [column_selector(doc, columns[0]) for doc in self.true]
            y_pred = []
            for col in columns:
                y_pred += [column_selector(doc, col) for doc in self.pred]
        except AttributeError:
            msg = f"Missing columns {columns} in the system response file '{self.f_pred}' or the gold standard."
            logging.error(msg)
            raise AssertionError(msg)

        true_tags = get_all_tags(y_true)
        pred_tags = get_all_tags(y_pred)

        if tags:
            logging.info(f"Evaluation is limited to the provided tag set: {tags}")
            self.check_spurious_tags(tags, pred_tags, columns)

            # take the union of the actual gold standard labels and  TODO: check if we want this behavior for 2022
            # labels of the response file that are valid even when not included
            # in gold standard of this particular column
            # Other spurious tags are treated as non-entity ('O' tag).

            tags = true_tags | {tag for tag in pred_tags if tag in tags}

        elif eval_type == "nerc":
            # For NERC, only tags which are covered by the gold standard are considered
            tags = true_tags
            self.check_spurious_tags(true_tags, pred_tags, columns)

            if not pred_tags:
                msg = f"No tags in the column '{columns}' of the system response file: '{self.f_pred}'"
                logging.warning(msg)

        elif eval_type == "nel":
            # For NEL, any tag in gold standard or predictions are considered
            tags = true_tags | pred_tags

        logging.info(f"Evaluating on the following tags: {tags}")

        return tags

    def check_spurious_tags(self, tags_true: set, tags_pred: set, columns: list):
        """Log any tags of the system response which are not in the gold standard.

        :param list tags_true: a set of true labels".
        :param list tags_pred: a set of system labels".
        :return: None.
        :rtype: None

        """

        for pred in tags_pred:
            if pred not in tags_true:
                msg = f"Spurious entity label '{pred}' in column {columns} of system response file: '{self.f_pred}'. \
                As the tag is not part of the gold standard, it is ignored in the evaluation."
                logging.warning(msg)


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
    """Update the counts of possible and actual based on evaluation.

    :param dict results: results with updated evaluation counts.
    :return: the results dict with actual, possible updated.
    :rtype: dict

    """

    correct = results["correct"]
    incorrect = results["incorrect"]
    partial = results["partial"]
    missed = results["missed"]
    spurious = results["spurious"]

    # Possible: number annotations in the gold-standard which contribute to the
    # final score
    possible = correct + incorrect + partial + missed  # TODO: could come from pred (?)

    # Actual: number of annotations produced by the NER system
    actual = correct + incorrect + partial + spurious

    results["actual"] = actual
    results["possible"] = possible

    results["TP"] = correct  # TODO comment: TP/FP/FN values are not used to compute metrics, only in reporting
    results["FP"] = actual - correct
    results["FN"] = possible - correct

    return results


def compute_precision_recall(results, partial=False):
    """ Compute the micro scores for Precision, Recall, F1.

    :param dict results: evaluation results.
    :param bool partial: option to half the reward of partial matches.
    :return: Description of returned object.
    :rtype: updated results

    """

    actual = results["actual"]
    possible = results["possible"]
    partial = results["partial"]
    correct = results["correct"]

    if partial:
        precision = (correct + 0.5 * partial) / actual if actual > 0 else 0
        recall = (correct + 0.5 * partial) / possible if possible > 0 else 0

    else:
        precision = correct / actual if actual > 0 else 0
        recall = correct / possible if possible > 0 else 0

    results["P_micro"] = precision
    results["R_micro"] = recall
    results["F1_micro"] = (
        2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    )

    return results


def compute_precision_recall_wrapper(results):
    """
    Wraps the compute_precision_recall function and runs it for each evaluation scenario in results
    """

    results_a = {
        key: compute_precision_recall(value, True)
        for key, value in results.items()
        if key in ["partial"]
    }

    # in the entity type matching scenario (fuzzy),
    # overlapping entities and entities with strict boundary matches are rewarded equally
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
    """Compute the macro scores for Precision, Recall, F1 across entity types.


    There are different ways to comput the macro F1-scores across class.
    Please see the explanations at:
    https://towardsdatascience.com/a-tale-of-two-macro-f1s-8811ddcf8f04

    :param dict results: evaluation results.
    :param dict results_per_type: evaluation results per type.
    :return: updated results and results per type.
    :rtype: Tuple(dict, dict)

    """

    for eval_schema in results:
        precision_sum = 0
        recall_sum = 0
        f1_sum = 0

        n_tags = len(results_per_type)

        for tag in results_per_type:
            precision_sum += results_per_type[tag][eval_schema]["P_micro"]
            recall_sum += results_per_type[tag][eval_schema]["R_micro"]
            f1_sum += results_per_type[tag][eval_schema]["F1_micro"]

        precision_macro = precision_sum / n_tags if n_tags > 0 else 0
        recall_macro = recall_sum / n_tags if n_tags > 0 else 0
        f1_macro_mean = f1_sum / n_tags if n_tags > 0 else 0
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
    """Compute the macro scores for Precision, Recall, F1 across documents.

    The score is a simple average across documents.

    :param dict results: evaluation results.
    :return: updated evaluation results.
    :rtype: dict

    """

    metrics = ("P_macro_doc", "R_macro_doc", "F1_macro_doc")

    for eval_schema in results:
        for metric in metrics:
            vals = results[eval_schema][metric]
            results[eval_schema][metric] = float(np.mean(vals)) if len(vals) > 0 else None
            results[eval_schema][metric + "_std"] = float(np.std(vals)) if len(vals) > 0 else None

    return results


def compute_slot_error_rate(results, results_per_type):
    """
    https://pdfs.semanticscholar.org/451b/61b390b86ae5629a21461d4c619ea34046e0.pdf
    https://www.aclweb.org/anthology/I11-1058.pdf
    """

    raise NotImplementedError
