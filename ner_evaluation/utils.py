#!/usr/bin/env python3
# coding: utf-8

import csv
from collections import namedtuple
import logging

Entity = namedtuple("Entity", "e_type start_offset end_offset")


class TokAnnotation:
    """
    The annotation of a token comprises an arbitrary number of attributes.
    The name of the attributes are specified during run-time.
    """

    def __init__(self, properties: dict):
        self.fieldnames = [col for col in properties]

        # columns are set as class variables
        for k, v in properties.items():
            if k.upper() != "TOKEN":
                try:
                    v = v.upper()
                except AttributeError:
                    msg = f"Empty values in column '{k}'. They get replaced by an underscore."
                    logging.warning(msg)
                    v = "_"

            setattr(self, k, v)

    def __repr__(self):
        return "TokAnnotation({!r})".format(self.get_values())

    def get_values(self):
        return {k: v for k, v in self.__dict__.items() if k in self.fieldnames}


def get_all_tags(y_true):
    """
    Return a set of all tags excluding non-annotations (i.e. "_", "O", "-")

    :param list y_true: a nested list of labels with the structure "[docs [sents [tokens]]]".
    :return: set of all labels.
    :rtype: set
    """

    # keep only primary annotation when separated by a pipe
    tags = {label.split("|")[0].split("-")[-1] for doc in y_true for seg in doc for label in seg}

    non_tags = [
        "_",
        "-",
        "O",
    ]

    for symbol in non_tags:
        try:
            tags.remove(symbol)
        except KeyError:
            pass

    return tags


def check_spurious_tags(tags_true: set, tags_pred: set, columns: list):
    """Log any tags of the system response which are not in the gold standard.

    :param list tags_true: a set of true labels".
    :param list tags_pred: a set of system labels".
    :return: None.
    :rtype: None

    """

    for pred in tags_pred:
        if pred not in tags_true:
            msg = f"Spurious entity label '{pred}' in column {columns} in system response, which is part of the gold standard. Tag is ignored in the evaluation."
            logging.error(msg)


def read_conll_annotations(fname, glueing_col_pairs=None, structure_only=False):
    """
    Read the token annotations from a tsv file (CLEF format).

    :param str fname: file name that contains the annotation in the CLEF format.
    :param list glueing_col_pairs: concat the annotation of two columns (list with tuples).
    :param bool structure_only: read file without the actual annotation.
    :return: a nested list of TokAnnotation with the structure "[docs [sents [tokens]]]
    :rtype: list

    """
    annotations = []
    sent_annotations = []
    doc_annotations = []

    with open(fname) as csvfile:
        csvreader = csv.DictReader(csvfile, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar="")
        fieldnames = csvreader.fieldnames

        for row in csvreader:
            first_item = row[fieldnames[0]]
            # skip empty lines
            if not first_item:
                continue
            elif first_item.startswith("#"):
                # segmenting lines
                if first_item.startswith("# segment") and sent_annotations:
                    doc_annotations.append(sent_annotations)
                    sent_annotations = []

                # segmenting documents
                elif first_item.startswith("# document") and sent_annotations:
                    doc_annotations.append(sent_annotations)
                    annotations.append(doc_annotations)
                    sent_annotations = []
                    doc_annotations = []

                # other lines starting with # are dismissed

            else:

                # discard annotation and keep only structure
                if structure_only:
                    token = row[fieldnames[0]]
                    row = {k: "" for k in row}
                    row[fieldnames[0]] = token

                # perform post-hoc annotation changes
                if glueing_col_pairs:
                    for col_1, col_2 in glueing_col_pairs:
                        if row[col_2] != "O":
                            _, col_1_label = row[col_1].split("-")
                            col_2_iob, col_2_label = row[col_2].split("-")
                            new_col_2_label = f"{col_2_iob}-{col_1_label}.{col_2_label}"
                            row[col_2] = new_col_2_label

                # add final annotation
                tok_annot = TokAnnotation(row)
                sent_annotations.append(tok_annot)

    # add last document and segment as well
    if sent_annotations:
        doc_annotations.append(sent_annotations)
        annotations.append(doc_annotations)

    return annotations


def column_selector(doc, attribute):
    return [[getattr(tok, attribute) for tok in sent] for sent in doc]


def collect_named_entities(tokens: [TokAnnotation], cols: list):
    """
    Collect a list of all entities, storing the entity type and the onset and the
    offset in a named-tuple.
    For named entity, multiple annotation alternatives are not allowed.

    :param [TokAnnotation] tokens: a list of tokens of the type TokAnnotation.
    :param list cols: name of columns from which the annotation is taken.
    :return: a nested list of Entity named-tuples
    """

    named_entities = []
    start_offset = None
    end_offset = None
    ent_type = None

    for offset, token in enumerate(tokens):

        token_tag = getattr(token, cols[0])

        if token_tag == "O":
            if ent_type is not None and start_offset is not None:
                end_offset = offset - 1
                named_entities.append(Entity(ent_type, start_offset, end_offset))
                start_offset = None
                end_offset = None
                ent_type = None

        elif ent_type is None:
            ent_type = token_tag[2:]
            start_offset = offset

        elif ent_type != token_tag[2:] or (ent_type == token_tag[2:] and token_tag[:1] == "B"):

            end_offset = offset - 1
            named_entities.append(Entity(ent_type, start_offset, end_offset))

            # start of a new entity
            ent_type = token_tag[2:]
            start_offset = offset
            end_offset = None

    # catches an entity that goes up until the last token
    if ent_type and start_offset and end_offset is None:
        named_entities.append(Entity(ent_type, start_offset, len(tokens) - 1))

    # align shape of NE and link objects as the latter allows alternative annotations
    named_entities = [[ne] for ne in named_entities]

    return named_entities


def collect_link_objects(tokens, cols, n_best=1):
    """
    Collect a list of all link objects, storing the link itself and the onset
    and the offset in a named-tuple.

    Link alternatives may be provided either in separate columns using the
    cols attribute or as pipe-separated values within the the cell.

    :param [TokAnnotation] tokens: a list of tokens of the type TokAnnotation.
    :param list cols: name of column from which the annotation is taken.
    :param int n_best: the number of alternative links that should be considered (pipe-separated cell).
    :return: a nested list of Entity named-tuples that may comprise link alternatives
    """

    links = []
    start_offset = None
    end_offset = None
    ent_type = None

    if len(cols) > 1 and n_best > 1:
        msg = (
            "NEL evaluation is undefined when both a alternative column is provided as well as a n-best list within the cell."
            + "Please restrict to a single schema comprising the alternatives."
        )
        logging.error(msg)
        raise AssertionError(msg)

    for offset, token in enumerate(tokens):

        token_tag = getattr(token, cols[0])

        if token_tag in ("_", "-"):
            # end of a nel object
            if ent_type is not None and start_offset is not None:
                end_offset = offset - 1
                links.append(Entity(ent_type, start_offset, end_offset))
                start_offset = None
                end_offset = None
                ent_type = None

        # start of a new nel object
        elif ent_type is None:
            ent_type = token_tag
            start_offset = offset

        # start of a new nel object without a gap
        elif ent_type != token_tag:
            end_offset = offset - 1
            links.append(Entity(ent_type, start_offset, end_offset))

            # start of a new entity
            ent_type = token_tag
            start_offset = offset
            end_offset = None

    # catches an entity that goes up until the last token
    if ent_type and start_offset and end_offset is None:
        links.append(Entity(ent_type, start_offset, len(tokens) - 1))

    # allow alternative annotations with the same on/offset as the primary one
    links_union = []

    if len(cols) > 1:
        # alternative annotations provided in separate columns
        for link in links:
            union = []

            for col in cols:
                token_tag = getattr(tokens[link.start_offset], col)
                union.append(Entity(token_tag, link.start_offset, link.end_offset))

            links_union.append(union)
    else:
        # standard NEL evaluation that considers a n_best links of a ranked list
        # links are separated by pipes in a single cell.
        for link in links:
            union = []
            n_best_links = link.e_type.split("|")[:n_best]

            for tag in n_best_links:
                union.append(Entity(tag, link.start_offset, link.end_offset))

            links_union.append(union)

    return links_union
