#!/usr/bin/env python3
# coding: utf-8

import csv
from collections import namedtuple
import logging

Entity = namedtuple("Entity", "e_type start_offset end_offset")


def get_all_tags(y_true):

    tags = {label.split("-")[-1] for doc in y_true for seg in doc for label in seg}
    if "_" in tags:
        tags.remove("_")
    if "O" in tags:
        tags.remove("O")

    return tags


def check_tag_selection(y_cand, tags_ref):

    tags_cand = get_all_tags(y_cand)

    remove_tags = set()

    for tag in tags_ref:
        if tag not in tags_cand:
            logging.info(
                f"Selected tag '{tag}' is not covered by the gold data set and ignored for in the evaluation."
            )

            remove_tags.add(tag)

    return tags_cand


def check_spurious_tags(y_true, y_pred):

    tags_true = get_all_tags(y_true)
    tags_pred = get_all_tags(y_pred)

    for pred in tags_pred:
        if pred not in tags_true:

            logging.error(
                f"Spurious entity label '{pred}' in predictions. Tag is not part of the gold standard and ignored for in the evaluation."
            )


def read_conll_annotations(fname, glueing_col_pairs=None):
    annotations = []
    sent_annotations = []
    doc_annotations = []

    with open(fname) as csvfile:
        csvreader = csv.DictReader(csvfile, delimiter="\t")
        fieldnames = csvreader.fieldnames

        attributes = [attr.replace("-", "_") for attr in fieldnames]

        TokAnnotation = namedtuple("TokAnnotation", attributes)

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
                    annotations.append(doc_annotations)
                    doc_annotations = []
                # other lines starting with # are dismissed

            else:
                if glueing_col_pairs:
                    for col_1, col_2 in glueing_col_pairs:
                        if row[col_2] != "O":
                            _, col_1_label = row[col_1].split("-")
                            col_2_iob, col_2_label = row[col_2].split("-")
                            new_col_2_label = f"{col_2_iob}-{col_1_label}.{col_2_label}"
                            row[col_2] = new_col_2_label

                row = [val.upper() for key, val in row.items()]
                tok_annot = TokAnnotation(*row)
                sent_annotations.append(tok_annot)

    # add last document and segment as well
    if sent_annotations:
        doc_annotations.append(sent_annotations)
        annotations.append(doc_annotations)

    return annotations


def column_selector(doc, attribute):
    return [[getattr(tok, attribute) for tok in sent] for sent in doc]


def collect_named_entities(tokens):
    """
    Creates a list of Entity named-tuples, storing the entity type and the start and end
    offsets of the entity.

    :param tokens: a list of tags
    :return: a list of Entity named-tuples
    """

    named_entities = []
    start_offset = None
    end_offset = None
    ent_type = None

    for offset, token_tag in enumerate(tokens):

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

        elif ent_type != token_tag[2:] or (
            ent_type == token_tag[2:] and token_tag[:1] == "B"
        ):

            end_offset = offset - 1
            named_entities.append(Entity(ent_type, start_offset, end_offset))

            # start of a new entity
            ent_type = token_tag[2:]
            start_offset = offset
            end_offset = None

    # catches an entity that goes up until the last token

    if ent_type and start_offset and end_offset is None:
        named_entities.append(Entity(ent_type, start_offset, len(tokens) - 1))

    return named_entities


def collect_link_objects(tokens):
    """
    Creates a list of Entity named-tuples, storing the entity type and the start and end
    offsets of the entity.

    :param tokens: a list of tags
    :return: a list of Entity named-tuples
    """

    link_objects = []
    start_offset = None
    end_offset = None
    ent_type = None

    for offset, token_tag in enumerate(tokens):

        if token_tag == "_":
            # end of a nel object
            if ent_type is not None and start_offset is not None:
                end_offset = offset - 1
                link_objects.append(Entity(ent_type, start_offset, end_offset))
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
            link_objects.append(Entity(ent_type, start_offset, end_offset))

            # start of a new entity
            ent_type = token_tag
            start_offset = offset
            end_offset = None

    # catches an entity that goes up until the last token
    if ent_type and start_offset and end_offset is None:
        link_objects.append(Entity(ent_type, start_offset, len(tokens) - 1))

    return link_objects
