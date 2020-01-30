#!/usr/bin/env python3
# coding: utf-8

import csv
from collections import namedtuple

Entity = namedtuple("Entity", "e_type start_offset end_offset")


def get_all_tags(annotations):

    labels = {label.split("-")[-1] for sent in annotations for label in sent}
    if "_" in labels:
        labels.remove("_")
    if "O" in labels:
        labels.remove("O")

    return labels


def read_conll_annotations(fname):
    annotations = []
    sent_annotations = []

    with open(fname) as csvfile:
        reader = csv.reader(csvfile, delimiter="\t")
        attributes = next(reader)

        attributes = [attr.replace("-", "_") for attr in attributes]

        TokAnnotation = namedtuple("TokAnnotation", attributes)

        for row in reader:
            # skip empty lines and comments including meta data
            if not list(filter(None, row)):
                continue
            elif row[0].startswith("#"):
                if sent_annotations:
                    annotations.append(sent_annotations)
                    sent_annotations = []
                continue
            else:
                row = [item.upper() for item in row]
                tok_annot = TokAnnotation(*row)
                sent_annotations.append(tok_annot)

    return annotations


def segment2labels(sent, attribute):
    return [getattr(tok, attribute) for tok in sent]


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
