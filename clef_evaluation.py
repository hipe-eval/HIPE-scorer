#!/usr/bin/env python3
# coding: utf-8


"""
Script to produce the evaluation for the HIPE Shared Task
"""

from ner_evaluation.ner_eval import Evaluator


def evaluation(task):
    pass


evaluator = Evaluator("data/HIPE-data-v01-sample-de.tsv", "data/HIPE-data-v01-sample-de_pred.tsv")


# NERC-FINE
labels = {
    "LOC.ADD.ELEC",
    "LOC.ADD.PHYS",
    "LOC.ADM.NAT",
    "LOC.ADM.REG",
    "LOC.ADM.SUP",
    "LOC.ADM.TOWN",
    "LOC.FAC",
    "LOC.ORO",
    "LOC.PHYS.ASTRO",
    "LOC.PHYS.GEO",
    "LOC.PHYS.HYDRO",
    "LOC.UNK",
    "ORG.ADM",
    "ORG.ENT",
    "ORG.ENT.PRESSAGENCY",
    "PERS.COLL",
    "PERS.IND",
    "PERS.IND.ARTICLEAUTHOR",
    "PROD.DOCTR",
    "PROD.MEDIA",
    "TIME.DATE.ABS",
}
anno_types = {"NE_FINE_LIT", "NE_FINE_METO", "NE_FINE_COMP", "NE_NESTED"}

nerc_fine = {}
nerc_fine_per_tag = {}

for anno_type in anno_types:
    nerc_fine[anno_type], nerc_fine_per_tag[anno_type] = evaluator.evaluate(
        anno_type, eval_type="nerc", tags=labels
    )


# NERC-Coarse
labels = {"LOC", "ORG", "PERS", "PROD", "TIME"}
anno_types = {"NE_COARSE_LIT", "NE_COARSE_METO"}

nerc_coarse = {}
nerc_coarse_per_tag = {}

for anno_type in anno_types:
    nerc_coarse[anno_type], nerc_coarse_per_tag[anno_type] = evaluator.evaluate(
        anno_type, eval_type="nerc", tags=labels
    )


# NEL
anno_types = {"NEL_LIT", "NEL_METO"}
nel = {}
nel_per_type = {}
for anno_type in anno_types:
    nel[anno_type], nel_per_type[anno_type] = evaluator.evaluate(
        anno_type, eval_type="nel", tags=None
    )
