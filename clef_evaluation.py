#!/usr/bin/env python3
# coding: utf-8


"""
Script to produce the evaluation for the HIPE Shared Task
"""

from ner_evaluation.ner_eval import Evaluator
import argparse
import logging
import csv
import pathlib
import json


def parse_args():
    """Parse the arguments given with program call"""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-g",
        "--file_gold",
        required=True,
        action="store",
        dest="f_gold",
        help="path to gold standard file in CONNL-U format",
    )

    parser.add_argument(
        "-p",
        "--file_pred",
        required=True,
        action="store",
        dest="f_pred",
        help="path to system prediction file in CONNL-U format",
    )

    parser.add_argument(
        "-l",
        "--log",
        action="store",
        default="clef_evaluation.log",
        dest="f_log",
        help="name of log file",
    )

    parser.add_argument(
        "-t",
        "--task",
        required=True,
        action="store",
        dest="task",
        help="type of evaluation",
        choices={"nerc_fine", "nerc_coarse", "nel"},
    )

    return parser.parse_args()


def evaluation_nerc_fine(evaluator):

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
            anno_type, eval_type="nerc", tags=None
        )

    return nerc_fine, nerc_fine_per_tag


def evaluation_nerc_coarse(evaluator):

    labels = {"LOC", "ORG", "PERS", "PROD", "TIME"}
    anno_types = {"NE_COARSE_LIT", "NE_COARSE_METO"}

    nerc_coarse = {}
    nerc_coarse_per_tag = {}

    for anno_type in anno_types:
        nerc_coarse[anno_type], nerc_coarse_per_tag[anno_type] = evaluator.evaluate(
            anno_type, eval_type="nerc", tags=labels
        )

    return nerc_coarse, nerc_coarse_per_tag


def evaluation_nel(evaluator):
    anno_types = {"NEL_LIT", "NEL_METO"}
    nel = {}
    nel_per_type = {}
    for anno_type in anno_types:
        nel[anno_type], nel_per_type[anno_type] = evaluator.evaluate(
            anno_type, eval_type="nel", tags=None
        )
    return nel, nel_per_type


def evaluation(args):

    f_tsv = str(pathlib.Path(args.f_pred).parents[0] / "results.tsv")
    f_json = str(pathlib.Path(args.f_pred).parents[0] / "results_all.json")
    rows_output = []

    evaluator = Evaluator(args.f_gold, args.f_pred)

    header = [
        "Evaluation",
        "Label",
        "P",
        "R",
        "F1",
        "TP",
        "FP",
        "FN",
    ]

    metrics = ["precision", "recall", "f1", "correct", "spurious", "missed"]

    if args.task.startswith("nerc"):
        regimes = ["fuzzy", "strict"]

        if args.task == "nerc_fine":
            eval_global, eval_per_tag = evaluation_nerc_fine(evaluator)

        elif args.task == "nerc_coarse":
            eval_global, eval_per_tag = evaluation_nerc_coarse(evaluator)

        # assemble output for nerc
        for anno_type in eval_per_tag:

            for regime in regimes:

                eval_regime = f"{anno_type}-micro-{regime}"
                # mapping terminology fuzzy->type
                regime = "ent_type" if regime == "fuzzy" else regime

                # collect metrics per type
                for tag in eval_per_tag[anno_type]:
                    results = [eval_regime, tag]
                    for metric in metrics:
                        results.append(eval_per_tag[anno_type][tag][regime][metric])

                    rows_output.append(results)

                # collect aggregated metrics
                results = [eval_regime, "ALL"]
                for metric in metrics:
                    results.append(eval_global[anno_type][regime][metric])

                rows_output.append(results)

    elif args.task == "nel":
        eval_global, eval_per_tag = evaluation_nel(evaluator)

        # assemble output for nel
        for anno_type in eval_per_tag:
            eval_regime = anno_type + "-micro-fuzzy"

            # collect aggregated metrics
            results = [eval_regime, "ALL"]
            for metric in metrics:
                results.append(eval_global[anno_type]["ent_type"][metric])
            rows_output.append(results)

    with open(f_tsv, "w") as csvfile:
        writer = csv.writer(csvfile, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows_output)

    results_all = {"aggregated": eval_global, "per_type": eval_per_tag}
    with open(f_json, "w") as jsonfile:
        json.dump(
            results_all, jsonfile, indent=4,
        )


def main():

    args = parse_args()

    logging.basicConfig(
        filename=args.f_log,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    evaluation(args)


################################################################################
if __name__ == "__main__":
    main()
    # "data/HIPE-data-v01-sample-de.tsv", "data/HIPE-data-v01-sample-de_pred.tsv"
