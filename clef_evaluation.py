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


FINE_COLUMNS = {"NE_FINE_LIT", "NE_FINE_METO", "NE_FINE_COMP", "NE_NESTED"}
COARSE_COLUMNS = {"NE_COARSE_LIT", "NE_COARSE_METO"}
NEL_COLUMNS = {"NEL_LIT", "NEL_METO"}


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

    parser.add_argument(
        "--glueing_cols",
        required=False,
        action="store",
        dest="glueing_cols",
        help="provide two columns separated by a plus (+) whose label are glued together for the evaluation (e.g. COL1_LABEL.COL2_LABEL). \
        When glueing more than one pair, separate by comma",
    )

    parser.add_argument(
        "-s" "--skip_check",
        required=False,
        action="store_true",
        dest="skip_check",
        help="skip check that ensures the prediction file is in line with submission requirements",
    )

    return parser.parse_args()


def enforce_filename(fname):

    try:
        f_obj = pathlib.Path(fname.lower())
        submission = f_obj.stem
        suffix = f_obj.suffix
        team, bundle, lang, n_submission = submission.split("_")
        bundle = int(bundle.lstrip("bundle"))

        assert suffix == ".tsv"
        assert lang in ("de", "fr", "en")
        assert bundle in range(1, 6)

    except (ValueError, AssertionError) as e:
        raise e
        raise AssertionError(
            "Filename needs to comply with shared task requirements. "
            "Please rename accordingly: TEAMNAME_TASKBUNDLEID_LANG_RUNNUMBER.tsv",
        )

    return submission


def evaluation_wrapper(evaluator, eval_type, cols):
    eval_global = {}
    eval_per_tag = {}

    for col in cols:
        eval_global[col], eval_per_tag[col] = evaluator.evaluate(
            col, eval_type=eval_type, tags=None, merge_lines=True
        )

        # add aggregated stats across types as artificial tag
        eval_per_tag[col]["ALL"] = eval_global[col]

    return eval_per_tag


def get_results(args):

    if not args.skip_check:
        submission = enforce_filename(args.f_pred)
    else:
        submission = args.f_pred

    f_tsv = str(pathlib.Path(args.f_pred).parents[0] / f"results_{args.task}.tsv")
    f_json = str(pathlib.Path(args.f_pred).parents[0] / f"results_{args.task}_all.json")

    if args.glueing_cols:
        glueing_pairs = args.glueing_cols.split(",")
        glueing_col_pairs = [pair.split("+") for pair in glueing_pairs]
    else:
        glueing_col_pairs = None

    evaluator = Evaluator(args.f_gold, args.f_pred, glueing_col_pairs)

    if args.task == "nerc_fine":
        eval_stats = evaluation_wrapper(evaluator, eval_type="nerc", cols=FINE_COLUMNS)
        assemble_tsv_output(submission, f_tsv, eval_stats)

    elif args.task == "nerc_coarse":
        eval_stats = evaluation_wrapper(evaluator, eval_type="nerc", cols=COARSE_COLUMNS)
        assemble_tsv_output(submission, f_tsv, eval_stats)

    elif args.task == "nel":
        eval_stats = evaluation_wrapper(evaluator, eval_type="nel", cols=NEL_COLUMNS)
        assemble_tsv_output(submission, f_tsv, eval_stats, regimes=["fuzzy"], only_aggregated=True)

    with open(f_json, "w") as jsonfile:
        json.dump(
            eval_stats, jsonfile, indent=4,
        )


def assemble_tsv_output(
    submission, f_tsv, eval_stats, regimes=["fuzzy", "strict"], only_aggregated=False
):

    metrics = ("P", "R", "F1")
    figures = ("TP", "FP", "FN")
    aggregations = ("micro", "macro_doc")

    fieldnames = [
        "System",
        "Evaluation",
        "Label",
        "P",
        "R",
        "F1",
        "F1_std",
        "P_std",
        "R_std",
        "TP",
        "FP",
        "FN",
    ]

    rows = []

    for col in sorted(eval_stats):
        for aggr in aggregations:
            for regime in regimes:

                eval_regime = f"{col}-{aggr}-{regime}"
                # mapping terminology fuzzy->type
                regime = "ent_type" if regime == "fuzzy" else regime

                # collect metrics
                for tag in sorted(eval_stats[col]):

                    # collect only aggregated metrics
                    if only_aggregated and tag != "ALL":
                        continue

                    results = {}
                    results["System"] = submission
                    results["Evaluation"] = eval_regime
                    results["Label"] = tag
                    for metric in metrics:
                        mapped_metric = f"{metric}_{aggr}"
                        results[metric] = eval_stats[col][tag][regime][mapped_metric]

                    # add TP/FP/FN for micro analysis
                    if aggr == "micro":
                        for fig in figures:
                            results[fig] = eval_stats[col][tag][regime][fig]

                    if "macro" in aggr:
                        for metric in metrics:
                            mapped_metric = f"{metric}_{aggr}_std"
                            results[metric + "_std"] = eval_stats[col][tag][regime][mapped_metric]

                    for metric, fig in results.items():
                        try:
                            results[metric] = round(fig, 3)
                        except TypeError:
                            # some values are empty
                            pass

                    rows.append(results)

    with open(f_tsv, "w") as csvfile:
        writer = csv.DictWriter(csvfile, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():

    args = parse_args()

    logging.basicConfig(
        filename=args.f_log,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    get_results(args)


################################################################################
if __name__ == "__main__":
    main()
    # "data/HIPE-data-v01-sample-de.tsv", "data/HIPE-data-v01-sample-de_pred.tsv"
