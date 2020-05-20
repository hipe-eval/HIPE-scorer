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
import sys


FINE_COLUMNS = {"NE-FINE-LIT", "NE-FINE-METO", "NE-FINE-COMP", "NE-NESTED"}
COARSE_COLUMNS = {"NE-COARSE-LIT", "NE-COARSE-METO"}
NEL_COLUMNS = ["NEL-LIT", "NEL-METO"]


def parse_args():
    """Parse the arguments given with program call"""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-r",
        "--ref",
        required=True,
        action="store",
        dest="f_ref",
        help="path to gold standard file in CONNL-U format",
    )

    parser.add_argument(
        "-p",
        "--pred",
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
        "-n",
        "--n_best",
        required=False,
        action="store",
        dest="n_best",
        help="limit the number of alternative entity links if multiple separate by a comma. Alternative links are separated by a pipe in a single cell",
    )

    parser.add_argument(
        "-u",
        "--union",
        required=False,
        action="store_true",
        dest="union",
        help="consider the union of the metonymic and literal annotation for the evaluation of NEL",
    )

    parser.add_argument(
        "-s",
        "--skip_check",
        required=False,
        action="store_true",
        dest="skip_check",
        help="skip check that ensures the prediction file is in line with submission requirements",
    )

    parser.add_argument(
        "-o",
        "--outdir",
        action="store",
        default=".",
        dest="outdir",
        help="name of output directory",
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

    except (ValueError, AssertionError):
        msg = (
            "The filename of the system response '{self.f_pred}' needs to comply with the shared task requirements. "
            + "Rename according to the following scheme: TEAMNAME_TASKBUNDLEID_LANG_RUNNUMBER.tsv"
        )
        logging.error(msg)
        raise AssertionError(msg)

    return submission, lang


def evaluation_wrapper(evaluator, eval_type, cols, n_best=1):
    eval_global = {}
    eval_per_tag = {}

    for col in cols:
        eval_global[col], eval_per_tag[col] = evaluator.evaluate(
            col, eval_type=eval_type, tags=None, merge_lines=True, n_best=n_best
        )

        # add aggregated stats across types as artificial tag
        eval_per_tag[col]["ALL"] = eval_global[col]

    return eval_per_tag


def get_results(
    f_ref, f_pred, task, skip_check=False, glueing_cols=None, n_best=[1], union=False, outdir="."
):

    if not skip_check:
        submission, lang = enforce_filename(f_pred)
    else:
        submission = f_pred
        lang = "LANG"

    f_sub = pathlib.Path(f_pred)
    f_tsv = str(pathlib.Path(outdir) / f_sub.name.replace(".tsv", f"_{task}_results.tsv"))
    f_json = str(pathlib.Path(outdir) / f_sub.name.replace(".tsv", f"_{task}_results.json"))

    if glueing_cols:
        glueing_pairs = glueing_cols.split(",")
        glueing_col_pairs = [pair.split("+") for pair in glueing_pairs]
    else:
        glueing_col_pairs = None

    evaluator = Evaluator(f_ref, f_pred, glueing_col_pairs)

    if task == "nerc_fine":
        eval_stats = evaluation_wrapper(evaluator, eval_type="nerc", cols=FINE_COLUMNS)
        fieldnames, rows = assemble_tsv_output(submission, eval_stats)

    elif task == "nerc_coarse":
        eval_stats = evaluation_wrapper(evaluator, eval_type="nerc", cols=COARSE_COLUMNS)
        fieldnames, rows = assemble_tsv_output(submission, eval_stats)

    elif task == "nel" and not union:
        rows = []
        # evaluate for various n-best
        for n in n_best:
            eval_stats = evaluation_wrapper(evaluator, eval_type="nel", cols=NEL_COLUMNS, n_best=n)
            fieldnames, rows_n_best = assemble_tsv_output(
                submission, eval_stats, regimes=["fuzzy"], only_aggregated=True, suffix=f"best@{n}",
            )
            rows += rows_n_best

    elif task == "nel" and union:
        eval_global, eval_per_tag = evaluator.evaluate(
            NEL_COLUMNS, eval_type="nel", tags=None, merge_lines=True, n_best=1
        )
        # add aggregated stats across types as artificial tag
        eval_per_tag["ALL"] = eval_global
        eval_stats = {"|".join(NEL_COLUMNS): eval_global}

        fieldnames, rows = assemble_tsv_output(
            submission,
            eval_stats,
            regimes=["fuzzy"],
            only_aggregated=True,
            suffix=f"union@lit-meto",
        )

    with open(f_tsv, "w") as csvfile:
        writer = csv.DictWriter(csvfile, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(f_json, "w") as jsonfile:
        json.dump(
            eval_stats, jsonfile, indent=4,
        )


def assemble_tsv_output(
    submission, eval_stats, regimes=["fuzzy", "strict"], only_aggregated=False, suffix=""
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

    if suffix:
        suffix = "-" + suffix

    for col in sorted(eval_stats):
        for aggr in aggregations:
            for regime in regimes:

                eval_regime = f"{col}-{aggr}-{regime}{suffix}"
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

    return fieldnames, rows


def check_validity_of_arguments(args):
    if args.task != "nel" and (args.union or args.n_best):
        msg = "The provided arguments are not valid. Alternative annotations are only allowed for the NEL evaluation."
        logging.error(msg)
        raise AssertionError(msg)

    if args.union and args.n_best:
        msg = "The provided arguments are not valid. Restrict to a single evaluation schema for NEL, either a ranked n-best list or the union of the metonymic and literal column."
        logging.error(msg)
        raise AssertionError(msg)


def main():
    args = parse_args()

    logging.basicConfig(
        filename=args.f_log,
        filemode="w",
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        check_validity_of_arguments(args)
    except Exception as e:
        print(e)
        sys.exit(1)

    if not args.n_best:
        n_best = [1]
    else:
        n_best = [int(n) for n in args.n_best.split(",")]

    try:
        get_results(
            args.f_ref,
            args.f_pred,
            args.task,
            args.skip_check,
            args.glueing_cols,
            n_best,
            args.union,
            args.outdir,
        )
    except AssertionError as err:
        print(err)


################################################################################
if __name__ == "__main__":
    main()
    # "data/HIPE-data-v01-sample-de.tsv", "data/HIPE-data-v01-sample-de_pred.tsv"
