#!/usr/bin/env python3
# coding: utf-8


"""
Evaluate the systems for the HIPE Shared Task

Usage:
  clef_evaluation --pred=<fpath> --ref=<fpath> ( --task=nerc_coarse | --task=nerc_fine ) [options]
  clef_evaluation --pred=<fpath> --ref=<fpath> --task=nel [--n_best=<kn>] [options]
  clef_evaluation -h | --help


Options:
    -h --help             Show this screen.
    -r --ref=<fpath>      Path to gold standard file in CONLL-U-style format.
    -p --pred=<fpath>     Path to system prediction file in CONLL-U-style format.
    -o --outdir=<dir>     Path to output directory.
    -l --log=<fpath>      Path to log file.
    -t --task=<task>      Type of evaluation task (nerc_fine, nerc_coarse, nel).
    -n, --n_best=<kn>     Evaluate NEL at particular cutoff value(s) provided with a ranked list of entity links, separate with a comma if multiple cutoffs. Link lists use a pipe as separator [default: 1].
    --noise-level         Evaluate NEL or NERC also on particular noise levels according to normalized Levenshtein distance of their manual OCR transcript. Example: 0.0-0.1,0.1-1.0",
    --time-period         Evaluate NEL or NERC also on particular time periods. Example: 1900-1950,1950-2000
    --glue=<str>          Provide two columns separated by a plus (+) whose label are glued together for the evaluation (e.g. COL1_LABEL.COL2_LABEL). When glueing more than one pair, separate by comma.
    --skip_check          Skip check that ensures the prediction file is in line with submission requirements.
    --tagset=<fpath>      Path to file containing the valid tagset.
    --suffix=<str>        Suffix that is appended to output file names and evaluation keys.
"""


import argparse
import logging
import csv
import pathlib
import json
import sys

import itertools
from collections import defaultdict

from datetime import datetime


from ner_evaluation.ner_eval import Evaluator

FINE_COLUMNS = ["NE-FINE-LIT", "NE-FINE-METO", "NE-FINE-COMP", "NE-NESTED"]
COARSE_COLUMNS = ["NE-COARSE-LIT", "NE-COARSE-METO"]
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
        help="path to gold standard file in CONLL-U-style format",
    )

    parser.add_argument(
        "-p",
        "--pred",
        required=True,
        action="store",
        dest="f_pred",
        help="path to system prediction file in CONLL-U-style format",
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
        help="evaluate at particular cutoff value(s) for an ordered list of entity links, separate with a comma if multiple cutoffs. Link lists use a pipe as separator.",
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
    parser.add_argument(
        "--suffix",
        action="store",
        default="",
        dest="suffix",
        help="Suffix to append at output file names",
    )

    parser.add_argument(
        "--tagset", action="store", dest="f_tagset", help="file containing the valid tagset",
    )

    parser.add_argument(
        "--noise-level",
        action="store",
        dest="noise_level",
        help="evaluate NEL or NERC also on particular noise levels according to normalized Levenshtein distance of their manual OCR transcript. Example: 0.0-0.1,0.1-1.0",
    )

    parser.add_argument(
        "--time-period",
        action="store",
        dest="time_period",
        help="evaluate NEL or NERC also on particular time periods. Example: 1900-1950,1950-2000",
    )

    return parser.parse_args()


def enforce_filename(fname: str):

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
            f"The filename of the system response '{fname}' needs to comply with the shared task requirements. "
            + "Rename according to the following scheme: TEAMNAME_TASKBUNDLEID_LANG_RUNNUMBER.tsv"
        )
        logging.error(msg)
        raise AssertionError(msg)

    return submission, lang


def evaluation_wrapper(
    evaluator,
    cols: list,
    eval_type: str,
    n_best: int = 1,
    noise_levels: list = [None],
    time_periods: list = [None],
    tags: set = None,
):
    def recursive_defaultdict():
        return defaultdict(recursive_defaultdict)

    results = recursive_defaultdict()

    for col, noise_level, time_period in itertools.product(cols, noise_levels, time_periods):
        eval_global, eval_per_tag = evaluator.evaluate(
            col,
            eval_type=eval_type,
            merge_lines=True,
            n_best=n_best,
            noise_level=noise_level,
            time_period=time_period,
            tags=tags,
        )

        time_period = define_time_label(time_period)
        noise_level = define_noise_label(noise_level)

        # add aggregated stats across types as artificial tag
        results[col][time_period][noise_level] = eval_per_tag
        results[col][time_period][noise_level]["ALL"] = eval_global

    return results


def get_results(
    f_ref: str,
    f_pred: str,
    task: str,
    skip_check: bool = False,
    glueing_cols: str = None,
    n_best: list = [1],
    outdir: str = ".",
    suffix: str = "",
    f_tagset: str = None,
    noise_levels: list = [None],
    time_periods: list = [None],
):

    if not skip_check:
        submission, lang = enforce_filename(f_pred)
    else:
        submission = f_pred
        lang = "LANG"

    if glueing_cols:
        glueing_pairs = glueing_cols.split(",")
        glueing_col_pairs = [pair.split("+") for pair in glueing_pairs]
    else:
        glueing_col_pairs = None

    if f_tagset:
        with open(f_tagset) as f_in:
            tagset = set(f_in.read().upper().splitlines())
    else:
        tagset = None

    evaluator = Evaluator(f_ref, f_pred, glueing_col_pairs)

    if task in ("nerc_fine", "nerc_coarse"):
        columns = FINE_COLUMNS if task == "nerc_fine" else COARSE_COLUMNS

        eval_stats = evaluation_wrapper(
            evaluator,
            eval_type="nerc",
            cols=columns,
            tags=tagset,
            noise_levels=noise_levels,
            time_periods=time_periods,
        )

        fieldnames, rows = assemble_tsv_output(submission, eval_stats, suffix=suffix)

    elif task == "nel":
        rows = []
        eval_stats = {}

        for n in n_best:
            eval_stats[n] = evaluation_wrapper(
                evaluator,
                eval_type="nel",
                cols=NEL_COLUMNS,
                n_best=n,
                noise_levels=noise_levels,
                time_periods=time_periods,
            )

            fieldnames, rows_temp = assemble_tsv_output(
                submission,
                eval_stats[n],
                n_best=n,
                regimes=["fuzzy"],
                only_aggregated=True,
                suffix=suffix,
            )

            rows += rows_temp

    if suffix:
        suffix = "_" + suffix

    f_sub = pathlib.Path(f_pred)
    f_tsv = str(pathlib.Path(outdir) / f_sub.name.replace(".tsv", f"_{task}{suffix}.tsv"))
    f_json = str(pathlib.Path(outdir) / f_sub.name.replace(".tsv", f"_{task}{suffix}.json"))

    # write condesed results to tsv
    with open(f_tsv, "w") as csvfile:
        writer = csv.DictWriter(csvfile, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # write detailed results to json
    with open(f_json, "w") as jsonfile:
        json.dump(
            eval_stats, jsonfile, indent=4,
        )


def define_noise_label(noise_level):
    if noise_level:
        noise_lower, noise_upper = noise_level
        return f"LED-{noise_lower}-{noise_upper}"
    else:
        return "LED-ALL"


def define_time_label(time_period):
    if time_period:
        date_start, date_end = time_period

        if all([True for date in [date_start, date_end] if date.day == 1 and date.month == 1]):
            # shorten label if only a year was provided (no particular month or day)
            date_start, date_end = date_start.strftime("%Y"), date_end.strftime("%Y")
        else:
            date_start, date_end = date_start.strftime("%Y"), date_end.strftime("%Y")

        return f"TIME-{date_start}-{date_end}"
    else:
        return "TIME-ALL"


def assemble_tsv_output(
    submission, eval_stats, n_best=1, regimes=["fuzzy", "strict"], only_aggregated=False, suffix="",
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

    # dirty lookup of unknown keys to avoid for-loops
    col = next(iter(eval_stats))
    time_periods = list(iter(eval_stats[col]))
    noise_levels = list(iter(eval_stats[col][time_periods[0]]))

    for col, time_period, noise_level, aggr, regime in itertools.product(
        sorted(eval_stats), time_periods, noise_levels, aggregations, regimes
    ):

        n_best_suffix = f"-@{n_best}" if "NEL" in col else ""
        eval_regime = (
            f"{col}-{aggr}-{regime}-"
            + f"{suffix + '-' if suffix else ''}"
            + time_period
            + "-"
            + noise_level
            + n_best_suffix
        )

        # mapping terminology fuzzy->type
        regime = "ent_type" if regime == "fuzzy" else regime

        eval_handle = eval_stats[col][time_period][noise_level]

        # collect metrics
        for tag in sorted(eval_handle):

            # collect only aggregated metrics
            if only_aggregated and tag != "ALL":
                continue

            results = {}
            results["System"] = submission
            results["Evaluation"] = eval_regime
            results["Label"] = tag

            for metric in metrics:
                mapped_metric = f"{metric}_{aggr}"
                results[metric] = eval_handle[tag][regime][mapped_metric]

            # add TP/FP/FN for micro analysis
            if aggr == "micro":
                for fig in figures:
                    results[fig] = eval_handle[tag][regime][fig]

            if "macro" in aggr:
                for metric in metrics:
                    mapped_metric = f"{metric}_{aggr}_std"
                    results[metric + "_std"] = eval_handle[tag][regime][mapped_metric]

            for key, val in results.items():
                try:
                    results[key] = round(val, 3)
                except TypeError:
                    # some values are empty
                    pass

            rows.append(results)

    return fieldnames, rows


def check_validity_of_arguments(args):
    if args.task != "nel" and (args.n_best):
        msg = "The provided arguments are not valid. Alternative annotations are only allowed for the NEL evaluation."
        logging.error(msg)
        raise AssertionError(msg)


def main():
    args = parse_args()

    # log to file
    logging.basicConfig(
        filename=args.f_log,
        filemode="w",
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # log errors also to console
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.ERROR)
    logging.getLogger().addHandler(handler)

    try:
        check_validity_of_arguments(args)
    except Exception as e:
        print(e)
        sys.exit(1)

    if args.n_best:
        n_best = [int(n) for n in args.n_best.split(",")]
    else:
        n_best = [1]

    if args.noise_level:
        noise_levels = [level.split("-") for level in args.noise_level.split(",")]
        noise_levels = [tuple([float(lower), float(upper)]) for lower, upper in noise_levels]

        # add case to evaluate on all entities regardless of noise
        noise_levels = [None] + noise_levels

    else:
        noise_levels = [None]

    if args.time_period:
        time_periods = [period.split("-") for period in args.time_period.split(",")]
        try:
            time_periods = [
                (datetime.strptime(period[0], "%Y"), datetime.strptime(period[1], "%Y"))
                for period in time_periods
            ]
        except ValueError:
            time_periods = [
                (datetime.strptime(period[0], "%Y/%m/%d"), datetime.strptime(period[1], "%Y/%m/%d"))
                for period in time_periods
            ]
        # add case to evaluate on all entities regardless of period
        time_periods = [None] + time_periods
    else:
        time_periods = [None]

    try:
        get_results(
            args.f_ref,
            args.f_pred,
            args.task,
            args.skip_check,
            args.glueing_cols,
            n_best,
            args.outdir,
            args.suffix,
            args.f_tagset,
            noise_levels,
            time_periods,
        )
    except AssertionError as e:
        # don't interrupt the pipeline
        print(e)


################################################################################
if __name__ == "__main__":
    main()
