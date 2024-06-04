#!/usr/bin/env python3
# coding: utf-8


"""
Evaluate the systems for the HIPE Shared Task

Usage:
  clef_evaluation.py --pred=<fpath> --ref=<fpath> --task=nerc_coarse [options]
  clef_evaluation.py --pred=<fpath> --ref=<fpath> --task=nerc_fine [options]
  clef_evaluation.py --pred=<fpath> --ref=<fpath> --task=nel [--n_best=<n>] [options]
  clef_evaluation.py -h | --help


Options:
    -h --help               Show this screen.
    -t --task=<type>        Type of evaluation task (nerc_coarse, nerc_fine, nel).
    -e --hipe_edition=<str> Specify the HIPE edition (triggers different set of columns to be considered during eval). Possible values: hipe-2020, hipe-2022 [default: hipe-2020]
    -r --ref=<fpath>        Path to gold standard file in CONLL-U-style format.
    -p --pred=<fpath>       Path to system prediction file in CONLL-U-style format.
    -o --outdir=<dir>       Path to output directory [default: .].
    -l --log=<fpath>        Path to log file.
    -g --original_nel       It splits the NEL boundaries using original CLEF algorithm.
    -n, --n_best=<n>        Evaluate NEL at particular cutoff value(s) when provided with a ranked list of entity links. Example: 1,3,5 [default: 1].
    --noise-level=<str>     Evaluate NEL or NERC also on particular noise levels (normalized Levenshtein distance of their manual OCR transcript). Example: 0.0-0.1,0.1-1.0,
    --time-period=<str>     Evaluate NEL or NERC also on particular time periods. Example: 1900-1950,1950-2000.
    --glue=<str>            Provide two columns separated by a plus (+) whose label are glued together for the evaluation (e.g. COL1_LABEL.COL2_LABEL). When glueing more than one pair, separate by comma.
    --skip-check            Skip check that ensures that the files name is in line with submission requirements.
    --tagset=<fpath>        Path to file containing the valid tagset of CLEF-HIPE.
    --suffix=<str>          Suffix that is appended to output file names and evaluation keys.
"""


import logging
import csv
import pathlib
import json
import sys

import itertools
from collections import defaultdict

from datetime import datetime
from docopt import docopt


from hipe_evaluation.ner_eval import Evaluator

# FINE_COLUMNS = ["NE-FINE-LIT", "NE-FINE-METO", "NE-FINE-COMP", "NE-NESTED"]
# COARSE_COLUMNS = ["NE-COARSE-LIT", "NE-COARSE-METO"]
# NEL_COLUMNS = ["NEL-LIT", "NEL-METO"]

COARSE_COLUMNS_HIPE2020 = ["NE-COARSE-LIT", "NE-COARSE-METO"]
FINE_COLUMNS_HIPE2020 = ["NE-FINE-LIT", "NE-FINE-METO", "NE-FINE-COMP", "NE-NESTED"]
NEL_COLUMNS_HIPE2020 = ["NEL-LIT", "NEL-METO"]

COARSE_COLUMNS_HIPE2022 = ["NE-COARSE-LIT"]
FINE_COLUMNS_HIPE2022 = ["NE-FINE-LIT", "NE-NESTED"]
NEL_COLUMNS_HIPE2022 = ["NEL-LIT"]

HIPE_EDITIONS = ["HIPE-2020", "HIPE-2022"]


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
            f"The filename of the system response '{fname}' needs to comply with the HIPE 2020 shared task requirements. "
            + "Rename according to the following scheme: TEAMNAME_TASKBUNDLEID_LANG_RUNNUMBER.tsv"
        )
        logging.error(msg)
        raise AssertionError(msg)

    return submission, lang


def enforce_filename_2022(fname: str):
    """
    Check if filename comply with the HIPE2022 convention:
    TEAMNAME_TASKBUNDLEID_DATASETALIAS_LANG_RUNNUMBER.tsv
    """
    try:
        f_obj = pathlib.Path(fname.lower())
        submission = f_obj.stem
        suffix = f_obj.suffix
        team, bundle, dataset, lang, run_nb = submission.split("_")
        logging.info(
            f"team {team} bundle {bundle} dataset {dataset} lang {lang} run_nb {run_nb}"
        )
        bundle = int(bundle.lstrip("bundle"))

        assert suffix == ".tsv", f"Problem with file suffix {suffix}"
        assert bundle in range(1, 6), f"Problem with file bundle {bundle}"
        assert dataset in {
            "ajmc",
            "newseye",
            "hipe2020",
            "topres19th",
            "sonar",
            "letemps",
        }, f"Problem with dataset {dataset}"
        assert lang in {"de", "fr", "en", "sv", "fi"}, f"Problem with language {lang}"
        assert int(run_nb) in range(1, 3), f"Problem with run number {run_nb}"

    except (ValueError, AssertionError) as e:
        logging.error(e)
        msg = (
            f"The filename of the system response '{fname}' needs to comply with the HIPE 2022 shared task requirements. "
            + "Rename according to the following scheme: TEAMNAME_TASKBUNDLEID_DATASETALIAS_LANG_RUNNUMBER.tsv"
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
    additional_cols: list = None,  # TODO: find a better name
):
    def recursive_defaultdict():
        return defaultdict(recursive_defaultdict)

    results = recursive_defaultdict()

    if additional_cols is not None:
        try:
            assert len(cols) == len(additional_cols)
        except AssertionError:
            msg = f"Additional columns must have the same size that columns. Got {cols} and {additional_cols}."
            logging.error(msg)
            raise AssertionError(msg)

    for (col_id, col), noise_level, time_period in itertools.product(
        enumerate(cols), noise_levels, time_periods
    ):
        additional_col = None
        if additional_cols is not None:
            additional_col = additional_cols[col_id]

        eval_global, eval_per_tag = (
            evaluator.evaluate(  # TODO: reorder passed args to match order of eval function def
                col,
                eval_type=eval_type,
                merge_lines=True,  # TODO: should be false for all hipe 2022
                n_best=n_best,
                noise_level=noise_level,
                time_period=time_period,
                tags=tags,
                additional_columns=additional_col,
            )
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
    edition: str,
    skip_check: bool = False,
    glueing_cols: str = None,
    n_best: list = [1],
    outdir: str = ".",
    suffix: str = "",
    f_tagset: str = None,
    noise_levels: list = [None],
    time_periods: list = [None],
    original_nel: bool = False,
):

    if not skip_check:
        if edition == "HIPE-2020":
            submission, lang = enforce_filename(f_pred)
        elif edition == "HIPE-2022":
            submission, lang = enforce_filename_2022(f_pred)

    else:
        submission = f_pred
        lang = "LANG"  # TODO: rm (?) not used afterwards it seems.

    if glueing_cols:
        glueing_pairs = glueing_cols.split(",")
        glueing_col_pairs = [pair.split("+") for pair in glueing_pairs]
    else:
        glueing_col_pairs = None

    if f_tagset:  # TODO: adapt for different tagsets (?) would be stricter.
        with open(f_tagset) as f_in:
            tagset = set(f_in.read().upper().splitlines())
    else:
        tagset = None

    evaluator = Evaluator(f_ref, f_pred, glueing_col_pairs)

    if task in ("nerc_fine", "nerc_coarse"):
        if edition == "HIPE-2022":
            ner_columns = (
                FINE_COLUMNS_HIPE2022
                if task == "nerc_fine"
                else COARSE_COLUMNS_HIPE2022
            )
        elif edition == "HIPE-2020":
            ner_columns = (
                FINE_COLUMNS_HIPE2020
                if task == "nerc_fine"
                else COARSE_COLUMNS_HIPE2020
            )

        eval_stats = evaluation_wrapper(
            evaluator,
            eval_type="nerc",
            cols=ner_columns,
            tags=tagset,
            noise_levels=noise_levels,
            time_periods=time_periods,
        )

        fieldnames, rows = assemble_tsv_output(submission, eval_stats, suffix=suffix)

    elif task == "nel":
        rows = []
        eval_stats = {}

        nel_columns = (
            NEL_COLUMNS_HIPE2020 if edition == "HIPE-2020" else NEL_COLUMNS_HIPE2022
        )

        if original_nel:
            nel_additional_cols = None
        else:
            nel_additional_cols = (
                COARSE_COLUMNS_HIPE2020
                if edition == "HIPE-2020"
                else COARSE_COLUMNS_HIPE2022
            )

        for n in n_best:
            eval_stats[n] = evaluation_wrapper(
                evaluator,
                eval_type="nel",
                cols=nel_columns,
                additional_cols=nel_additional_cols,
                n_best=n,
                noise_levels=noise_levels,
                time_periods=time_periods,
            )

            fieldnames, rows_temp = assemble_tsv_output(
                submission,
                eval_stats[n],
                n_best=n,
                # regimes=["fuzzy"],
                only_aggregated=True,
                suffix=suffix,
            )

            rows += rows_temp

    suffix = "_" + suffix if suffix else ""

    f_sub = pathlib.Path(f_pred)
    f_tsv = str(
        pathlib.Path(outdir) / f_sub.name.replace(".tsv", f"_{task}{suffix}.tsv")
    )
    f_json = str(
        pathlib.Path(outdir) / f_sub.name.replace(".tsv", f"_{task}{suffix}.json")
    )

    # write condensed results to tsv
    with open(f_tsv, "w") as csvfile:
        writer = csv.DictWriter(csvfile, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # write detailed results to json
    with open(f_json, "w") as jsonfile:
        json.dump(
            eval_stats,
            jsonfile,
            indent=4,
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

        if all(
            [
                True
                for date in [date_start, date_end]
                if date.day == 1 and date.month == 1
            ]
        ):
            # shorten label if only a year was provided (no particular month or day)
            date_start, date_end = date_start.strftime("%Y"), date_end.strftime("%Y")
        else:
            date_start, date_end = date_start.strftime("%Y"), date_end.strftime("%Y")

        return f"TIME-{date_start}-{date_end}"
    else:
        return "TIME-ALL"


def assemble_tsv_output(
    submission,
    eval_stats,
    n_best=1,
    regimes=["fuzzy", "strict"],
    only_aggregated=False,
    suffix="",
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


def main(args):

    f_ref = args["--ref"]
    f_pred = args["--pred"]
    outdir = args["--outdir"]
    hipe_edition = args["--hipe_edition"].upper()  # mandatory option
    f_log = args["--log"]
    task = args["--task"]
    original_nel = args["--original_nel"]
    n_best = args["--n_best"]
    noise_level = args["--noise-level"]
    time_period = args["--time-period"]
    glueing_cols = args["--glue"]
    skip_check = args["--skip-check"]
    f_tagset = args["--tagset"]
    suffix = args["--suffix"]

    log_fmt = f"%(asctime)s - %(levelname)s - {f_pred} - %(message)s"
    logging.basicConfig(fmt=log_fmt)
    # log warnings to file
    handler1 = logging.FileHandler(f_log, mode="w")
    handler1.setLevel(logging.WARNING)
    handler1.setFormatter(logging.Formatter(fmt=log_fmt))
    logging.getLogger().addHandler(handler1)

    # log errors also to console
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.WARNING)
    handler.setFormatter(logging.Formatter(fmt=log_fmt))
    logging.getLogger().addHandler(handler)

    if hipe_edition not in HIPE_EDITIONS:
        msg = f"Hipe edition was not or incorrectly set. Use --hipe_edition=hipe-2022 or --hipe_edition=hipe-2022. '"
        logging.error(msg)
        sys.exit(1)

    if n_best:
        n_best = [int(n) for n in n_best.split(",")]
    else:
        n_best = [1]

    if noise_level:
        noise_levels = [level.split("-") for level in noise_level.split(",") if level]
        logging.warning(f"noise_level `{noise_level}` noise_levels {noise_levels}")
        assert (
            len(noise_levels[0]) == 2
        ), f"found invalid noise level argument {noise_level} leading to {noise_levels}"
        noise_levels = [
            tuple([float(lower), float(upper)]) for lower, upper in noise_levels
        ]

        # add case to evaluate on all entities regardless of noise
        noise_levels = [None] + noise_levels

    else:
        noise_levels = [None]

    if time_period:
        time_periods = [period.split("-") for period in time_period.split(",")]
        try:
            time_periods = [
                (datetime.strptime(period[0], "%Y"), datetime.strptime(period[1], "%Y"))
                for period in time_periods
            ]
        except ValueError:
            time_periods = [
                (
                    datetime.strptime(period[0], "%Y/%m/%d"),
                    datetime.strptime(period[1], "%Y/%m/%d"),
                )
                for period in time_periods
            ]
        # add case to evaluate on all entities regardless of period
        time_periods = [None] + time_periods
    else:
        time_periods = [None]

    try:
        get_results(
            f_ref,
            f_pred,
            task,
            hipe_edition,
            skip_check,
            glueing_cols,
            n_best,
            outdir,
            suffix,
            f_tagset,
            noise_levels,
            time_periods,
            original_nel,
        )
    except AssertionError as e:
        # don't interrupt the pipeline
        print(e)


################################################################################
if __name__ == "__main__":
    args = docopt(__doc__)

    tasks = ("nerc_coarse", "nerc_fine", "nel")
    if args["--task"] not in tasks:
        msg = "Please restrict to one of the available evaluation tasks: " + ", ".join(
            tasks
        )
        logging.error(msg)
        sys.exit(1)
    logging.debug(f"ARGUMENTS {args}")
    main(args)
