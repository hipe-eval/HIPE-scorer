#!/usr/bin/env python
# coding: utf-8

"""
Normalize entity linking by remapping links according to an external file

Usage:
    normalize_linking.py -i=<fpath> -o=<fpath> [--norm-time (--norm-histo --map=<fpath>) --union-meto-lit] [--hipe_edition=<str>]
    normalize_linking.py -h | --help

Options:
    -h --help               Show this screen.
    -i --in=<fpath>         File path to original system response.
    -o --out=<fpath>        File path to normalized system response.
    -m --map=<fpath>        File path to link historical mapping resource.
    --norm-time             Normalize NEL for time mentions by linking to NIL.
    --norm-histo            Normalize NEL for historical entities
    --union-meto-lit        Unionize literal and metonymic columns (apply on both columns).
    -e --hipe_edition=<str> Specify the HIPE edition. Ignores METO columns if set to hipe-2022.  Possible values: hipe-2020, hipe-2022 [default: hipe-2020]


All file path can be local or remote URLs.

"""

import csv
import itertools
import pandas as pd
from docopt import docopt

HIPE_EDITIONS = ["HIPE-2020", "HIPE-2022"]

def get_mappings(f_map):
    df_mapping = pd.read_csv(f_map, delimiter="\t")
    df_mapping = df_mapping.melt("Main").drop("variable", axis=1)
    df_mapping["Main"] = df_mapping["Main"].str.extract("(Q[0-9]+)")
    df_mapping["value"] = df_mapping["value"].str.extract("(Q[0-9]+)")
    mapping = dict(df_mapping[["value", "Main"]].values)

    return mapping


def normalize_n_to_n(df: pd.DataFrame, mapping: dict):
    """
    Remap linking alternatives to a common main identity.
    """

    try:
        # remap literal NEL column
        df["NEL-LIT"] = df["NEL-LIT"].str.split("|")
        df["NEL-LIT"] = df["NEL-LIT"].apply(
            lambda row: [mapping[k] if mapping.get(k) else k for k in row]
        )
        df["NEL-LIT"] = df["NEL-LIT"].str.join("|")

        # remap metonymic NEL column
        df["NEL-METO"] = df["NEL-METO"].str.split("|")
        df["NEL-METO"] = df["NEL-METO"].apply(
            lambda row: [mapping[k] if mapping.get(k) else k for k in row]
        )
        df["NEL-METO"] = df["NEL-METO"].str.join("|")
    except KeyError:
        pass

    return df


def unionize_meto_lit(df: pd.DataFrame):
    """
    Unionize the metonymic and the literal columns (apply on both columns).

    The order is kept and "EMPTY" is used as placeholder in case of mismatching
    list length.

    """

    def union(list1, list2):
        if list1[0]:
            return list(
                itertools.chain.from_iterable(
                    itertools.zip_longest(list1, list2, fillvalue="EMPTY")
                )
            )
        else:
            return [""]

    try:
        df["NEL-LIT-LIST"] = df["NEL-LIT"].str.split("|")
        df["NEL-METO-LIST"] = df["NEL-METO"].str.split("|")

        # unionize the literal and metonymic columns as ranked list
        df["NEL-LIT-UNION"] = (
            df[["NEL-LIT-LIST", "NEL-METO-LIST"]]
            .dropna()
            .apply(lambda x: union(x[0], x[1]), axis=1)
        )
        df["NEL-METO-UNION"] = (
            df[["NEL-METO-LIST", "NEL-LIT-LIST"]]
            .dropna()
            .apply(lambda x: union(x[0], x[1]), axis=1)
        )
        df["NEL-LIT-UNION"] = df["NEL-LIT-UNION"].str.join("|")
        df["NEL-METO-UNION"] = df["NEL-METO-UNION"].str.join("|")

        # keep the original _ for non-annotations
        # as there may be literal annotations lacking a metonymic sense
        # may be vice-versa
        df.loc[df["NEL-METO"] == "_", "NEL-METO-UNION"] = "_"
        df.loc[df["NEL-METO"] == "-", "NEL-METO-UNION"] = "-"

        df.loc[df["NEL-LIT"] == "_", "NEL-LIT-UNION"] = "_"
        df.loc[df["NEL-LIT"] == "-", "NEL-LIT-UNION"] = "-"

        df["NEL-LIT"] = df["NEL-LIT-UNION"]
        df["NEL-METO"] = df["NEL-METO-UNION"]

        # remove intermediate results
        df = df.drop(columns=["NEL-LIT-UNION", "NEL-METO-UNION", "NEL-LIT-LIST", "NEL-METO-LIST"])

    except KeyError:
        pass

    return df


def remove_time_linking(df, replacement="NIL",map_meto=True):
    try:
        df.loc[df["NE-COARSE-LIT"].str.contains("time"), "NEL-LIT"] = replacement
        if map_meto:
            df.loc[df["NE-COARSE-LIT"].str.contains("time"), "NEL-METO"] = replacement
    except KeyError:
        pass

    return df


def main(args):

    f_in = args["--in"]
    f_out = args["--out"]
    f_map = args["--map"]
    norm_time = args["--norm-time"]
    norm_histo = args["--norm-histo"]
    unionize = args["--union-meto-lit"]
    hipe_edition = args["--hipe_edition"].upper()  # mandatory option

    if hipe_edition not in HIPE_EDITIONS:
        msg = f"Hipe edition was not or incorrectly set. Use --hipe_edition=hipe-2022 or --hipe_edition=hipe-2022. '"
        logging.error(msg)
        sys.exit(1)

    df = pd.read_csv(f_in, sep="\t", quoting=csv.QUOTE_NONE, quotechar="", skip_blank_lines=False)
    df = df.fillna(value={"NE-COARSE-LIT": "", "NEL-LIT": "", "NEL-METO": ""})

    if norm_histo:
        mappings = get_mappings(f_map)
        df = normalize_n_to_n(df, mappings)

    if norm_time:
        df = remove_time_linking(df,map_meto=hipe_edition == 'HIPE-2020')

    if unionize:
        df = unionize_meto_lit(df)

    df.to_csv(f_out, index=False, sep="\t", quoting=csv.QUOTE_NONE, quotechar="")


if __name__ == "__main__":
    arguments = docopt(__doc__)
    main(arguments)
