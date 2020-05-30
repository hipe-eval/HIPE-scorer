#!/usr/bin/env python
# coding: utf-8

"""
Normalize entity linking by remapping linkgs according to an external file

Usage:
    lib/normalize_linking.py -i=<fpath> -o=<fpath> --norm-time [options]
    lib/normalize_linking.py -i=<fpath> -o=<fpath> --map=<fpath> --norm-histo [options]
    lib/normalize_linking.py -i=<fpath> -o=<fpath> --norm-time --map=<fpath> --norm-histo [options]

Options:
    -h --help               Show this screen.
    -i --in=<fpath>         File path to original system response.
    -o --out=<fpath>        File path to normalized system response.
    -m --map=<fpath>        File path to link mapping.
    --norm-time             Normalize NEL for time mentions.
    --norm-histo            Normalize NEL for historical entities
    --union-meto-lit        Unionize literal and metonymic columns (apply on both columns)
"""

import csv
import pandas as pd
from docopt import docopt
import itertools


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
        df["NEL-LIT"] = df["NEL-LIT"].str.split("|")
        df["NEL-METO"] = df["NEL-METO"].str.split("|")


        df["NEL-LIT-UNION"] = (
            df[["NEL-LIT", "NEL-METO"]].dropna().apply(lambda x: union(x[0], x[1]), axis=1)
        )
        df["NEL-METO-UNION"] = (
            df[["NEL-METO", "NEL-LIT"]].dropna().apply(lambda x: union(x[0], x[1]), axis=1)
        )

        df["NEL-LIT"] = df["NEL-LIT-UNION"].str.join("|")
        df["NEL-METO"] = df["NEL-METO-UNION"].str.join("|")

        # remove intermediate results
        df.drop(columns=['NEL-LIT-UNION', 'NEL-METO-UNION'])

    except KeyError:
        pass

    return df


def remove_time_linking(df, replacement="NIL"):
    try:
        df.loc[df["NE-COARSE-LIT"].str.contains("time"), "NEL-LIT"] = replacement
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

    df = pd.read_csv(f_in, sep="\t", quoting=csv.QUOTE_NONE, quotechar="", skip_blank_lines=False)
    df = df.fillna(value={"NE-COARSE-LIT": "", "NEL-LIT": "", "NEL-METO": ""})

    if norm_histo:
        mappings = get_mappings(f_map)
        df = normalize_n_to_n(df, mappings)

    if norm_time:
        df = remove_time_linking(df)

    if unionize:
        df = unionize_meto_lit(df)

    df.to_csv(f_out, index=False, sep="\t", quoting=csv.QUOTE_NONE, quotechar="")


if __name__ == "__main__":
    arguments = docopt(__doc__)
    main(arguments)
