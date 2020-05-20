#!/usr/bin/env python
# coding: utf-8
"""
Normalize entity linking by remapping linkgs according to an external file

Usage:
    lib/normalize_linking.py -i=<fpath> -o=<fpath> -m=<fpath>

Options:
    -h --help               Show this screen.
    -i --in=<fpath>         File path to original system response.
    -o --out=<fpath>        File path to normalized system response.
    -m --map=<fpath>        File path to link mapping.

"""

import csv
import pandas as pd
from docopt import docopt


def get_mappings(f_map):
    df_mapping = pd.read_csv(f_map, delimiter="\t")
    df_mapping = df_mapping.melt("Main").drop("variable", axis=1)
    df_mapping["Main"] = df_mapping["Main"].str.extract("(Q[0-9]+)")
    df_mapping["value"] = df_mapping["value"].str.extract("(Q[0-9]+)")
    mapping = dict(df_mapping[["value", "Main"]].values)

    return mapping


def normalize_n_to_n(f_in: str, f_out: str, mapping: dict):
    """
    Read a dataset, remap linking alternatives to a common main identity,
    and save as new dataset.
    """

    df_subm = pd.read_csv(f_in, sep="\t", quoting=csv.QUOTE_NONE, quotechar="")
    df_subm = df_subm.fillna(value={"NEL-LIT": "", "NEL-METO": ""})

    # remap literal NEL column
    df_subm["NEL-LIT"] = df_subm["NEL-LIT"].str.split("|")
    df_subm["NEL-LIT"] = df_subm["NEL-LIT"].apply(
        lambda row: [mapping[k] if mapping.get(k) else k for k in row]
    )
    df_subm["NEL-LIT"] = df_subm["NEL-LIT"].str.join("|")

    # remap metonymic NEL column
    df_subm["NEL-METO"] = df_subm["NEL-METO"].str.split("|")
    df_subm["NEL-METO"] = df_subm["NEL-METO"].apply(
        lambda row: [mapping[k] if mapping.get(k) else k for k in row]
    )
    df_subm["NEL-METO"] = df_subm["NEL-METO"].str.join("|")

    df_subm.to_csv(f_out, index=False, sep="\t", quoting=csv.QUOTE_NONE, quotechar="")


def main(args):

    f_in = args["--in"]
    f_out = args["--out"]
    f_map = args["--map"]

    mappings = get_mappings(f_map)

    normalize_n_to_n(f_in, f_out, mappings)


if __name__ == "__main__":
    arguments = docopt(__doc__)
    main(arguments)
