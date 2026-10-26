"""Read the sample sheet.

Reads config/samples.tsv (one row per sample) and returns it as a DataFrame
indexed by sample_id. Writes nothing.
"""
import pandas as pd


def read_samples(path="config/samples.tsv"):
    # drop=False keeps sample_id as a column as well as the index.
    return pd.read_csv(path, sep="\t").set_index("sample_id", drop=False)
