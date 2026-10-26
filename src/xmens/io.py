import pandas as pd
def read_samples(path="config/samples.tsv"):
    return pd.read_csv(path, sep="\t").set_index("sample_id", drop=False)
