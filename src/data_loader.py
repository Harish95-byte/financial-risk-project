import pandas as pd


def load_data(path):

    # Load only 50,000 rows
    df = pd.read_csv(path, nrows=50000)

    return df