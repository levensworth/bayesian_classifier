import pandas as pd
import numpy as np


def load_data(path, separator=','):
    df = pd.read_csv(path, sep= separator)
    return df

def normalize_data(cateogires):
    """
    categories should be a dict containing each category with the percentage to use
    ie:
    {
        "destacadas": 0.2,
        "deportes": 0.3
    }

    Notice, each percentage is based on the total amount for that particular category
    """
    pass

