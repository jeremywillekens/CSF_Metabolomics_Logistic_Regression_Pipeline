import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from helpers import split_columns



    
def log_transform_dataframe(dataframe: pd.DataFrame, number_labeled_columns: int) -> pd.DataFrame:
    """
    Apply a log10 transformation to the metabolite values in a dataframe.

    Parameters:
        dataframe (pd.DataFrame): The input dataframe containing metabolite values and group information, with metabolites starting from {number_labeled_columns} onward.
        number_label_columns: number of columns containing categorical values used as labels (eg. group, sex, cognitive status etc)
    Returns:
        pd.DataFrame: A new dataframe with the same structure as the input, but with log10-transformed metabolite values.
    """

    labels, metabolite_columns = split_columns(dataframe, number_labeled_columns)

    # Ensure metabolite_columns is not empty
    if not metabolite_columns:
        raise ValueError("No metabolite columns found for log transformation.")

    # Create a copy of the dataframe to avoid modifying the original
    log_transformed = dataframe.copy()

    # Apply a log10 transformation to the metabolite columns
    log_transformed[metabolite_columns] = np.log10(log_transformed[metabolite_columns])     # Assumes metabolite values are strictly positive

    return log_transformed



def standardize_dataframe(dataframe:pd.DataFrame, number_labeled_columns: int) -> pd.DataFrame:
    """
    Standardize metabolite values in a dataframe by scaling them to have a mean of 0 and a standard deviation of 1.

    Parameters:
        dataframe (pd.DataFrame): The input dataframe containing metabolite values and other metadata (e.g., 'Timepoint').
        number_label_columns: number of columns containing categorical values used as labels (eg. group, sex, cognitive status etc)

    Returns:
        pd.DataFrame: A new dataframe with standardized metabolite values.
    """

    labels, metabolite_columns = split_columns(dataframe, number_labeled_columns)

    # Apply standardization of the metabolite data and retain the group column
    scaler = StandardScaler()
    scaled_transformed = dataframe.copy()   # Create a copy to preserve the original data
    scaled_transformed[metabolite_columns] = scaler.fit_transform(scaled_transformed[metabolite_columns])   

    return scaled_transformed

