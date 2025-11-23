import matplotlib.pyplot as plt
import pandas as pd

import re
import time




def split_columns(dataframe:pd.DataFrame, number_label_columns:int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
        Split the columns of a dataframe into labeled columns and columns of interest (e.g., metabolites).

        Parameters:
            dataframe (pd.DataFrame): The input dataframe containing both labeled data (e.g., group, sex, cognitive status) and other numerical data (e.g., metabolite values).
            number_label_columns (int): The number of columns at the start of the dataframe that represent labeled data. These columns will be separated from the rest of the dataframe.

        Returns:
            tuple[list[str], list[str]]: A tuple containing:
                - A list of column names for the labeled data.
                - A list of column names for the numerical data (e.g., metabolites).
        """
    
    # Identify columns of interest based on the specified number of label columns
    labeled_columns = dataframe.columns[:number_label_columns]
    columns_of_interest = dataframe.columns[number_label_columns:]


    return list(labeled_columns), list(columns_of_interest)



def save_plot(graph_type: str, destination_directory: str):
    """
    Save a matplotlib plot to a specified directory with a unique, timestamped filename.

    Parameters:
        graph_type (str): A descriptor for the type of plot being saved (e.g., "density", "boxplot").
        destination_directory (str): The directory where the plot will be saved.

    Raises:
        FileNotFoundError: If the specified directory does not exist.
    """

    def sanitize_filename(name: str) -> str:
        """
        Remove or replace invalid characters from a filename.

        Parameters:
            name (str): The original filename.

        Returns:
            str: A sanitized filename safe for filesystem usage.
        """
        return re.sub(r'[<>:"/\\|?*]', '_', name)  # Replace invalid characters with underscores

    def pngfile_name_generator(graph_type: str) -> str:
        """
        Generate a unique file name for the plot based on the current timestamp.

        Parameters:
            graph_type (str): A descriptor for the type of plot (used in the file name).

        Returns:
            str: A unique file name for the plot, formatted as '{graph_type} - {timestamp}.png'.
        """
        time_string = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())  # Format time as a string
        sanitized_name = sanitize_filename(graph_type)  # Sanitize the graph type
        return f"{sanitized_name} - {time_string}.png"


    file_name = pngfile_name_generator(graph_type)  # Generate a unique name for the output file
    full_path = f"{destination_directory}/{file_name}"  # Combine directory and file name

    try:
        plt.savefig(full_path, bbox_inches='tight')  # Save the plot
        print(f"Plot saved: {full_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")