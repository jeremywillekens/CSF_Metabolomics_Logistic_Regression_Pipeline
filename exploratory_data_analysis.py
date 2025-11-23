import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


from helpers import save_plot, split_columns



def melt_data(dataframe: pd.DataFrame, variable_of_interest:str, number_labeled_columns:int) -> pd.DataFrame:
    """
    Transform a wide-format dataframe of metabolite values into a long-format dataframe.

    Parameters:
        dataframe (pd.DataFrame): The input dataframe containing metabolite values, with {variable_of_interest} as one of the columns.
        variable_of_interest: The categorical variable (goup) you are studying
        number_label_columns: number of columns containing categorical values used as labels (eg. group, sex, cognitive status etc)

    Returns:
        pd.DataFrame: A melted dataframe with the following columns:
                      - {variable_of_interest}: The original variable_of_interest group.
                      - 'Metabolite': The name of the metabolite.
                      - 'Value': The corresponding value for each metabolite and variable_of_interest.
    """

    label_columns, metabolite_columns = split_columns(dataframe, number_labeled_columns)

    # Melt the dataframe, retaining the 'Group' column
    melted_data = dataframe.melt(id_vars=[variable_of_interest],             # Columns to retain in the melted format
                                value_vars=metabolite_columns,      # Columns to unpivot (metabolites)
                                var_name='Metabolite',              # Name of the new column for metabolite names
                                value_name='Value'                  # Name of the new column for metabolite values
                                )

    return melted_data



def exploratory_data_analysis(dataframe: pd.DataFrame,
                              variable_of_interest: str,
                              data_type: str,
                              number_labeled_columns: int,
                              savefig: bool = False, 
                              destination_directory: str = None):
    """
    Perform exploratory data analysis (EDA) by visualizing density and box plots for a given dataframe.

    Parameters:
        dataframe (pd.DataFrame): The input dataframe containing metabolite values and group information for analysis.
        variable_of_interest (str): The categorical variable (e.g., "group") for comparison in the plots.
        data_type (str): A label to describe the type of data (e.g., "Raw Values", "Log-transformed Values"), used in plot titles and axis labels.
        number_label_columns: number of columns containing categorical values used as labels (eg. group, sex, cognitive status etc)
        savefig (bool, optional): Whether to save the generated plots as PNG files. Defaults to False.
        destination_directory (str, optional): The directory where the plots will be saved if savefig is True. Must be specified if savefig=True.

    Raises:
        ValueError: If savefig is True but destination_directory is not provided.
    """

    # Ensure destination_directory is specified when savefig is True
    if savefig and not destination_directory:
        raise ValueError("destination_directory must be specified when savefig=True.")
    
    
    # Melt the dataframe into a long format for visualization
    melted_data = melt_data(dataframe, variable_of_interest, number_labeled_columns)
    
    # Ensure {variable_of_interest} is treated as a categorical variable for consistent plotting
    melted_data[variable_of_interest] = melted_data[variable_of_interest].astype('category')

    # Generate a density plot to visualize data distribution for all metabolites
    plt.figure(figsize=(8, 6))
    sns.kdeplot(
        data=melted_data,
        x='Value',
        hue=variable_of_interest,
        common_norm=False,  # Normalize densities separately for each group
        fill=True,          # Fill the area under the curves
        alpha=0.5,          # Set transparency for better visualization
        palette='husl'      # Use the 'husl' color palette for distinct group colors
    )
    plt.xlabel(f"{data_type}")  # Label the x-axis
    plt.ylabel("Density")       # Label the y-axis
    plt.title(f"Density Plot for All Metabolites ({data_type})")  # Add a title

    # Save the density plot if savefig is True
    if savefig:
        save_plot("density", destination_directory)

    plt.show()  # Display the density plot

    # Generate a box plot to visualize data distribution for each group
    plt.figure(figsize=(8, 6))
    sns.boxplot(
        data=melted_data,
        x=variable_of_interest,
        y='Value',
        hue=variable_of_interest,  # Add a legend for each group
        palette='husl'             # Use the same 'husl' palette for consistent coloring
    )
    plt.xlabel(variable_of_interest)  # Label the x-axis
    plt.ylabel(f"{data_type}")        # Label the y-axis
    plt.title(f"Box Plot of Metabolites by Group ({data_type})")  # Add a title

    # Save the box plot if savefig is True
    if savefig:
        save_plot("distribution", destination_directory)

    plt.show()  # Display the box plot