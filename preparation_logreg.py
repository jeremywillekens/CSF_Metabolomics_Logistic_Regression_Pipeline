import pandas as pd


def prep_log_regression_file(
    data: pd.DataFrame, 
    metabolites_of_interest: list, 
    label_to_predict: str, 
    positive_class: str, 
    negative_class: str, 
    save_check: str = None) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepares feature (`X`) and target (`y`) datasets for logistic regression by selecting relevant metabolites 
    and encoding the target variable.

    This function checks if all specified metabolites exist in the dataset, extracts the relevant data, 
    and encodes the target variable as binary (1 = `positive_class`, 0 = `negative_class`).

    Args:
        data (pd.DataFrame): The input dataset containing metabolites and metadata.
        metabolites_of_interest (list): List of metabolite names to include in the dataset.
        label_to_predict (str): Column name representing the target variable.
        positive_class (str): The category in `label_to_predict` that should be encoded as 1 (the outcome to predict).
        negative_class (str): The category in `label_to_predict` that should be encoded as 0.
        save_check (str, optional): Directory path to save `X` and `y` files for verification. Defaults to None.

    Returns:
        tuple[pd.DataFrame, pd.Series]: 
            - `X`: DataFrame containing selected metabolite features.
            - `y`: Series containing the encoded target variable.
    """

    # Checks that all metabolites from {metabolites_of_interest} exist in the {data} dataframe
    missing_metabolites = [m for m in metabolites_of_interest if m not in data.columns]
    if missing_metabolites:
        raise ValueError(
            f"The following metabolites were not found in the dataset: {missing_metabolites}\n"
            f"Check the spelling, capitalization, or whether they exist in this file."
        )

        
    # Isolates the metadata (= target to predict)
    metadata = data[[label_to_predict]]  # Double bracket here to have a df rather than a series

    # Gets the abundance of metabolites of interest for all the samples
    data_of_interest = data[metabolites_of_interest]

    # Merges the metadata and the abundance of the metabolites of interest
    concatenated_data = pd.concat([metadata, data_of_interest], axis=1)

    # Machine learning algorithms work with 1 and 0, rather than strings (1 = what you want to predict)
    # Replaces negative_class with 0 and positive_class with 1
    concatenated_data.loc[concatenated_data[label_to_predict] == negative_class, label_to_predict] = 0
    concatenated_data.loc[concatenated_data[label_to_predict] == positive_class, label_to_predict] = 1
    concatenated_data[label_to_predict] = concatenated_data[label_to_predict].astype(int)  # Ensures pandas treats 0/1 as integers

    # Check class distribution
    print(concatenated_data[label_to_predict].value_counts())  

    X = concatenated_data.drop(columns=[label_to_predict])  # Features (DataFrame)
    y = concatenated_data[label_to_predict]  # Target (Series)
    
    # Save check outputs if required
    if save_check:
        X.to_csv(f"{save_check}/X_save_check_output.csv", index=False)  
        print(f"The file X_save_check_output.csv has been saved in {save_check}")

        y.to_csv(f"{save_check}/y_save_check_output.csv", index=False)  
        print(f"The file y_save_check_output.csv has been saved in {save_check}")
    
    return X, y