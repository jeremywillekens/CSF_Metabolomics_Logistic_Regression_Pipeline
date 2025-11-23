import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression

from helpers import save_plot




def extract_and_plot_coefficients(X: pd.DataFrame, y: pd.Series, parameters: dict, title: str, destination_directory: str, savefig: bool = False) -> pd.DataFrame:
    """
    Trains a logistic regression model on the given dataset, extracts feature coefficients,
    and plots them in a bar chart.

    This function trains a logistic regression model on the provided dataset, retrieves
    feature importance through model coefficients, and visualizes them in a horizontal
    bar plot. Optionally, it can save the plot to a specified directory.

    Args:
        - X (pd.DataFrame): Feature matrix containing selected features.
        - y (pd.Series): Target variable (binary classification).
        - parameters (dict): Dictionary containing model hyperparameters:
            - `"penalty"` (str): Regularization type (`"l1"`, `"l2"`, `"elasticnet"`, or `"none"`).
            - `"solver"` (str): Solver algorithm for logistic regression (e.g., `"liblinear"`, `"saga"`).
            - `"C"` (float): Inverse of regularization strength.
        - title (str): Title for the coefficient plot, indicating the step in the pipeline.
        - destination_directory (str): Path to save the plot if `savefig=True`.
        - savefig (bool, optional): Whether to save the coefficient plot as an image. Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame containing:
            - `"Metabolite"`: The feature names.
            - `"Coefficient"`: The corresponding logistic regression coefficients.
            - `"Abs_Coefficient"`: The absolute values of the coefficients (for sorting purposes).

    Notes:
        - A positive coefficient means the metabolite increases the likelihood of the positive class.
        - A negative coefficient suggests the metabolite is associated with the negative class.
    """

    plt.close("all")  # Close all open figures before starting a new plot


    if parameters['penalty'] == 'elasticnet':

        # Train Logistic Regression on one split (not CV-based)
        model = LogisticRegression(penalty=parameters['penalty'], 
                                solver=parameters['solver'], 
                                C=parameters['C'],
                                l1_ratio = parameters['l1_ratio'], 
                                random_state=42)

    else:
        # Train Logistic Regression on one split (not CV-based)
        model = LogisticRegression(penalty=parameters['penalty'], 
                                solver=parameters['solver'], 
                                C=parameters['C'],
                                random_state=42)

    model.fit(X, y)  # Train on full dataset (one-shot fit)

    # Extract coefficients
    coefficients = model.coef_.flatten()
    coeff_df = pd.DataFrame({'Metabolite': X.columns, 'Coefficient': coefficients})
    coeff_df['Abs_Coefficient'] = coeff_df['Coefficient'].abs()
    coeff_df = coeff_df.sort_values(by='Abs_Coefficient', ascending=False)

    # Print the top 20 features for reference
    print(f"\nTop 20 Features by Absolute Coefficient ({title}):")
    print(coeff_df[['Metabolite', 'Coefficient']].head(20))

    
    # Plot coefficients
    plt.figure(figsize=(20, 12))
    plt.barh(coeff_df['Metabolite'], coeff_df['Coefficient'], color = 'blue')

    # Increase tick label sizes
    plt.xticks(fontsize=18)  # x-axis numbers
    plt.yticks(fontsize=22)  # metabolite names

    plt.xlabel('Coefficient Value', fontsize=22, labelpad=15)
    plt.title(f"Logistic Regression Coefficients {title}")
    plt.axvline(0, color='red', linestyle='--', linewidth=2)
    plt.gca().invert_yaxis()  # Highest coefficient at the top
    plt.tight_layout()
    plt.grid(alpha=0.4)

    # Save the figure with the correct filename
    if savefig:
        save_plot("rfecv", destination_directory)

    plt.show()
    plt.close()  # Closes the current figure


    return coeff_df



def plot_avg_roc(cv_logistic_regression_results: dict, parameters: dict, destination_directory: str, 
                 savefig: bool = False, smooth: bool = False, window_size: int = 7) -> None:
    """
    Plots the mean ROC curve with a confidence band from repeated stratified cross-validation.
    Optionally applies a rolling average to smooth the curve and the confidence band.

    Args:
        - cv_logistic_regression_results (dict): Dictionary containing ROC-related metrics, including mean FPR, mean TPR, standard deviation, and AUC scores.
        - parameters (dict): Dictionary with cross-validation parameters, specifying the number of splits (`"n_splits"`) and repeats (`"n_repeats"`).
        - destination_directory (str): Path to the directory where the plot should be saved.
        - savefig (bool, optional): Whether to save the plot as an image. Defaults to False.
        - smooth (bool, optional): Whether to apply rolling average smoothing. Defaults to False.
        - window_size (int, optional): Window size for the rolling average. Defaults to 5.

    Returns:
        None: Displays the ROC curve and saves it if `savefig=True`.
    """

    plt.close("all")  # Close all open figures

    mean_fpr = cv_logistic_regression_results["mean_fpr"]
    mean_tpr = cv_logistic_regression_results["mean_tpr"]
    std_tpr = cv_logistic_regression_results["std_tpr"]

    if smooth:
        # Convert to Pandas Series to apply rolling average smoothing
        smoothed_tpr = pd.Series(mean_tpr).rolling(window=window_size, min_periods=1).mean().values
        smoothed_std_tpr = pd.Series(std_tpr).rolling(window=window_size, min_periods=1).mean().values
    else:
        smoothed_tpr = mean_tpr  # No smoothing applied
        smoothed_std_tpr = std_tpr  # Keep original standard deviation

    # Plot the Mean ROC Curve
    plt.figure(figsize=(12, 12))
    plt.plot(mean_fpr, smoothed_tpr, color="blue", 
             label=f"Mean ROC (AUC = {cv_logistic_regression_results['mean_auc']:.2f} ± {cv_logistic_regression_results['std_auc']:.2f})")
    
    # Increase tick label sizes
    plt.xticks(fontsize=18)  # x-axis numbers
    plt.yticks(fontsize=16)  # y-axis numbers

    # Smooth the shaded confidence band as well
    plt.fill_between(mean_fpr, smoothed_tpr - smoothed_std_tpr, smoothed_tpr + smoothed_std_tpr, color="blue", alpha=0.1)

    plt.plot([0, 1], [0, 1], color="red", linestyle="--")  # Random classifier line
    plt.xlabel("False Positive Rate", fontsize=22, labelpad=15)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=22, labelpad=15)
    plt.title(f"Mean ROC Curve Over {parameters['n_repeats']} Repeated {parameters['n_splits']}-Fold Splits", fontsize=20)
    plt.legend(fontsize = 18, loc = 'best')

    if savefig:
        save_plot("mean_roc_curve", destination_directory)

    plt.show()






def plot_distrib_predicted_probs(cv_logistic_regression_results: dict,
                                 parameters: dict,
                                 destination_directory: str,
                                 savefig: bool = False) -> None:
    """
    Plots the distribution of predicted probabilities from cross-validation results.

    This function visualizes the spread of predicted probabilities for the positive class 
    (e.g., Low WMI) obtained from logistic regression. It includes a kernel density estimate (KDE)
    and marks the decision threshold.

    Args:
        cv_logistic_regression_results (dict): Dictionary containing predicted probabilities (`"all_y_probs"`).
        parameters (dict): Dictionary with cross-validation parameters, including `"threshold"`, which defines the classification cutoff.
        destination_directory (str): Path to the directory where the plot should be saved.
        savefig (bool, optional): Whether to save the plot as an image. Defaults to False.

    Returns:
        None: Displays the probability distribution and saves it if `savefig=True`.
    """
    
    plt.close("all")  # Close all open figures before starting a new plot

    # Plot probability distribution
    plt.figure(figsize=(16, 12))
    sns.histplot(cv_logistic_regression_results["all_y_probs"], bins=20, kde=True, color="blue")
    plt.axvline(0.5, color="red", linestyle="--", label=f"Threshold = {parameters['threshold']}")
    plt.xlabel("Predicted Probability for Low WMI (Class 1)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Predicted Probabilities")
    plt.legend()
    
    # Save the figure with the correct filename
    if savefig:
        save_plot("predicted_probs_distrib", destination_directory)

    plt.show()
    plt.close()



def plot_confusion_matrix(cv_logistic_regression_results: dict, destination_directory: str, savefig: bool = False) -> None:
    """
    Plots the confusion matrix from cross-validation results, displaying both raw values and percentages.

    This function visualizes:
    - The **raw summed confusion matrix** from all cross-validation folds.
    - The **normalized confusion matrix**, where all values sum to 100% for interpretability.

    Args:
        cv_logistic_regression_results (dict): Dictionary containing:
            - `"conf_matrices"`: The raw summed confusion matrix.
            - `"conf_matrix_normalized"`: The normalized confusion matrix (sums to 100%).
        destination_directory (str): Path to the directory where the plot should be saved.
        savefig (bool, optional): Whether to save the plot as an image. Defaults to False.

    Returns:
        None: Displays the confusion matrix and saves it if `savefig=True`.
    """

    plt.close("all")  # Close all open figures before starting a new plot

    # Extract raw and normalized confusion matrices
    raw_conf_matrix = cv_logistic_regression_results["conf_matrices"]
    norm_conf_matrix = cv_logistic_regression_results["conf_matrix_normalized"]

    # Define labels showing both raw counts and percentages
    matrix_labels = np.array([
        [f"True Negative\n{raw_conf_matrix[0, 0]:.0f} ({norm_conf_matrix[0, 0]:.1f}%)", 
         f"False Negative\n{raw_conf_matrix[0, 1]:.0f} ({norm_conf_matrix[0, 1]:.1f}%)"], 
        
        [f"False Positive\n{raw_conf_matrix[1, 0]:.0f} ({norm_conf_matrix[1, 0]:.1f}%)", 
         f"True Positive\n{raw_conf_matrix[1, 1]:.0f} ({norm_conf_matrix[1, 1]:.1f}%)"]
    ])

    # Plot the confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(norm_conf_matrix, annot=matrix_labels, fmt="", cmap="Blues", 
                xticklabels=["Predicted: High WMI (0)", "Predicted: Low WMI (1)"], 
                yticklabels=["Actual: High WMI (0)", "Actual: Low WMI (1)"],
                linewidths=0.5, linecolor="black", cbar=False, 
                annot_kws={"size": 20})  # Increases the font size of annotations
    
    # Increase tick label size
    plt.xticks(fontsize=18)  # adjust as needed
    plt.yticks(fontsize=18)

    # Labels and title
    plt.xlabel("Predicted Label", fontsize=22, labelpad=15)
    plt.ylabel("Actual Label", fontsize=22, labelpad=15)
    plt.title("Confusion Matrix with Raw Counts and Percentages", fontsize=24, pad=20)

    # Save the figure with the correct filename
    if savefig:
        save_plot("confusion_matrix", destination_directory)
    
    plt.show()
    plt.close()