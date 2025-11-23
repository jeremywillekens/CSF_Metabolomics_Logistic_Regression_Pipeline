"""
Interactive analysis pipeline for logistic regression.
Run cells sequentially. Several steps require human decisions
after inspecting plots (see steps 6–8).
"""



#%% 1- LIBRARY IMPORT

import pandas as pd

from transformation import log_transform_dataframe, standardize_dataframe
from preparation_logreg import prep_log_regression_file
from exploratory_data_analysis import exploratory_data_analysis

from logreg_plotting import extract_and_plot_coefficients, plot_avg_roc, plot_distrib_predicted_probs, plot_confusion_matrix
from logistic_regression import cv_logistic_regression, rec_feature_elim_with_cv, logreg_C_hyperparameter_tuning, compare_models, get_model_hyperparameters, threshold_optimization





#%% 2- INITIAL DATA LOADING AND TRANSFORMATION

path = "example_data.csv"         # Path to the dataset (default example)
output = "results"                # Directory where outputs will be saved


number_labeled_columns = 4      # First columns that define metadata

# Load the data
raw_data = pd.read_csv(path)
exploratory_data_analysis(raw_data, "Cognition", "log-transformed standardized", number_labeled_columns, savefig=False, destination_directory = output)

# Data transformation & check
log_transformed_data = log_transform_dataframe(raw_data, number_labeled_columns)
standardized_data = standardize_dataframe(log_transformed_data, number_labeled_columns)
exploratory_data_analysis(standardized_data, "Cognition", "log-transformed standardized", number_labeled_columns, savefig=False, destination_directory = output)



# List of metabolites to include in the model
metabolites_of_interest = [
    "1-stearoyl-2-docosahexaenoyl-GPC (18:0/22:6)",
    "succinylcarnitine (C4-DC)",
    "1-stearoyl-2-arachidonoyl-GPC (18:0/20:4)",
    "1-arachidonoyl-GPC (20:4n6)*",
    "cholesterol",
    "1-palmitoyl-2-stearoyl-GPC (16:0/18:0)",
    "1-palmitoyl-GPC (16:0)",
    "glycerate",
    "1-stearoyl-2-oleoyl-GPC (18:0/18:1)",
    "N-acetylasparagine",
    "N,N-dimethyl-pro-pro",
    "histidine betaine (hercynine)*",
    "sphingomyelin (d18:1/18:1, d18:2/18:0)",
    "1-palmitoyl-2-docosahexaenoyl-GPC (16:0/22:6)",
    "1-stearoyl-2-arachidonoyl-GPE (18:0/20:4)",
    "1-palmitoyl-2-dihomo-linolenoyl-GPC (16:0/20:3n3 or 6)*",
    "1-oleoyl-GPC (18:1)",
    "1-(1-enyl-stearoyl)-2-docosahexaenoyl-GPE (P-18:0/22:6)*",
    "beta-citrylglutamate",
    "cysteinylglycine"]

#################################################
######  /!\ To include all the metabolites in the model (computationnally intense):
#metabolites_of_interest = standardized_data.columns[number_labeled_columns:]   
#################################################


# Column containing the label you want to predict, e.g. "Cognition" (I do it that way for flexibility)
label_to_predict = "Cognition"      # Alternative: "Timepoint"
positive_class = "Low WMI"          # T3
negative_class = "High WMI"         # T1


# Default parameters that will be used for the first logregs. Then, each parameter will be adjusted (and a new parameter dict created to avoid confusion)
parameters = {
    "n_splits":4,
    "n_repeats":5,
    "penalty": "l2",
    "solver": "liblinear",
    "C": 1.0,
    "l1_ratio": 0.5,
    "threshold": 0.5,
    "empirical_p_value": None}



# Prepares feature (`X`) and target (`y`) datasets for logistic regression by selecting relevant metabolites and encoding the target variable.
X, y = prep_log_regression_file(standardized_data, metabolites_of_interest, label_to_predict, positive_class, negative_class, save_check = None)



#%% 3- INITIAL LOGISTIC REGRESSION, EVALUATION AND PLOTTING

# Performs repeated stratified cross-validation for logistic regression and computes relevant performance metrics
# on the initial features
cv_logistic_regression_results = cv_logistic_regression(X, y, parameters)

# Plots the mean ROC curve with a confidence band from repeated stratified cross-validation
plot_avg_roc(cv_logistic_regression_results, parameters, output, savefig=False)

# Plots the distribution of predicted probabilities from cross-validation results
plot_distrib_predicted_probs(cv_logistic_regression_results, parameters, destination_directory=output, savefig=False )

# Plots the confusion matrix from cross-validation results, displaying both raw values and percentages
plot_confusion_matrix(cv_logistic_regression_results, output, savefig=False)

# Trains a logistic regression model on the given dataset, extracts feature coefficients, and plots them in a bar chart
# on the initial features
initial_coeffs = extract_and_plot_coefficients(X, y, parameters, "(Initial)", output, savefig=False)



# %% 4- FEATURE SELECTION

# Performs Recursive Feature Elimination with Cross-Validation (RFECV) to select the optimal number of features
selected_features = rec_feature_elim_with_cv(X, y, parameters, output, savefig=True, info = True)

# X_selected will now replace X as it corresponds to the X of the selected features
X_selected = X[selected_features]  # Only keep best features

# Performs repeated stratified cross-validation for logistic regression and computes relevant performance metrics
# only on the selected features
logreg_rfecv = cv_logistic_regression(X_selected, y, parameters)

# Plots the mean ROC curve with a confidence band from repeated stratified cross-validation (Selected Features)
plot_avg_roc(logreg_rfecv, parameters, output, savefig=False)

# Trains a logistic regression model on the given dataset, extracts feature coefficients, and plots them in a bar chart
# only on the selected features
coeffs_after_feature_selection = extract_and_plot_coefficients(X_selected, y, parameters, "(After Feature Selection)", output, savefig=False)



# %% 5- BEST C HYPERPARAMETER DETERMINATION

# Performs hyperparameter tuning for logistic regression using L1, L2, and ElasticNet regularization
# only on the selected features
best_C_param = logreg_C_hyperparameter_tuning(X_selected, y, parameters, info = True)



# %% 6- COMPARISON OF THE DIFFERENT MODELS WITH THE SELECTED FEATURES AND THE BEST C HYPERPARAMETER

# Compare models and manually choose the best one
models = ["L1 (Lasso)", "L2 (Ridge)", "ElasticNet", "No Penalty"]
compare_models(X_selected, y, best_C_param, parameters, output, models, savefig=False, info=True)



# %% 7- MANUAL MODEL SELECTION AT THIS STEP AND EVALUATION

# HERE YOU HAVE TO CHOOSE THE MODEL MANUALLY 
model_choice = 'elasticnet'     # Use the penalty name that you picked on step 6: "l1", "l2", "elasticnet" or None

# Retrieve hyperparameters for the selected model
best_model_parameters = get_model_hyperparameters(parameters, best_C_param, model_choice)

# Extract and plot coefficients using the selected model’s parameters
coefficients_after_selection = extract_and_plot_coefficients(
    X_selected, y, best_model_parameters, "", output, savefig=False)



#%% 8- THRESHOLD OPTIMIZATION ON THE SELECTED FEATURES, WITH THE BEST C PARAMETER, AND THE BEST MODEL

#Optimizes the decision threshold for logistic regression using repeated stratified K-fold cross-validation
final_parameters = threshold_optimization(X_selected, y, best_model_parameters, output, savefig = False, info = True)



#%% 9- FINAL  LOGISTIC REGRESSION, EVALUATION AND PLOTTING

# Performs repeated stratified cross-validation for logistic regression and computes relevant performance metrics
# only on the selected features, with the best hyperparameters and model
final_log_reg_results = cv_logistic_regression(X_selected, y, final_parameters)

# Plots the mean ROC curve with a confidence band from repeated stratified cross-validation (Selected Features)
plot_avg_roc(final_log_reg_results, final_parameters, output, savefig=True, smooth=False)

# Plots the final confusion matrix from cross-validation results, displaying both raw values and percentages
plot_confusion_matrix(final_log_reg_results, output, savefig=False)

# Trains a logistic regression model on the given dataset, extracts feature coefficients, and plots them in a bar chart
# on the selected features, best hyperparameters and selected mode
initial_coeffs = extract_and_plot_coefficients(X_selected, y, final_parameters, "(Final)", output, savefig=True)
# %%
