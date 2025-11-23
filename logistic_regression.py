import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (RepeatedStratifiedKFold, GridSearchCV)
from sklearn.metrics import (roc_curve, auc, confusion_matrix, precision_recall_curve)
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV

from helpers import save_plot

import joblib
n_jobs = min(2, joblib.cpu_count() - 1)



def cv_logistic_regression(X: pd.DataFrame, y: pd.Series, parameters: dict) -> dict:
    """
    Performs repeated stratified cross-validation for logistic regression and computes relevant performance metrics.

    This function applies a logistic regression model using repeated stratified k-fold cross-validation.
    It calculates key classification metrics such as the mean ROC curve, AUC scores, confusion matrices, 
    and predicted probabilities across different validation folds.

    Args:
        X (pd.DataFrame): The feature matrix (independent variables) containing selected metabolites.
        y (pd.Series): The target variable (dependent variable), which should be binary-encoded (0 or 1)
        parameters (dict): Dictionary containing hyperparameters for the cross-validation and logistic regression model:
            - `"n_splits"` (int): Number of folds for stratified k-fold cross-validation
            - `"n_repeats"` (int): Number of times the cross-validation process is repeated
            - `"penalty"` (str): Regularization type for logistic regression (e.g., `"l2"`, `"l1"`, `"none"`)
            - `"solver"` (str): Optimization algorithm for logistic regression (e.g., `"liblinear"`, `"saga"`)
            - `"threshold"` (float): Decision threshold for classification. Defaults to `0.5`, but can be optimized based on precision-recall tradeoff.
            - `"C"` (float): Inverse of regularization strength; must be positive

    Returns:
        dict: A dictionary containing the following cross-validation results:
            - `"mean_fpr"` (np.ndarray): The mean false positive rate (FPR) values for ROC curve interpolation.
            - `"mean_tpr"` (np.ndarray): The mean true positive rate (TPR) across cross-validation folds.
            - `"std_tpr"` (np.ndarray): The standard deviation of the TPR across folds, representing variability.
            - `"mean_auc"` (float): The mean area under the curve (AUC) score across all folds.
            - `"std_auc"` (float): The standard deviation of the AUC scores across folds.
            - `"all_y_probs"` (list): List of predicted probabilities for the positive class across all folds.
            - `"conf_matrices"` (list of np.ndarray): List of confusion matrices for each fold
            - '"conf_matrix_normalized"'(list of np.ndarray): List of normalized confusion matrices for each fold (sum to 100%)

    Notes:
        - The function assumes that `y` is already binary-encoded (i.e., contains only 0s and 1s).
        - The `"random_state"` is set to `42` to ensure reproducibility across runs.
        - The ROC curve is interpolated to allow consistent averaging across folds.
    """


    # Define Cross-Validation Strategy
    cv = RepeatedStratifiedKFold(n_splits=parameters["n_splits"], n_repeats=parameters["n_repeats"], random_state=42)

    # Initialize Model
    if parameters["penalty"] == "elasticnet":
        model = LogisticRegression(penalty=parameters["penalty"], solver=parameters["solver"], C=parameters["C"], l1_ratio=parameters["l1_ratio"], max_iter=5000, random_state=42)
    else:
        model = LogisticRegression(penalty=parameters["penalty"], solver=parameters["solver"], C=parameters["C"], random_state=42)
    
    
    # Create Storage for Metrics
    mean_fpr = np.linspace(0, 1, 100)  # Create a fixed set of FPR values for interpolating ROC curves across folds
    tprs = []  # Store interpolated True Positive Rates (TPRs) for averaging ROC curves
    aucs = []  # AUC scores across folds
    conf_matrices = []  # Store confusion matrices for each fold
    all_y_probs = []  # Store predicted probabilities for the positive class (user-defined)

    # Perform Cross-Validation
    for train_idx, test_idx in cv.split(X, y):
        # Get Training and Test Sets
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Train Model on the Current Fold
        model.fit(X_train, y_train)

        # Predict Probabilities for ROC Curve
        y_prob = model.predict_proba(X_test)[:, 1]

        # Apply custom threshold to make class predictions
        y_pred = (y_prob >= parameters["threshold"]).astype(int)

        # Compute ROC Curve and AUC
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = auc(fpr, tpr)
        
        # Interpolate TPR values for averaging ROC curves across different CV folds
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0  # Ensure ROC starts at (0,0)
        aucs.append(auc_score)

        # Store Confusion Matrix using the threshold-optimized predictions
        conf_matrices.append(confusion_matrix(y_test, y_pred))  # Now correctly uses custom threshold


        # Store predicted probabilities for the positive class (user-defined)
        all_y_probs.extend(y_prob)


    # Sum confusion matrices across folds
    sum_conf_matrix = np.sum(conf_matrices, axis=0)

    # Normalize so the total adds up to 100%
    total = np.sum(sum_conf_matrix)
    conf_matrix_normalized = (sum_conf_matrix / total) * 100  # Now sums to 100%



    # Aggregate results: Compute mean ROC, AUC, and other metrics across CV folds
    cv_logistic_regression_results = {
        "mean_fpr": mean_fpr,
        "mean_tpr": np.mean(tprs, axis=0),
        "std_tpr": np.std(tprs, axis=0),
        "mean_auc": np.mean(aucs),
        "std_auc": np.std(aucs),
        "all_y_probs": all_y_probs,
        "conf_matrices": sum_conf_matrix,  # Return the raw summed confusion matrix
        "conf_matrix_normalized": conf_matrix_normalized}  # New normalized version (sums to 100%)

    print(f"number of metrics: {len(aucs)}")
    return cv_logistic_regression_results





def rec_feature_elim_with_cv(X: pd.DataFrame, y: pd.Series, parameters: dict, destination_directory: str, savefig: bool = False, info: bool = False) -> list:
    """
    Performs Recursive Feature Elimination with Cross-Validation (RFECV) to select the optimal number of features.

    This function applies RFECV using a logistic regression model to iteratively remove less important features
    while optimizing the Area Under the Curve (AUC) score. It then plots the AUC score as a function of 
    the number of selected features and highlights the optimal point.

    Args:
        X (pd.DataFrame): Feature matrix containing independent variables.
        y (pd.Series): Target variable (binary classification).
        parameters (dict): Dictionary containing hyperparameters for the logistic regression model and cross-validation:
            - `"penalty"` (str): Regularization type (`"l1"`, `"l2"`, `"elasticnet"`, or `"none"`).
            - `"solver"` (str): Solver algorithm for logistic regression (e.g., `"liblinear"`, `"saga"`).
            - `"C"` (float): Inverse of regularization strength.
            - `"n_splits"` (int): Number of folds for cross-validation.
        destination_directory (str): Path to save the RFECV plot if `savefig=True`.
        savefig (bool, optional): Whether to save the feature selection plot as an image. Defaults to False.
        info (bool, optional): Whether to print the optimal number of selected features and their names. Defaults to False.

    Returns:
        list: A list of selected feature names chosen by RFECV.

    Notes:
        - The function selects features based on cross-validation performance, using AUC as the evaluation metric.
        - The `info` argument can be used to print the selected features if needed.
        - The `savefig` argument saves the generated plot to `destination_directory` if enabled.
    """

    plt.close("all")  # Close all open figures before starting a new plot

    # Initialize the logistic regression model
    model = LogisticRegression(penalty=parameters['penalty'], solver=parameters['solver'], C=parameters['C'], random_state=42)

    # Set up RFECV to iteratively remove features and optimize AUC score
    rfecv = RFECV(estimator=model,
                  step=1,
                  cv=parameters["n_splits"],
                  scoring='roc_auc',
                  n_jobs=n_jobs)

    # Fit RFECV on the full dataset
    rfecv.fit(X, y)

    # Identify features selected by RFECV based on cross-validation results
    selected_features = X.columns[rfecv.support_]
    num_selected_features = rfecv.n_features_

    # Display optimal number of features and selected features if info=True
    if info:
        print(f"Optimal number of features: {num_selected_features}")
        print("Selected Features:", list(selected_features))


    # Find optimal number of features (xmax) corresponding to highest AUC score (ymax) across CV folds
    ymax = max(rfecv.cv_results_["mean_test_score"])  # Best AUC score
    xmax = list(rfecv.cv_results_["mean_test_score"]).index(ymax) + 1  # Feature count at ymax


    # Plot AUC Score vs. Number of Features
    plt.figure(figsize=(16, 12))
    plt.plot(range(1, len(X.columns) + 1), 
             rfecv.cv_results_["mean_test_score"], marker='o', label="AUC Score")
    plt.xticks(np.arange(1, len(X.columns) + 1, 1.0))

    # Highlight the optimal number of features and its corresponding AUC score
    plt.axvline(xmax, color="red", linestyle="--", alpha=0.7, label=f"Optimal Features = {xmax}")  
    plt.axhline(ymax, color="red", linestyle=":", alpha=0.7, label=f"Optimal AUC = {ymax:.3f}")  
    plt.legend(loc='best')


    # Labels & Title
    plt.xlabel("Number of Features Selected")
    plt.ylabel("Mean AUC Score")
    plt.title("RFECV: AUC vs Number of Features")

    # Save the figure with the correct filename
    if savefig:
        save_plot("rfecv", destination_directory)
    
    plt.show()
    plt.close()

    # Return the list of selected features chosen by RFECV
    return list(selected_features)



def logreg_C_hyperparameter_tuning(X: pd.DataFrame, y: pd.Series, parameters: dict, info: bool = False) -> dict:
    """
    Performs hyperparameter tuning for logistic regression using L1, L2, and ElasticNet regularization.

    This function searches for the optimal regularization strength (`C`) for L1 (Lasso), L2 (Ridge), 
    and ElasticNet logistic regression models using cross-validation. For ElasticNet, it also determines 
    the best `l1_ratio` to balance L1 and L2 penalties.

    Args:
        - X (pd.DataFrame): Feature matrix.
        - y (pd.Series): Target variable (binary classification).
        - parameters (dict): Dictionary containing cross-validation settings:
            - `"n_splits"` (int): Number of CV folds.
        - info (bool, optional): Whether to print the best hyperparameters. Defaults to False.

    Returns:
        dict: A dictionary containing the best hyperparameters:
            - `"best_C_l1"` (float): Best `C` for L1 regularization.
            - `"best_C_l2"` (float): Best `C` for L2 regularization.
            - `"best_C_elasticnet"` (float): Best `C` for ElasticNet regularization.
            - `"best_l1_ratio"` (float): Best `l1_ratio` for ElasticNet.

    Notes:
        - Uses `GridSearchCV` with AUC (`roc_auc`) as the evaluation metric.
        - L1 & L2 models use the `"liblinear"` solver, while ElasticNet uses `"saga"`.
        - The function does NOT determine which regularization method is best; it only retrieves 
          the optimal hyperparameters for each type.
    """

    
    # L1 & L2 Regularization: Define grid for hyperparameter tuning
    param_grid_l1_l2 = {
        'C': np.logspace(-4, 4, 50),  # Search space for C
        'penalty': ['l1', 'l2'],      # Regularization types
        'solver': ['liblinear'],      # 'liblinear' is efficient for small datasets
        'max_iter': [10000]            # Increased iterations inside GridSearch
    }

    # Run GridSearch for L1 and L2
    grid_search_l1_l2 = GridSearchCV(LogisticRegression(random_state=42), 
                                    param_grid_l1_l2, cv= parameters["n_splits"], scoring='roc_auc', n_jobs=n_jobs)
    grid_search_l1_l2.fit(X, y)

    # Extract best `C` separately for L1 and L2
    cv_results = grid_search_l1_l2.cv_results_
    best_C_l1, best_C_l2 = None, None
    best_score_l1, best_score_l2 = -np.inf, -np.inf

    for i, param in enumerate(cv_results["params"]):
        if param["penalty"] == "l1" and cv_results["mean_test_score"][i] > best_score_l1:
            best_C_l1 = param["C"]
            best_score_l1 = cv_results["mean_test_score"][i]
        elif param["penalty"] == "l2" and cv_results["mean_test_score"][i] > best_score_l2:
            best_C_l2 = param["C"]
            best_score_l2 = cv_results["mean_test_score"][i]

    # ElasticNet requires 'saga' solver (not 'liblinear'), so define a separate search grid
    param_grid_elasticnet = {
        'C': np.logspace(-4, 4, 10),   # Search space for C
        'penalty': ['elasticnet'],     
        'solver': ['saga'],            # 'saga' supports ElasticNet
        'l1_ratio': np.arange(0, 1.05, 0.05),  # 21 steps evenly spaced from 0 to 1
        'max_iter': [10000]  # Ensure sufficient iterations for convergence
    }

    # Run GridSearch for ElasticNet
    grid_search_elasticnet = GridSearchCV(LogisticRegression(random_state=42), 
                                        param_grid_elasticnet, cv= parameters["n_splits"], scoring='roc_auc', n_jobs=n_jobs)
    grid_search_elasticnet.fit(X, y)

    # Retrieve best C and l1_ratio for ElasticNet
    best_C_elasticnet = grid_search_elasticnet.best_params_['C']
    best_l1_ratio = grid_search_elasticnet.best_params_['l1_ratio']

    # Store the best hyperparameters in a dictionary
    best_hyperparameters = {
        "best_C_l1": best_C_l1,
        "best_C_l2": best_C_l2,
        "best_C_elasticnet": best_C_elasticnet,
        "best_l1_ratio": best_l1_ratio
    }

    # Print results if needed
    if info:
        print("Best C values:")
        print(f"L1 (Lasso) → Best C: {best_C_l1:.4f}")
        print(f"L2 (Ridge) → Best C: {best_C_l2:.4f}")
        print(f"ElasticNet → Best C: {best_C_elasticnet:.4f}, Best L1 Ratio: {best_l1_ratio:.4f}")
    
    return best_hyperparameters



def compare_models(X: pd.DataFrame,
                   y: pd.Series,
                   best_hyperparameters: dict,
                   parameters: dict,
                   destination_directory: str,
                   models_to_test: list = None,
                   savefig: bool = False,
                   info: bool = False) -> None:
    """
    Compares different logistic regression models using ROC curves and AUC scores.

    This function evaluates L1, L2, ElasticNet, and unregularized logistic regression models
    using cross-validation. It plots the mean ROC curve for each model and reports their
    respective AUC scores.

    Args:
        - X (pd.DataFrame): Feature matrix.
        - y (pd.Series): Target variable (binary classification).
        - best_hyperparameters (dict): Dictionary containing the best C values and l1_ratio.
        - parameters (dict): Dictionary containing cross-validation parameters:
            - "n_splits": Number of folds.
            - "n_repeats": Number of CV repetitions.
        - destination_directory (str): Directory to save the ROC comparison plot.
        - models_to_test (list, optional): Specific models to evaluate. Defaults to None (uses all models).
        - savefig (bool, optional): Whether to save the ROC curve plot. Defaults to False.
        - info (bool, optional): Whether to print AUC scores. Defaults to False.
    
    Returns:
        None. The function generates and optionally saves a ROC curve plot.
    """

    plt.close("all")  # Ensure no lingering plots from previous function calls

    # Retrieve the best hyperparameters
    best_C_l1 = best_hyperparameters["best_C_l1"]
    best_C_l2 = best_hyperparameters["best_C_l2"]
    best_C_elasticnet = best_hyperparameters["best_C_elasticnet"]
    best_l1_ratio = best_hyperparameters["best_l1_ratio"]

    # Define logistic regression models with optimized hyperparameters
    models = {
        "L1 (Lasso)": LogisticRegression(penalty='l1', solver='liblinear', C=best_C_l1, max_iter=5000, random_state=42),
        "L2 (Ridge)": LogisticRegression(penalty='l2', solver='liblinear', C=best_C_l2, max_iter=5000, random_state=42),
        "ElasticNet": LogisticRegression(penalty='elasticnet', solver='saga', C=best_C_elasticnet, l1_ratio=best_l1_ratio, max_iter=5000, random_state=42),
        "No Penalty": LogisticRegression(penalty=None, max_iter=5000, random_state=42)
    }

    # If specific models are selected, keep only those
    if models_to_test:
        models = {k: v for k, v in models.items() if k in models_to_test}

    # Set up cross-validation strategy
    cv = RepeatedStratifiedKFold(n_splits=parameters["n_splits"], n_repeats=parameters["n_repeats"], random_state=42)

    # Dictionary to store AUC scores for each model
    model_results = {}

    plt.figure(figsize=(16, 12))  # Initialize figure for ROC curves

    # Iterate through models and evaluate their performance
    for name, model in models.items():
        mean_fpr = np.linspace(0, 1, 1000)  # Standardized x-axis for ROC curves
        tprs = []  # Store true positive rates for averaging
        aucs = []  # Store AUC scores

        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Train model
            model.fit(X_train, y_train)

            # Predict probabilities for ROC curve
            y_prob = model.predict_proba(X_test)[:, 1]

            # Compute ROC curve and AUC
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc_score = auc(fpr, tpr)

            # Store results
            tprs.append(np.interp(mean_fpr, fpr, tpr))  # Interpolate TPR for consistent plotting
            aucs.append(auc_score)

        # Compute mean and standard deviation of ROC curves
        mean_tpr = np.mean(tprs, axis=0)
        std_tpr = np.std(tprs, axis=0)
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)

        # Store AUC scores
        model_results[name] = (mean_auc, std_auc)  # Store both mean and std AUC

        # Plot the mean ROC curve with confidence interval
        plt.plot(mean_fpr, mean_tpr, label=f"{name} (AUC = {mean_auc:.4f} ± {std_auc:.4f})")
        plt.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, alpha=0.1)

    # Finalize ROC Curve Plot
    plt.plot([0, 1], [0, 1], color="red", linestyle="--")  # Random guess line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title("Comparison of Logistic Regression Models")
    plt.legend()

    # Print AUC scores if requested
    if info:
        print("Final AUC Scores (Mean ± Std from Repeated CV):")
        for model_name, (mean_auc, std_auc) in model_results.items():
            print(f"{model_name}: {mean_auc:.4f} ± {std_auc:.4f}")

    # Save the figure if requested
    if savefig:
        save_plot("comparison_roc", destination_directory)

    plt.show()




def get_model_hyperparameters(old_parameters, C_tuning_results: dict, model_name: str) -> dict:
    """
    Retrieves the optimal C and l1_ratio (if applicable) for the selected model.

    Args:
        old_parameters (dict): Dictionary containing existing hyperparameters.
        C_tuning_results (dict): Dictionary containing the best C values for different penalties.
        model_name (str): The selected logistic regression model type.

    Returns:
        dict: A dictionary containing updated hyperparameters.
    """

    model_params = {
        "n_splits": old_parameters["n_splits"],
        "n_repeats": old_parameters["n_repeats"],
        "penalty": model_name,
        "solver": old_parameters["solver"],
        "C": 1.0,  # Placeholder, will be updated
        "threshold": 0.5,
        "l1_ratio": 0.5  # Placeholder for ElasticNet
    }

    # Assign the correct C value based on the model choice
    if model_name == "l1":
        model_params["C"] = C_tuning_results["best_C_l1"]

    elif model_name == "l2":
        model_params["C"] = C_tuning_results["best_C_l2"]

    elif model_name == "elasticnet":
        model_params["solver"] = 'saga'  # ElasticNet requires 'saga'
        model_params["C"] = C_tuning_results["best_C_elasticnet"]
        model_params["l1_ratio"] = C_tuning_results["best_l1_ratio"]  # Include l1_ratio

    elif model_name is None:  # No penalty (equivalent to model without regularization)
        model_params["penalty"] = None  # No penalty in logistic regression
        model_params["C"] = 1.0  # Default C for unpenalized model
        model_params["solver"] = "lbfgs"  # ✅ Change solver to 'lbfgs'

    else:
        raise ValueError(f"Invalid model selection: {model_name}. Choose from ['l1', 'l2', 'elasticnet', None]")

    return model_params




def threshold_optimization(X: pd.DataFrame,
                           y: pd.Series,
                           parameters: dict,
                           destination_directory: str,
                           savefig: bool = False,
                           info: bool = False) -> dict:
    """
    Optimizes the decision threshold for logistic regression using repeated stratified K-fold cross-validation.

    This function determines the optimal classification threshold by maximizing the F1-score across 
    repeated stratified K-fold validation folds. It evaluates the tradeoff between Precision and Recall, 
    computes Precision-Recall curves, and selects the threshold that yields the highest F1-score.

    Additionally, it visualizes:
    - The **Precision-Recall curve**, showing the averaged Precision and Recall at different thresholds.
    - A **histogram of the best thresholds found** across CV folds.

    The optimal threshold is stored in `parameters["threshold"]`, ensuring seamless integration 
    with the rest of the ML pipeline.

    Args:
        X (pd.DataFrame): The feature matrix containing selected features.
        y (pd.Series): The binary target variable (must be encoded as 0 and 1).
        parameters (dict): Dictionary containing model hyperparameters:
            - `"penalty"` (str): Regularization type (`"l1"`, `"l2"`, `"elasticnet"`, `"none"`).
            - `"solver"` (str): Optimization algorithm for logistic regression (e.g., `"liblinear"`, `"saga"`).
            - `"C"` (float): Inverse of regularization strength.
            - `"l1_ratio"` (float, optional): ElasticNet mixing parameter (only used if `"penalty"` is `"elasticnet"`).
            - `"n_splits"` (int): Number of folds for repeated stratified K-fold cross-validation.
            - `"n_repeats"` (int): Number of times cross-validation is repeated.
            - `"threshold"` (float): Initial classification threshold (default is `0.5` but gets updated).
        destination_directory (str): Path to save the generated plots if `savefig=True`.
        savefig (bool, optional): If `True`, saves the plots in `destination_directory`. Defaults to `False`.
        info (bool, optional): If `True`, prints the optimal threshold and its corresponding F1-score. Defaults to `False`.

    Returns:
        dict: The updated `parameters` dictionary with the optimized `"threshold"` value.

    Notes:
        - The function interpolates Precision-Recall curves to ensure smooth averaging across folds.
        - The final threshold is computed as the **mean of the best thresholds found in each CV fold**.
        - ROC curve calculations are **not** included here since the focus is on Precision-Recall optimization.
        - Ensure that `savefig=True` only if `destination_directory` is valid.

    Example Usage:
        >>> final_params = threshold_optimization(X_selected, y, best_model_parameters, "results/", savefig=True, info=True)
        >>> print(final_params["threshold"])  # Access the optimized threshold
    """

    
    plt.close("all")

    penalty = parameters["penalty"]
    solver = parameters["solver"]
    C = parameters["C"]
    n_splits = parameters["n_splits"]
    n_repeats = parameters["n_repeats"]
    l1_ratio = parameters["l1_ratio"]

    # Define cross-validation strategy
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

    thresholds = []
    precision_curves = []
    recall_curves = []
    mean_thresholds = np.linspace(0, 1, 100)  # Fixed range of thresholds

    if penalty == "elasticnet":
        model = LogisticRegression(penalty="elasticnet", solver=solver, C=C, l1_ratio=l1_ratio, max_iter=5000, random_state=42)
    else:
        model = LogisticRegression(penalty=penalty, solver=solver, C=C, max_iter=5000, random_state=42)

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)  # Train the model

        # Get probability scores for the positive class
        y_prob = model.predict_proba(X_test)[:, 1]

        # Compute precision-recall curve
        precisions, recalls, thresholds_pr = precision_recall_curve(y_test, y_prob)

        # Ensure thresholds_pr has the same length as precisions and recalls
        thresholds_pr = np.append(thresholds_pr, 1)  # Extend to ensure matching length

        # Interpolate precision and recall
        precision_interp = np.interp(mean_thresholds, thresholds_pr[::-1], precisions[::-1])
        recall_interp = np.interp(mean_thresholds, thresholds_pr[::-1], recalls[::-1])


        precision_curves.append(precision_interp)
        recall_curves.append(recall_interp)

        # Compute F1-score for each threshold
        f1_scores = [(2 * (p * r) / (p + r)) if (p + r) > 0 else 0 for p, r in zip(precisions, recalls)]
        best_f1_index = np.argmax(f1_scores)
        best_threshold = thresholds_pr[best_f1_index]
        thresholds.append(best_threshold)

    # Compute mean and std for Precision-Recall curves
    mean_precision = np.mean(precision_curves, axis=0)
    std_precision = np.std(precision_curves, axis=0)
    mean_recall = np.mean(recall_curves, axis=0)
    std_recall = np.std(recall_curves, axis=0)

    # Compute mean best threshold across folds
    optimal_threshold = np.mean(thresholds)

    # Find the closest actual threshold to optimal_threshold
    closest_threshold_idx = (np.abs(thresholds_pr - optimal_threshold)).argmin()
    f1_at_optimal_threshold = f1_scores[closest_threshold_idx]

    ### **Plot Averaged Precision-Recall Curve Across CV Folds**
    plt.figure(figsize=(16, 12))
    plt.plot(mean_thresholds, mean_precision, label="Mean Precision", color="blue")
    plt.plot(mean_thresholds, mean_recall, label="Mean Recall", color="green")
    plt.fill_between(mean_thresholds, mean_precision - std_precision, mean_precision + std_precision, color="blue", alpha=0.1)
    plt.fill_between(mean_thresholds, mean_recall - std_recall, mean_recall + std_recall, color="green", alpha=0.1)
    plt.axvline(optimal_threshold, color="red", linestyle="--", label=f"Optimal Threshold = {optimal_threshold:.2f}")
    plt.xlabel("Decision Threshold")
    plt.ylabel("Score")
    plt.title("Averaged Precision-Recall Curve Across CV Folds")
    plt.legend()

    # Save the figure with the correct filename
    if savefig:
        save_plot("prec_recall", destination_directory)

    plt.show()


    ### **Plot Histogram of Best Thresholds Across CV Folds**
    plt.figure(figsize=(16, 12))
    plt.hist(thresholds, bins=20, color="blue", alpha=0.7, edgecolor="black")
    plt.axvline(optimal_threshold, color="red", linestyle="--", label=f"Optimal Threshold = {optimal_threshold:.2f}")
    plt.xlabel("Best Thresholds Across CV Folds")
    plt.ylabel("Frequency")
    plt.title("Threshold Distribution Across Cross-Validation")
    plt.legend()
    
    # Save the figure with the correct filename
    if savefig:
        save_plot("threshold_distrib", destination_directory)
    
    plt.show()



    if info:
        # Print the optimal threshold
        print(f"Optimal Decision Threshold (Averaged Across Folds): {optimal_threshold:.2f} (F1-score: {f1_at_optimal_threshold:.3f})")

    # Updates the parameters dictionary with the best threshold
    parameters["threshold"] = optimal_threshold
    
    return parameters