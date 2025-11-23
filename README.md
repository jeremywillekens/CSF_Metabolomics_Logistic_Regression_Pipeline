# **CSF Metabolomics Logistic Regression Pipeline**

This repository contains the code used to generate the results for the manuscript:

**“CSF metabolomic signature during therapy for childhood acute lymphoblastic leukemia predicts subsequent working memory impairment.”**

It provides an **interactive workflow** for logistic regression modeling, feature selection, cross-validation, hyperparameter tuning, and threshold optimization applied to CSF metabolomics data.

---

## **Pipeline Overview**

This reproducible workflow includes:

- data transformation (log-transformation and standardization)  
- exploratory data analysis  
- interactive model building  
- repeated stratified cross-validation  
- recursive feature elimination (RFECV)  
- hyperparameter tuning (C and l1_ratio)  
- model comparison  
- threshold optimization  
- coefficient visualization  
- ROC and confusion matrix plotting  

The workflow is **interactive** and uses **VS Code cells (`#%%`)**.  
Users are expected to run each section sequentially and inspect output before proceeding.

---

## **Repository Structure**

```
main_interactive.py            # Step-by-step pipeline (run this file)
helpers.py                     # Utility functions (plot saving, column handling)
transformation.py              # Log10 transform + standardization
exploratory_data_analysis.py   # Density and boxplot visualizations
preparation_logreg.py          # Builds X and y matrices for modeling
logistic_regression.py         # CV, RFECV, hyperparameter tuning, threshold optimization
logreg_plotting.py             # ROC, coefficients, confusion matrices, probability plots

example_data.csv               # Example dataset (metadata + 20 metabolites)
requirements.txt               # Package dependencies
LICENSE                        # MIT License granting reuse permissions
CITATION.cff                   # Formal citation metadata for GitHub & Zenodo
.gitignore                     # Ignore Python cache files
README.md                      # Documentation
```


---

## **How to Run the Pipeline**

### **1. Open `main_interactive.py` in VS Code**

The script is divided into numbered sections using `#%%` cells:

```
#%% 1 — LIBRARY IMPORT
#%% 2 — DATA LOADING AND TRANSFORMATION
#%% 3 — INITIAL LOGISTIC REGRESSION
#%% 4 — FEATURE SELECTION (RFECV)
#%% 5 — BEST C DETERMINATION
#%% 6 — MODEL COMPARISON
#%% 7 — MANUAL MODEL SELECTION
#%% 8 — THRESHOLD OPTIMIZATION
#%% 9 — FINAL MODEL EVALUATION
```

Run each cell sequentially and inspect the outputs.

---

## **Example Data**

The file **`example_data.csv`** contains:

- raw abundance values for the 20 metabolites selected for the predictive signature  
- associated metadata for each CSF sample from pediatric ALL patients  

### **Required Input Format**

```
Sample | Metadata#2 | ... | Metadata#n | Metabolite#1 | ... | Metabolite#n
```

### **Important**

In `main_interactive.py`, update the following variable according to the number of metadata columns in your dataset:

```python
number_labeled_columns = <number_of_metadata_columns>
```

This ensures correct separation between metadata and metabolite values.

---

## **Outputs**

Running the pipeline produces:

- **ROC curves** with confidence intervals  
- **Predicted probability distributions**  
- **Confusion matrices** (raw and normalized)  
- **Coefficient bar plots**  
- **RFECV feature selection plots**  
- **Hyperparameter tuning summaries**  
- **Final model performance metrics**  

If `savefig=True`, plots are saved automatically to the folder defined in:

```python
output = "path/to/output_directory"
```

---

## **Requirements**

Install dependencies using:

```bash
pip install -r requirements.txt
```

The required packages are:

```
joblib==1.5.1
matplotlib==3.10.3
numpy==2.3.0
pandas==2.3.0
seaborn==0.13.2
scikit-learn==1.7.1
```

Python version: **3.10+ recommended**.

---

## **Reproducibility Notes**

- Cross-validation uses **RepeatedStratifiedKFold** with `random_state=42` for reproducibility.  
- The workflow is **human-guided**: decisions between steps (e.g., selecting the final model) must be made manually.  
- The pipeline avoids silent failure — missing metabolites or misformatted input will raise clear errors.  

---

## **Citation**

If you use this code in your research, please cite:

> 10.5281/zenodo.17693235

---

## **Contact**

For questions or clarifications, please contact:

**Jeremy Willekens**  
Rutgers Cancer Institute 
Email: *jeremy.willekens@hotmail.fr*  


---

## **License**

This project is licensed under the MIT License.  
See the `LICENSE` file for details.

