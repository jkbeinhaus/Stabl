import warnings
warnings.filterwarnings('ignore')
# Libraries
## Basic libraries
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone

from stabl.stabl import Stabl, plot_stabl_path, plot_fdr_graph, save_stabl_results
from stabl.preprocessing import LowInfoFilter, remove_low_info_samples

## Stabl Pipelines
from stabl.multi_omic_pipelines import multi_omic_stabl, multi_omic_stabl_cv
from stabl.single_omic_pipelines import single_omic_stabl, single_omic_stabl_cv
from stabl.pipelines_utils import compute_features_table
# Data
## Training data
# Importing the training data
X_train = pd.read_csv('../Sample Data/COVID-19/Training/Proteomics.csv',index_col="sampleID")
y_train = pd.read_csv("../Sample Data/COVID-19/Training/Mild&ModVsSevere.csv", index_col=0).iloc[:, 0]
## Validation Data
X_val = pd.read_csv("../Sample Data/COVID-19/Validation/Validation_proteomics.csv", index_col=0)
y_val = ~pd.read_csv("../Sample Data/COVID-19/Validation/Validation_outcome(WHO.0 ≥ 5).csv", index_col=0).iloc[:,0]
# Result folder name
result_folder = "./Results COVID-19"
# Single-omic Training-CV
stabl = Stabl(
    lambda_grid=np.linspace(0.01, 5, 10),
    n_bootstraps=1000,
    artificial_type="random_permutation",
    replace=False,
    fdr_threshold_range=np.arange(0.1, 1, 0.01),
    sample_fraction=.5,
    random_state=42
)

stability_selection = clone(stabl).set_params(hard_threshold=.1, artificial_type = None)

outer_splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=42)
predictions_dict = single_omic_stabl_cv(
    X=X_train,
    y=y_train,
    outer_splitter=outer_splitter,
    stabl=stabl,
    stability_selection=stability_selection,
    task_type="binary",
    save_path=result_folder,
    outer_groups=None
)
# Tables of features
selected_features_dict = dict()
for model in ["STABL", "Lasso", "Lasso 1SE", "ElasticNet", "SS 03", "SS 05", "SS 08"]:
    path = Path(result_folder, "Training-Validation", f"{model} coefficients.csv")
    try:
        selected_features_dict[model] = list(pd.read_csv(path, index_col=0).iloc[:, 0].index)
    except:
        selected_features_dict[model] = []
features_table = compute_features_table(
    selected_features_dict,
    X_train=X_train,
    y_train=y_train,
    X_test=X_val,
    y_test=y_val,
    task_type="binary")
features_table.to_csv(Path(result_folder, "Training-Validation", "Table of features.csv"))