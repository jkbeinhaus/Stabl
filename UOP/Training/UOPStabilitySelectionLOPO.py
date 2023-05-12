import warnings
warnings.filterwarnings('ignore')
# Libraries
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone
from sklearn.linear_model import Lasso

from stabl.stabl import Stabl, plot_stabl_path, plot_fdr_graph, save_stabl_results, export_stabl_to_csv
from stabl.preprocessing import LowInfoFilter, remove_low_info_samples

from stabl.multi_omic_pipelines import multi_omic_stabl, multi_omic_stabl_cv, late_fusion_lasso_cv
from stabl.single_omic_pipelines import single_omic_stabl, single_omic_stabl_cv
from stabl.pipelines_utils import compute_features_table
# Data
X_Celldensities = pd.read_csv('./UOPfinal_celldensities.csv',index_col=0)
X_Function = pd.read_csv('./UOPfinal_functional.csv',index_col=0)
X_Metavariables = pd.read_csv('./UOPfinal_metavariables.csv',index_col=0)
X_Neighborhood = pd.read_csv('./UOPfinal_neighborhood.csv',index_col=0)

Val_Celldensities = pd.read_csv('../Validation/Val_celldensities.csv',index_col=0)
Val_Function = pd.read_csv('../Validation/Val_functional.csv',index_col=0)
Val_Metavariables = pd.read_csv('../Validation/Val_metavariables.csv',index_col=0)
Val_Neighborhood = pd.read_csv('../Validation/Val_neighborhood.csv',index_col=0)

y = pd.read_csv('./UOPfinal_outcome.csv',index_col=0)
y = y.grade-1

y_test = pd.read_csv('../Validation/Val_outcome.csv',index_col=0)
y_test = y_test.grade-1
train_data_dict = {
    "Celldensities": X_Celldensities, 
    "Function": X_Function,
    "Neighborhood": X_Neighborhood,
    "Metavariables": X_Metavariables
    }

test_data_dict = {
    "Celldensities": Val_Celldensities, 
    "Function": Val_Function,
    "Neighborhood": Val_Neighborhood,
    "Metavariables": Val_Metavariables
}
# Define Leave-one-patient-out CV
groups = pd.read_csv('./UOPfinal_patient_groups.csv', index_col=0)
outer_groups = groups.to_numpy().flatten()
# Results folder
result_folder = "./Results UOPStabilitySelectionLOPO"
# Main script
for omic_name, X_omic in train_data_dict.items():
    X_omic = remove_low_info_samples(X_omic)
    train_data_dict[omic_name] = X_omic
stabl = Stabl(
    lambda_name='C',
    lambda_grid=np.linspace(0.01, 5, 10),
    n_bootstraps=500,
    artificial_type="random_permutation",
    artificial_proportion=1.,
    replace=False,
    fdr_threshold_range=np.arange(0.2, 1, 0.01),
    sample_fraction=.5,
    random_state=1
 )

outer_splitter = LeaveOneGroupOut()

stability_selection = clone(stabl).set_params(artificial_type=None, hard_threshold=0.3)
# Multi-omic Training-CV
np.random.seed(1)
predictions_dict = multi_omic_stabl_cv(
    data_dict=train_data_dict,
    y=y,
    outer_splitter=outer_splitter,
    stabl=stabl,
    stability_selection=stability_selection,
    task_type="binary",
    save_path=Path(result_folder),
    outer_groups=outer_groups)
# Multiomic Training to derive coefficients
np.random.seed(1)
stabl_multi = Stabl(
    lambda_grid=np.linspace(0.01, 5, 30),
    n_bootstraps=5000,
    artificial_proportion=1.,
    artificial_type="random_permutation",
    hard_threshold=None,
    replace=False,
    fdr_threshold_range=np.arange(0.2, 1, 0.01),
    sample_fraction=.5,
    random_state=1
)

stability_selection = clone(stabl_multi).set_params(artificial_type=None, hard_threshold=.3)
predictions_dict = multi_omic_stabl(
    data_dict=train_data_dict,
    y=y_test,
    stabl=stabl_multi,
    stability_selection=stability_selection,
    task_type="binary",
    save_path=Path(result_folder),
    X_test=pd.concat(test_data_dict.values(),axis=1),
    y_test=y_test
)
# Late fusion lasso
late_fusion_lasso_cv(
    train_data_dict=train_data_dict,
    y=y,
    outer_splitter=outer_splitter,
    task_type="binary",
    save_path=result_folder,
    groups=outer_groups
)
# Features Table
selected_features_dict = dict()
for model in ["STABL", "EF Lasso", "SS 03", "SS 05", "SS 08"]:
    path = Path(result_folder, "Training-Validation", f"{model} coefficients.csv")
    try:
        selected_features_dict[model] = list(pd.read_csv(path, index_col=0).iloc[:, 0].index)
    except:
        selected_features_dict[model] = []
features_table = compute_features_table(
    selected_features_dict,
    X_train=pd.concat(train_data_dict.values(), axis=1),
    y_train=y,
    X_test=pd.concat(test_data_dict.values(), axis=1),
    y_test=y_test,
    task_type="binary"
)
features_table.to_csv(Path(result_folder, "Training-Validation", "Table of features.csv"))