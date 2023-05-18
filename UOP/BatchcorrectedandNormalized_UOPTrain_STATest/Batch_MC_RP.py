import warnings
warnings.filterwarnings('ignore')
# Libraries
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone
from sklearn.linear_model import Lasso

from stabl.stabl import Stabl, plot_stabl_path, plot_fdr_graph, save_stabl_results, export_stabl_to_csv
from stabl.preprocessing import LowInfoFilter, remove_low_info_samples

%config InlineBackend.figure_formats=['retina']
from stabl.multi_omic_pipelines import multi_omic_stabl, multi_omic_stabl_cv, late_fusion_lasso_cv
from stabl.single_omic_pipelines import single_omic_stabl, single_omic_stabl_cv
from stabl.pipelines_utils import compute_features_table
# Data
UOP_Celldensities =     pd.read_csv('../DataBatchcorrected/UOPfinal_celldensities_corr.csv', index_col=0)
UOP_Function =          pd.read_csv('../DataBatchcorrected/UOPfinal_functional_corr.csv', index_col=0)
UOP_Metavariables =     pd.read_csv('../DataBatchcorrected/UOPfinal_metavariables_corr.csv', index_col=0)
UOP_Neighborhood =      pd.read_csv('../DataBatchcorrected/UOPfinal_neighborhood_corr.csv', index_col=0)

UOP_data = {
    'UOP_Celldensities': UOP_Celldensities,
    'UOP_Function': UOP_Function,
    'UOP_Metavariables': UOP_Metavariables,
    'UOP_Neighborhood': UOP_Neighborhood
}

for data_name, data_frame in UOP_data.items():
    numeric_columns = data_frame.select_dtypes(include=['float64', 'int64']).columns
    UOP_data[data_name][numeric_columns] = UOP_data[data_name][numeric_columns].apply(zscore)

UOP_y = pd.read_csv('../DataTraining/UOPfinal_outcome.csv',index_col=0)
UOP_y = UOP_y.grade-1
# Compute mean and standard deviation for each data layer in UOP_data
train_means = {}
train_stds = {}
for data_name, data_frame in UOP_data.items():
    numeric_columns = data_frame.select_dtypes(include=['float64', 'int64']).columns
    train_means[data_name] = data_frame[numeric_columns].mean()
    train_stds[data_name] = data_frame[numeric_columns].std()
Val_Celldensities =     pd.read_csv('../DataBatchcorrected/Val_celldensities_corr.csv', index_col=0)
Val_Function =          pd.read_csv('../DataBatchcorrected/Val_functional_corr.csv', index_col=0)
Val_Metavariables =     pd.read_csv('../DataBatchcorrected/Val_metavariables_corr.csv', index_col=0)
Val_Neighborhood =      pd.read_csv('../DataBatchcorrected/Val_neighborhood_corr.csv', index_col=0)

val_data = {
    'Val_Celldensities': Val_Celldensities,
    'Val_Function': Val_Function,
    'Val_Metavariables': Val_Metavariables,
    'Val_Neighborhood': Val_Neighborhood
}

# Normalize each data layer in val_data using the mean and standard deviation from UOP_data
for data_name, data_frame in val_data.items():
    if data_name in UOP_data:
        numeric_columns = data_frame.select_dtypes(include=['float64', 'int64']).columns
        data_frame[numeric_columns] = (data_frame[numeric_columns] - train_means[data_name]) / train_stds[data_name]
        
Val_y = pd.read_csv('../DataValidation/Val_outcome.csv',index_col=0)
Val_y = Val_y.grade-1
train_data_dict = UOP_data
test_data_dict = val_data
# Results folder
result_folder = "./Batch_MC_RP"
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
    random_state=111
 )

outer_splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=1)

stability_selection = clone(stabl).set_params(artificial_type=None, hard_threshold=0.5)
# Multi-omic Training-CV
np.random.seed(111)
predictions_dict = multi_omic_stabl_cv(
    data_dict=train_data_dict,
    y=UOP_y,
    outer_splitter=outer_splitter,
    stabl=stabl,
    stability_selection=stability_selection,
    task_type="binary",
    save_path=Path(result_folder)
)
# Multiomic Training to derive coefficients
np.random.seed(111)
stabl_multi = Stabl(
    lambda_grid=np.linspace(0.01, 5, 30),
    n_bootstraps=5000,
    artificial_proportion=1.,
    artificial_type="random_permutation",
    hard_threshold=None,
    replace=False,
    fdr_threshold_range=np.arange(0.2, 1, 0.01),
    sample_fraction=.5,
    random_state=111
)

stability_selection = clone(stabl_multi).set_params(artificial_type=None, hard_threshold=.3)
predictions_dict = multi_omic_stabl(
    data_dict=train_data_dict,
    y=UOP_y,
    stabl=stabl_multi,
    stability_selection=stability_selection,
    task_type="binary",
    save_path=Path(result_folder),
    X_test=pd.concat(test_data_dict.values(),axis=1),
    y_test=Val_y
)
# Late fusion lasso
late_fusion_lasso_cv(
    train_data_dict=train_data_dict,
    y=UOP_y,
    outer_splitter=outer_splitter,
    task_type="binary",
    save_path=result_folder,
    groups=None
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
    y_train=UOP_y,
    task_type="binary"
)
features_table.to_csv(Path(result_folder, "Training-Validation", "Table of features.csv"))