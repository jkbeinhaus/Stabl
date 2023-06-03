import warnings
warnings.filterwarnings('ignore')
# Libraries
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold, LeaveOneGroupOut
from sklearn.base import clone
from sklearn.linear_model import Lasso

from stabl.stabl import Stabl, plot_stabl_path, plot_fdr_graph, save_stabl_results, export_stabl_to_csv
from stabl.preprocessing import LowInfoFilter, remove_low_info_samples

from stabl.multi_omic_pipelines import multi_omic_stabl, multi_omic_stabl_cv, late_fusion_lasso_cv
from stabl.single_omic_pipelines import single_omic_stabl, single_omic_stabl_cv
from stabl.pipelines_utils import compute_features_table
# Data
Val_Celldensities =     pd.read_csv('../DataMedians/Stanford_celldensities_median.csv', index_col=0)
Val_Function =          pd.read_csv('../DataMedians/Stanford_functional_median.csv', index_col=0)
Val_Metavariables =     pd.read_csv('../DataMedians/Stanford_metavariables_median.csv', index_col=0)
Val_Neighborhood =      pd.read_csv('../DataMedians/Stanford_neighborhood_median.csv', index_col=0)

val_data = {
    'Val_Celldensities': Val_Celldensities,
    'Val_Function': Val_Function,
    'Val_Metavariables': Val_Metavariables,
    'Val_Neighborhood': Val_Neighborhood
}

for data_name, data_frame in val_data.items():
    numeric_columns = data_frame.select_dtypes(include=['float64', 'int64']).columns
    val_data[data_name][numeric_columns] = val_data[data_name][numeric_columns].apply(zscore)

Val_y = pd.read_csv('../DataValidation/Val_outcome.csv',index_col=0)
Val_y['site'] = 'Stanford'
#Val_y = Val_y.grade-1
UOP_Celldensities =     pd.read_csv('../DataMedians/UOP_celldensities_median.csv', index_col=0)
UOP_Function =          pd.read_csv('../DataMedians/UOP_functional_median.csv', index_col=0)
UOP_Metavariables =     pd.read_csv('../DataMedians/UOP_metavariables_median.csv', index_col=0)
UOP_Neighborhood =      pd.read_csv('../DataMedians/UOP_neighborhood_median.csv', index_col=0)

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
UOP_y['site'] = 'UOP'
X_Celldensities = pd.concat([Val_Celldensities, UOP_Celldensities])
X_Function = pd.concat([Val_Function, UOP_Function])
X_Metavariables = pd.concat([Val_Metavariables, UOP_Metavariables])
X_Neighborhood = pd.concat([Val_Neighborhood, UOP_Neighborhood])
y = pd.concat([Val_y, UOP_y])
y['patient_id'] = y.index.str.split('_').str.get(0)

data = {
    'Celldensities': X_Celldensities,
    'Function': X_Function,
    'Metavariables': X_Metavariables,
    'Neighborhood': X_Neighborhood
}
y
from scipy.stats import ttest_ind, mannwhitneyu

for key, df in data.items():
    univariate = pd.DataFrame()
    for column in df.select_dtypes(include=['float64', 'int64']):
        x = df[column].dropna()
        outcome = y.loc[x.index, 'grade']
        
        group1 = x[outcome == 1]  # Values in x corresponding to y.grade == 1
        group2 = x[outcome == 2]  # Values in x corresponding to y.grade == 2
        
        statistic, p_value_ttest = ttest_ind(group1, group2)
        statistic, p_value_mannwhitneyu = mannwhitneyu(group1, group2)
        univariate.loc[column, 'Statistic'] = statistic
        univariate.loc[column, 'p-value ttest'] = p_value_ttest
        univariate.loc[column, 'p-value mannwhitney'] = p_value_mannwhitneyu

    filename = f'univariate_results_{key}.csv'
    univariate.to_csv(filename)
# Results folder
result_folder = "./Modelonalldata"
# Main script
groups = y.patient_id
outer_groups = groups.to_numpy().flatten()

outcome = y.grade-1
for omic_name, X_omic in data.items():
    X_omic = remove_low_info_samples(X_omic)
    data[omic_name] = X_omic
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

outer_splitter =  LeaveOneGroupOut()

stability_selection = clone(stabl).set_params(artificial_type=None, hard_threshold=0.5)
# Multi-omic Training-CV
np.random.seed(111)
predictions_dict = multi_omic_stabl_cv(
    data_dict=data,
    y=outcome,
    outer_splitter=outer_splitter,
    stabl=stabl,
    stability_selection=stability_selection,
    task_type="binary",
    save_path=Path(result_folder),
    outer_groups=outer_groups
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
    y_train=train_outcome,
    task_type="binary"
)
features_table.to_csv(Path(result_folder, "Training-Validation", "Table of features.csv"))