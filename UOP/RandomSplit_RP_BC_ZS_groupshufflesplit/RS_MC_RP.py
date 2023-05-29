import warnings
warnings.filterwarnings('ignore')
# Libraries
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GroupShuffleSplit
from sklearn.base import clone
from sklearn.linear_model import Lasso

from stabl.stabl import Stabl, plot_stabl_path, plot_fdr_graph, save_stabl_results, export_stabl_to_csv
from stabl.preprocessing import LowInfoFilter, remove_low_info_samples

from stabl.multi_omic_pipelines import multi_omic_stabl, multi_omic_stabl_cv, late_fusion_lasso_cv
from stabl.single_omic_pipelines import single_omic_stabl, single_omic_stabl_cv
from stabl.pipelines_utils import compute_features_table
# Data
Celldensities =     pd.read_csv('../DataBatchcorrected/Celldensities.csv', index_col=0)
Function =          pd.read_csv('../DataBatchcorrected/Function.csv', index_col=0)
Metavariables =     pd.read_csv('../DataBatchcorrected/Metavariables.csv', index_col=0)
Neighborhood =      pd.read_csv('../DataBatchcorrected/Neighborhood.csv', index_col=0)

data = {
    'Celldensities': Celldensities,
    'Function': Function,
    'Metavariables': Metavariables,
    'Neighborhood': Neighborhood
}

y = pd.read_csv('../DataBatchcorrected/Outcome.csv',index_col=0)
y['patient_id'] = y.index.str.split('_').str.get(0)
y['site'] = y.index.to_series().apply(lambda x: 'Stanford' if x.startswith('S') else 'UOP')
y
groups = y.patient_id
outer_groups = groups.to_numpy().flatten()
unique_patients = pd.DataFrame(y['patient_id'].unique(), columns=['patient_id'])
unique_patients = unique_patients.merge(y, on='patient_id', how = 'left') 
unique_patients = unique_patients.drop_duplicates(subset=['patient_id', 'grade', 'site'])
unique_patients
# Split the dataframe into training and validation sets
train_df, val_df = train_test_split(unique_patients, test_size=0.25, stratify=unique_patients[['site', 'grade']], random_state=1)
train_df
# Print the shapes of the training and validation sets
print("Training set shape:", train_df.shape)
print("Validation set shape:", val_df.shape)
train_indices = y[y['patient_id'].isin(train_df['patient_id'])].index
train_indices
# Split each dataframe in the data dictionary into train and test
train_data_dict = {}
test_data_dict = {}

for key, df in data.items():
    train_df = df.loc[train_indices]  # Select rows from the dataframe based on train indices
    test_df = df.drop(train_indices)  # Drop rows from the dataframe based on train indices
    
    train_data_dict[key] = train_df
    test_data_dict[key] = test_df

train_outcome = y.loc[train_indices].grade-1
test_outcome = y.drop(train_indices).grade-1
#Zscore training data
for data_name, data_frame in train_data_dict.items():
    numeric_columns = data_frame.select_dtypes(include=['float64', 'int64']).columns
    train_data_dict[data_name][numeric_columns] = train_data_dict[data_name][numeric_columns].apply(zscore)

# Calculate mean and SD from training data
train_means = {}
train_stds = {}
for data_name, data_frame in train_data_dict.items():
    numeric_columns = data_frame.select_dtypes(include=['float64', 'int64']).columns
    train_means[data_name] = data_frame[numeric_columns].mean()
    train_stds[data_name] = data_frame[numeric_columns].std()

# Normalize each data layer in testing data using the mean and standard deviation from training data
for data_name, data_frame in test_data_dict.items():
    if data_name in train_data_dict:
        numeric_columns = data_frame.select_dtypes(include=['float64', 'int64']).columns
        data_frame[numeric_columns] = (data_frame[numeric_columns] - train_means[data_name]) / train_stds[data_name]
# Results folder
result_folder = "./RS_MC_RP"
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

outer_splitter = GroupShuffleSplit(n_splits=5, n_repeats=20, random_state=1)

stability_selection = clone(stabl).set_params(artificial_type=None, hard_threshold=0.5)
# Multi-omic Training-CV
np.random.seed(111)
predictions_dict = multi_omic_stabl_cv(
    data_dict=train_data_dict,
    y=train_outcome,
    outer_splitter=outer_splitter,
    stabl=stabl,
    stability_selection=stability_selection,
    task_type="binary",
    save_path=Path(result_folder),
    outer_groups=outer_groups
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
    y=train_outcome,
    stabl=stabl_multi,
    stability_selection=stability_selection,
    task_type="binary",
    save_path=Path(result_folder),
    X_test=pd.concat(test_data_dict.values(),axis=1),
    y_test=test_outcome
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