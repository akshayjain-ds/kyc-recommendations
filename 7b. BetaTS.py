# Databricks notebook source
# MAGIC %run ../set_up

# COMMAND ----------

import numpy as np
import pandas as pd
from typing import Optional
from tecton import conf
from datetime import date, datetime, timedelta
import tecton
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as f
import pickle
from datetime import datetime
from scipy.spatial.distance import cdist
from scipy.stats import beta
import copy
from tqdm.notebook import tqdm
from numpy.linalg import pinv
from scipy.linalg import pinvh
from numpy.random import multivariate_normal as mvn
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import IncrementalPCA
import pickle 
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics import silhouette_score


# COMMAND ----------

encoded_dataset_train = pd.read_parquet("/dbfs/FileStore/kyc_decisioning/self_learning/encoded_dataset_train.parquet")
encoded_dataset_test = pd.read_parquet("/dbfs/FileStore/kyc_decisioning/self_learning/encoded_dataset_test.parquet")
encoded_dataset_train.shape, encoded_dataset_test.shape

# COMMAND ----------

for col in encoded_dataset_test.columns:
  encoded_dataset_train[col]  = pd.to_numeric(encoded_dataset_train[col], errors='ignore')
  encoded_dataset_test[col]  = pd.to_numeric(encoded_dataset_test[col], errors='ignore')

# COMMAND ----------

varlist = ['age_at_completion_rev',
 'applicant_postcode_High',
#  'applicant_postcode_Low',
 'applicant_postcode_Medium',
#  'applicant_id_type_Driving_Licence',
 'applicant_id_type_Other_ID',
 'applicant_id_type_Passport',
 'applicant_id_type_Provisional_Licence',
 'company_type_High',
#  'company_type_Low',
 'company_type_Prohibited',
 'company_icc_High',
#  'company_icc_Low',
 'company_icc_Medium',
 'company_icc_Prohibited',
 'company_sic_High',
#  'company_sic_Low',
 'company_sic_Medium',
 'company_sic_Prohibited',
 'company_keywords_High',
#  'company_keywords_Low',
 'company_keywords_Medium',
 'company_keywords_Prohibited',
 'applicant_country_High',
 'applicant_country_Medium',
 'applicant_country_Low',
 'applicant_id_type_RP_Nid',
 'fraud_fail',
 'individual_blocklist',
 'individual_identity_address',
 'individual_sanctions_pep',
 'individual_id_scan',
 'business_internal_checks',
 'rc_rule_in_fbr']

arm = 'decision_outcome' # akyc/mkyc
mkyc_arms = ['mkyc_approved', 'mkyc_rejected']
akyc_arms = ['mkyc_approved', 'auto_approved']

reward_1 = 'is_rejected_inv_at_onb'
reward_2 = 'is_rejected_inv_af_app'
reward_3 = 'is_sar90'

len(varlist)

# COMMAND ----------

# pca_varlist = [f"pca_{i}" for i in range(1, pca_count+1)]
# X_train = pd.DataFrame(pca.transform(encoded_dataset_train[varlist]), columns=pca_varlist, index=encoded_dataset_train.index)
# X_test = pd.DataFrame(pca.transform(encoded_dataset_test[varlist]), columns=pca_varlist, index=encoded_dataset_test.index)
# X_train[['decision', arm, reward_1, reward_2, reward_3]] = encoded_dataset_train[['decision', arm, reward_1, reward_2, reward_3]]
# X_test[['decision', arm, reward_1, reward_2, reward_3]] = encoded_dataset_test[['decision', arm, reward_1, reward_2, reward_3]]

pca_varlist = varlist.copy()
X_train = encoded_dataset_train.copy()
X_test = encoded_dataset_test.copy()
X_train.shape, X_test.shape

# COMMAND ----------

with open("/dbfs/FileStore/kyc_decisioning/self_learning/kmeans_objects.pkl", "rb") as f:
  kmeans_objects = pickle.load(f)

with open("/dbfs/FileStore/kyc_decisioning/self_learning/kmeans_distortion.pkl", "rb") as f:
  distortion = pickle.load(f)

with open("/dbfs/FileStore/kyc_decisioning/self_learning/kmeans_silhouette.pkl", "rb") as f:
  silhouette_coefficients = pickle.load(f)

# COMMAND ----------

with open("/dbfs/FileStore/kyc_decisioning/self_learning/kmeans_objects_metrics.pkl", "rb") as f:
  kmeans_objects_metrics = pickle.load(f)

# COMMAND ----------

clusters = 250
np.around(kmeans_objects_metrics.get(clusters).get('distortion'), decimals=2), np.around(kmeans_objects_metrics.get(clusters).get('silhouette'), decimals=2)

# COMMAND ----------

kmeans_object = kmeans_objects_metrics.get(clusters).get('model')
kmeans_object

# COMMAND ----------

X_test = X_test.sample(frac=1)
X_test.shape

# COMMAND ----------

X_train['cluster'] = kmeans_object.predict(X_train[pca_varlist])
X_test['cluster'] = kmeans_object.predict(X_test[pca_varlist])

# COMMAND ----------

reward_1_mkyc_metrics = X_train[X_train[arm].isin(mkyc_arms)].groupby(['cluster'])[[reward_1]].agg(["count", "sum"])[reward_1]
reward_1_mkyc_metrics = reward_1_mkyc_metrics.rename(columns = {"count": f"{reward_1}_count", "sum": f"{reward_1}_success"})
reward_1_mkyc_metrics[f'{reward_1}_failure'] = reward_1_mkyc_metrics[f'{reward_1}_count'] - reward_1_mkyc_metrics[f'{reward_1}_success']
reward_1_mkyc_metrics.drop(columns=[f'{reward_1}_count'], inplace=True)

reward_2_akyc_metrics = X_train[X_train[arm].isin(akyc_arms)].groupby(['cluster'])[[reward_2]].agg(["count", "sum"])[reward_2]
reward_2_akyc_metrics = reward_2_akyc_metrics.rename(columns = {"count": f"{reward_2}_count", "sum": f"{reward_2}_success"})
reward_2_akyc_metrics[f'{reward_2}_failure'] = reward_2_akyc_metrics[f'{reward_2}_count'] - reward_2_akyc_metrics[f'{reward_2}_success']
reward_2_akyc_metrics.drop(columns=[f'{reward_2}_count'], inplace=True)

reward_3_akyc_metrics = X_train[X_train[arm].isin(akyc_arms)].groupby(['cluster'])[[reward_3]].agg(["count", "sum"])[reward_3]
reward_3_akyc_metrics = reward_3_akyc_metrics.rename(columns = {"count": f"{reward_3}_count", "sum": f"{reward_3}_success"})
reward_3_akyc_metrics[f'{reward_3}_failure'] = reward_3_akyc_metrics[f'{reward_3}_count'] - reward_3_akyc_metrics[f'{reward_3}_success']
reward_3_akyc_metrics.drop(columns=[f'{reward_3}_count'], inplace=True)

train_cluster_metrics_beta_TS = pd.concat([reward_1_mkyc_metrics, reward_2_akyc_metrics, reward_3_akyc_metrics], axis=1)
train_cluster_metrics_beta_TS = train_cluster_metrics_beta_TS.to_dict(orient='index')
len(train_cluster_metrics_beta_TS)

# COMMAND ----------


result_dict_betaTS = {}
for company_id, context in tqdm(X_test.iterrows()):
  
  # get nearest cluster
  cluster_num = context['cluster']
  # cluster_num = kmeans_object.predict(context[pca_varlist].values.reshape(-1,1).T)[0]

  # get rows of nearest cluster from historical data
  nearest_cluster_metrics = train_cluster_metrics_beta_TS.get(cluster_num)

  # get #success and #failures for reward_1
  success_1 = np.nan_to_num(nearest_cluster_metrics.get(f'{reward_1}_success'), 0)
  failure_1 = np.nan_to_num(nearest_cluster_metrics.get(f'{reward_1}_failure'), 0)

  # get #success and #failures for reward_2 and reward_3
  success_2 = np.nan_to_num(nearest_cluster_metrics.get(f'{reward_2}_success'), 0)
  failure_2 = np.nan_to_num(nearest_cluster_metrics.get(f'{reward_2}_failure'), 0)
  success_3 = np.nan_to_num(nearest_cluster_metrics.get(f'{reward_3}_success'), 0)
  failure_3 = np.nan_to_num(nearest_cluster_metrics.get(f'{reward_3}_failure'), 0)

  # get beta probabilities (bootsrapping)
  p_1 = beta.rvs(success_1 + 1, failure_1 + 1, size = 10, random_state=int(company_id)).mean()
  p_2 = beta.rvs(success_2 + 1, failure_2 + 1, size = 10, random_state=int(company_id)).mean()
  p_3 = beta.rvs(success_3 + 1, failure_3 + 1, size = 10, random_state=int(company_id)).mean()

  # save company_id, probabilities, actual decisons and rewards
  result_dict_betaTS[company_id] = {}
  result_dict_betaTS[company_id]['cluster'] = cluster_num
  result_dict_betaTS[company_id][f'{reward_1}_score'] = np.around(p_1*1000, decimals=0)
  result_dict_betaTS[company_id][f'{reward_2}_score'] = np.around(p_2*1000, decimals=0)
  result_dict_betaTS[company_id][f'{reward_3}_score'] = np.around(p_3*1000, decimals=0)
  result_dict_betaTS[company_id]['decision'] = context['decision']
  result_dict_betaTS[company_id][arm] = context[arm]
  result_dict_betaTS[company_id][reward_1] = context[reward_1]
  result_dict_betaTS[company_id][reward_2] = context[reward_2]
  result_dict_betaTS[company_id][reward_3] = context[reward_3]


# COMMAND ----------

result_df_betaTS = pd.DataFrame(result_dict_betaTS).T
results = result_df_betaTS.copy()
results.shape

# COMMAND ----------

results.head()

# COMMAND ----------

for col in results.columns:
  results[col]  = pd.to_numeric(results[col], errors='ignore')

# COMMAND ----------

results.describe()

# COMMAND ----------

results_1 = results[results[arm].isin(mkyc_arms)]
results_2 = results[results[arm].isin(akyc_arms)]
results_1.shape, results_2.shape

# COMMAND ----------

results[[f'{reward_1}_score', f'{reward_2}_score', f'{reward_3}_score']].corr('spearman')

# COMMAND ----------

fpr_1, tpr_1, thresholds_1 = roc_curve(results_1[reward_1], results_1[f'{reward_1}_score'])
fpr_2, tpr_2, thresholds_2 = roc_curve(results_2[reward_2], results_2[f'{reward_2}_score'])
fpr_3, tpr_3, thresholds_3 = roc_curve(results_2[reward_3], results_2[f'{reward_3}_score'])
# fpr_3, tpr_3, thresholds_3 = roc_curve(results_2[reward_3], results_2[f'{reward_2}_score'])

# COMMAND ----------

np.around(auc(fpr_1, tpr_1), decimals=2), np.around(auc(fpr_2, tpr_2), decimals=2), np.around(auc(fpr_3, tpr_3), decimals=2)

# COMMAND ----------

result_df_betaTS.to_parquet("/dbfs/FileStore/kyc_decisioning/self_learning/result_df_betaTS.parquet")

# COMMAND ----------

