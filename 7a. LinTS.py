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
from numpy.random import beta
import copy
from tqdm.notebook import tqdm
from scipy.linalg import pinv, pinvh
from scipy.stats import multivariate_normal as mvn
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import IncrementalPCA
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

 varlist = [
#  'age_at_completion_rev',
 'directors_avg_age_at_completion_rev',
 'years_to_id_expiry',
#  'count_failed_business_rules',
 'applicant_postcode_High',
#  'applicant_postcode_Low',
 'applicant_postcode_Medium',
 'applicant_idcountry_issue_High',
#  'applicant_idcountry_issue_Low',
 'applicant_idcountry_issue_Medium',
 'applicant_nationality_High',
#  'applicant_nationality_Low',
 'applicant_nationality_Medium',
#  'applicant_id_type_Driving_Licence',
 'applicant_id_type_Other_ID',
 'applicant_id_type_Passport',
 'applicant_id_type_Provisional_Licence',
 'applicant_device_type_android',
 'applicant_email_domain_gmail.com',
 'applicant_email_domain_hotmail.com',
 'applicant_email_domain_outlook.com',
 'applicant_email_domain_yahoo.com',
 'applicant_email_domain_hotmail.co.uk',
 'applicant_email_domain_icloud.com',
 'applicant_email_domain_yahoo.co.uk',
 'applicant_email_domain_live.co.uk',
 'company_type_High',
#  'company_type_Low',
 'company_type_Prohibited',
 'company_nob_High',
#  'company_nob_Low',
 'company_nob_Medium',
 'company_nob_Prohibited',
 'company_keywords_High',
#  'company_keywords_Low',
 'company_keywords_Medium',
 'company_keywords_Prohibited',
 'company_postcode_High',
#  'company_postcode_Low',
 'company_postcode_Medium',
 'company_status_Active',
#  'company_status_Undefined',
#  'company_directors_count_1',
#  'company_directors_count_2',
#  'company_directors_count_3+',
#  'company_structurelevelwise_1',
 'company_structurelevelwise_2',
 'company_structurelevelwise_3+',
#  'age_of_company_bucket_< 1M',
 'age_of_company_bucket_< 6M',
 'age_of_company_bucket_< 12M',
 'age_of_company_bucket_>= 12M',
 'applicant_id_type_RP_Nid',
 'fraud_fail',
 'individual_blocklist',
 'individual_identity_address',
 'individual_sanctions_pep',
 'individual_id_scan',
 'business_internal_checks',
#  'rc_rule_in_fbr',
 ]

arm = 'decision_outcome' # akyc/mkyc
mkyc_arms = ['mkyc_approved', 'mkyc_rejected']
akyc_arms = ['mkyc_approved', 'auto_approved']

reward_1 = 'is_rejected_inv_at_onb'
reward_2 = 'is_rejected_inv_af_app'
reward_3 = 'is_sar90'
reward_4 = 'is_app_fraud'

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

# with open("/dbfs/FileStore/kyc_decisioning/self_learning/kmeans_objects.pkl", "rb") as f:
#   kmeans_objects = pickle.load(f)

# with open("/dbfs/FileStore/kyc_decisioning/self_learning/kmeans_distortion.pkl", "rb") as f:
#   distortion = pickle.load(f)

# with open("/dbfs/FileStore/kyc_decisioning/self_learning/kmeans_silhouette.pkl", "rb") as f:
#   silhouette_coefficients = pickle.load(f)

# COMMAND ----------

with open("/dbfs/FileStore/kyc_decisioning/self_learning/kmeans_objects_metrics.pkl", "rb") as f:
  kmeans_objects_metrics = pickle.load(f)

# COMMAND ----------

clusters = 200
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

def is_pos_def(X):
  try:
    _ = np.linalg.cholesky(X)
    return True
  except np.linalg.LinAlgError:
    return False

def nearest_pos_def(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    if is_pos_def(A):
      return A
    else:
      B = (A + A.T) / 2
      _, s, V = np.linalg.svd(B)
      H = np.dot(V.T, np.dot(np.diag(s), V))
      A2 = (B + H) / 2
      A3 = (A2 + A2.T) / 2

      if is_pos_def(A3):
        return A3
      
      else:
        spacing = np.spacing(np.linalg.norm(A))
        I = np.eye(A.shape[0])
        k = 1
        while not is_pos_def(A3):
          mineig = np.min(np.real(np.linalg.eigvals(A3)))
          A3 += I * (-mineig * k**2 + spacing)
          k += 1
        return A3

def sigmoid(x, floor = -709, cap = 709):

  x = np.where(x >= cap, cap, np.where(x <= floor, floor, x))

  return 1.0 / (1.0 + np.exp(-1.0 * x))

def cross_entropy_loss(y_true, y_pred, clip_val=1e-10):
  y_pred = np.clip(y_pred, clip_val, 1-clip_val)
  loss = np.multiply(y_true,np.log(y_pred))
  total_loss = -np.sum(loss,axis=0)
  return total_loss

# Iterated Reweighted Least Squares (IRLS/IWLS)
def get_b_cov_b(train_X, train_y, b_init,
                epoch=100, tol=1e-4):
  N, D = train_X.shape
  betas = b_init
  losses = []
  for e in range(epoch):
      y = sigmoid(np.matmul(train_X, betas))
      W = np.diag(np.ravel(y * (1 - y)))
      grad = np.matmul(train_X.T, (y - train_y))
      H = np.matmul(np.matmul(train_X.T, W), train_X) + 0.001 * np.eye(D)
      inv_H = nearest_pos_def(pinv(H))
      betas -= np.matmul(inv_H, grad)
      loss = cross_entropy_loss(train_y, sigmoid(np.matmul(train_X, betas)))
      losses.append(loss)
      if len(losses) >= 3:
        if all([losses[-2] - losses[-1] < tol, 
                losses[-3] - losses[-2] < tol]):
          break
  return betas, inv_H

# COMMAND ----------

train_cluster_metrics_linTS = {}

for i in tqdm(range(clusters)):
  
  train_cluster_metrics_linTS[i] = {}
  cluster_rows = X_train[X_train['cluster'] == i]

  X_y = cluster_rows[cluster_rows[arm].isin(mkyc_arms)][pca_varlist + [reward_1]]
 
  train_X = np.hstack((np.ones((X_y.shape[0],1)),  X_y[pca_varlist].values))
  train_y = X_y[[reward_1]].values
  b_init = np.zeros((train_X.shape[1],1))
  train_cluster_metrics_linTS[i][f'{reward_1}_success'] = train_y.sum()
  train_cluster_metrics_linTS[i][f'{reward_1}_failure'] = train_y.shape[0] - train_y.sum()
  train_cluster_metrics_linTS[i][f"{reward_1}_beta"], train_cluster_metrics_linTS[i][f"{reward_1}_cov_beta"] = get_b_cov_b(train_X, train_y, b_init)

  X_y = cluster_rows[cluster_rows[arm].isin(akyc_arms)][pca_varlist + [reward_2]]
  train_X = np.hstack((np.ones((X_y.shape[0],1)),  X_y[pca_varlist].values))
  b_init = np.zeros((train_X.shape[1],1))
  train_y = X_y[[reward_2]].values
  train_cluster_metrics_linTS[i][f'{reward_2}_success'] = train_y.sum()
  train_cluster_metrics_linTS[i][f'{reward_2}_failure'] = train_y.shape[0] - train_y.sum()
  train_cluster_metrics_linTS[i][f"{reward_2}_beta"], train_cluster_metrics_linTS[i][f"{reward_2}_cov_beta"] = get_b_cov_b(train_X, train_y, b_init)

  X_y = cluster_rows[cluster_rows[arm].isin(akyc_arms)][pca_varlist + [reward_3]]
  train_X = np.hstack((np.ones((X_y.shape[0],1)),  X_y[pca_varlist].values))
  b_init = np.zeros((train_X.shape[1],1))
  train_y = X_y[[reward_3]].values
  train_cluster_metrics_linTS[i][f'{reward_3}_success'] = train_y.sum()
  train_cluster_metrics_linTS[i][f'{reward_3}_failure'] = train_y.shape[0] - train_y.sum()
  train_cluster_metrics_linTS[i][f"{reward_3}_beta"], train_cluster_metrics_linTS[i][f"{reward_3}_cov_beta"] = get_b_cov_b(train_X, train_y, b_init)

  X_y = cluster_rows[cluster_rows[arm].isin(akyc_arms)][pca_varlist + [reward_4]]
  train_X = np.hstack((np.ones((X_y.shape[0],1)),  X_y[pca_varlist].values))
  b_init = np.zeros((train_X.shape[1],1))
  train_y = X_y[[reward_4]].values
  train_cluster_metrics_linTS[i][f'{reward_4}_success'] = train_y.sum()
  train_cluster_metrics_linTS[i][f'{reward_4}_failure'] = train_y.shape[0] - train_y.sum()
  train_cluster_metrics_linTS[i][f"{reward_4}_beta"], train_cluster_metrics_linTS[i][f"{reward_4}_cov_beta"] = get_b_cov_b(train_X, train_y, b_init)


# COMMAND ----------


result_dict_linTS = {}
for company_id, context in tqdm(X_test.iterrows()):
  
  # get nearest cluster
  cluster_num = context['cluster']
  x = context[pca_varlist].values.reshape(1,-1)
  # cluster_num = kmeans_object.predict(context[pca_varlist].values.reshape(-1,1).T)[0]

  # get rows of nearest cluster from historical data
  nearest_cluster_metrics = train_cluster_metrics_linTS.get(cluster_num)
  
  # get beta and cov_beta for reward_1
  b, cov_b = nearest_cluster_metrics.get(f"{reward_1}_beta").flatten(), nearest_cluster_metrics.get(f"{reward_1}_cov_beta")
  b_sample = mvn(b, cov_b, allow_singular=False).rvs(size=10, random_state=int(company_id)).mean(axis=0)
  log_odds_1 = np.dot(np.hstack((np.ones((x.shape[0],1)),  x)), b_sample)[0]

  # get beta and cov_beta for reward_2
  b, cov_b = nearest_cluster_metrics.get(f"{reward_2}_beta").flatten(), nearest_cluster_metrics.get(f"{reward_2}_cov_beta")
  b_sample = mvn(b, cov_b, allow_singular=False).rvs(size=10, random_state=int(company_id)).mean(axis=0)
  log_odds_2 = np.dot(np.hstack((np.ones((x.shape[0],1)),  x)), b_sample)[0]

  # get beta and cov_beta for reward_3
  b, cov_b = nearest_cluster_metrics.get(f"{reward_3}_beta").flatten(), nearest_cluster_metrics.get(f"{reward_3}_cov_beta")
  b_sample = mvn(b, cov_b, allow_singular=False).rvs(size=10, random_state=int(company_id)).mean(axis=0)
  log_odds_3 = np.dot(np.hstack((np.ones((x.shape[0],1)),  x)), b_sample)[0]

  # get beta and cov_beta for reward_4
  b, cov_b = nearest_cluster_metrics.get(f"{reward_4}_beta").flatten(), nearest_cluster_metrics.get(f"{reward_4}_cov_beta")
  b_sample = mvn(b, cov_b, allow_singular=False).rvs(size=10, random_state=int(company_id)).mean(axis=0)
  log_odds_4 = np.dot(np.hstack((np.ones((x.shape[0],1)),  x)), b_sample)[0]

  # get beta probabilities (bootsrapping)
  p_1 = sigmoid(log_odds_1)
  p_2 = sigmoid(log_odds_2)
  p_3 = sigmoid(log_odds_3)
  p_4 = sigmoid(log_odds_4)

  # save company_id, probabilities, actual decisons and rewards
  result_dict_linTS[company_id] = {}
  result_dict_linTS[company_id]['nearest_cluster'] = cluster_num
  result_dict_linTS[company_id][f'{reward_1}_score'] = np.around(p_1*1000, decimals=0)
  result_dict_linTS[company_id][f'{reward_2}_score'] = np.around(p_2*1000, decimals=0)
  result_dict_linTS[company_id][f'{reward_3}_score'] = np.around(p_3*1000, decimals=0)
  result_dict_linTS[company_id][f'{reward_4}_score'] = np.around(p_4*1000, decimals=0)
  result_dict_linTS[company_id]['decision'] = context['decision']
  result_dict_linTS[company_id][arm] = context[arm]
  result_dict_linTS[company_id][reward_1] = context[reward_1]
  result_dict_linTS[company_id][reward_2] = context[reward_2]
  result_dict_linTS[company_id][reward_3] = context[reward_3]
  result_dict_linTS[company_id][reward_4] = context[reward_4]


# COMMAND ----------

result_df_linTS = pd.DataFrame(result_dict_linTS).T
results = result_df_linTS.copy()
results.shape

# COMMAND ----------

results.head()

# COMMAND ----------

for col in results.columns:
  results[col]  = pd.to_numeric(results[col], errors='ignore')

# COMMAND ----------

results[[f'{reward_1}_score', f'{reward_2}_score', f'{reward_3}_score', f'{reward_4}_score']].corr('spearman')

# COMMAND ----------

results_1 = results[results[arm].isin(mkyc_arms)]
results_2 = results[results[arm].isin(akyc_arms)]
results_1.shape, results_2.shape

# COMMAND ----------

fpr_1, tpr_1, thresholds_1 = roc_curve(results_1[reward_1], results_1[f'{reward_1}_score'])

fpr_2, tpr_2, thresholds_2 = roc_curve(results_2[reward_2], results_2[f'{reward_2}_score'])

fpr_3, tpr_3, thresholds_3 = roc_curve(results_2[reward_3], results_2[f'{reward_3}_score'])
# fpr_3, tpr_3, thresholds_3 = roc_curve(results_2[reward_3], results_2[f'{reward_2}_score'])

fpr_4, tpr_4, thresholds_4 = roc_curve(results_2[reward_4], results_2[f'{reward_4}_score'])

# COMMAND ----------

np.around(auc(fpr_1, tpr_1), decimals=2), np.around(auc(fpr_2, tpr_2), decimals=2), np.around(auc(fpr_3, tpr_3), decimals=2), np.around(auc(fpr_4, tpr_4), decimals=2)

# COMMAND ----------

result_df_linTS.to_parquet("/dbfs/FileStore/kyc_decisioning/self_learning/result_df_linTS.parquet")

# COMMAND ----------

