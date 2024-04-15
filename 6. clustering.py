# Databricks notebook source
# MAGIC %run ../set_up

# COMMAND ----------

# MAGIC %pip install kneed

# COMMAND ----------

# MAGIC %run ./feature_selection

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
from kneed import KneeLocator
from scipy.spatial.distance import cdist
from numpy.random import beta
import copy
from tqdm.notebook import tqdm
from numpy.linalg import pinv
from scipy.linalg import pinvh
from numpy.random import multivariate_normal as mvn
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import IncrementalPCA
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import pickle

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

len(varlist)

# COMMAND ----------

variance_inflation(encoded_dataset_train[varlist])

# COMMAND ----------

# pca_count = 28
# pca = IncrementalPCA(pca_count)
# pca.fit(encoded_dataset_train[varlist])

# COMMAND ----------

# cum_explained_variance = pca.explained_variance_ratio_.cumsum()
# cum_explained_variance

# COMMAND ----------

# plt.style.use("fivethirtyeight")
# plt.plot(range(1,pca_count+1), cum_explained_variance)
# plt.xticks(range(1,pca_count+1))
# plt.xlabel("Number of Principal Components")
# plt.ylabel("Explained Variance")
# plt.show()

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

X_train.head()

# COMMAND ----------

range_object = range(25, 625, 25)
len(range_object)

# COMMAND ----------

kmeans_objects_metrics = {}
for k in tqdm(range_object):
    kmeans_model = KMeans(init="k-means++",
                          n_init=10,
                          max_iter=300,
                          random_state=42, 
                          n_clusters=k)
    kmeans_model.fit(X_train[pca_varlist])

    kmeans_objects_metrics[k] = {}
    kmeans_objects_metrics[k]['model'] = copy.deepcopy(kmeans_model)

    kmeans_objects_metrics[k]['distortion'] = cdist(X_train[pca_varlist], kmeans_model.cluster_centers_, 'euclidean').min(axis=1).sum() / X_train[pca_varlist].shape[0]
    kmeans_objects_metrics[k]['distortion_test'] = cdist(X_test[pca_varlist], kmeans_model.cluster_centers_, 'euclidean').min(axis=1).sum() / X_test[pca_varlist].shape[0]

    sample = X_train[pca_varlist].sample(frac = 0.2, random_state=42)
    sample['cluster'] = kmeans_model.predict(sample[pca_varlist])
    kmeans_objects_metrics[k]['silhouette'] = silhouette_score(sample[pca_varlist].values, sample['cluster'].values)
    sample = X_test[pca_varlist].sample(frac = 0.2, random_state=42)
    sample['cluster'] = kmeans_model.predict(sample[pca_varlist])
    kmeans_objects_metrics[k]['silhouette_test']  = silhouette_score(sample[pca_varlist].values, sample['cluster'].values)


# COMMAND ----------

k_range = list(kmeans_objects_metrics)
distortion_scores = [kmeans_objects_metrics.get(k).get('distortion') for k in k_range]
silhouette_scores = [kmeans_objects_metrics.get(k).get('silhouette') for k in k_range]
distortion_scores_test = [kmeans_objects_metrics.get(k).get('distortion_test') for k in k_range]
silhouette_scores_test = [kmeans_objects_metrics.get(k).get('silhouette_test') for k in k_range]
print(min(distortion_scores), max(silhouette_scores))
print(min(distortion_scores_test), max(silhouette_scores_test))

# COMMAND ----------

plt.style.use("fivethirtyeight")
plt.plot(k_range, distortion_scores)
plt.plot(k_range, distortion_scores_test)
plt.xticks(k_range)
plt.xlabel("Number of Clusters")
plt.ylabel("Distortion, Silhouette Score")
plt.show()

# COMMAND ----------

plt.style.use("fivethirtyeight")
plt.plot(k_range, silhouette_scores)
plt.plot(k_range, silhouette_scores_test)
plt.xticks(k_range)
plt.xlabel("Number of Clusters")
plt.ylabel("Distortion, Silhouette Score")
plt.show()

# COMMAND ----------

distortion_kl = KneeLocator(
    k_range, distortion_scores, curve="convex", direction="decreasing"
)

silhouette_kl = KneeLocator(
    k_range, silhouette_scores, curve="concave", direction="increasing"
)

distortion_kl.elbow, silhouette_kl.elbow

# COMMAND ----------

distortion_kl = KneeLocator(
    k_range, distortion_scores_test, curve="convex", direction="decreasing"
)

silhouette_kl = KneeLocator(
    k_range, silhouette_scores_test, curve="concave", direction="increasing"
)

distortion_kl.elbow, silhouette_kl.elbow

# COMMAND ----------

with open("/dbfs/FileStore/kyc_decisioning/self_learning/kmeans_objects_metrics.pkl", "wb") as f:
  pickle.dump(kmeans_objects_metrics, f, pickle.HIGHEST_PROTOCOL)

# COMMAND ----------

with open("/dbfs/FileStore/kyc_decisioning/self_learning/kmeans_objects_metrics.pkl", "rb") as f:
  kmeans_objects_metrics = pickle.load(f)

# COMMAND ----------

clusters = 200
print(np.around(kmeans_objects_metrics.get(clusters).get('distortion'), decimals=2), np.around(kmeans_objects_metrics.get(clusters).get('silhouette'), decimals=2))
print(np.around(kmeans_objects_metrics.get(clusters).get('distortion_test'), decimals=2), np.around(kmeans_objects_metrics.get(clusters).get('silhouette_test'), decimals=2))

# COMMAND ----------

kmeans_object = kmeans_objects_metrics[clusters].get('model')
kmeans_object

# COMMAND ----------

X_train['cluster'] = kmeans_object.predict(X_train[pca_varlist])
X_test['cluster'] = kmeans_object.predict(X_test[pca_varlist])

# COMMAND ----------

from itertools import islice

def take(n, iterable):
    """Return the first n items of the iterable as a list."""
    return list(islice(iterable, n))

mkyc_metrics = X_train[X_train[arm].isin(mkyc_arms)].groupby(['cluster'])[['cluster']].agg(["count"])['cluster']
mkyc_metrics = mkyc_metrics.rename(columns = {"count": "mkyc_count"})

approved_metrics = X_train[X_train[arm].isin(akyc_arms)].groupby(['cluster'])[['cluster']].agg(["count"])['cluster']
approved_metrics = approved_metrics.rename(columns = {"count": "approved_count"})

count_metrics = X_train.groupby(['cluster'])[['cluster']].agg(["count"])['cluster']
count_metrics = count_metrics.rename(columns = {"count": "member_count"})

train_cluster_metrics = pd.concat([count_metrics, approved_metrics, mkyc_metrics], axis=1)
train_cluster_metrics['mkyc_rate'] = np.around(train_cluster_metrics['mkyc_count'] / train_cluster_metrics['member_count'], decimals=2)
train_cluster_metrics['cluster_center'] = kmeans_object.cluster_centers_.tolist()
train_cluster_metrics = train_cluster_metrics.to_dict(orient='index')
len(train_cluster_metrics)

cluster_varlist_means = X_train.groupby(['cluster'])[pca_varlist].mean()
varlist_means = X_train[varlist].mean().to_dict()

for c, row in cluster_varlist_means.iterrows():
  var_explains = {var: np.around(row[var]/varlist_means.get(var), decimals=2) for var in pca_varlist}
  # var_explains_sum = sum(var_explains.values())
  # var_explains = {key: value/var_explains_sum for key, value in var_explains.items()}
  var_explains = sorted(var_explains.items(), key=lambda x:x[1], reverse=True)
  var_explains = dict(var_explains)
  train_cluster_metrics[c]['feature_importance'] = take(3, var_explains.items())

# COMMAND ----------

train_cluster_metrics[26]

# COMMAND ----------

min([train_cluster_metrics.get(k).get('member_count') for k in list(train_cluster_metrics)]), max([train_cluster_metrics.get(k).get('member_count') for k in list(train_cluster_metrics)])

# COMMAND ----------

min([train_cluster_metrics.get(k).get('mkyc_count') for k in list(train_cluster_metrics)]), max([train_cluster_metrics.get(k).get('mkyc_count') for k in list(train_cluster_metrics)])

# COMMAND ----------

min([train_cluster_metrics.get(k).get('approved_count') for k in list(train_cluster_metrics)]), max([train_cluster_metrics.get(k).get('approved_count') for k in list(train_cluster_metrics)])

# COMMAND ----------

{k: train_cluster_metrics.get(k).get('mkyc_count') for k in list(train_cluster_metrics)}

# COMMAND ----------

[(k, train_cluster_metrics.get(k).get('approved_count')) for k in list(train_cluster_metrics)]

# COMMAND ----------

