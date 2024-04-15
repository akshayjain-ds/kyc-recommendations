# Databricks notebook source
# MAGIC %run ../set_up
# MAGIC

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

# COMMAND ----------

base_dataset_train = pd.read_parquet("/dbfs/FileStore/kyc_decisioning/self_learning/base_dataset_train.parquet")
base_dataset_test = pd.read_parquet("/dbfs/FileStore/kyc_decisioning/self_learning/base_dataset_test.parquet")
base_dataset_train.shape, base_dataset_test.shape

# COMMAND ----------

base_dataset_train.head()

# COMMAND ----------

base_dataset_train.columns

# COMMAND ----------

base_dataset_train[base_dataset_train['age_at_completion'] < 18].shape

# COMMAND ----------

base_dataset_train[base_dataset_train['age_at_completion'] >= 100].shape

# COMMAND ----------

base_dataset_train[base_dataset_train['directors_avg_age_at_completion'] < 18].shape

# COMMAND ----------

base_dataset_train[base_dataset_train['directors_avg_age_at_completion'] >= 100].shape

# COMMAND ----------

base_dataset_train = base_dataset_train[base_dataset_train['age_at_completion'] >= 18]
base_dataset_train = base_dataset_train[base_dataset_train['directors_avg_age_at_completion'] >= 18]
base_dataset_train.shape

# COMMAND ----------

encoded_dataset_train = pd.DataFrame(index = base_dataset_train.index)
encoded_dataset_test = pd.DataFrame(index = base_dataset_test.index)
min_age, max_age = 18, 100
encoded_dataset_train['age_at_completion_rev'] = base_dataset_train['age_at_completion'].apply(lambda x:  1 - (x - min_age)/(max_age-min_age))
encoded_dataset_test['age_at_completion_rev'] = base_dataset_test['age_at_completion'].apply(lambda x: 1 - (x - min_age)/(max_age-min_age))
encoded_dataset_train['age_at_completion_rev'].describe()

# COMMAND ----------

encoded_dataset_train['directors_avg_age_at_completion_rev'] = base_dataset_train['directors_avg_age_at_completion'].apply(lambda x: 1 - (x - min_age)/(max_age-min_age))
encoded_dataset_test['directors_avg_age_at_completion_rev'] = base_dataset_test['directors_avg_age_at_completion'].apply(lambda x: 1 - (x - min_age)/(max_age-min_age))

# COMMAND ----------

min_expiry, max_expiry = 0, 10
encoded_dataset_train['years_to_id_expiry'] = base_dataset_train['years_to_id_expiry'].apply(lambda x:  (x - min_expiry)/(max_expiry-min_expiry))
encoded_dataset_test['years_to_id_expiry'] = base_dataset_test['years_to_id_expiry'].apply(lambda x: (x - min_expiry)/(max_expiry-min_expiry))

# COMMAND ----------

min_fbr, max_fbr = 0, 5
encoded_dataset_train['count_failed_business_rules'] = base_dataset_train['count_failed_business_rules'].apply(lambda x:  (x - min_fbr)/(max_fbr-min_fbr))
encoded_dataset_test['count_failed_business_rules'] = base_dataset_test['count_failed_business_rules'].apply(lambda x: (x - min_fbr)/(max_fbr-min_fbr))

# COMMAND ----------

encoded_dataset_test.head()

# COMMAND ----------

feature_list = ['applicant_postcode_High','applicant_postcode_Low','applicant_postcode_Medium'] + \
['applicant_idcountry_issue_High', 'applicant_idcountry_issue_Low', 'applicant_idcountry_issue_Medium'] + \
['applicant_nationality_High', 'applicant_nationality_Low', 'applicant_nationality_Medium'] + \
['applicant_id_type_Driving_Licence', 'applicant_id_type_National_ID', 'applicant_id_type_Other_ID',
'applicant_id_type_Passport', 'applicant_id_type_Provisional_Licence', 'applicant_id_type_Residence_Permit'] + \
['applicant_device_type_android'] + \
['applicant_email_domain_gmail.com', 'applicant_email_domain_hotmail.com', 'applicant_email_domain_outlook.com',
 'applicant_email_domain_yahoo.com', 'applicant_email_domain_hotmail.co.uk', 'applicant_email_domain_icloud.com',
 'applicant_email_domain_yahoo.co.uk', 'applicant_email_domain_live.co.uk'] + \
['company_type_High','company_type_Low', 'company_type_Prohibited'] + \
['company_nob_High', 'company_nob_Low', 'company_nob_Medium','company_nob_Prohibited'] + \
['company_keywords_High', 'company_keywords_Low', 'company_keywords_Medium','company_keywords_Prohibited'] + \
['company_postcode_High','company_postcode_Low','company_postcode_Medium'] + \
['company_status_Active','company_status_Undefined'] + \
['company_directors_count_1','company_directors_count_2','company_directors_count_3+'] + \
['company_structurelevelwise_1','company_structurelevelwise_2','company_structurelevelwise_3+'] + \
['age_of_company_bucket_< 1M', 'age_of_company_bucket_< 6M', 'age_of_company_bucket_< 12M', 'age_of_company_bucket_>= 12M']
len(feature_list)

# COMMAND ----------

encoded_dataset_train[feature_list] = pd.get_dummies(base_dataset_train[['applicant_postcode', 
                                   'applicant_idcountry_issue',
                                   'applicant_nationality', 
                                   'applicant_id_type',
                                   'applicant_device_type',
                                   'applicant_email_domain',
                                   'company_type', 
                                   'company_nob', 
                                   'company_keywords',
                                   'company_postcode',
                                   'company_status',
                                   'company_directors_count',
                                   'company_structurelevelwise',
                                   'age_of_company_bucket']])[feature_list]

encoded_dataset_test[feature_list] = pd.get_dummies(base_dataset_test[['applicant_postcode', 
                                   'applicant_idcountry_issue',
                                   'applicant_nationality', 
                                   'applicant_id_type',
                                   'applicant_device_type',
                                   'applicant_email_domain',
                                   'company_type', 
                                   'company_nob', 
                                   'company_keywords',
                                   'company_postcode',
                                   'company_status',
                                   'company_directors_count',
                                   'company_structurelevelwise',
                                   'age_of_company_bucket']])[feature_list]
encoded_dataset_train.shape, encoded_dataset_test.shape

# COMMAND ----------

encoded_dataset_train['applicant_id_type_RP_Nid'] = encoded_dataset_train[['applicant_id_type_Residence_Permit', 'applicant_id_type_National_ID']].max(axis=1)
encoded_dataset_train.drop(columns=['applicant_id_type_Residence_Permit', 'applicant_id_type_National_ID'], inplace=True)

encoded_dataset_test['applicant_id_type_RP_Nid'] = encoded_dataset_test[['applicant_id_type_Residence_Permit', 'applicant_id_type_National_ID']].max(axis=1)
encoded_dataset_test.drop(columns=['applicant_id_type_Residence_Permit', 'applicant_id_type_National_ID'], inplace=True)

encoded_dataset_train.shape, encoded_dataset_test.shape


# COMMAND ----------

encoded_dataset_train.head()

# COMMAND ----------

encoded_dataset_train[ ['fraud_fail',
 'individual_blocklist',
 'individual_identity_address',
 'individual_sanctions_pep',
 'individual_id_scan',
 'business_internal_checks', 
 'rc_rule_in_fbr']] = base_dataset_train[['fraud_fail',
                                                    'individual_blocklist',
                                                    'individual_identity_address',
                                                    'individual_sanctions_pep',
                                                    'individual_id_scan',
                                                    'business_internal_checks',
                                                    'rc_rule_in_fbr']]
encoded_dataset_test[['fraud_fail',
 'individual_blocklist',
 'individual_identity_address',
 'individual_sanctions_pep',
 'individual_id_scan',
 'business_internal_checks',
 'rc_rule_in_fbr']] = base_dataset_test[['fraud_fail',
                                                   'individual_blocklist',
                                                    'individual_identity_address',
                                                    'individual_sanctions_pep',
                                                    'individual_id_scan',
                                                    'business_internal_checks',
                                                    'rc_rule_in_fbr']]

encoded_dataset_train.shape, encoded_dataset_test.shape


# COMMAND ----------

encoded_dataset_train.head()

# COMMAND ----------

encoded_dataset_train.describe()

# COMMAND ----------

import seaborn as sb
heat = sb.heatmap(encoded_dataset_train.corr("spearman"))

# COMMAND ----------

encoded_dataset_train[['decision','decision_outcome', 'is_rejected_inv_at_onb','is_rejected_inv_af_app','is_sar90', 'is_app_fraud', 'app_fraud_amount']] = base_dataset_train[['decision','decision_outcome', 'is_rejected_inv_at_onb','is_rejected_inv_af_app','is_sar90', 'is_app_fraud', 'app_fraud_amount']]
encoded_dataset_test[['decision','decision_outcome', 'is_rejected_inv_at_onb','is_rejected_inv_af_app','is_sar90', 'is_app_fraud', 'app_fraud_amount']] = base_dataset_test[['decision','decision_outcome', 'is_rejected_inv_at_onb','is_rejected_inv_af_app','is_sar90', 'is_app_fraud', 'app_fraud_amount']]

encoded_dataset_train.shape, encoded_dataset_test.shape

# COMMAND ----------

encoded_dataset_train.head()

# COMMAND ----------

encoded_dataset_train.columns.tolist()

# COMMAND ----------

encoded_dataset_test['decision'] = encoded_dataset_test['decision'].fillna("mkyc")

# COMMAND ----------

encoded_dataset_train.to_parquet("/dbfs/FileStore/kyc_decisioning/self_learning/encoded_dataset_train.parquet")
encoded_dataset_test.to_parquet("/dbfs/FileStore/kyc_decisioning/self_learning/encoded_dataset_test.parquet")

# COMMAND ----------

