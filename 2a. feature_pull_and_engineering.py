# Databricks notebook source
# MAGIC %run ../set_up

# COMMAND ----------

import pandas as pd
from typing import Optional
from tecton import conf
from datetime import date, datetime
import tecton
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as f
import pickle

SCOPE = "tecton"
SNOWFLAKE_DATABASE = "TIDE"
SNOWFLAKE_USER = dbutils.secrets.get(scope=SCOPE, key="SNOWFLAKE_DATABRICKS_USER")
SNOWFLAKE_PASSWORD = dbutils.secrets.get(scope=SCOPE, key="SNOWFLAKE_DATABRICKS_PASSWORD")
SNOWFLAKE_ACCOUNT = dbutils.secrets.get(scope=SCOPE, key="SNOWFLAKE_DATABRICKS_ACCOUNT")
SNOWFLAKE_ROLE = "DATABRICKS_ROLE"
SNOWFLAKE_WAREHOUSE = "DATABRICKS_WH"
CONNECTION_OPTIONS = dict(sfUrl=SNOWFLAKE_ACCOUNT,#"https://tv61388.eu-west-2.aws.snowflakecomputing.com/",
                           sfUser=SNOWFLAKE_USER,
                          sfPassword=SNOWFLAKE_PASSWORD,
                          sfDatabase=SNOWFLAKE_DATABASE,
                          sfWarehouse=SNOWFLAKE_WAREHOUSE,
                          sfRole=SNOWFLAKE_ROLE)

def get_spark():
    return SparkSession.builder.appName('strawberries').getOrCreate()

def spark_connector(query_string: str)-> DataFrame:
  return spark.read.format("snowflake").options(**CONNECTION_OPTIONS).option("query", query_string).load().cache()

def get_dataset(query_string):
  output = spark_connector(query_string)
  output = output.rename(columns={col: col.lower() for col in list(output.columns)})
  spine =  get_spark().createDataFrame(output)
  return spine

# COMMAND ----------

import sys
import tempfile
import warnings
from xgboost import plot_importance
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
warnings.filterwarnings("ignore")
from tqdm import tqdm, tqdm_notebook
from typing import Tuple
from sklearn.calibration import CalibratedClassifierCV
tqdm.pandas()
from sklearn import metrics
import json
from sklearn.model_selection import (
  KFold,
  StratifiedKFold,
  TimeSeriesSplit,
)
from sklearn.metrics import recall_score
from sklearn.metrics import (
  roc_curve,
  auc,
  precision_recall_curve,
  PrecisionRecallDisplay
)
import shap
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce
import copy


# COMMAND ----------

workspace_name = 'akshayjain'
raw_service_name = 'membership_completed_v_data_source'
feature_service_name = 'uk_ifre_feature_service'
my_experiment_workspace = tecton.get_workspace(workspace_name)

# COMMAND ----------

def diff_month(d1: str, d2: str):

  d1 = pd.to_datetime(d1)
  d2 = pd.to_datetime(d2)

  return (d1.year - d2.year) * 12 + d1.month - d2.month
    
start_date, end_date = '2022-01-01', '2023-03-31'
diff_month(end_date, start_date)

# COMMAND ----------

ds = my_experiment_workspace.get_data_source('membership_completed_v_data_source')
member_data = ds.get_dataframe(start_time=pd.to_datetime(start_date), end_time = pd.to_datetime(end_date)).to_spark()

# COMMAND ----------

df = member_data.persist()

# COMMAND ----------

df = df.toPandas()
df.shape

# COMMAND ----------

df['company_id'].duplicated().sum()

# COMMAND ----------

df.set_index('company_id', inplace=True)

# COMMAND ----------

df.head()

# COMMAND ----------

pd.isnull(df).sum()/df.shape[0]

# COMMAND ----------

df['age_of_company'] = (pd.to_datetime(df['company_created_on']).apply(lambda x : x.date()) - pd.to_datetime(df['company_incorporation_date_rawdata']).apply(lambda x : x.date()))/np.timedelta64(1, 'M')

# COMMAND ----------

df['age_of_company_bucket'] = df['age_of_company'].apply(lambda x: "Undefined" if pd.isnull(x) else ("< 1M" if x <= 1 else ("< 6M" if x < 6 else ("< 12M" if x < 12 else ">= 12M"))))
df['age_of_company_bucket'].value_counts()

# COMMAND ----------

df['applicant_device_type'] = df['applicant_device_type_rawdata'].apply(lambda x: str(x).lower())
df['applicant_device_type'].value_counts()

# COMMAND ----------

df['applicant_email_domain'] = df['applicant_email_rawdata'].apply(lambda x: x.split("@")[-1]).apply(lambda x: x.split("#")[-1])
df['applicant_email_domain'].value_counts()/df.shape[0]

# COMMAND ----------

# rule_applicant_singlename
rule_applicant_singlename = df[['applicant_id_firstname_rawdata', 'applicant_id_lastname_rawdata']].apply(lambda row: all([pd.isnull(row['applicant_id_firstname_rawdata']), 
                                                                                                            pd.isnull(row['applicant_id_lastname_rawdata'])])
                                                                                                       or
                                                                                                       all([bool(row['applicant_id_firstname_rawdata']), 
                                                                                                            bool(row['applicant_id_lastname_rawdata'])])
                                                                                                            , axis=1)
rule_applicant_singlename = ~rule_applicant_singlename                                                                                                           
rule_applicant_singlename.mean()                                                                                                   

# COMMAND ----------

# rule_industry_animal_breeder
rule_industry_animal_breeder = df['company_icc_rawdata'].apply(lambda x: str(x).lower().__contains__("animal_breeder"))
rule_industry_animal_breeder.mean()

# COMMAND ----------

# rule_idcountry_russia
rule_idcountry_russia = df['applicant_idcountry_issue_rawdata'].apply(lambda x: any([str(x).upper().__contains__("RU"), str(x).upper().__contains__("RUS")]))
rule_idcountry_russia.mean()

# COMMAND ----------

# rule_idcountry_ukraine
rule_idcountry_ukraine = df['applicant_idcountry_issue_rawdata'].apply(lambda x: any([str(x).upper().__contains__("UA"), str(x).upper().__contains__("UKR")]))
rule_idcountry_ukraine.mean()

# COMMAND ----------

# rule_idcountry_belarus
rule_idcountry_belarus = df['applicant_idcountry_issue_rawdata'].apply(lambda x: any([str(x).upper().__contains__("BY"), str(x).upper().__contains__("BLR")]))
rule_idcountry_belarus.mean()

# COMMAND ----------

# rule_idcountry_romania
feature_list = ['applicant_idcountry_issue_rawdata', 'applicant_id_type_rawdata', 'company_type_rawdata', 
                'applicant_postcode_rawdata', 'company_icc_rawdata', 'company_sic_rawdata']

rule_idcountry_romania = df[feature_list].apply(lambda row: all([any([str(row['applicant_idcountry_issue_rawdata']).upper().__contains__("RO"), 
                                                                str(row['applicant_idcountry_issue_rawdata']).upper().__contains__("ROU")])
                                                                 ,
                                                                  str(row['applicant_id_type_rawdata']).__contains__("PASSPORT"),
                                                                  any([str(row['company_type_rawdata']).__contains__("LTD"), 
                                                                       str(row['company_type_rawdata']).__contains__("null"), 
                                                                       str(row['company_type_rawdata']).__contains__("None")])
                                                                ])
                                                                and
                                                                any([
                                                                any([str(row['applicant_postcode_rawdata']).startswith("E"),
                                                                  str(row['applicant_postcode_rawdata']).startswith("B"),
                                                                  str(row['applicant_postcode_rawdata']).startswith("IP"),
                                                                  str(row['applicant_postcode_rawdata']).startswith("IG"),
                                                                  str(row['applicant_postcode_rawdata']).startswith("ST"),
                                                                  str(row['applicant_postcode_rawdata']).startswith("CV"),
                                                                  str(row['applicant_postcode_rawdata']).startswith("PR1"),
                                                                  str(row['applicant_postcode_rawdata']).startswith("PR2")])
                                                                ,
                                                                any([str(row["company_icc_rawdata"]).strip().__contains__("category.construction"),
                                                                 str(row["company_icc_rawdata"]).strip().__contains__("category.cleaner"),
                                                                 str(row["company_icc_rawdata"]).strip().__contains__("category.transport_and_storage"),
                                                                 str(row["company_icc_rawdata"]).strip().__contains__("category.decorator_\u0026_painter"),
                                                                 str(row["company_icc_rawdata"]).strip().__contains__("category.painter_\u0026_decorator"),
                                                                 str(row["company_icc_rawdata"]).strip().__contains__("category.furniture_removal"),
                                                                 str(row["company_icc_rawdata"]).strip().__contains__("category.domestic_cleaner"),
                                                                 str(row["company_icc_rawdata"]).strip().__contains__("category.household_cleaning_services"),
                                                                 str(row["company_icc_rawdata"]).strip().__contains__("category.industrial_cleaning_services"),
                                                                 str(row["company_icc_rawdata"]).strip().__contains__("category.hygiene_and_cleansing_services"),
                                                                 str(row["company_icc_rawdata"]).strip().__contains__("category.carpenter_/_carpentry"),
                                                                 str(row["company_icc_rawdata"]).strip().__contains__("category.hygiene_and_cleansing_services"),
                                                                 str(row["company_icc_rawdata"]).strip().__contains__("category.plumbing_/_plumber"),
                                                                 str(row["company_icc_rawdata"]).strip().__contains__("category.builder")])
                                                                ,
                                                                any([str(row['company_sic_rawdata']).__contains__("59111"),
                                                                    str(row['company_sic_rawdata']).__contains__("96090"),
                                                                    str(row['company_sic_rawdata']).__contains__("59112"),
                                                                    str(row['company_sic_rawdata']).__contains__("59200"),
                                                                    str(row['company_sic_rawdata']).__contains__("74209"),
                                                                    str(row['company_sic_rawdata']).__contains__("74202"),
                                                                    str(row['company_sic_rawdata']).__contains__("74201")
                                                                    ])
                                                            ])
                                                              , axis=1)
rule_idcountry_romania.mean()

# COMMAND ----------

# rule_idcountry_portugal
feature_list = ['applicant_idcountry_issue_rawdata', 'applicant_id_type_rawdata', 'company_is_registered_rawdata', 
                'applicant_postcode_rawdata', 'applicant_email_rawdata', 'applicant_device_type_rawdata']

rule_idcountry_portugal = df[feature_list].apply(lambda row: all([
                                                                any([str(row['applicant_idcountry_issue_rawdata']).upper().__contains__("PT"), 
                                                                     str(row['applicant_idcountry_issue_rawdata']).upper().__contains__("PRT")])
                                                                ,
                                                                any([str(row['applicant_id_type_rawdata']).__contains__("PASSPORT"),
                                                                    str(row['applicant_id_type_rawdata']).__contains__("ID_CARD")])
                                                                ,
                                                                row['company_is_registered_rawdata']
                                                                ,
                                                                any([str(row['applicant_postcode_rawdata']).startswith("M3"),
                                                                    str(row['applicant_postcode_rawdata']).startswith("M6"),
                                                                    str(row['applicant_postcode_rawdata']).startswith("M7"),
                                                                    str(row['applicant_postcode_rawdata']).startswith("M8"),
                                                                    str(row['applicant_postcode_rawdata']).startswith("G20"),
                                                                    str(row['applicant_postcode_rawdata']).startswith("G21"),
                                                                    str(row['applicant_postcode_rawdata']).startswith("G22"),
                                                                    str(row['applicant_postcode_rawdata']).startswith("G31"),
                                                                    str(row['applicant_postcode_rawdata']).startswith("G51"),
                                                                    str(row['applicant_postcode_rawdata']).startswith("G51"),
                                                                    str(row['applicant_postcode_rawdata']).startswith("ML1"),
                                                                    str(row['applicant_postcode_rawdata']).startswith("PA1")])
                                                                ,
                                                                str(row['applicant_email_rawdata']).lower().__contains__("@gmail.com")
                                                                ,
                                                                str(row['applicant_device_type_rawdata']).lower().__contains__("ios")
                                                                ])
                                                              ,axis=1)
rule_idcountry_portugal.mean()

# COMMAND ----------

# rule_company_bank

feature_list = ['company_trading_name_rawdata', 'company_sic_rawdata', 'company_icc_rawdata']
rule_company_bank = df[feature_list].apply(lambda row: any([str(row['company_trading_name_rawdata']).lower().__contains__("bank"), 
                                                                                                 str(row['company_trading_name_rawdata']).lower().__contains__("banking"),
                                                                                                 str(row['company_trading_name_rawdata']).lower().__contains__("banker"),
                                                                                                 str(row['company_icc_rawdata']).lower().__contains__("category.bank_(the_business_you_own_is_a_bank)"),
                                                                                                 str(row['company_sic_rawdata']).lower().__contains__("64191"), 
                                                                                                 str(row['company_sic_rawdata']).lower().__contains__("64110")]),
                                                                                axis=1)
rule_company_bank.mean()

# COMMAND ----------

# rule_company_bank
rule_company_bank = df['company_trading_name_rawdata'].apply(lambda x: any([str(x).lower().__contains__("bank"), 
                                                                                  str(x).lower().__contains__("banking"),
                                                                                  str(x).lower().__contains__("banker"),
                                                                                  str(x).lower().__contains__("category.bank_(the_business_you_own_is_a_bank)"),
                                                                                  str(x).lower().__contains__("64191"), 
                                                                                  str(x).lower().__contains__("64110")]))
rule_company_bank.mean()

# COMMAND ----------

rules_dataset = df[['company_created_on']]
rules_dataset['rule_applicant_singlename'] = rule_applicant_singlename*1
rules_dataset['rule_industry_animal_breeder'] = rule_industry_animal_breeder*1
rules_dataset['rule_idcountry_russia'] = rule_idcountry_russia*1
rules_dataset['rule_idcountry_ukraine'] = rule_idcountry_ukraine*1
rules_dataset['rule_idcountry_belarus'] = rule_idcountry_belarus*1
rules_dataset['rule_idcountry_romania'] = rule_idcountry_romania*1
rules_dataset['rule_idcountry_portugal'] = rule_idcountry_portugal*1
rules_dataset['rule_company_bank'] = rule_company_bank*1
rules_dataset['rc_rule_in_fbr'] = rules_dataset.max(axis=1)
rules_dataset['count_rc_rule_in_fbr'] = rules_dataset.sum(axis=1)
rules_dataset.drop(columns = ['company_created_on'], inplace=True)
rules_dataset.shape

# COMMAND ----------

rules_dataset.head()

# COMMAND ----------

rules_dataset.rc_rule_in_fbr.mean(), rules_dataset.count_rc_rule_in_fbr.mean()

# COMMAND ----------

df.head()

# COMMAND ----------

compamies = df[['timestamp']]
compamies['company_id'] = df.index
compamies.reset_index(drop=True, inplace=True)
compamies.head()

# COMMAND ----------

fs = my_experiment_workspace.get_feature_service(feature_service_name)
feature_dataset = fs.get_historical_features(spine=compamies, from_source=True).to_spark()
type(feature_dataset)

# COMMAND ----------

feature_dataset = feature_dataset.persist()

# COMMAND ----------

feature_dataset = feature_dataset.toPandas()
feature_dataset.rename(columns={col: col.split('__')[-1] for col in list(feature_dataset.columns)}, inplace=True)
feature_dataset.set_index('company_id', inplace=True)
feature_dataset.shape

# COMMAND ----------

feature_dataset.head()

# COMMAND ----------

pd.isnull(feature_dataset).sum()

# COMMAND ----------

feature_dataset.dropna(inplace=True)
feature_dataset.shape

# COMMAND ----------

for col in [
  'age_at_completion',
  'applicant_postcode',
  'applicant_idcountry_issue',
  'applicant_nationality',
  'company_keywords',
  'company_type',
  'applicant_id_type',
  'company_icc',
  'company_sic',
  'company_nob',
  'manual_approval_triggers',
  'manual_approval_triggers_unexpected',
  'company_industry_bank',
  'company_keywords_bank',
  'applicant_fraud_pass',
  'director_fraud_pass',
  'shareholder_fraud_pass']:
  
  # feature_dataset[f'{col}_input'] = feature_dataset[col].apply(lambda x: json.loads(x).get("input"))
  # feature_dataset[f'{col}_value'] = feature_dataset[col].apply(lambda x: json.loads(x).get("output").get("value"))
  # feature_dataset[f'{col}_error'] = feature_dataset[col].apply(lambda x: json.loads(x).get("output").get("error"))
  # feature_dataset.drop(columns=[col], inplace=True)

  feature_dataset[f'{col}'] = feature_dataset[col].apply(lambda x: json.loads(x).get("output").get("value"))

feature_dataset.drop(columns=['timestamp'], inplace=True)
feature_dataset.shape

# COMMAND ----------

feature_dataset.tail()

# COMMAND ----------

feature_dataset = feature_dataset.merge(rules_dataset, left_index=True, right_index=True)
feature_dataset.shape

# COMMAND ----------

feature_dataset['fraud_fail'] = 1 - feature_dataset[['applicant_fraud_pass', 'director_fraud_pass', 'shareholder_fraud_pass']].min(axis=1)*1
feature_dataset['count_rc_rule_in_fbr'] = feature_dataset['count_rc_rule_in_fbr'] + feature_dataset['fraud_fail']


# COMMAND ----------

feature_dataset.drop(columns = [ 'applicant_fraud_pass', 'director_fraud_pass', 'shareholder_fraud_pass', 
                                'company_industry_bank','company_keywords_bank', 
                                'company_icc', 'company_sic'], inplace=True)


# COMMAND ----------

feature_dataset = feature_dataset.merge(df[['applicant_device_type', 'applicant_email_domain', 'age_of_company_bucket']], left_index=True, right_index=True)
feature_dataset.shape

# COMMAND ----------

feature_dataset.tail()

# COMMAND ----------

feature_dataset['applicant_nationality'].value_counts()/feature_dataset.shape[0]

# COMMAND ----------

feature_dataset['applicant_email_domain'] = feature_dataset['applicant_email_domain'].apply(lambda x: x if x in ['gmail.com',
 'hotmail.com',
 'outlook.com',
 'yahoo.com',
 'hotmail.co.uk',
 'icloud.com',
 'yahoo.co.uk',
 'live.co.uk'] else "other")
feature_dataset['applicant_email_domain'].value_counts()

# COMMAND ----------

mat = feature_dataset['manual_approval_triggers'].apply(lambda x: np.unique([str(x).upper().replace(st.upper(), ' INDIVIDUAL_BLOCKLIST ') for st in ['bl001', 'bl002', 'bl004', 'bl005', 'bl006', 'bl007', 'bl008']])[0])

mat = mat.apply(lambda x: str(x).upper().replace('applicant_address_fail'.upper(), ' INDIVIDUAL_IDENTITY_ADDRESS '))

mat = mat.apply(lambda x: np.unique([str(x).upper().replace(st.upper(), ' INDIVIDUAL_SANCTIONS_PEP ') for st in ['applicant_pep', 'sanction_fail']])[0])


mat = mat.apply(lambda x: np.unique([str(x).upper().replace(st.upper(), ' INDIVIDUAL_ID_SCAN ') for st in ['facematch_verification_failed', 'id_scan_images_download_failed', 
                                                                                                                       'id_scan_verification_failed', 'idscan_mismatch']])[0])

mat = mat.apply(lambda x: np.unique([str(x).upper().replace(st.upper(), ' BUSINESS_INTERNAL_CHECKS ') for st in ['missing_country_shareholder', 'director_mismatch',
                                                                                                                             'missing_number_or_name_registered_address', 
                                                                                                                             'missing_shareholder', 'missing_sic_codes']])[0])

                                                                                                       

# COMMAND ----------

from sklearn.feature_extraction.text import CountVectorizer

# COMMAND ----------

triggers_to_use = ['individual_blocklist',
 'individual_identity_address',
 'individual_sanctions_pep',
 'individual_id_scan',
 'business_internal_checks']

triggers_vec = CountVectorizer(binary=True, vocabulary=triggers_to_use)                                              
triggers_vec.fit(mat)


# COMMAND ----------

triggers_df = pd.DataFrame(triggers_vec.transform(mat).toarray(), 
                           columns=triggers_vec.get_feature_names_out().tolist(), 
                           index=feature_dataset.index)

triggers_df.shape

# COMMAND ----------

triggers_df.mean()

# COMMAND ----------

import seaborn as sb
heat = sb.heatmap(triggers_df.corr())

# COMMAND ----------

feature_dataset[triggers_df.columns.tolist()] = triggers_df

# COMMAND ----------

feature_dataset['count_failed_business_rules'] = feature_dataset['count_rc_rule_in_fbr'] + feature_dataset[triggers_df.columns.tolist()].sum(axis=1)
feature_dataset['count_failed_business_rules'].value_counts()

# COMMAND ----------

feature_dataset['count_failed_business_rules'] = np.clip(feature_dataset['count_failed_business_rules'], a_min=0, a_max=5)
feature_dataset['count_failed_business_rules'].value_counts()

# COMMAND ----------

feature_dataset.drop(columns = ['manual_approval_triggers',
       'manual_approval_triggers_unexpected'], inplace=True)
feature_dataset.shape

# COMMAND ----------

feature_dataset.drop(columns = [
                                'rule_industry_animal_breeder', 'rule_company_bank', 'rule_applicant_singlename', 
                                'rule_idcountry_belarus', 'rule_idcountry_portugal', 'rule_idcountry_russia', 'rule_idcountry_ukraine', 'rule_idcountry_romania', 'count_rc_rule_in_fbr'], inplace=True)
feature_dataset.shape

# COMMAND ----------

feature_dataset.head()

# COMMAND ----------

feature_dataset['year_month'] = pd.to_datetime(feature_dataset['company_created_on']).apply(lambda x: str(x.date())[:7])

# COMMAND ----------

feature_dataset.shape

# COMMAND ----------

pd.isnull(feature_dataset).sum()

# COMMAND ----------

feature_dataset.to_csv("/dbfs/FileStore/kyc_decisioning/self_learning/tecton_feature_dataset.csv")

# COMMAND ----------

