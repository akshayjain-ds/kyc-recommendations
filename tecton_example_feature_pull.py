# Databricks notebook source
# MAGIC %run ./set_up

# COMMAND ----------

import numpy as np
import pandas as pd
from typing import Optional
from tecton import conf
from datetime import date, datetime, timedelta
import tecton
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
import json

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

def spark_connector(query_string: str)-> DataFrame:
  return spark.read.format("snowflake").options(**CONNECTION_OPTIONS).option("query", query_string).load().cache()
tecton.__version__

# COMMAND ----------

workspace_name = 'akshayjain'
my_experiment_workspace = tecton.get_workspace(workspace_name)
feature_service_name = 'uk_ifre_feature_service'

# COMMAND ----------

example_companies = [
  '1218780', '1348160', '1348160', '1407038', '1508146', '1512497', '1512752', '1617146', '1704783', '1728419'
]

spine_df = pd.DataFrame(example_companies, columns=['company_id'])
spine_df['timestamp'] = pd.to_datetime("2023-09-01")

# COMMAND ----------

from pyspark.sql import SparkSession
#Create PySpark SparkSession
spark = SparkSession.builder \
    .master("local[1]") \
    .appName("SparkByExamples.com") \
    .getOrCreate()
#Create PySpark DataFrame from Pandas
spine_df=spark.createDataFrame(spine_df) 
spine_df.printSchema()
spine_df.show()

# COMMAND ----------

fs = my_experiment_workspace.get_feature_service(feature_service_name)
feature_dataset = fs.get_historical_features(spine=spine_df, from_source=True).to_pandas()

# COMMAND ----------

feature_dataset

# COMMAND ----------

feature_dataset.rename(columns={col: col.split('__')[-1] for col in list(feature_dataset.columns)}, inplace=True)
feature_dataset.set_index('company_id', inplace=True)

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
  
  feature_dataset[f'{col}_input'] = feature_dataset[col].apply(lambda x: json.loads(x).get("input"))
  feature_dataset[f'{col}_value'] = feature_dataset[col].apply(lambda x: json.loads(x).get("output").get("value"))
  feature_dataset[f'{col}_error'] = feature_dataset[col].apply(lambda x: json.loads(x).get("output").get("error"))

# COMMAND ----------

feature_dataset

# COMMAND ----------

