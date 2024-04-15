# Databricks notebook source
# MAGIC %run ../set_up

# COMMAND ----------

import pandas as pd
from typing import Optional
from tecton import conf
from datetime import date, datetime, timedelta
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

from datetime import date
from dateutil.relativedelta import relativedelta

# start_date, end_date = '2022-01-01', str(date.today() + relativedelta(months=-6) )
start_date, end_date = '2022-01-01', '2023-03-31'
start_date, end_date

# COMMAND ----------

workspace_name = 'akshayjain'
raw_service_name = 'membership_completed_v_data_source'
feature_service_name = 'uk_ifre_feature_service'
my_experiment_workspace = tecton.get_workspace(workspace_name)

# COMMAND ----------

member_query = """
with company as
(
select 
  c.company_id::varchar as company_id,
  m.member_id::varchar as member_id,
  c.created_at::timestamp as created_at,
  c.accounts_next_due_at as comnpany_accounts_next_due_at, 
  c.is_accounts_overdue as comnpany_accounts_overdue, 
  c.company_status,
  rank() OVER (PARTITION BY c.company_id ORDER BY c.created_at::timestamp DESC) AS rnk
from 
  KYX_PROD.PRES_KYX.memberships m 
  inner join
  KYX_PROD.PRES_KYX.companies c
  on c.member_id = m.member_id
where 
  m.is_completed=1 
  and nvl(m.approved_at_clean::date, c.created_at::date) <= '{to_date}'::date
  and 
  ((c.created_at::date between '{from_date}'::date and '{to_date}'::date)
  or 
  ((m.approved_at_clean::date between '{from_date}'::date and '{to_date}'::date) 
  and (m.approved_at_clean::date between c.created_at::date and c.created_at::date + interval '3 months'))
  )
),
async as
(select 
  member_id, id_type, id_subtype, id_country, id_first_name, id_last_name, id_expiry_at,
  rank() OVER (PARTITION BY member_id ORDER BY created_at DESC) AS rnk
from 
  KYX_PROD.CHNL_KYX.VERIFICATION_ID_SCAN
where 
  verification_type in ('ID_SCAN','ID_SCAN_VALIDATIONS') 
  and verification_usage IN ('REGISTRATION', 'ACCOUNT_RECOVERY') 
  ),
dob as 
(SELECT
    member_id,
    date_of_birth,
    rank() OVER (PARTITION BY member_id ORDER BY created_at::timestamp) AS rnk
FROM 
    KYX_PROD.PRES_KYX.users
)
select distinct
  c.company_id, 
  c.created_at,
  c.comnpany_accounts_next_due_at, 
  c.comnpany_accounts_overdue, 
  async.id_expiry_at, 
  async.id_first_name,
  async.id_last_name,
  dob.date_of_birth,
  datediff('years', c.created_at::date, async.id_expiry_at::date) as years_to_id_expiry
from 
  company c 
  left join 
  async
  on c.member_id = async.member_id 
  and c.rnk = 1 and async.rnk = 1
  left join 
  dob 
  on c.member_id = dob.member_id and dob.rnk=1
"""

# COMMAND ----------

member_df = spark_connector(member_query.format(from_date=start_date, to_date=end_date))
member_df = member_df.toDF(*[c.lower().split('.')[-1] for c in member_df.columns])
member_df.count()

# COMMAND ----------

member_df = member_df.toPandas()
member_df['company_id'].duplicated().sum()

# COMMAND ----------

member_df.head()

# COMMAND ----------

if member_df['company_id'].duplicated().sum() == 0:
  member_df.set_index('company_id', inplace=True)
  print("company_id set as index")
member_df.shape

# COMMAND ----------

pd.isnull(member_df).sum()/member_df.shape[0]

# COMMAND ----------

member_df['years_to_id_expiry'] = pd.to_numeric(member_df['years_to_id_expiry'], errors='ignore')

# COMMAND ----------

member_df['years_to_id_expiry'].describe(percentiles=np.linspace(0,1,101))

# COMMAND ----------

member_df['years_to_id_expiry'] = np.clip(member_df['years_to_id_expiry'], a_min=0, a_max=10)

# COMMAND ----------

member_df['years_to_id_expiry'] = member_df['years_to_id_expiry'].fillna(member_df['years_to_id_expiry'].median()) 

# COMMAND ----------

duedil_query = """select distinct * from 
(select distinct
  c.company_id::varchar as company_id, 
  c.member_id::varchar as member_id,
  c.created_at::timestamp as created_at,
  case when COMPANIES_HOUSE_NUMBER = to_varchar(try_parse_json(json_col):data:companyNumber) then 1 else 0 end as duedil_hit,
  COMPANIES_HOUSE_NUMBER,
  try_parse_json(json_col):metadata:created_at::timestamp as duedil_created_at,
  try_parse_json(json_col):data as duedil_payload,
  try_parse_json(json_col):data:address:postcode::varchar as company_postcode,
  try_parse_json(json_col):data:address:countryCode::varchar as company_countrycode,
  try_parse_json(json_col):data:charitableIdentityCount as charitableIdentityCount,
  try_parse_json(json_col):data:financialSummary as financialSummary,
  try_parse_json(json_col):data:incorporationDate as incorporationDate,
  try_parse_json(json_col):data:numberOfEmployees as numberOfEmployees,
  try_parse_json(json_col):data:recentStatementDate as recentStatementDate,
  try_parse_json(json_col):data:majorShareholders as majorShareholders,
  try_parse_json(json_col):data:directorsTree as directorsTree,
  try_parse_json(json_col):data:shareholderTree as shareholderTree,
  try_parse_json(json_col):data:personsOfSignificantControl as personsOfSignificantControl,
  try_parse_json(json_col):data:structureDepth as structureDepth,
  try_parse_json(json_col):data:structureLevelWise:"1" as structureLevelWise,
  try_parse_json(json_col):data:status::varchar as status,
  rank() OVER (PARTITION BY COMPANIES_HOUSE_NUMBER ORDER BY duedil_created_at) AS rnk
from 
  KYX_PROD.PRES_KYX.companies c
  inner join 
  KYX_PROD.PRES_KYX.memberships m
  on c.member_id = m.member_id and m.is_completed=1
  left join 
  TIDE.DUEDIL_INTEGRATION.uk_registered_companies d
  on COMPANIES_HOUSE_NUMBER = to_varchar(try_parse_json(json_col):data:companyNumber)
where c.created_at::date between '{from_date}'::date and '{to_date}'::date
order by company_id, created_at)
where rnk = 1"""

# COMMAND ----------

dd_df = spark_connector(duedil_query.format(from_date=start_date, to_date="2023-06-30"))
dd_df = dd_df.toDF(*[c.lower().split('.')[-1] for c in dd_df.columns])
dd_df.count()

# COMMAND ----------

dd_df = dd_df.toPandas()
dd_df.head()

# COMMAND ----------

dd_df['company_id'].duplicated().sum()

# COMMAND ----------

dd_df.set_index('company_id', inplace=True)

# COMMAND ----------

dd_df['duedil_hit'].mean()

# COMMAND ----------

dd_df['company_status'] = dd_df['status'].apply(lambda x: "Undefined" if pd.isnull(x) else str(x)).astype(str)
dd_df['company_status'].value_counts()

# COMMAND ----------

from feature_extractors.kyc.risk_extractors.mappings.postcodes import PostcodeMapping

def validate_and_get_postcode(raw_input: str):
    if raw_input is None or (isinstance(raw_input, str) and not raw_input.strip()):
        return repr(ValueError("empty value for the postcode is not allowed"))

    if not raw_input or not isinstance(raw_input, str):
        return repr(TypeError(f"postcode must be a string. Got {type(raw_input).__name__}"))

    raw_input = raw_input.strip()

    if len(raw_input) <= 4:
        outward = raw_input.strip()
    else:
        raw_input = raw_input.replace(' ', '')
        inward_len = 3
        outward, inward = raw_input[:-inward_len].strip(), raw_input[-inward_len:].strip()

    if not 2 <= len(outward) <= 4:
        return repr(ValueError(
            f'Postcode {raw_input} is not valid - outward is expected to be b/w 2-4 chars. got outward_code={outward}'))

    return outward

def postcode_value_mapper(raw_input: str, processed_value: str, mapping: dict = PostcodeMapping.get_postcode_mapping_v2()) -> str:
    import json

    if processed_value.__contains__("Error") or processed_value.__contains__("Exception"):
        return json.dumps({"input": {"postcode": raw_input},
                            "output": {"value": "Undefined",
                                      "error": repr(processed_value)}})

    if len(processed_value) == 0:
        return json.dumps({"input": {"postcode": raw_input},
                            "output": {"value": "Low",
                                      "error": None}})

    if processed_value in mapping:
        return json.dumps({"input": {"postcode": raw_input},
                            "output": {"value": mapping[processed_value],
                                      "error": None}})
    else:
        return postcode_value_mapper(raw_input, processed_value[:-1], mapping)
      
company_postcode = dd_df['company_postcode'].apply(lambda x: postcode_value_mapper(x, validate_and_get_postcode(x)))
company_postcode_risk = company_postcode.apply(lambda x: json.loads(x).get("output").get("value"))
dd_df['company_postcode'] = company_postcode_risk.apply(lambda x: 'High' if x == 'high' else x)

# COMMAND ----------

dd_df['company_postcode'].value_counts()/dd_df.shape[0]

# COMMAND ----------

member_df = member_df.merge(dd_df[['company_status', 'company_postcode']], left_index=True, right_index=True)
member_df.shape

# COMMAND ----------

member_df.head()

# COMMAND ----------

def get_director_info(payload: str) -> {}:

  try:

    directorstree = json.loads(payload)
    directors = {}
    for i, officer in enumerate(directorstree):

      officer_id = directorstree[i].get("officerId", '')
      directors[officer_id] = {}

      directors[officer_id]['dd_first_name'] = directorstree[i].get("person", {}).get("firstName", '').lower()
      directors[officer_id]['dd_last_name'] = directorstree[i].get("person", {}).get("lastName", '').lower()
      dd_dob = pd.to_datetime(directorstree[i].get("person", {}).get("dateOfBirth", ''))
      directors[officer_id]['dd_dob_month'] = str(dd_dob.date().month)
      directors[officer_id]['dd_dob_year'] = str(dd_dob.date().year)
      directors[officer_id]['dd_nationalities'] = directorstree[i].get("person", {}).get("nationalities", [])
    
    return directors

  except:
    return None

# COMMAND ----------

dd_df['directors_info'] = dd_df['directorstree'].apply(lambda x: get_director_info(x))

# COMMAND ----------

pd.isnull(dd_df['directors_info']).sum()/dd_df['directors_info'].shape[0]

# COMMAND ----------

member_df.head()

# COMMAND ----------

applicant_director_info = member_df[['created_at', 'id_first_name', 'id_last_name', 'date_of_birth']].merge(dd_df[['directors_info']], left_index=True, right_index=True)
applicant_director_info.shape

# COMMAND ----------

applicant_director_info.head()

# COMMAND ----------

from fuzzywuzzy import fuzz
def get_applicant_director_nationality(row: pd.Series, similarity_threshold=90):

  if bool(row['directors_info']):
  
    applicant_id_firstname = str(row['id_first_name']).lower()
    applicant_id_lasttname = str(row['id_last_name']).lower()
    applicant_dob = pd.to_datetime(row['date_of_birth']).date()
    applicant_dob_month = str(applicant_dob.month)
    applicant_dob_year = str(applicant_dob.year)
    directors_info = row['directors_info']

    for officer in directors_info.keys():
      director = directors_info.get(officer, {})

      if (fuzz.partial_token_set_ratio(applicant_id_firstname, director.get('dd_first_name', '')) >= similarity_threshold) and \
        (fuzz.partial_token_set_ratio(applicant_id_lasttname, director.get('dd_last_name', '')) >= similarity_threshold) and \
          applicant_dob_month == director.get('dd_dob_month', '') \
            and applicant_dob_year == director.get('dd_dob_year', ''):
              return director.get('dd_nationalities', None)
      
  return None

# COMMAND ----------

def get_applicant_directors_avg_age(row: pd.Series):
  
  company_created_on = pd.to_datetime(row['created_at'])

  if bool(row['directors_info']):
  
    directors_info = row['directors_info']

    directors_age = []
    for officer in directors_info.keys():
      director = directors_info.get(officer, {})

      director_dob = pd.to_datetime(f"{director.get('dd_dob_year', '')}-{director.get('dd_dob_month', '')}-01")
      age = (company_created_on - director_dob)/np.timedelta64(1, 'Y')
      directors_age.append(age)

      return int(np.average(directors_age))
    
  else:

    applicant_dob = pd.to_datetime(row['date_of_birth'])
    age = (company_created_on - applicant_dob)/np.timedelta64(1, 'Y')

    return int(age)

      
  return None

# COMMAND ----------

applicant_director_nationality = applicant_director_info.apply(lambda row: get_applicant_director_nationality(row), axis=1)
applicant_directors_age = applicant_director_info.apply(lambda row: get_applicant_directors_avg_age(row), axis=1)
applicant_director_nationality.shape, applicant_directors_age.shape

# COMMAND ----------

from feature_extractors.kyc.risk_extractors.risk_extractors import ApplicantIdCountry
country_mapping = ApplicantIdCountry().country_mappings
applicant_nationality = applicant_director_nationality.apply(lambda x: [country_mapping.get(country, "Undefined") for country in x] if bool(x) else "Undefined")
applicant_nationality.value_counts()

# COMMAND ----------

applicant_dd_nationality = applicant_nationality.apply(lambda x: "High" if "High" in x else ("Medium" if "Medium" in x else ("Low" if "Low" in x else "Undefined")))

# COMMAND ----------

applicant_dd_nationality = pd.DataFrame(applicant_dd_nationality, columns=['applicant_dd_nationality'])
applicant_directors_age = pd.DataFrame(applicant_directors_age, columns=['directors_avg_age_at_completion'])
applicant_dd_nationality.shape, applicant_directors_age.shape

# COMMAND ----------

applicant_dd_nationality['applicant_dd_nationality'].value_counts()/applicant_dd_nationality.shape[0]

# COMMAND ----------

applicant_directors_age.describe()

# COMMAND ----------

member_df = member_df.merge(applicant_dd_nationality, left_index=True, right_index=True).merge(applicant_directors_age, left_index=True, right_index=True)
member_df.shape

# COMMAND ----------

member_df['applicant_dd_nationality'].value_counts()/member_df.shape[0]

# COMMAND ----------

dd_df['company_directors_count'] = dd_df['directors_info'].apply(lambda x: len(x) if bool(x) else np.nan)
dd_df['company_directors_count'] = dd_df['company_directors_count'].apply(lambda x: "1" if x == 1 else ("2" if x == 2 else ("3+" if x >= 3 else "Undefined")))
dd_df['company_directors_count'].value_counts()

# COMMAND ----------

member_df = member_df.merge(dd_df['company_directors_count'], left_index=True, right_index=True)
member_df.shape


# COMMAND ----------

dd_df['company_structurelevelwise'] = dd_df['structurelevelwise'].apply(lambda x: "Undefined" if pd.isnull(x) else "1" if x == "1" else ("2" if x == "2" else "3+")).astype(str)
dd_df['company_structurelevelwise'].value_counts()

# COMMAND ----------

member_df = member_df.merge(dd_df[['company_structurelevelwise']], left_index=True, right_index=True)
member_df.shape

# COMMAND ----------

feature_dataset = pd.read_csv("/dbfs/FileStore/kyc_decisioning/self_learning/tecton_feature_dataset.csv",
                 dtype={"company_id": "str"})
feature_dataset.set_index('company_id', inplace=True)
feature_dataset['company_created_on'] = pd.to_datetime(feature_dataset['company_created_on']).apply(lambda x: x.date())
feature_dataset.shape

# COMMAND ----------

member_df.columns.tolist()

# COMMAND ----------

feature_dataset.columns.tolist()

# COMMAND ----------

feature_dataset = feature_dataset.merge(member_df[['years_to_id_expiry',
                                                'company_status',
                                                'company_postcode',
                                                'applicant_dd_nationality',
                                                'directors_avg_age_at_completion',
                                                'company_directors_count',
                                                'company_structurelevelwise']], left_index=True, right_index=True)
feature_dataset.shape

# COMMAND ----------

feature_dataset['applicant_nationality'] = feature_dataset[['applicant_nationality', 'applicant_dd_nationality']].apply(lambda row:
  "High" if any([row['applicant_nationality'] == 'High', row['applicant_dd_nationality'] == 'High']) else (
    "Medium" if any([row['applicant_nationality'] == 'Medium', row['applicant_dd_nationality'] == 'Medium']) else (
      "Low" if any([row['applicant_nationality'] == 'Low', row['applicant_dd_nationality'] == 'Low']) else "Undefined"
    )
  ),
  axis=1)

feature_dataset.drop(columns=['applicant_dd_nationality'], inplace=True)

# COMMAND ----------

feature_dataset['applicant_nationality'].value_counts()/feature_dataset.shape[0]

# COMMAND ----------

feature_dataset['applicant_idcountry_idnationality'] = feature_dataset[['applicant_idcountry_issue', 'applicant_nationality']].apply(lambda row:
  "High" if any([row['applicant_idcountry_issue'] == 'High', row['applicant_nationality'] == 'High']) else (
    "Medium" if any([row['applicant_idcountry_issue'] == 'Medium', row['applicant_nationality'] == 'Medium']) else (
      "Low" if any([row['applicant_idcountry_issue'] == 'Low', row['applicant_nationality'] == 'Low']) else "Undefined"
    )
  ),
  axis=1)

# COMMAND ----------

feature_dataset.shape

# COMMAND ----------

feature_dataset.head()

# COMMAND ----------

pd.isnull(feature_dataset).sum()/feature_dataset.shape[0]

# COMMAND ----------

feature_dataset_train = feature_dataset[pd.to_datetime(feature_dataset['company_created_on']).apply(lambda x: x.date()) <= (pd.to_datetime(end_date).date() - timedelta(days=90))]
feature_dataset_test = feature_dataset[pd.to_datetime(feature_dataset['company_created_on']).apply(lambda x: x.date()) > (pd.to_datetime(end_date).date() - timedelta(days=90))]
feature_dataset_train.shape, feature_dataset_test.shape

# COMMAND ----------

feature_dataset_train['company_created_on'].min(), feature_dataset_train['company_created_on'].max()

# COMMAND ----------

feature_dataset_test['company_created_on'].min(), feature_dataset_test['company_created_on'].max()

# COMMAND ----------

feature_dataset_train.drop(columns=['company_created_on'], inplace=True)
feature_dataset_test.drop(columns=['company_created_on'], inplace=True)

# COMMAND ----------

feature_dataset_train.head()

# COMMAND ----------

feature_dataset_train.to_parquet("/dbfs/FileStore/kyc_decisioning/self_learning/tecton_feature_dataset_train.parquet")
feature_dataset_test.to_parquet("/dbfs/FileStore/kyc_decisioning/self_learning/tecton_feature_dataset_test.parquet")

# COMMAND ----------

