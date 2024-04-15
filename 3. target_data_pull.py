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

def get_dataset(query_string):

  output = spark_connector(query_string)
  output = output.rename(columns={col: col.lower() for col in list(output.columns)})
  spine =  get_spark().createDataFrame(output)
  return spine

spine_query = """
with closure_type as (
SELECT
  to_varchar(company_id) as company_id,
  "timestamp"::timestamp as action_time,
  action_type::varchar as closure_type,
  "action"::varchar as closure_reason,
  CASE
    WHEN "action" IN ('CA_NOT_NEEDED','CA_DISSATISFIED','CA_LIMITED_FEATURES','CA_MULTI_USER', 'CA_VOLUNTARY_OTHER', 'CA_DISSATISFIED') THEN 'Voluntary'
    WHEN "action" IN ('CA_ADVERSE_MEDIA', 'CA_APPLICATION_REJECTED','CA_RISK_NOB','CA_CONFIRMED_FRAUD','CA_NOT_DIRECTOR','CA_FRAUD_FAKE_ID','CA_ELG_INVALID_SELFIE','CA_RISK_COUNTRY','CA_ELG_NOT_REGISTERED','CA_GEN_NOT_ACTIVE','CA_ELG_NOT_SOLE_TRADER','CA_ELG_NOT_OF_AGE','CA_ELG_SECOND_ST_APPLICATION','CA_ELG_NOT_REGISTERED_CHARITY', 'CA_RISK_BLACKLIST', 'RA_RISK_NOB', 'CA_INVOLUNTARY_OTHER', 'CA_FINCRIME_OTHER', 'RA_GEN_NOT_ACTIVE', 'CA_RISK_OTHER', 'RA_RISK_BLACKLIST', 'RA_FRAUD_FAKE_ID', 'CA_MONEY_LAUNDERING', 'RA_ELG_INVALID_SELFIE', 'CA_CONFIRMED_FRAUD_1ST_PARTY', 'CA_CONFIRMED_FRAUD_3RD_PARTY', 'CA_SANCTIONS_HIT', 'RA_ELG_NOT_OF_AGE', 'RA_NOT_DIRECTOR', 'RA_ELG_NOT_REGISTERED', 'RA_ELG_NOT_REGISTERED_CHARITY', 
    'CA_TERRORIST_FINANCING', 'CA_TAX_EVASION', 'RA_ELG_NOT_SOLE_TRADER', 'RA_RISK_COUNTRY', 'CA_HUMAN_TRAFFICKING') THEN 'Involuntary'
    WHEN "action" IN ('CA_ST_TO_LTD','CA_LTD_TO_ST','CA_DUPLICATE', 'RA_DUPLICATE', 'RA_ST_TO_LTD', 'RA_LTD_TO_ST') THEN 'Multiple Accounts'
  END as closure_type_2, 
  rank() OVER (PARTITION BY company_id ORDER BY "timestamp"::timestamp desc) AS rnk
from db_tide.tide_account_staff_action
),
members as (
select distinct
  member_id,
  created_at,
  completed_at,
  approved_at_clean,
  case
    when (approved_at_clean is not null and approved_at_clean::date - created_at::date <= 90) then 1
    else 0 end as is_approved,
  case
    when (rejected_at_clean is not null and rejected_at_clean::date - created_at::date <= 90) then 1
    else 0 end as is_rejected,
  case
    when (approved_at_clean is null and rejected_at_clean is not null and rejected_at_clean::date - created_at::date <= 90) then 1
    else 0 end as is_rejected_at_onb
from
  kyx_prod.pres_kyx.memberships
where
  created_at::date between '{from_date}'::date and '{to_date}'::date
  and is_completed = 1
  and (approved_at_clean::date - created_at::date <= 90 or rejected_at_clean::date - created_at::date <= 90)
  ),
sar_data as
(SELECT distinct
    m.member_id, to_varchar(A.company_id) as company_id, 
    min(datediff('days', m.approved_at_clean, A.sar_created_date)) as days_to_sar
FROM 
    (SELECT  
        REGEXP_SUBSTR(TRIM(jira_tickets.ticket_summary),'[0-9]{{4,}}') AS company_id, 
        TO_DATE(DATE_TRUNC('DAY', MIN(jira_ticket_changes.change_at))) AS sar_created_date
    FROM    
        TIDE.PRES_JIRA.JIRA_TICKETS AS jira_tickets
        LEFT JOIN TIDE.PRES_JIRA.JIRA_TICKET_CHANGES AS jira_ticket_changes 
        ON jira_tickets.TICKET_ID = jira_ticket_changes.TICKET_ID
    WHERE  jira_tickets.PROJECT_KEY = 'RCM' AND TRIM(jira_tickets.issue_type) IN ('TM alert', 'Risk case') AND
          (jira_tickets.JIRA_TICKET_STATUS IS NULL OR jira_tickets.JIRA_TICKET_STATUS <> 'Duplicates') AND
          (NOT (jira_tickets.is_subtask = 1 ) OR (jira_tickets.is_subtask = 1 ) IS NULL) AND
          jira_ticket_changes.NEW_VALUE IN ('SAR', 'Tide Review', 'PPS Review', 'Submit to NCA', 'NCA Approval', 
                                            'NCA Refusal', 'Clear funds', 'Off-board','customer')
    GROUP BY 1
    ) A 
    JOIN 
    kyx_prod.pres_kyx.companies c 
    ON A.company_id = c.company_id 
    JOIN 
    kyx_prod.pres_kyx.MEMBERSHIPS m 
    ON c.member_id = m.member_id
WHERE 
    m.is_completed = 1
group by 1,2),
decisions as 
(select 
        try_parse_json(data):identity:membership_id as member_id,
        try_parse_json(data):metadata:created_at::timestamp as created_at,
        replace(try_parse_json(data):decision:risk_category::varchar, '"') as risk_category,
        replace(try_parse_json(data):decision:final_decision::varchar, '"') as final_decision,
        try_parse_json(data):decision:failed_business_rules::varchar as failed_business_rules,
        rank() OVER (PARTITION BY member_id 
                     ORDER BY created_at::timestamp DESC) AS rnk
      from raw.kyc_decisioning.decisions 
      where try_parse_json(data):metadata:created_at::date between '{from_date}'::date and '{to_date}'::date
     )
select distinct 
  to_varchar(c.company_id) as company_id, 
  to_varchar(d.member_id) as member_id,
  m.created_at as company_created_on,
  d.final_decision as decision,
  case 
    when d.final_decision = 'mkyc' and m.is_approved = 1 then 'mkyc_approved'
    when d.final_decision = 'mkyc' and m.is_rejected_at_onb = 1 then 'mkyc_rejected'
    when d.final_decision = 'approved' and m.is_approved = 1 then 'auto_approved'
    else d.final_decision end as decision_outcome,
  m.is_rejected_at_onb * case when ct.closure_type = 'InVoluntary' and (ct.action_time::date between m.created_at::date and m.created_at::date + interval '90 days' ) then 1 else 0 end as is_rejected_inv_at_onb,
  m.is_approved * case when ct.closure_type = 'InVoluntary' and (ct.action_time::date between m.approved_at_clean::date and m.approved_at_clean::date + interval '90 days' ) then 1 else 0 end as is_rejected_inv_af_app,
  case when m.is_approved = 1 and sar.days_to_sar <= 90 then 1 else 0 end as is_sar90
from 
  kyx_prod.pres_kyx.companies c 
  inner join members m
  on c.member_id = m.member_id
  inner join 
  decisions d 
  on m.member_id = d.member_id
  left join closure_type ct
  on c.company_id = ct.company_id and ct.rnk = 1
  left join 
  sar_data sar 
  on c.company_id = sar.company_id
where 
  d.rnk = 1
  and not (d.final_decision = 'approved' and m.is_rejected_at_onb = 1)
"""

# COMMAND ----------

def diff_month(d1: str, d2: str):

  d1 = pd.to_datetime(d1)
  d2 = pd.to_datetime(d2)

  return (d1.year - d2.year) * 12 + d1.month - d2.month
    
start_date, end_date = '2022-01-01', '2023-03-31'
diff_month(end_date, start_date)

# COMMAND ----------

df = spark_connector(spine_query.format(from_date=start_date, to_date=end_date))
df = df.toDF(*[c.lower().split('.')[-1] for c in df.columns])
df.count()

# COMMAND ----------

df = df.toPandas()
df.shape

# COMMAND ----------

df['company_id'].duplicated().sum()

# COMMAND ----------

df.set_index('company_id', inplace=True)

# COMMAND ----------

for reward in ['is_rejected_inv_at_onb', 'is_rejected_inv_af_app', 'is_sar90']:
  df[reward] = pd.to_numeric(df[reward])

# COMMAND ----------

df.info()

# COMMAND ----------

df.head()

# COMMAND ----------

df['month_year'] = df['company_created_on'].apply(lambda x: str(x.date())[:7])

# COMMAND ----------

appf_query = """ 
with data as 
(SELECT 
    jira_fincrime.ticket_key,
    jira_fincrime.company_id,
    jira_fincrime.ticket_id,
    min(CASE WHEN jira_ticket_changes.change_at IS NOT NULL THEN jira_ticket_changes.change_at ELSE jira_fincrime.ticket_created_at END) as reported_date
FROM 
    pres_jira.jira_tickets  AS jira_fincrime
    LEFT JOIN pres_jira.jira_ticket_changes jira_ticket_changes 
    ON jira_fincrime.ticket_id = jira_ticket_changes.ticket_id AND jira_ticket_changes.change_field = 'Number of fraud reports'
WHERE 
    jira_fincrime.project_key = 'RCM' 
    AND jira_fincrime.number_of_fraud_report IS NOT NULL
    AND jira_fincrime.is_subtask <> 1 
    AND fraud_type_new  IN ('Invoice scams', 'Mail boxes and multiple post redirections', 'Safe account scam', 'Mule', 'Telephone banking scam', 'HMRC scam', 'Impersonation scam', 'Investment scam', 'Cryptocurrency investment fraud', 'Advance fee fraud', '419 emails and letters', 'Romance scam', 'Purchase scam', 'Other', 'Not provided')
    AND (DATE(CASE WHEN jira_ticket_changes.change_at IS NOT NULL THEN jira_ticket_changes.change_at ELSE jira_fincrime.ticket_created_at END) 
        BETWEEN '{from_date}'::date and '{to_date}'::date + interval '6 months')
    AND jira_fincrime.jira_ticket_status not in ('Duplicate', 'Duplicates')
group by 1,2,3
having
    reported_date between '{from_date}'::date and '{to_date}'::date + interval '6 months'
)
, txns as (
select 
    ticket_key,
    REGEXP_REPLACE(txn_ref_all, '[\\s,"\]', '') as txn_ref_all
from (
    SELECT ticket_key, trim(one_fraud_report_transaction_reference) AS txn_ref_all FROM pres_jira.jira_tickets
    UNION 
    SELECT ticket_key,  trim(two_fraud_report_transaction_reference) FROM pres_jira.jira_tickets
    UNION 
    SELECT ticket_key, trim(three_fraud_report_transaction_reference) FROM pres_jira.jira_tickets
    UNION 
    SELECT ticket_key,  trim(four_fraud_report_transaction_reference) FROM pres_jira.jira_tickets
    UNION 
    SELECT ticket_key,  trim(five_fraud_report_transaction_reference) FROM pres_jira.jira_tickets
    UNION 
    SELECT ticket_key, trim(six_fraud_report_transaction_reference) FROM pres_jira.jira_tickets
    UNION 
    SELECT ticket_key, trim(seven_fraud_report_transaction_reference) FROM pres_jira.jira_tickets
    UNION 
    SELECT ticket_key, trim(eight_fraud_report_transaction_reference) FROM pres_jira.jira_tickets
    UNION 
    SELECT ticket_key, trim(nine_fraud_report_transaction_reference) FROM pres_jira.jira_tickets
    UNION 
    SELECT ticket_key, trim(ten_fraud_report_transaction_reference) FROM pres_jira.jira_tickets)
where txn_ref_all is not null)
, tickets as
(
select distinct
    coalesce(a.company_id::varchar, b.company_id::varchar) as company_id,
    case when a.parent_key is  null then a.ticket_key else a.parent_key end as ticket_key,
    case when a.parent_key is  null then a.fraud_type_new else sub.fraud_type_new end as fraud_type_new, 
    b.txn_ref as txnref,
    b.transaction_at,
    b.amount,
    comp.is_registered,
    comp.first_transaction_at,
    comp.industry_classification_description,
    mem.approved_at_clean,
    mem.last_touch_attribution_marketing_channel,
    mem.risk_rating,
    mem.risk_band,
    comp.is_cofo,
    CASE WHEN mem.manual_approval_triggers <> ''  THEN 'Yes' ELSE 'No' END as manual_kyc,
    incorporation_at
from 
    (select * from pres_jira.jira_tickets where project_key = 'RCM') a 
    left join txns t on a.ticket_key = t.ticket_key
    left join pres_jira.jira_tickets sub on a.parent_key = sub.ticket_key
    join pres_core.cleared_transactions b on t.txn_ref_all = b.txn_ref
    left join KYX_PROD.PRES_KYX.companies comp on b.company_id = comp.company_id
    left join KYX_PROD.PRES_KYX.memberships mem on comp.member_id = mem.member_id
where year(transaction_at) >= 2022
qualify row_number() over(partition by txn_ref order by a.ticket_created_at asc) = 1 )
, jira as
(select  t.* 
from tickets t
join  
data d 
on t.ticket_key = d.ticket_key 
), 
appf as
(select *, --count( distinct ticket_key ) as no_of_tickets
case when fraud_type_new IN ('Invoice scams', 'Mail boxes and multiple post redirections') THEN 'Invoice and Mandate scam'
     when fraud_type_new IN ('Safe account scam', 'Mule') THEN 'Safe account scam'  
     when fraud_type_new IN ('HMRC scam', 'Telephone banking scam', 'Impersonation scam') THEN 'Impersonation scam'
     when fraud_type_new IN ('Investment scam', 'Cryptocurrency investment fraud') THEN 'Investment scam'
     when fraud_type_new IN ('Advance fee fraud', '419 emails and letters') THEN 'Advance fee scam'
     when fraud_type_new IN ('Romance scam') THEN 'Romance scam'
     when fraud_type_new IN ('Purchase scam') THEN 'Purchase scam'
     when fraud_type_new IN ('Other', 'Not provided') THEN 'Unknown scam'
     else fraud_type_new end as mapped_fraud_type
from jira
where mapped_fraud_type in ('Invoice and Mandate scam', 'Impersonation scam', 'Safe account scam', 'Investment scam', 'Advance fee scam', 'Romance scam', 'Purchase scam', 'Unknown scam')
order by company_id, transaction_at)
select 
    c.member_id::varchar as membership_id,
    c.company_id::varchar as company_id, 
    c.created_at::timestamp as timestamp,
    m.approved_at_clean::timestamp as approved_at,
    case when m.approved_at_clean is not null then 1 else 0 end as is_approved,
    max(case when appf.transaction_at::date between m.approved_at_clean::date and m.approved_at_clean::date + interval '6 months' then 1 else 0 end) as is_app_fraud,
    sum(case when appf.transaction_at::date between m.approved_at_clean::date and m.approved_at_clean::date + interval '6 months' then appf.amount else 0 end) as app_fraud_amount
from 
    KYX_PROD.PRES_KYX.memberships m
    inner join 
    KYX_PROD.PRES_KYX.companies c 
    on m.member_id::varchar = c.member_id::varchar
    left join 
    appf 
    on c.company_id::varchar = appf.company_id::varchar
where 
  m.is_completed=1 
  and nvl(m.approved_at_clean::date, c.created_at::date) <= '{to_date}'::date
  and 
  ((c.created_at::date between '{from_date}'::date and '{to_date}'::date)
  or 
  ((m.approved_at_clean::date between '{from_date}'::date and '{to_date}'::date) 
  and (m.approved_at_clean::date between c.created_at::date and c.created_at::date + interval '3 months'))
  )
group by 1, 2, 3, 4, 5
"""

# COMMAND ----------

appf_df = spark_connector(appf_query.format(from_date=start_date, to_date=end_date))
appf_df = appf_df.toDF(*[c.lower().split('.')[-1] for c in appf_df.columns])
appf_df.count()

# COMMAND ----------

dfapp = appf_df.toPandas()
dfapp.shape

# COMMAND ----------

dfapp.index.duplicated().sum()

# COMMAND ----------

dfapp.set_index('company_id', inplace=True)

# COMMAND ----------

df = df.merge(dfapp[['is_app_fraud', 'app_fraud_amount']], left_index=True, right_index=True, how='left')
df.shape

# COMMAND ----------

pd.isnull(df).sum()/df.shape[0]

# COMMAND ----------

df['is_app_fraud'] = pd.to_numeric(df['is_app_fraud'], errors='ignore')
df[['is_app_fraud', 'app_fraud_amount']] = df[['is_app_fraud', 'app_fraud_amount']].fillna(0)
df['app_fraud_amount'] = df['app_fraud_amount'].apply(lambda x: 0 if x<0 else x)

# COMMAND ----------

df.describe()

# COMMAND ----------

base_dataset_train = df[pd.to_datetime(df['company_created_on']).apply(lambda x: x.date()) <= (pd.to_datetime(end_date).date() - timedelta(days=90))]
base_dataset_test = df[pd.to_datetime(df['company_created_on']).apply(lambda x: x.date()) > (pd.to_datetime(end_date).date() - timedelta(days=90))]
base_dataset_train.shape, base_dataset_test.shape

# COMMAND ----------

base_dataset_train.groupby(['decision'])[['is_rejected_inv_at_onb','is_rejected_inv_af_app','is_sar90', 
                                          'is_app_fraud', 'app_fraud_amount']].mean()

# COMMAND ----------

base_dataset_train.groupby(['decision_outcome'])[['is_rejected_inv_at_onb','is_rejected_inv_af_app','is_sar90',
                                                  'is_app_fraud', 'app_fraud_amount']].mean()

# COMMAND ----------

df.head()

# COMMAND ----------

feature_dataset_train = pd.read_parquet("/dbfs/FileStore/kyc_decisioning/self_learning/tecton_feature_dataset_train.parquet")
feature_dataset_test = pd.read_parquet("/dbfs/FileStore/kyc_decisioning/self_learning/tecton_feature_dataset_test.parquet")
feature_dataset_train.shape, feature_dataset_test.shape

# COMMAND ----------

feature_dataset_train.index.duplicated().sum(), feature_dataset_test.index.duplicated().sum()

# COMMAND ----------

feature_dataset_train.head()

# COMMAND ----------

base_dataset_train = base_dataset_train.merge(feature_dataset_train, left_index=True, right_index=True)
base_dataset_test = base_dataset_test.merge(feature_dataset_test, left_index=True, right_index=True, how='right')
base_dataset_train.shape, base_dataset_test.shape

# COMMAND ----------

base_dataset_train.head()

# COMMAND ----------

base_dataset_test.head()

# COMMAND ----------

base_dataset_train.to_parquet("/dbfs/FileStore/kyc_decisioning/self_learning/base_dataset_train.parquet")
base_dataset_test.to_parquet("/dbfs/FileStore/kyc_decisioning/self_learning/base_dataset_test.parquet")

# COMMAND ----------

