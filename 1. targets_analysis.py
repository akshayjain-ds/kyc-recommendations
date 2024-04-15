# Databricks notebook source
# MAGIC %run ../set_up

# COMMAND ----------

import numpy as np
import pandas as pd
from typing import Optional
from tecton import conf
from datetime import date, datetime
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
  "action"::varchar as staff_action,
  CASE
    WHEN "action" IN ('CA_NOT_NEEDED','CA_DISSATISFIED','CA_LIMITED_FEATURES','CA_MULTI_USER', 'CA_VOLUNTARY_OTHER', 'CA_DISSATISFIED') THEN 'Voluntary'
    WHEN "action" IN ('CA_ADVERSE_MEDIA', 'CA_APPLICATION_REJECTED','CA_RISK_NOB','CA_CONFIRMED_FRAUD','CA_NOT_DIRECTOR','CA_FRAUD_FAKE_ID','CA_ELG_INVALID_SELFIE','CA_RISK_COUNTRY','CA_ELG_NOT_REGISTERED','CA_GEN_NOT_ACTIVE','CA_ELG_NOT_SOLE_TRADER','CA_ELG_NOT_OF_AGE','CA_ELG_SECOND_ST_APPLICATION','CA_ELG_NOT_REGISTERED_CHARITY', 'CA_RISK_BLACKLIST', 'RA_RISK_NOB', 'CA_INVOLUNTARY_OTHER', 'CA_FINCRIME_OTHER', 'RA_GEN_NOT_ACTIVE', 'CA_RISK_OTHER', 'RA_RISK_BLACKLIST', 'RA_FRAUD_FAKE_ID', 'CA_MONEY_LAUNDERING', 'RA_ELG_INVALID_SELFIE', 'CA_CONFIRMED_FRAUD_1ST_PARTY', 'CA_CONFIRMED_FRAUD_3RD_PARTY', 'CA_SANCTIONS_HIT', 'RA_ELG_NOT_OF_AGE', 'RA_NOT_DIRECTOR', 'RA_ELG_NOT_REGISTERED', 'RA_ELG_NOT_REGISTERED_CHARITY', 
    'CA_TERRORIST_FINANCING', 'CA_TAX_EVASION', 'RA_ELG_NOT_SOLE_TRADER', 'RA_RISK_COUNTRY', 'CA_HUMAN_TRAFFICKING') THEN 'Involuntary'
    WHEN "action" IN ('CA_ST_TO_LTD','CA_LTD_TO_ST','CA_DUPLICATE', 'RA_DUPLICATE', 'RA_ST_TO_LTD', 'RA_LTD_TO_ST') THEN 'Multiple Accounts'
  END as closure_type, 
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
  pres_core.memberships
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
    TIDE.PRES_CORE.COMPANIES c 
    ON A.company_id = c.company_id 
    JOIN 
    TIDE.PRES_CORE.MEMBERSHIPS m 
    ON c.member_id = m.member_id
WHERE 
    m.is_completed = 1
group by 1,2),
decisions as 
(select 
        try_parse_json(data):identity:membership_id as membership_id,
        try_parse_json(data):metadata:created_at::timestamp as created_at,
        replace(try_parse_json(data):decision:risk_category::varchar, '"') as risk_category,
        replace(try_parse_json(data):decision:final_decision::varchar, '"') as final_decision,
        try_parse_json(data):decision:failed_business_rules::varchar as failed_business_rules,
        rank() OVER (PARTITION BY membership_id 
                     ORDER BY created_at::timestamp DESC) AS rnk
      from raw.kyc_decisioning.decisions 
      where try_parse_json(data):metadata:created_at::date between '{from_date}'::date and '{to_date}'::date
     )
select distinct 
  to_varchar(c.companyid) as company_id, 
  to_varchar(d.membership_id) as membership_id,
  m.created_at as company_created_on,
  d.final_decision as decision,
  m.is_approved, 
  m.is_rejected, 
  m.is_rejected_at_onb,
  case 
    when d.final_decision = 'mkyc' and m.is_approved = 1 then 'mkyc_approved'
    when d.final_decision = 'mkyc' and m.is_rejected_at_onb = 1 then 'mkyc_rejected'
    when d.final_decision = 'approved' and m.is_approved = 1 then 'auto_approved'
    else d.final_decision end as decision_outcome,
  m.is_rejected_at_onb * case when ct.closure_type = 'Involuntary' then 1 else 0 end as is_rejected_inv_at_onb,
  m.is_approved * case when ct.closure_type = 'Involuntary' and (ct.action_time::date between m.approved_at_clean::date and m.approved_at_clean::date + interval '90 days') then 1 else 0 end as is_rejected_inv_af_app,
  case when m.is_approved = 1 and sar.days_to_sar <= 90 then 1 else 0 end as is_sar90,
  case when (ct.action_time::date between m.created_at::date and m.created_at::date + interval '90 days') or (ct.action_time::date between m.approved_at_clean::date and m.approved_at_clean::date + interval '90 days') then staff_action else '' end staff_action
from 
  db_tide.company c 
  inner join members m
  on c.membership_id = m.member_id
  inner join 
  decisions d 
  on m.member_id = d.membership_id
  left join closure_type ct
  on c.companyid = ct.company_id and ct.rnk = 1
  left join 
  sar_data sar 
  on c.companyid = sar.company_id
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

df.info()

# COMMAND ----------

for col in ['is_approved', 'is_rejected', 'is_rejected_at_onb', 'is_rejected_inv_at_onb', 'is_rejected_inv_af_app', 'is_sar90']:
  df[col] = pd.to_numeric(df[col], errors='ignore')

# COMMAND ----------

df.head()

# COMMAND ----------

print(np.around(df[['is_rejected', 'is_rejected_at_onb', 'is_rejected_inv_at_onb', 'is_rejected_inv_af_app', 'is_sar90']].mean()*100, decimals=2))

# COMMAND ----------

df[['is_rejected', 'is_rejected_at_onb', 'is_rejected_inv_at_onb', 'is_rejected_inv_af_app', 'is_sar90']].corr('spearman')

# COMMAND ----------

df.groupby(['is_approved'])[['is_rejected','is_rejected_at_onb', 'is_rejected_inv_at_onb', 'is_rejected_inv_af_app', 'is_sar90']].agg(['mean'])

# COMMAND ----------

df.groupby(['decision'])[['decision']].agg(['count'])

# COMMAND ----------

df.groupby(['decision'])[['is_rejected_inv_at_onb', 'is_rejected_inv_af_app', 'is_sar90']].agg(['mean'])

# COMMAND ----------

df.groupby(['decision_outcome'])[['decision_outcome']].agg(['count'])

# COMMAND ----------


df.groupby(['decision_outcome'])[['is_rejected_inv_at_onb', 'is_rejected_inv_af_app', 'is_sar90']].agg(['mean'])

# COMMAND ----------

r = (df[df['is_rejected']==1].groupby(['closure_reason'])['closure_reason'].count()/df[df['is_rejected']==1].shape[0]).sort_values(ascending=False)
r.sum(), r


# COMMAND ----------

r = (df[df['is_rejected_at_onb']==1].groupby(['closure_reason'])['closure_reason'].count()/df[df['is_rejected_at_onb']==1].shape[0]).sort_values(ascending=False)
r.sum(), r

# COMMAND ----------

(df[df['is_rejected_inv_at_onb']==1].groupby(['closure_reason'])['closure_reason'].count()/df[df['is_rejected_inv_at_onb']==1].shape[0]).sort_values(ascending=False)

# COMMAND ----------

((df[df['is_rejected_inv_at_onb']==1].groupby(['closure_reason'])['closure_reason'].count()/df[df['is_rejected_inv_at_onb']==1].shape[0]).sort_values(ascending=False).head(8)).plot.pie(y='closure_reason', figsize=(5,5))

# COMMAND ----------

r = (df[df['is_rejected_inv_af_app']==1].groupby(['closure_reason'])['closure_reason'].count()/df[df['is_rejected_inv_af_app']==1].shape[0]).sort_values(ascending=False)
r.sum(), r


# COMMAND ----------

((df[df['is_rejected_inv_af_app']==1].groupby(['staff_action'])['staff_action'].count()/df[df['is_rejected_inv_af_app']==1].shape[0]).sort_values(ascending=False).head(5)).plot.pie(y='staff_action', figsize=(5,5))

# COMMAND ----------

r = (df[df['is_sar90'] == 1].groupby(['staff_action'])['staff_action'].count()/df[df['is_sar90'] == 1].shape[0]).sort_values(ascending=False)
r.sum(), r

# COMMAND ----------

((df[df['is_sar90']==1].groupby(['staff_action'])['staff_action'].count()/df[df['is_sar90']==1].shape[0]).sort_values(ascending=False).head(2)).plot.pie(y='staff_action', figsize=(5,5))

# COMMAND ----------

df.groupby(['is_rejected_inv_af_app'])['is_sar90'].mean()

# COMMAND ----------

df.groupby(['is_rejected_inv_af_app'])['is_sar90'].sum()/df['is_sar90'].sum()


# COMMAND ----------

df['month_year'] = df['company_created_on'].apply(lambda x: str(x.date())[:7])

# COMMAND ----------

df[df['decision_outcome'].isin(['mkyc_approved', 'mkyc_rejected'])].groupby(['month_year'])['is_rejected_inv_at_onb'].mean().plot()

# COMMAND ----------

df[df['decision_outcome'].isin(['mkyc_approved', 'auto_approved'])].groupby(['month_year'])['is_rejected_inv_af_app'].mean().plot()

# COMMAND ----------

df[df['decision_outcome'].isin(['mkyc_approved', 'auto_approved'])].groupby(['month_year'])['is_sar90'].mean().plot()

# COMMAND ----------

(df[df['decision_outcome'].isin(['mkyc_approved', 'auto_approved']) & (df['is_rejected_inv_af_app']==1)].groupby(['month_year'])['is_sar90'].sum()/df[df['decision_outcome'].isin(['mkyc_approved', 'auto_approved'])].groupby(['month_year'])['is_sar90'].sum()).plot()

# COMMAND ----------

