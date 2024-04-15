# Databricks notebook source
# DBTITLE 1,KYC Risk Engine Training Set Up
# MAGIC %md 
# MAGIC This notebook sets up the environment for the KYC onboarding risk engine training repo
# MAGIC

# COMMAND ----------

# MAGIC %pip install --upgrade pip

# COMMAND ----------

# MAGIC %pip install /Workspace/Shared/Decisioning/Strawberries/packages/ds_python_utils-0.1.14-py3-none-any.whl
# MAGIC %pip install /Workspace/Shared/Decisioning/Strawberries/packages/kyc_decisioning_common-0.5.1-py3-none-any.whl
# MAGIC %pip install /Workspace/Shared/Decisioning/Strawberries/packages/feature_extractors-0.2.16-py3-none-any.whl
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %pip install mlflow==1.30.0
# MAGIC %pip install hyperopt
# MAGIC %pip install category-encoders==2.5.1.post0
# MAGIC %pip install xgboost==1.3.3
# MAGIC %pip install pandas==1.3.5
# MAGIC %pip install shap
# MAGIC %pip install cloudpickle==2.2.0
# MAGIC %pip install rollbar==0.15.2
# MAGIC %pip install scikit-learn==1.2.1

# COMMAND ----------

