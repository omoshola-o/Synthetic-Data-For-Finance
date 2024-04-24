# Databricks notebook source
# MAGIC %pip install Faker
# MAGIC %pip install openpyxl

# COMMAND ----------

import pyspark
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import col, sum
from pyspark.sql.functions import col, isnull
import matplotlib.pyplot as plt
from faker import Faker
from pyspark.sql.types import StringType, DoubleType
from pyspark.sql.functions import udf


# COMMAND ----------

spark = SparkSession.builder \
  .master('local[1]') \
  .appName('cra_mvp') \
  .getOrCreate()

df = spark.read.table('hive_metastore.default.crs_transformed')
df.printSchema()
display(df)

# COMMAND ----------

df.columns

# COMMAND ----------

fake = Faker()

# Customer name
customer_name = udf(lambda: fake.name(), StringType())  

# Financial columns  
tnw = udf(lambda: fake.random_int(min=1, max=500), DoubleType())
profits_perc = udf(lambda: fake.random_int(min=1, max=100), DoubleType())
positive_wc = udf(lambda: fake.random_int(min=0, max=1), DoubleType())
tnw_exposure = udf(lambda: fake.pydecimal(left_digits=5, right_digits=2, positive=True), DoubleType())  
fleet_size = udf(lambda: fake.random_int(min=1, max=500), DoubleType())
total_exposure = udf(lambda: fake.pydecimal(left_digits=8, right_digits=2, positive=True), DoubleType())
revenue = udf(lambda: fake.pydecimal(left_digits=8, right_digits=2, positive=True), DoubleType())
ebit = udf(lambda: fake.pydecimal(left_digits=5, right_digits=2, positive=True), DoubleType())
depreciation = udf(lambda: fake.pydecimal(left_digits=5, right_digits=2, positive=True), DoubleType())
net_profit = udf(lambda: fake.pydecimal(left_digits=5, right_digits=2, positive=True), DoubleType())
fixed_assets = udf(lambda: fake.pydecimal(left_digits=8, right_digits=2, positive=True), DoubleType())  
intangible_assets = udf(lambda: fake.pydecimal(left_digits=8, right_digits=2, positive=True), DoubleType())
current_assets = udf(lambda: fake.pydecimal(left_digits=8, right_digits=2, positive=True), DoubleType())  
tangible_net_worth = udf(lambda: fake.pydecimal(left_digits=8, right_digits=2, positive=True), DoubleType())
long_term_liab = udf(lambda: fake.pydecimal(left_digits=8, right_digits=2, positive=True), DoubleType())
long_term_credit = udf(lambda: fake.pydecimal(left_digits=8, right_digits=2, positive=True), DoubleType())
short_term_liab = udf(lambda: fake.pydecimal(left_digits=8, right_digits=2, positive=True), DoubleType())
cr_rating_score = udf(lambda: fake.random_int(min=1, max=10), DoubleType())
pmt_discipline_score = udf(lambda: fake.random_int(min=1, max=10), DoubleType())
debt_equity_ratio = udf(lambda: fake.pydecimal(left_digits=3, right_digits=2, positive=True), DoubleType()) 
debt_asset_ratio = udf(lambda: fake.pydecimal(left_digits=3, right_digits=2, positive=True), DoubleType())
current_ratio = udf(lambda: fake.pydecimal(left_digits=3, right_digits=2, positive=True), DoubleType())
return_on_assets = udf(lambda: fake.pydecimal(left_digits=5, right_digits=2, positive=True), DoubleType())

target = udf(lambda: fake.random_int(min=0, max=1), DoubleType())

df = spark.range(100).withColumn("customer_name", customer_name()) \
  .withColumn("TNW_in_MEUR", tnw()) \
  .withColumn("profits_perc_TNW", profits_perc()) \
  .withColumn("Positive_WC", positive_wc()) \
  .withColumn("TNW_to_T-Exposure", tnw_exposure()) \
  .withColumn("fleet_size", fleet_size()) \
  .withColumn("total_exposure", total_exposure()) \
  .withColumn("revenue", revenue()) \
  .withColumn("EBIT", ebit()) \
  .withColumn("depreciation", depreciation()) \
  .withColumn("net_profit", net_profit()) \
  .withColumn("fixed_assets", fixed_assets()) \
  .withColumn("intangible_assets", intangible_assets()) \
  .withColumn("current_assets", current_assets()) \
  .withColumn("tangible_net_worth", tangible_net_worth()) \
  .withColumn("long_term_liab", long_term_liab()) \
  .withColumn("long_term_credit", long_term_credit()) \
  .withColumn("short_term_liab", short_term_liab()) \
  .withColumn("CR_Rating_score", cr_rating_score()) \
  .withColumn("pmt_discipline_score", pmt_discipline_score()) \
  .withColumn("debt_equity_ratio", debt_equity_ratio()) \
  .withColumn("debt_asset_ratio", debt_asset_ratio()) \
  .withColumn("current_ratio", current_ratio()) \
  .withColumn("return_on_assets", return_on_assets()) \
  .withColumn("target_variable", target())

df.display()

# COMMAND ----------

import pandas as pd
import numpy as np
from faker import Faker

num_records = 20000


#This is the best typ of the
data = {
  'customer_name': [Faker().company() for _ in range(num_records)],
  'credit_rating': [np.random.choice(['A','B','C','D']) for _ in range(num_records)],
  'years_in_business': [str(np.random.randint(1,50)) for _ in range(num_records)],
  'TNW_in_MEUR': [str(np.random.randint(100,10000)) for _ in range(num_records)],
  'profits_perc_TNW': [str(np.random.randint(1,100)) for _ in range(num_records)],
  'positive_working_capital': [np.random.choice(['Yes','No']) for _ in range(num_records)],
  'TNW_to_total_exposure': [str(np.random.uniform(0,1)) for _ in range(num_records)],
  'legal_form': [np.random.choice(['LLC','Corporation','Partnership']) for _ in range(num_records)],
  'fundation_year': [str(np.random.randint(1900,2020)) for _ in range(num_records)],
  'fundation_month': [str(np.random.randint(1,13)) for _ in range(num_records)],
  'fundation_day': [str(np.random.randint(1,29)) for _ in range(num_records)],
  'financing_currency': [np.random.choice(['EUR','USD','GBP']) for _ in range(num_records)],
  'fleet_size': [str(np.random.randint(1,500)) for _ in range(num_records)],
  'fleet_own_trucks': [str(np.random.randint(0,500)) for _ in range(num_records)],
  'fleet_vfs_trucks': [str(np.random.randint(0,500)) for _ in range(num_records)],
  'fleet_other_trucks': [str(np.random.randint(0,500)) for _ in range(num_records)],
  'fleet_own_EC': [str(np.random.randint(0,500)) for _ in range(num_records)],
  'fleet_vfs_EC': [str(np.random.randint(0,500)) for _ in range(num_records)],
  'fleet_other_EC': [str(np.random.randint(0,500)) for _ in range(num_records)],
  'Offer_number_1': [str(np.random.randint(10000,99999)) for _ in range(num_records)],
  'Offer_number_2': [str(np.random.randint(10000,99999)) for _ in range(num_records)],
  'Offer_number_3': [str(np.random.randint(10000,99999)) for _ in range(num_records)],
  'financial_performance_currency': [np.random.choice(['EUR','USD','GBP']) for _ in range(num_records)],
  'type_of_request': [np.random.choice(['New','Renewal','Increase']) for _ in range(num_records)],
  'flag_domestic_use': [np.random.choice(['Yes','No']) for _ in range(num_records)],
  'vfs_customer': [np.random.choice(['Yes','No']) for _ in range(num_records)],
  'vfs_known_since': [str(np.random.randint(2010,2022)) for _ in range(num_records)],
  'payment_discipline': [np.random.choice(['Excellent','Good','Fair','Poor']) for _ in range(num_records)],
  'total_exposure': [str(np.random.randint(1000,1000000)) for _ in range(num_records)],
  'quote_id': [str(np.random.randint(1000000,9999999)) for _ in range(num_records)],
  'current_period': [str(np.random.randint(2010,2023)) for _ in range(num_records)],
  'current_period_number_months': [str(np.random.randint(1,13)) for _ in range(num_records)],
  'current_period_revenue': [str(np.random.randint(1000,10000000)) for _ in range(num_records)],
  'current_period_EBIT': [str(np.random.randint(100,1000000)) for _ in range(num_records)],
  'current_period_depreciation': [str(np.random.randint(10,100000)) for _ in range(num_records)],
  'current_period_net_profit': [str(np.random.randint(10,1000000)) for _ in range(num_records)],
  'current_period_fixed_assets': [str(np.random.randint(100,10000000)) for _ in range(num_records)],
  'current_period_intangible_assets': [str(np.random.randint(10,1000000)) for _ in range(num_records)],
  'current_period_current_assets': [str(np.random.randint(100,10000000)) for _ in range(num_records)],
  'current_period_tangible_net_worth': [str(np.random.randint(100,10000000)) for _ in range(num_records)],
  'current_period_long_term_liabilities': [str(np.random.randint(0,1000000)) for _ in range(num_records)],
  'current_period_long_term_credit': [str(np.random.randint(0,1000000)) for _ in range(num_records)],
  'current_period_short_term_liabilities': [str(np.random.randint(1000,10000000)) for _ in range(num_records)],
  'current_period_short_term_credit': [str(np.random.randint(0,1000000)) for _ in range(num_records)],
  'current_period_off_balance_liabilities': [str(np.random.randint(0,1000000)) for _ in range(num_records)],
  'last_period': [str(np.random.randint(2009,2022)) for _ in range(num_records)],
  'last_period_number_months': [str(np.random.randint(1,13)) for _ in range(num_records)],
  'last_period_revenue': [str(np.random.randint(1000,10000000)) for _ in range(num_records)],
  'last_period_EBIT': [str(np.random.randint(100,1000000)) for _ in range(num_records)],
  'last_period_depreciation': [str(np.random.randint(10,100000)) for _ in range(num_records)],
  'last_period_net_profit': [str(np.random.randint(10,1000000)) for _ in range(num_records)],
  'last_period_fixed_assets': [str(np.random.randint(100,10000000)) for _ in range(num_records)],
  'last_period_intangible_assets': [str(np.random.randint(10,1000000)) for _ in range(num_records)],
  'last_period_current_assets': [str(np.random.randint(100,10000000)) for _ in range(num_records)],
  'last_period_tangible_net_worth': [str(np.random.randint(100,10000000)) for _ in range(num_records)],
  'last_period_long_term_liabilities': [str(np.random.randint(0,1000000)) for _ in range(num_records)],
  'last_period_long_term_credit': [str(np.random.randint(0,1000000)) for _ in range(num_records)],
  'last_period_short_term_liabilities': [str(np.random.randint(1000,10000000)) for _ in range(num_records)],
  'last_period_short_term_credit': [str(np.random.randint(0,1000000)) for _ in range(num_records)],
  'last_period_off_balance_liabilities': [str(np.random.randint(0,1000000)) for _ in range(num_records)],
  'period_before_last_period': [str(np.random.randint(2008,2021)) for _ in range(num_records)],
  'period_before_last_period_number_months': [str(np.random.randint(1,13)) for _ in range(num_records)],
  'period_before_last_period_revenue': [str(np.random.randint(1000,10000000)) for _ in range(num_records)],
  'period_before_last_period_EBIT': [str(np.random.randint(100,1000000)) for _ in range(num_records)],
  'period_before_last_period_depreciation': [str(np.random.randint(10,100000)) for _ in range(num_records)],
  'period_before_last_period_net_profit': [str(np.random.randint(10,1000000)) for _ in range(num_records)],
  'period_before_last_period_fixed_assets': [str(np.random.randint(100,10000000)) for _ in range(num_records)],
  'period_before_last_period_intangible_assets': [str(np.random.randint(10,1000000)) for _ in range(num_records)],
  'period_before_last_period_current_assets': [str(np.random.randint(100,10000000)) for _ in range(num_records)],
  'period_before_last_period_tangible_net_worth': [str(np.random.randint(100,10000000)) for _ in range(num_records)],
  'period_before_last_period_long_term_liabilities': [str(np.random.randint(0,1000000)) for _ in range(num_records)],
  'period_before_last_period_long_term_credit': [str(np.random.randint(0,1000000)) for _ in range(num_records)],
  'period_before_last_period_short_term_liabilities': [str(np.random.randint(1000,10000000)) for _ in range(num_records)],
  'period_before_last_period_short_term_credit': [str(np.random.randint(0,1000000)) for _ in range(num_records)],
  'period_before_last_period_off_balance_liabilities': [str(np.random.randint(0,1000000)) for _ in range(num_records)],
  'market': [np.random.choice(['US','Europe','Asia']) for _ in range(num_records)],
  'type_of_data': ['Synthetic' for _ in range(num_records)]
}

df = pd.DataFrame(data)
df.display()

# # Output to CSV
# df.to_csv('synthetic_data.csv', index=False)
