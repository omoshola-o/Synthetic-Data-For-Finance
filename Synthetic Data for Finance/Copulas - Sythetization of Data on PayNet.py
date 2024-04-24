# Databricks notebook source
# MAGIC %pip install openpyxl
# MAGIC %pip install shap
# MAGIC %pip install scikit-learn

# COMMAND ----------

import pandas as pd
from functools import reduce
from pyspark.sql import DataFrame
from pyspark.sql.functions import col
from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_replace
from pyspark.sql.functions import expr, lit
from sklearn import model_selection
from sklearn import ensemble
import pyspark.sql.functions as F

# COMMAND ----------

def read_excel_sheets(file_path, sheet_names):
    dfs = {}
    xl = pd.ExcelFile(file_path)

    for sheet in sheet_names:
        df = xl.parse(sheet)
        
        # add PD_status column based on sheet name
        if sheet == "Performing":
            df['PD_status'] = "performing"
        elif sheet == "PreviouslyDefaulted":
            df['PD_status'] = "defaulted"
        
        dfs[sheet] = spark.createDataFrame(df)

    return dfs

sheet_names = ["Performing", "PreviouslyDefaulted"]
dfs = read_excel_sheets("/dbfs/FileStore/AbsolutePD_Client_Deliverable_1013_202309_Contract_Level.xlsx", sheet_names)

# Space for additional dfses

for name, df in dfs.items():
    print(name)
    display(df)

# COMMAND ----------

def unionAll(*dfs):
    return reduce(DataFrame.unionAll, dfs)

df_all = unionAll(dfs['Performing'], dfs['PreviouslyDefaulted'])
display(df_all)
# display(df_all.filter(col("PayNet_absolutepd_1q") == 1))

# COMMAND ----------

from pyspark.sql.functions import when
columns = ["PD_status"]

# defining a mapping for pd_status  to numerical values
pd_status_mapping = {"performing": 0, "defaulted": 1,}

# using the when and otherwise functions to create a new target_variable column
df_all = df_all.withColumn("pd_status_score", 
                   when(df_all["PD_status"] == "performing", pd_status_mapping["performing"])
                   .otherwise("defaulted"))

# show.
display(df_all)

# COMMAND ----------

# df_all = spark.createDataFrame(df_all)
df_imp = df_all.select('cur_bal_member_lender','total_gross_orig_receivable_amt', 'annualized_scheduled_payments', 'number_of_lenders', 'cnt_0130_dpd', 'cnt_3160_dpd', 'cnt_6190_dpd','cnt_over90_dpd', 'contract_balance','industry_segment', 'state','PD_status')

# COMMAND ----------

spark.conf.set("spark.sql.legacy.allowUntypedScalaUDF", "true")
spark.conf.set("spark.sql.execution.arrow.enabled","true")

# Increase the iteration count for SVD
spark.conf.set("spark.ml.linalg.svdMaxIter", "100")

# COMMAND ----------

df_imp.columns
df_imp.display()

# COMMAND ----------

from pyspark.sql.functions import col, isnan

# Iterate over each column and count the null values
null_counts = []
for col_name in df_imp.columns:
    null_count = df_imp.where(col(col_name).isNull() | isnan(col(col_name))).count()
    null_counts.append((col_name, null_count))

# Convert the results to a DataFrame for better visualization
null_counts_df = spark.createDataFrame(null_counts, ["Column", "Null Count"])

# Show the DataFrame containing null counts
null_counts_df.show(60)

# COMMAND ----------

df_imp.printSchema()

# COMMAND ----------

df_imp.columns

# COMMAND ----------

# Assuming df_all is your DataFrame containing the dataset
column_names = df_imp.columns

# Print column names and their indexes
for idx, col_name in enumerate(column_names):
    print(f"Index: {idx}, Name: {col_name}")

# COMMAND ----------

import csv 
from scipy.stats import norm
import numpy as np

# Convert DataFrame to CSV file
df_imp.toPandas().to_csv('df_imp.csv', index=False)

with open('df_imp.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    fields = next(reader)  # Reads header row as a list
    rows = list(reader)    # Reads all subsequent rows as a list of lists

# -- group by (industry_segment, state, PD_status)

groupCount = {}
groupList = {}
for obs in rows:
    group = obs[9] + "\t" + obs[10] + "\t" + obs[11]
    if group in groupCount:
        cnt = groupCount[group]
        groupList[(group, cnt)] = (obs[0], obs[1], obs[2], obs[3], obs[4], obs[5], obs[6], obs[7], obs[8])
        groupCount[group] += 1
    else:
        cnt = 0
        groupList[(group, cnt)] = (obs[0], obs[1], obs[2], obs[3], obs[4], obs[5], obs[6], obs[7], obs[8])
        groupCount[group] = 1

print(groupList)

# COMMAND ----------

# Understanding
# The key is the group identifier concatenated from the 1st, 4th and 5th indexes.
# The value contains the 0th, 2nd, 3rd and 6th indexes for each row of that group.

# COMMAND ----------

df_imp.columns

# COMMAND ----------

from scipy.stats import norm
from scipy import stats
import statsmodels.api as sm

#-- generate synthetic data customized to each group (Gaussian copula)

seed = 453
np.random.seed(seed)
OUT=open("paynet_synth.txt","w")
for group in groupCount:
    nobs = groupCount[group]
    cur_bal_member_lender = []
    total_gross_orig_receivable_amt = []
    annualized_scheduled_payments = []
    number_of_lenders = []
    cnt_0130_dpd = []
    cnt_3160_dpd = []
    cnt_6190_dpd = []
    cnt_over90_dpd = []
    contract_balance = []


    for cnt in range(nobs):
      features = groupList[(group,cnt)]   
      cur_bal_member_lender.append(float(features[0]))  
      total_gross_orig_receivable_amt.append(float(features[1]))
      annualized_scheduled_payments.append(float(features[2]))
      number_of_lenders.append(float(features[3]))
      cnt_0130_dpd.append(float(features[4]))
      cnt_3160_dpd.append(float(features[5]))
      cnt_6190_dpd.append(float(features[6]))
      cnt_over90_dpd.append(float(features[7]))
      contract_balance.append(float(features[8]))

    mu = [np.mean(cur_bal_member_lender), np.mean(cur_bal_member_lender),
          np.mean(total_gross_orig_receivable_amt), np.mean(total_gross_orig_receivable_amt),
          np.mean(annualized_scheduled_payments), np.mean(annualized_scheduled_payments),
          np.mean(number_of_lenders), np.mean(number_of_lenders),
          np.mean(cnt_0130_dpd), np.mean(cnt_0130_dpd),
          np.mean(cnt_3160_dpd), np.mean(cnt_3160_dpd),
          np.mean(cnt_6190_dpd), np.mean(cnt_6190_dpd),
          np.mean(cnt_over90_dpd), np.mean(cnt_over90_dpd),
          np.mean(contract_balance), np.mean(contract_balance)]

    zero = [0, 0, 0, 0, 0, 0, 0, 0, 0 ]
    
    z = np.stack((cur_bal_member_lender,total_gross_orig_receivable_amt,annualized_scheduled_payments, number_of_lenders, cnt_0130_dpd, cnt_3160_dpd, cnt_6190_dpd,cnt_over90_dpd,
 contract_balance,), axis = 0)

    # cov = np.cov(z)
    corr = np.corrcoef(z) # correlation matrix for Gaussian copula for this group

    # Check and handle NaN values in the correlation matrix
    corr[np.isnan(corr)] = 0

    # Add a small constant to the diagonal of the correlation matrix to handle singular matrices
    epsilon = 1e-6
    corr += np.eye(corr.shape[0]) * epsilon

    print("------------------")
    print("\n\nGroup: ",group,"[",cnt,"obs ]\n")  

    # Print the mean values and correlation matrix

    print("mean cur_bal_member_lender: %1.2f\nmean total_gross_orig_receivable_amt: %1.2f\nmean annualized_scheduled_payments: %1.2f\nmean number_of_lenders: %1.2f\nmean cnt_0130_dpd: %1.2f\nmean cnt_3160_dpd: %1.2f\nmean cnt_6190_dpd: %1.2f\nmean cnt_over90_dpd: %1.2f\nmean contract_balance: %1.2f\n"
          %(mu[0],mu[1],mu[2],mu[3],mu[4],mu[5],mu[6],mu[7],mu[8]))
   
    print("correlation matrix:\n")
    print(np.corrcoef(z), "\n")

    nobs_synth = nobs  # number of synthetic obs to create for this group
    gfg = np.random.multivariate_normal(zero, corr, nobs_synth)
    g_cur_bal_member_lender = gfg[:,0]
    g_total_gross_orig_receivable_amt = gfg[:,1]
    g_annualized_scheduled_payments = gfg[:,2]
    g_number_of_lenders = gfg[:,3]
    g_cnt_0130_dpd = gfg[:,4]
    g_cnt_3160_dpd = gfg[:,5]
    g_cnt_6190_dpd = gfg[:,6]
    g_cnt_over90_dpd = gfg[:,7]
    g_contract_balance = gfg[:,8]

     # generate nobs_synth observations for this group
    print("synthetic observations:\n")
    for k in range(nobs_synth):   
        u_cur_bal_member_lender = norm.cdf(g_cur_bal_member_lender[k])
        u_total_gross_orig_receivable_amt = norm.cdf(g_total_gross_orig_receivable_amt[k])
        u_annualized_scheduled_payments = norm.cdf(g_annualized_scheduled_payments[k])
        u_number_of_lenders = norm.cdf(g_number_of_lenders[k])
        u_cnt_0130_dpd = norm.cdf(g_cnt_0130_dpd[k])
        u_cnt_3160_dpd = norm.cdf(g_cnt_3160_dpd[k])
        u_cnt_6190_dpd = norm.cdf(g_cnt_6190_dpd[k])
        u_cnt_over90_dpd = norm.cdf(cnt_over90_dpd[k])
        u_contract_balance = norm.cdf(contract_balance[k])

        s_u_cur_bal_member_lender = np.quantile(cur_bal_member_lender, u_cur_bal_member_lender)                # synthesized  
        s_total_gross_orig_receivable_amt = np.quantile(total_gross_orig_receivable_amt, u_total_gross_orig_receivable_amt)  # synthesized 
        s_annualized_scheduled_payments = np.quantile(annualized_scheduled_payments, u_annualized_scheduled_payments)                # synthesized 
        s_number_of_lenders = np.quantile(number_of_lenders, u_number_of_lenders)
        s_cnt_0130_dpd = np.quantile(cnt_0130_dpd, u_cnt_0130_dpd)
        s_cnt_3160_dpd = np.quantile(cnt_3160_dpd, u_cnt_3160_dpd)
        s_cnt_6190_dpd = np.quantile(cnt_6190_dpd, u_cnt_6190_dpd)
        s_cnt_over90_dpd = np.quantile(cnt_over90_dpd, u_cnt_over90_dpd)
        s_contract_balance = np.quantile(contract_balance, u_contract_balance)

        line = group + "\t" + str(s_u_cur_bal_member_lender) + "\t" + str(s_total_gross_orig_receivable_amt) + "\t" + str(s_annualized_scheduled_payments) + "\t" + str(s_number_of_lenders) + "\n" + str(s_cnt_0130_dpd) + "\t" + str(s_cnt_3160_dpd) + "\t" + str(s_cnt_over90_dpd) + "\t" + str(s_contract_balance) + "\n"
        OUT.write(line)
        print("%3d. %d %d %d %d %d %d %d %d" %(k, s_u_cur_bal_member_lender, s_total_gross_orig_receivable_amt, s_annualized_scheduled_payments, s_number_of_lenders, s_cnt_0130_dpd, s_cnt_3160_dpd, s_cnt_over90_dpd, s_contract_balance))
OUT.close()

# COMMAND ----------

import numpy as np
from scipy.stats import norm

#-- Generate synthetic data customized to each group (Gaussian copula)

seed = 453
np.random.seed(seed)
synthetic_data = []

for group in groupCount:
    nobs = groupCount[group]
    cur_bal_member_lender = []
    total_gross_orig_receivable_amt = []
    annualized_scheduled_payments = []
    number_of_lenders = []
    cnt_0130_dpd = []
    cnt_3160_dpd = []
    cnt_6190_dpd = []
    cnt_over90_dpd = []
    contract_balance = []

    for cnt in range(nobs):
        features = groupList[(group,cnt)]   
        cur_bal_member_lender.append(float(features[0]))  
        total_gross_orig_receivable_amt.append(float(features[1]))
        annualized_scheduled_payments.append(float(features[2]))
        number_of_lenders.append(float(features[3]))
        cnt_0130_dpd.append(float(features[4]))
        cnt_3160_dpd.append(float(features[5]))
        cnt_6190_dpd.append(float(features[6]))
        cnt_over90_dpd.append(float(features[7]))
        contract_balance.append(float(features[8]))

    mu = [np.mean(cur_bal_member_lender), np.mean(cur_bal_member_lender),
          np.mean(total_gross_orig_receivable_amt), np.mean(total_gross_orig_receivable_amt),
          np.mean(annualized_scheduled_payments), np.mean(annualized_scheduled_payments),
          np.mean(number_of_lenders), np.mean(number_of_lenders),
          np.mean(cnt_0130_dpd), np.mean(cnt_0130_dpd),
          np.mean(cnt_3160_dpd), np.mean(cnt_3160_dpd),
          np.mean(cnt_6190_dpd), np.mean(cnt_6190_dpd),
          np.mean(cnt_over90_dpd), np.mean(cnt_over90_dpd),
          np.mean(contract_balance), np.mean(contract_balance)]

    zero = [0, 0, 0, 0, 0, 0, 0, 0, 0 ]
    
    z = np.stack((cur_bal_member_lender,total_gross_orig_receivable_amt,
                  annualized_scheduled_payments, number_of_lenders,
                  cnt_0130_dpd, cnt_3160_dpd, cnt_6190_dpd, cnt_over90_dpd,
                  contract_balance), axis = 0)

    # cov = np.cov(z)
    corr = np.corrcoef(z) # correlation matrix for Gaussian copula for this group

    # Check and handle NaN values in the correlation matrix
    corr[np.isnan(corr)] = 0

    # Add a small constant to the diagonal of the correlation matrix to handle singular matrices
    epsilon = 1e-6
    corr += np.eye(corr.shape[0]) * epsilon

    # Attempt SVD with robustness

    print("------------------")
    print("\n\nGroup: ", group, "[", cnt, "obs ]\n")  

    # Print the mean values and correlation matrix

    print("mean cur_bal_member_lender: %1.2f\nmean total_gross_orig_receivable_amt: %1.2f\nmean annualized_scheduled_payments: %1.2f\nmean number_of_lenders: %1.2f\nmean cnt_0130_dpd: %1.2f\nmean cnt_3160_dpd: %1.2f\nmean cnt_6190_dpd: %1.2f\nmean cnt_over90_dpd: %1.2f\nmean contract_balance: %1.2f\n"
          %(mu[0],mu[1],mu[2],mu[3],mu[4],mu[5],mu[6],mu[7],mu[8]))
   
    print("correlation matrix:\n")
    print(np.corrcoef(z), "\n")

    nobs_synth = nobs  # number of synthetic obs to create for this group
    gfg = np.random.multivariate_normal(zero, corr, nobs_synth)
    g_cur_bal_member_lender = gfg[:,0]
    g_total_gross_orig_receivable_amt = gfg[:,1]
    g_annualized_scheduled_payments = gfg[:,2]
    g_number_of_lenders = gfg[:,3]
    g_cnt_0130_dpd = gfg[:,4]
    g_cnt_3160_dpd = gfg[:,5]
    g_cnt_6190_dpd = gfg[:,6]
    g_cnt_over90_dpd = gfg[:,7]
    g_contract_balance = gfg[:,8]

     # Generate nobs_synth observations for this group
    print("synthetic observations:\n")
    for k in range(nobs_synth):   
        u_cur_bal_member_lender = norm.cdf(g_cur_bal_member_lender[k])
        u_total_gross_orig_receivable_amt = norm.cdf(g_total_gross_orig_receivable_amt[k])
        u_annualized_scheduled_payments = norm.cdf(g_annualized_scheduled_payments[k])
        u_number_of_lenders = norm.cdf(g_number_of_lenders[k])
        u_cnt_0130_dpd = norm.cdf(g_cnt_0130_dpd[k])
        u_cnt_3160_dpd = norm.cdf(g_cnt_3160_dpd[k])
        u_cnt_6190_dpd = norm.cdf(g_cnt_6190_dpd[k])
        u_cnt_over90_dpd = norm.cdf(cnt_over90_dpd[k])
        u_contract_balance = norm.cdf(contract_balance[k])

        s_u_cur_bal_member_lender = np.quantile(cur_bal_member_lender, u_cur_bal_member_lender)                # synthesized  
        s_total_gross_orig_receivable_amt = np.quantile(total_gross_orig_receivable_amt, u_total_gross_orig_receivable_amt)  # synthesized 
        s_annualized_scheduled_payments = np.quantile(annualized_scheduled_payments, u_annualized_scheduled_payments)                # synthesized 
        s_number_of_lenders = np.quantile(number_of_lenders, u_number_of_lenders)
        s_cnt_0130_dpd = np.quantile(cnt_0130_dpd, u_cnt_0130_dpd)
        s_cnt_3160_dpd = np.quantile(cnt_3160_dpd, u_cnt_3160_dpd)
        s_cnt_6190_dpd = np.quantile(cnt_6190_dpd, u_cnt_6190_dpd)
        s_cnt_over90_dpd = np.quantile(cnt_over90_dpd, u_cnt_over90_dpd)
        s_contract_balance = np.quantile(contract_balance, u_contract_balance)

        synthetic_data.append((group, s_u_cur_bal_member_lender, s_total_gross_orig_receivable_amt,
                               s_annualized_scheduled_payments, s_number_of_lenders,
                               s_cnt_0130_dpd, s_cnt_3160_dpd, s_cnt_over90_dpd, s_contract_balance))

# Create a DataFrame from the synthetic data
columns = ['cur_bal_member_lender', 'total_gross_orig_receivable_amt',
           'annualized_scheduled_payments', 'number_of_lenders',
           'cnt_0130_dpd', 'cnt_3160_dpd', 'cnt_over90_dpd', 'contract_balance']

df_synth = spark.createDataFrame(synthetic_data, columns)

# Display the synthetic data
display(df_synth)

# COMMAND ----------

display(df_imp.where((F.col("industry_segment")== "RETL") & (F.col("state")== "MN")))
