# Databricks notebook source
# MAGIC %pip install openpyxl
# MAGIC %pip install shap
# MAGIC %pip install scikit-learn
# MAGIC %pip install sdv
# MAGIC %pip install copulas
# MAGIC %pip install Faker
# MAGIC %pip install sdmetrics
# MAGIC %pip install fitter
# MAGIC %pip install sdv
# MAGIC %pip install synthpop
# MAGIC

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

columns_to_drop = ['as_of_date','PayNet_id','region', 'member_lender_branch','member_lender_business_unit', 'member_lender_business_unit', 'member_lender_risk_rating', 'default_4q_flg', 'default_8q_flg', 'borrower_high_credit',
'naics_code', 'primary_collateral_type', 'most_recent_0130', 'most_recent_3160', 'most_recent_6190', 'most_recent_over90_dpd', 'sic_code', 'PayNet_absolutepd_1q','PayNet_absolutepd_2q', 'PayNet_absolutepd_3q', 'PayNet_absolutepd_4q','PayNet_absolutepd_5q', 'PayNet_absolutepd_6q', 'PayNet_absolutepd_7q', 'PayNet_absolutepd_8q','contract_sbu', 'Portfolio', 'oldest_contract_start_date', 'newest_contract_start_date','contract_collateral', 'industry_segment','state','PayNet_risk_factor_1', 'PayNet_risk_factor_2', 'PayNet_risk_factor_3','cur_bal_0130_dpd', 'cur_bal_3160_dpd', 'cur_bal_6190_dpd','cur_bal_over90_dpd', 'amount_past_due_0130', 'amount_past_due_3160', 'amount_past_due_6190', 'amount_past_due_over90', 'est_exposure_at_default_4q', 'est_exposure_at_default_8q']

df_all = df_all.drop(*columns_to_drop)
df_all.display()

# COMMAND ----------

df_all.printSchema()
df_all.columns
df_all.display()


# COMMAND ----------

df_all = df_all.toPandas()  
df_all = df_all.groupby("PD_status", group_keys=False).apply(lambda x:x.sample(frac=1))
df_all.sort_values(['customer_number', 'contract_number'], inplace=True)
df_all.display()

# COMMAND ----------

df_all.columns

# COMMAND ----------

df_all["PD_status"].value_counts(True)

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# COMMAND ----------

df_data= df_all.filter(['customer_number', 'customer_name', 'cur_bal_member_lender',
       'total_gross_orig_receivable_amt', 'annualized_scheduled_payments',
       'number_of_lenders', 'cnt_0130_dpd', 'cnt_3160_dpd', 'cnt_6190_dpd',
       'cnt_over90_dpd', 'contract_balance', 'contract_number', 'PD_status'])
df_data['cur_bal_member_lender'] = df_data['cur_bal_member_lender'].astype('float64')
df_data['total_gross_orig_receivable_amt'] = df_data['total_gross_orig_receivable_amt'].astype('float64')
df_data['annualized_scheduled_payments'] = df_data['annualized_scheduled_payments'].astype('float64')
df_data['number_of_lenders'] = df_data['number_of_lenders'].astype('int64')
df_data['cnt_0130_dpd'] = df_data['cnt_0130_dpd'].astype('int64')
df_data['cnt_3160_dpd'] = df_data['cnt_3160_dpd'].astype('int64')
df_data['cnt_6190_dpd'] = df_data['cnt_6190_dpd'].astype('int64')
df_data['cnt_over90_dpd'] = df_data['cnt_over90_dpd'].astype('int64')
df_data['contract_balance'] = df_data['contract_balance'].astype('float64')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Distribution of what is the distribution of the features we want to generate in the syntheticdataset

# COMMAND ----------

import pandas as pd
import scipy.stats as stats
import numpy as np

# List of distributions to check
distributions = ['cauchy', 'chi2', 'expon', 'exponpow', 'gamma', 'lognorm', 'norm', 'powerlaw', 'rayleigh', 'uniform','beta','truncnorm']


# Create an empty DataFrame to store the results
results = pd.DataFrame(columns=['Distribution', 'Column', 'Sum of Square Error', 'p-value'])

# Iterate over each column in the dataset with numeric data types
for column in df_data.select_dtypes(include=['float64', 'int64']).columns:
    # Extract the column data and remove any missing values (NaNs)
    column_df_data = df_data[column].dropna()

    # Iterate over each distribution in the list
    for distribution in distributions:
        # Fit the distribution to the column data and get the parameters
        params = getattr(stats, distribution).fit(column_df_data)

        # Calculate the sum of square error (SSE) by performing the Kolmogorov-Smirnov test
        sse = stats.kstest(column_df_data, distribution, args=params)[0]

        # Perform the Kolmogorov-Smirnov test again to get the p-value
        p_value = stats.kstest(column_df_data, distribution, args=params)[1]

        # Append the results to the DataFrame
        results = results.append({'Distribution': distribution, 'Column': column,
                                  'Sum of Square Error': sse, 'p-value': p_value}, ignore_index=True)

# Sort the results based on p-value in ascending order
results = results.sort_values(by='p-value')

# Select the best distribution for each column by grouping and selecting the first row
best_distributions = results.groupby('Column').first()

# Format p-values with 4 decimal places
best_distributions['p-value'] = best_distributions['p-value'].apply(lambda x: format(x, '.4f'))

# Create a DataFrame with the best distributions for each column
pd.DataFrame(best_distributions)

# COMMAND ----------

# df_data= df_all.filter(['customer_number', 'cur_bal_member_lender',
#        'total_gross_orig_receivable_amt', 'annualized_scheduled_payments',
#        'number_of_lenders', 'cnt_0130_dpd', 'cnt_3160_dpd', 'cnt_6190_dpd', 'contract_balance', 'PD_status'])
# df_data['cur_bal_member_lender'] = df_data['cur_bal_member_lender'].astype('float64')
# df_data['total_gross_orig_receivable_amt'] = df_data['total_gross_orig_receivable_amt'].astype('float64')
# df_data['annualized_scheduled_payments'] = df_data['annualized_scheduled_payments'].astype('float64')
# df_data['number_of_lenders'] = df_data['number_of_lenders'].astype('int64')
# df_data['cnt_0130_dpd'] = df_data['cnt_0130_dpd'].astype('int64')
# df_data['cnt_3160_dpd'] = df_data['cnt_3160_dpd'].astype('int64')
# df_data['cnt_6190_dpd'] = df_data['cnt_6190_dpd'].astype('int64')
# df_data['contract_balance'] = df_data['contract_balance'].astype('float64')

# COMMAND ----------

data_to_synthesize = df_data[['cur_bal_member_lender','total_gross_orig_receivable_amt', 'annualized_scheduled_payments','number_of_lenders', 'cnt_0130_dpd', 'cnt_3160_dpd', 'cnt_6190_dpd','cnt_over90_dpd', 'contract_balance', ]]

# COMMAND ----------

display(data_to_synthesize)

# COMMAND ----------

data_unchanged = df_data[['customer_number', 'customer_name', 'contract_number', 'PD_status']]

# COMMAND ----------

display(data_unchanged)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Distribution of cur_bal_member_lender

# COMMAND ----------

import pandas as pd
from sklearn.datasets import load_diabetes
from fitter import Fitter, get_common_distributions

###########To get a list of all available distributions:
from fitter import get_distributions
#get_distributions()

cur_bal_member_lender = df_data['cur_bal_member_lender'].values
f = Fitter(cur_bal_member_lender,
distributions=['arcsine', 'cosine', 'expon', 'weibull_max', 'weibull_min', 'dweibull', 't', 'pareto', 'exponnorm', 'lognorm','beta',"norm", "exponweib", "weibull_max", "weibull_min", "pareto", "genextreme",'cauchy','truncnorm', 'logistic','loglaplace','lognorm','loguniform','uniform', 'gamma','gaussian_kde'])
f.fit()
f.summary()
# #Identifying the best distribution
f.get_best(method = 'sumsquare_error')
#    shape parameters (a, b) = [1.07, 774615432.549401]
#   location parameter (loc) = -3.7409851959627747
#   scale parameter (scale) = 46385930205687.164
# f.fitted_param["beta"]

# COMMAND ----------

total_gross_orig_receivable_amt = df_data['total_gross_orig_receivable_amt'].values
f = Fitter(total_gross_orig_receivable_amt,
distributions=['arcsine', 'cosine', 'expon', 'weibull_max', 'weibull_min', 'dweibull', 't', 'pareto', 'exponnorm', 'lognorm','beta',"norm", "exponweib", "weibull_max", "weibull_min", "pareto", "genextreme",'cauchy','truncnorm', 'logistic','loglaplace','lognorm','loguniform','uniform', 'gamma','gaussian_kde'])
f.fit()
f.summary()
#Identifying the best distribution
f.get_best(method = 'sumsquare_error')
   #shape parameters (a, b) = [1.07, 774615432.549401]
  #location parameter (loc) = -3.7409851959627747
  #scale parameter (scale) = 46385930205687.164
#f.fitted_param["beta"]

# COMMAND ----------

annualized_scheduled_payments = df_data['annualized_scheduled_payments'].values
f = Fitter(annualized_scheduled_payments,
distributions=['arcsine', 'cosine', 'expon', 'weibull_max', 'weibull_min', 'dweibull', 't', 'pareto', 'exponnorm', 'lognorm','beta',"norm", "exponweib", "weibull_max", "weibull_min", "pareto", "genextreme",'cauchy','truncnorm', 'logistic','loglaplace','lognorm','loguniform','uniform', 'gamma','gaussian_kde'])
f.fit()
f.summary()
#Identifying the best distribution
f.get_best(method = 'sumsquare_error')
   #shape parameters (a, b) = [1.07, 774615432.549401]
  #location parameter (loc) = -3.7409851959627747
  #scale parameter (scale) = 46385930205687.164
#f.fitted_param["beta"]

# COMMAND ----------

number_of_lenders = df_data['number_of_lenders'].values
f = Fitter(number_of_lenders,
distributions=['arcsine', 'cosine', 'expon', 'weibull_max', 'weibull_min', 'dweibull', 't', 'pareto', 'exponnorm', 'lognorm','beta',"norm", "exponweib", "weibull_max", "weibull_min", "pareto", "genextreme",'cauchy','truncnorm', 'logistic','loglaplace','lognorm','loguniform','uniform', 'gamma','gaussian_kde'])
f.fit()
f.summary()
#Identifying the best distribution
f.get_best(method = 'sumsquare_error')
   #shape parameters (a, b) = [1.07, 774615432.549401]
  #location parameter (loc) = -3.7409851959627747
  #scale parameter (scale) = 46385930205687.164
#f.fitted_param["beta"]

# COMMAND ----------

cnt_0130_dpd = df_data['cnt_0130_dpd'].values
f = Fitter(cnt_0130_dpd,
distributions=['arcsine', 'cosine', 'expon', 'weibull_max', 'weibull_min', 'dweibull', 't', 'pareto', 'exponnorm', 'lognorm','beta',"norm", "exponweib", "weibull_max", "weibull_min", "pareto", "genextreme",'cauchy','truncnorm', 'logistic','loglaplace','lognorm','loguniform','uniform', 'gamma','gaussian_kde'])
f.fit()
f.summary()
#Identifying the best distribution
f.get_best(method = 'sumsquare_error')
   #shape parameters (a, b) = [1.07, 774615432.549401]
  #location parameter (loc) = -3.7409851959627747
  #scale parameter (scale) = 46385930205687.164
#f.fitted_param["beta"]

# COMMAND ----------

cnt_3160_dpd = df_data['cnt_3160_dpd'].values
f = Fitter(cnt_3160_dpd,
distributions=['arcsine', 'cosine', 'expon', 'weibull_max', 'weibull_min', 'dweibull', 't', 'pareto', 'exponnorm', 'lognorm','beta',"norm", "exponweib", "weibull_max", "weibull_min", "pareto", "genextreme",'cauchy','truncnorm', 'logistic','loglaplace','lognorm','loguniform','uniform', 'gamma','gaussian_kde'])
f.fit()
f.summary()
#Identifying the best distribution
f.get_best(method = 'sumsquare_error')
   #shape parameters (a, b) = [1.07, 774615432.549401]
  #location parameter (loc) = -3.7409851959627747
  #scale parameter (scale) = 46385930205687.164
#f.fitted_param["beta"]

# COMMAND ----------

cnt_6190_dpd = df_data['cnt_6190_dpd'].values
f = Fitter(cnt_6190_dpd,
distributions=['arcsine', 'cosine', 'expon', 'weibull_max', 'weibull_min', 'dweibull', 't', 'pareto', 'exponnorm', 'lognorm','beta',"norm", "exponweib", "weibull_max", "weibull_min", "pareto", "genextreme",'cauchy','truncnorm', 'logistic','loglaplace','lognorm','loguniform','uniform', 'gamma','gaussian_kde'])
f.fit()
f.summary()
#Identifying the best distribution
f.get_best(method = 'sumsquare_error')
   #shape parameters (a, b) = [1.07, 774615432.549401]
  #location parameter (loc) = -3.7409851959627747
  #scale parameter (scale) = 46385930205687.164
#f.fitted_param["beta"]

# COMMAND ----------

contract_balance = df_data['contract_balance'].values
f = Fitter(contract_balance,
distributions=['arcsine', 'cosine', 'expon', 'weibull_max', 'weibull_min', 'dweibull', 't', 'pareto', 'exponnorm', 'lognorm','beta',"norm", "exponweib", "weibull_max", "weibull_min", "pareto", "genextreme",'cauchy','truncnorm', 'logistic','loglaplace','lognorm','loguniform','uniform', 'gamma','gaussian_kde'])
f.fit()
f.summary()
#Identifying the best distribution
f.get_best(method = 'sumsquare_error')
   #shape parameters (a, b) = [1.07, 774615432.549401]
  #location parameter (loc) = -3.7409851959627747
  #scale parameter (scale) = 46385930205687.164
#f.fitted_param["beta"]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create the synthesizer

# COMMAND ----------

import sdv
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.constraints import FixedCombinations
from sdv.constraints import Unique
from sdv.constraints import Unique
from sdv.constraints import FixedCombinations


# COMMAND ----------

from sdv.metadata import SingleTableMetadata
metadata = SingleTableMetadata()
df_data = data_to_synthesize
metadata.detect_from_dataframe(df_data)

# # correct metadata to use the same values as in the real data 
# metadata.update_column( column_name='customer_number', sdtype='numerical' ) 
# metadata.update_column( column_name='PD_status', sdtype='categorical' ) 

# COMMAND ----------

metadata.visualize

# COMMAND ----------

df_data.display()

# COMMAND ----------

# # save metadata to json 
# metadata.save_to_json('metadata.json')

# COMMAND ----------

python_dict = metadata.to_dict()
metadata.validate()

# COMMAND ----------

# MAGIC
# MAGIC %md
# MAGIC Creating the Synthesizer
# MAGIC A Synthesizer is an object we use to create synthetic data using machine learning.  
# MAGIC  Below are the steps:  
# MAGIC - You’ll start by creating a synthesizer based on your metadata  
# MAGIC - Next, you’ll train the synthesizer using real data. In this phase, the synthesizer will learn patterns from the real data.  
# MAGIC - Once your synthesizer is trained, you can use it to generate new, synthetic data.  
# MAGIC SDV supports multiple synthesizers, we are using Gaussian Copula Synthesizer.   
# MAGIC You can check supported synthesizers here for single table.

# COMMAND ----------

# MAGIC %md
# MAGIC ***From synthetic data we want to anonymise those features*** :  
# MAGIC - customer_number, 
# MAGIC - customer_name, 
# MAGIC - contract_number, 
# MAGIC - PD_status
# MAGIC

# COMMAND ----------

from sdv.constraints import Unique
unique_customer_constraint = Unique(column_names=['contract_number'])
from sdv.constraints import FixedCombinations
fixed_customer_constraint = FixedCombinations(column_names=['customer_number','contract_number','customer_name'])
from sdv.single_table import GaussianCopulaSynthesizer

# COMMAND ----------

# MAGIC
# MAGIC %md 
# MAGIC  Create the synthesizer

# COMMAND ----------

synthesizer = GaussianCopulaSynthesizer(metadata, enforce_min_max_values=True, enforce_rounding=True,
    numerical_distributions={
        'cur_bal_member_lender': 'gamma',
        'total_gross_orig_receivable_amt':'gamma',
        'annualized_scheduled_payments': 'gamma',
        'number_of_lenders': 'beta',
        'cnt_0130_dpd': 'beta', 
        'cnt_3160_dpd': 'beta',
        'cnt_6190_dpd': 'beta',
        'cnt_over90_dpd': 'gamma',
        'contract_balance': 'beta'
    },
    default_distribution='beta')

# COMMAND ----------

# MAGIC %md
# MAGIC         <!-- 'cur_bal_member_lender': 'norm',
# MAGIC         'annualized_scheduled_payments': 'norm',
# MAGIC         'number_of_lenders': 'beta',
# MAGIC         'cnt_0130_dpd': 'beta', 
# MAGIC         'cnt_3160_dpd': 'beta',
# MAGIC         'cnt_6190_dpd': 'beta',
# MAGIC         'contract_balance': 'beta' -->

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train the synthesizer

# COMMAND ----------

synthesizer.fit(df_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Generate synthetic data

# COMMAND ----------

synthetic_df_data= synthesizer.sample(num_rows=len(df_data)*10)
#synthetic_data = synthesizer.sample(num_rows=10000)
display(synthetic_df_data)

# COMMAND ----------

data_unchanged.rename({"customer_number": "old_customer_number", 
                      "customer_name" : "old_customer_name",
                      "contract_number" : "old_contract_number",
                      "PD_status" : "old_PD_status"
           }, axis='columns', inplace =True) # Renaming column A with 'new_a' and B with 'new_b'

# COMMAND ----------

display(data_unchanged)

# COMMAND ----------

final = pd.concat([synthetic_df_data, data_unchanged], axis=1)

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

final.sort_values(['old_customer_number', 'old_customer_number'], inplace=True)
final.display()

# COMMAND ----------

test_synthetic_data_7861655 = final[(final["old_contract_number"]=='00000503-7861655-002') & (final["old_contract_number"]=='00000503-7861655-002')]

display(test_synthetic_data_7861655)

# COMMAND ----------

test_synthetic_data = final[final["old_contract_number"].isin(final["old_contract_number"].unique())]
test_synthetic_data.sort_values(['old_customer_number', 'old_contract_number'], inplace=True)
display(test_synthetic_data)

test_real_data = df_all[df_all["contract_number"].isin(df_all["contract_number"].unique())]
test_real_data.sort_values(['customer_number', 'contract_number'], inplace=True)
display(test_real_data)

# COMMAND ----------

test_real_data_7861655 = df_all[(df_all["contract_number"]=='00000503-7861655-002')
                                #  & (df_all["contract_number"]=='00000503-7861655-002')
                                ]

display(test_real_data_7861655)

# COMMAND ----------

# save the data as a CSV 
#synthetic_data.to_csv('synthetic_df_data.csv', index=False)

# COMMAND ----------

from sdv.evaluation.single_table import run_diagnostic, evaluate_quality
from sdv.evaluation.single_table import get_column_plot
from sdmetrics.reports.single_table import QualityReport

# COMMAND ----------

# MAGIC %md
# MAGIC ##2 - Quality of the Synthetic data

# COMMAND ----------

from sdmetrics.reports.single_table import QualityReport
from sdv.evaluation.single_table import run_diagnostic, evaluate_quality
from sdv.evaluation.single_table import get_column_plot
diagnostic_report= run_diagnostic(real_data=df_data, 
                            synthetic_data=synthetic_df_data, 
                            metadata=metadata)

# COMMAND ----------

# MAGIC
# MAGIC %md 
# MAGIC ### Interpretation
# MAGIC The score should be 100%.   
# MAGIC The diagnostic report checks for basic data validity and data structure issues.   
# MAGIC The score to be perfect for any of the default SDV synthesizers.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2-2). Measure the statistical similarity and Data Quality

# COMMAND ----------

from sdv.evaluation.single_table import evaluate_quality
from sdv.evaluation.single_table import get_column_plot
from sdv.evaluation.single_table import get_column_pair_plot
quality_report = evaluate_quality(
    real_data=df_data,
    synthetic_data=synthetic_df_data,
    metadata=metadata)


# COMMAND ----------

# MAGIC %md 
# MAGIC ### Interpreting the Score

# COMMAND ----------

# MAGIC %md
# MAGIC ***Your score will vary from 0% to 100%***.   
# MAGIC > This value tells you how similar the synthetic data is to the real data.  
# MAGIC A 100% score means that the patterns are exactly the same.
# MAGIC
# MAGIC > For example, if we compared the real data with itself (identity), the score would be 100%.  
# MAGIC A 0% score means the patterns are as different as can be.   
# MAGIC This would entail that the synthetic data purposefully contains anti-patterns that are opposite from the real data.  
# MAGIC Any score in the middle can be interpreted along this scale.   
# MAGIC For example, a score of 80% means that the synthetic data is about 80% similar to the real data — about 80% of the trends are similar.  
# MAGIC
# MAGIC > The quality score is expected to vary, and you may never achieve exactly 100% quality. 
# MAGIC That's ok! The SDV synthesizers are designed to estimate patterns, meaning that they may smoothen, extrapolate, or noise certain parts of the data.

# COMMAND ----------

# MAGIC %md
# MAGIC ### plot the data

# COMMAND ----------

df_all.columns

# COMMAND ----------

# # 3. plot the data
# fig = get_column_plot(
#     real_data=df_data,
#     synthetic_data=synthetic_df_data,
#     metadata=metadata,
#     column_name='customer_number'
# )
    
# fig.show()

# COMMAND ----------

# 3. plot the data
fig = get_column_plot(
    real_data=df_data,
    synthetic_data=synthetic_df_data,
    metadata=metadata,
    column_name='cur_bal_member_lender'
)
    
fig.show()

# COMMAND ----------

# 3. plot the data
fig = get_column_plot(
    real_data=df_data,
    synthetic_data=synthetic_df_data,
    metadata=metadata,
    column_name='total_gross_orig_receivable_amt'
)
    
fig.show()

# COMMAND ----------

# 3. plot the data
fig = get_column_plot(
    real_data=df_data,
    synthetic_data=synthetic_df_data,
    metadata=metadata,
    column_name='annualized_scheduled_payments'
)
    
fig.show()

# COMMAND ----------

# 3. plot the data
fig = get_column_plot(
    real_data=df_data,
    synthetic_data=synthetic_df_data,
    metadata=metadata,
    column_name='number_of_lenders'
)
    
fig.show()

# COMMAND ----------

# 3. plot the data
fig = get_column_plot(
    real_data=df_data,
    synthetic_data=synthetic_df_data,
    metadata=metadata,
    column_name='cnt_0130_dpd'
)
    
fig.show()

# COMMAND ----------

# 3. plot the data
fig = get_column_plot(
    real_data=df_data,
    synthetic_data=synthetic_df_data,
    metadata=metadata,
    column_name='contract_balance'
)
    
fig.show()

# COMMAND ----------

# MAGIC %md ###Quality report

# COMMAND ----------

from sdmetrics.reports.single_table import QualityReport
quality_report.get_details(property_name='Column Shapes')

# COMMAND ----------

from sdmetrics.reports.single_table import QualityReport
quality_report.get_details(property_name='Column Pair Trends')

# COMMAND ----------

from sdv.evaluation.single_table import get_column_pair_plot

fig = get_column_pair_plot(
    real_data=df_data,
    synthetic_data=synthetic_df_data,
    metadata=metadata,
    column_names=['total_gross_orig_receivable_amt', 'contract_balance'],
)

fig.show()

# COMMAND ----------

# MAGIC %md ###Mapping

# COMMAND ----------

# from sdv.metadata import SingleTableMetadata
# metadata = SingleTableMetadata()
# df_data = df_data
# metadata.detect_from_dataframe(df_data)

# # correct metadata to use the same values as in the real data 
# metadata.update_column( column_name='customer_number', sdtype='numerical' ) 
# metadata.update_column( column_name='PD_status', sdtype='categorical' ) 

# # Create a secure mapping between real and synthetic primary keys
# import hashlib
# mapping = {}
# for i, row in df_data.iterrows():
#     real_pk = row['customer_number']  # Use row-specific value
#     synthetic_pk = synthetic_df_data.iloc[i]['customer_number']
#     hashed_real_pk = hashlib.sha256(str(real_pk).encode()).hexdigest()
#     mapping[hashed_real_pk] = synthetic_pk

# # Print the mapping
# for hashed_real_pk, synthetic_pk in mapping.items():
#     print(f"Hashed Real PK: {hashed_real_pk}, Synthetic PK: {synthetic_pk}")

# COMMAND ----------

# # Print df_data with hashed real PK
# hashed_df_data = df_data.copy()
# hashed_df_data['hashed_real_pk'] = hashed_df_data['customer_number'].apply(lambda x: hashlib.sha256(str(x).encode()).hexdigest())
# display(hashed_df_data)

# # Print synthetic_df_data with synthetic PK
# synthetic_df_data_with_pk = synthetic_df_data.copy()
# synthetic_df_data_with_pk['synthetic_pk'] = synthetic_df_data_with_pk['customer_number']
# display(synthetic_df_data_with_pk)

# COMMAND ----------

# hashed_real_pk_to_synthetic_pk = {}

# # Iterate through the mapping dictionary
# for hashed_real_pk, synthetic_pk in mapping.items():
#     # Add the mapping to the hashed_real_pk_to_synthetic_pk dictionary
#     hashed_real_pk_to_synthetic_pk[hashed_real_pk] = synthetic_pk

# # Print the hashed_real_pk_to_synthetic_pk dictionary
# print(hashed_real_pk_to_synthetic_pk)

# COMMAND ----------

# # Print df_data table
# print("df_data table:")
# print(df_data.columns)
# display(df_data)

# # Print synthetic_df_data table
# print("synthetic_df_data table:")
# print(synthetic_df_data.columns)
# display(synthetic_df_data)

# COMMAND ----------

# MAGIC %md #Model Methodology

# COMMAND ----------

# MAGIC %md
# MAGIC ###EXPLANATION OF HOW THE MODEL WORKS

# COMMAND ----------

# MAGIC %md
# MAGIC Imagine you have a dataset containing information about your customers, such as their age, income, and purchase history. This dataset is valuable for understanding your customer base, but sharing or using the actual customer data directly could raise privacy concerns.
# MAGIC
# MAGIC The GaussianCopulaSynthesizer model allows us to create a new, synthetic dataset that looks and behaves like the original dataset but doesn't contain any real customer information. It's like creating a realistic dummy dataset that preserves the patterns and relationships in the original data without exposing sensitive information.
# MAGIC
# MAGIC Here's how the model works:
# MAGIC
# MAGIC 1. **Understanding the Data**: The model first analyzes the original dataset to understand the characteristics of each variable (e.g., age, income, purchase history) and how they relate to each other. It looks for patterns and dependencies without needing to know the actual values.
# MAGIC
# MAGIC 2. **Building a Statistical Model**: Based on this analysis, the model creates a statistical representation of the data. It captures the essential properties of the data, such as the range of values for each variable and how they tend to vary together.
# MAGIC
# MAGIC 3. **Generating Synthetic Data**: Using this statistical model, the model can generate an entirely new dataset with the same properties as the original data but with completely different values. It's like creating a new set of customers that look and behave like your real customers but without revealing any individual's information.
# MAGIC
# MAGIC 4. **Preserving Privacy**: Because the synthetic data doesn't contain any real customer information, it can be safely shared or used for various purposes, such as analysis, testing, or training machine learning models, without risking customer privacy.
# MAGIC
# MAGIC The synthetic data generated by this model is incredibly useful for businesses like ours. It allows us to gain insights, test scenarios, and make data-driven decisions without compromising the privacy of our customers or exposing sensitive information.

# COMMAND ----------

# MAGIC %md
# MAGIC ###How GaussianCopulaSynthesizer model builds a statistical model

# COMMAND ----------

# MAGIC %md
# MAGIC **How GaussianCopulaSynthesizer model builds a statistical model**
# MAGIC
# MAGIC Let's say our company sells various products, and we have a dataset containing information about our customers, such as their age, income, and purchase history across different product categories.
# MAGIC
# MAGIC When building the statistical model, the GaussianCopulaSynthesizer model would analyze this dataset and capture the following:
# MAGIC
# MAGIC 1. **Age Distribution**: The model would look at the range of ages of our customers and how many customers fall into different age groups (e.g., 20-30, 31-40, 41-50, etc.). This information helps the model understand the typical age distribution of our customer base.
# MAGIC
# MAGIC 2. **Income Patterns**: The model would examine the income levels of our customers and how they are distributed. It might identify common income ranges or patterns, such as a large portion of customers falling within a certain income bracket.
# MAGIC
# MAGIC 3. **Purchase Behavior**: The model would analyze the purchase history data to understand how customers' buying behavior varies across different product categories. It might identify relationships like customers who buy a certain product being more likely to also purchase another related product.
# MAGIC
# MAGIC 4. **Relationships between Variables**: The model would look for dependencies and correlations between different variables. For example, it might find that customers in a particular age group and income range tend to purchase more from a specific product category.
# MAGIC
# MAGIC The statistical model captures all these patterns, distributions, and relationships without using the actual customer data values. It's like creating a blueprint or a set of instructions that describe the overall structure and characteristics of our customer base and their purchasing behavior.
# MAGIC
# MAGIC With this statistical model, the GaussianCopulaSynthesizer can then generate a new synthetic dataset that mimics the real customer data in terms of age distribution, income patterns, purchase behavior, and the relationships between these variables. However, the synthetic data will contain entirely new, fictional customer records that don't correspond to any real individuals.
# MAGIC
# MAGIC This synthetic dataset can then be used for various business purposes, such as testing marketing strategies, analyzing potential product offerings, or training machine learning models, all while preserving the privacy of our actual customers.

# COMMAND ----------

# MAGIC %md ### Validation Tests and Comparisons between Real data and the Synthetic data 

# COMMAND ----------

import pandas as pd
import numpy as np
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer
from scipy.stats import ks_2samp, anderson

for column in df_data.select_dtypes(include=['float64', 'int64']).columns:
    print(f"Distribution Comparison for {column}:")
    print(f"Kolmogorov-Smirnov Test: {ks_2samp(df_data[column], synthetic_df_data[column])}")
    print(f"Anderson-Darling Test: {anderson(df_data[column], 'norm')}")
    print()

# You can also perform machine learning model performance comparisons and domain-specific validation tests
# based on your requirements and data characteristics.

# COMMAND ----------

# MAGIC %md
# MAGIC ####Distribution Comparison

# COMMAND ----------

# MAGIC %md
# MAGIC Results of various statistical tests and comparisons performed between the real data and the synthetic data generated by the GaussianCopulaSynthesizer from the Synthetic Data Vault (SDV) library.
# MAGIC
# MAGIC 1. **Distribution Comparison**:
# MAGIC    - This section compares the distributions of individual features (columns) between the real and synthetic data using the Kolmogorov-Smirnov and Anderson-Darling tests.
# MAGIC    - For most features, the Kolmogorov-Smirnov test shows a statistic value close to 0.2 and a p-value of 0, indicating that the distributions of the real and synthetic data are significantly different.
# MAGIC    - The Anderson-Darling test also shows large statistic values, significantly higher than the critical values, suggesting that the distributions are not the same.
# MAGIC    - These results indicate that the synthetic data generated by the GaussianCopulaSynthesizer does not accurately capture the distributions of the individual features in the real data.

# COMMAND ----------

# Correlation Comparison
real_corr = df_data.corr()
synthetic_corr = synthetic_df_data.corr()
print("Correlation Comparison:")
print(f"Frobenius Norm: {np.linalg.norm(real_corr - synthetic_corr, ord='fro')}")



# COMMAND ----------

# MAGIC %md ####Correlation Comparison:

# COMMAND ----------

# MAGIC %md
# MAGIC 2. **Correlation Comparison**:
# MAGIC    - This part compares the correlation matrices of the real and synthetic data using the Frobenius norm.
# MAGIC    - A Frobenius norm of 2.014264363961712 is reported, which is a relatively high value, suggesting that the correlation structures of the real and synthetic data are quite different.

# COMMAND ----------

# Statistical Moment Comparison
for column in df_data.select_dtypes(include=['float64', 'int64']).columns:
    real_mean = df_data[column].mean()
    real_var = df_data[column].var()
    real_skew = df_data[column].skew()
    real_kurt = df_data[column].kurt()

    synthetic_mean = synthetic_df_data[column].mean()
    synthetic_var = synthetic_df_data[column].var()
    synthetic_skew = synthetic_df_data[column].skew()
    synthetic_kurt = synthetic_df_data[column].kurt()

    print(f"Statistical Moment Comparison for {column}:")
    print(f"Mean: Real={real_mean}, Synthetic={synthetic_mean}")
    print(f"Variance: Real={real_var}, Synthetic={synthetic_var}")
    print(f"Skewness: Real={real_skew}, Synthetic={synthetic_skew}")
    print(f"Kurtosis: Real={real_kurt}, Synthetic={synthetic_kurt}")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Statistical Moment Comparison:

# COMMAND ----------

# MAGIC %md
# MAGIC 3. **Statistical Moment Comparison**:
# MAGIC    - This section compares the mean, variance, skewness, and kurtosis of each feature between the real and synthetic data.
# MAGIC    - For most features, the mean, variance, skewness, and kurtosis values differ substantially between the real and synthetic data.
# MAGIC    - This indicates that the synthetic data does not accurately capture the statistical moments of the individual features in the real data.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Based on these results, it appears that the synthetic data generated by the GaussianCopulaSynthesizer does not adequately represent the original real data in terms of individual feature distributions, correlation structures, and statistical moments. The synthetic data exhibits significant deviations from the real data in these aspects.
# MAGIC
# MAGIC These results suggest that the GaussianCopulaSynthesizer may not be the most suitable synthesizer for your specific dataset, or that further tuning and adjustments may be required to improve the quality of the synthetic data. We may need to explore alternative synthesizers from the SDV library or consider different approaches to generate synthetic data that better aligns with the characteristics of our real data.
# MAGIC
# MAGIC It's important to note that achieving a perfect replication of the real data is often challenging, and some deviations are expected, especially when dealing with complex, high-dimensional datasets. The acceptability of the synthetic data depends on the intended use case and the trade-offs between data utility and privacy preservation.

# COMMAND ----------

# MAGIC %md
# MAGIC # Testing on other Sythetizers

# COMMAND ----------

# MAGIC %md
# MAGIC ### Synthetic data with method-2(CTGAN Synthesizer)

# COMMAND ----------

from sdv.single_table import CTGANSynthesizer

synthesizer_CTGAN = CTGANSynthesizer(
    metadata, # required
    enforce_rounding=False,
    epochs=500,
    batch_size=500,
    generator_lr=0.0002,
    verbose=True
)
synthesizer_CTGAN.fit(df_data)
CTGAN_synthetic_data = synthesizer_CTGAN.sample(num_rows=200)
display(CTGAN_synthetic_data)

# COMMAND ----------

from sdv.evaluation.single_table import evaluate_quality
quality_report_CTGAN = evaluate_quality(df_data,CTGAN_synthetic_data,metadata)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Synthetic data with method-3(TVAE Synthesizer)

# COMMAND ----------

from sdv.single_table import TVAESynthesizer

synthesizer_TVAE = TVAESynthesizer(
    metadata, # required
    enforce_min_max_values=True,
    enforce_rounding=False,
    epochs=500
)
synthesizer_TVAE.fit(df_data)
TVAE_synthetic_data = synthesizer_TVAE.sample(num_rows=200)
display(TVAE_synthetic_data)

# COMMAND ----------

from sdv.evaluation.single_table import evaluate_quality
quality_report_TVAE = evaluate_quality(df_data,TVAE_synthetic_data,metadata)
