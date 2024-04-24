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
import shap
from sklearn.ensemble import RandomForestRegressor



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

df_all.printSchema()

# COMMAND ----------

# from pyspark.sql.functions import when
# columns = ["PD_status"]

# # defining a mapping for pd_status  to numerical values
# pd_status_mapping = {"performing": 0, "defaulted": 1,}

# # using the when and otherwise functions to create a new target_variable column
# df_all = df_all.withColumn("pd_status_score", 
#                    when(df_all["PD_status"] == "performing", pd_status_mapping["performing"])
#                    .otherwise("defaulted"))

# # show.
# display(df_all)

# COMMAND ----------

from pyspark.sql.functions import when, col, lit

columns = ["PD_status"]

# defining a mapping for pd_status to numerical values
pd_status_mapping = {"performing": 0, "defaulted": 1}

# using the when and otherwise functions to create a new target_variable column
df_all = df_all.withColumn("pd_status_score",
                            when(col("PD_status") == "performing", lit(pd_status_mapping["performing"]))
                            .when(col("PD_status") == "defaulted", lit(pd_status_mapping["defaulted"]))
                            .otherwise(lit(None))
                            .cast("integer"))

# show
display(df_all)

# COMMAND ----------

df_all.printSchema()

# COMMAND ----------

columns_to_drop = ['as_of_date', 'region', 'member_lender_branch', 'member_lender_business_unit', 'member_lender_business_unit',
                   'member_lender_risk_rating', 'default_4q_flg', 'default_8q_flg', 'borrower_high_credit',
                   'naics_code', 'primary_collateral_type', 'most_recent_0130', 'most_recent_3160', 'most_recent_6190', 'most_recent_over90_dpd', 'sic_code', 'PayNet_absolutepd_1q','PayNet_absolutepd_2q', 'PayNet_absolutepd_3q', 'PayNet_absolutepd_4q','PayNet_absolutepd_5q', 'PayNet_absolutepd_6q', 'PayNet_absolutepd_7q', 'PayNet_absolutepd_8q','contract_sbu', 'Portfolio', 'oldest_contract_start_date', 'newest_contract_start_date','contract_collateral', 'customer_name', 'industry_segment', 'PD_status', 'state','PayNet_risk_factor_1', 'PayNet_risk_factor_2', 'PayNet_risk_factor_3', 'contract_number']

df_all = df_all.drop(*columns_to_drop)

# COMMAND ----------

df_all.printSchema()

# COMMAND ----------

df_all.columns
df_all.display()


# COMMAND ----------

# Convert PySpark DataFrame to pandas DataFrame
df_all = df_all.toPandas()

# Split the pandas DataFrame
X_train, X_test, y_train, y_test = model_selection.train_test_split(df_all, df_all['pd_status_score'], random_state=0)

# COMMAND ----------

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

# 1. Handle missing values in the input data
imputer = SimpleImputer(strategy='mean')
X_train_filled = imputer.fit_transform(X_train)

# 2. Scale or normalize the input data if necessary
# (assuming X_train_scaled and y_train_scaled are the scaled versions)

# 3. Ensure the correct format of the input data
# (assuming X_train_scaled and y_train_scaled are the correct format)

# 4. Create and train the Random Forest Regressor with the corrected input data
regressor = RandomForestRegressor()
regressor.fit(X_train_filled, y_train)

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np

# # Train your model
# regressor = RandomForestRegressor()
# regressor.fit(X_train, y_train)

# Get feature importances
feature_importances = regressor.feature_importances_

# Sort feature importances in descending order
indices = np.argsort(feature_importances)[::-1]

# Rearrange feature names so they match the sorted feature importances
names = [X_train.columns[i] for i in indices]

# Create a horizontal bar plot
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.barh(range(len(indices)), feature_importances[indices], color="r", align="center")
plt.yticks(range(len(indices)), names, rotation='horizontal')
plt.xlabel("Relative Importance")
plt.tight_layout()
plt.show()

# COMMAND ----------

# Create object that can calculate shap values
explainer = shap.TreeExplainer(regressor)
# Calculate Shap values
shap_values = explainer.shap_values(X_train)

# COMMAND ----------

df_all.columns


# COMMAND ----------

feature_names = X_train.columns.tolist()
shap.summary_plot(shap_values, X_train, feature_names=feature_names, plot_type="bar")

# COMMAND ----------


df_all = spark.createDataFrame(df_all)
df_imp = df_all.select('PayNet_absolutepd_1q','PayNet_absolutepd_2q', 'PayNet_absolutepd_3q', 
                       'PayNet_absolutepd_4q','PayNet_absolutepd_5q', 'PayNet_absolutepd_6q', 
                       'PayNet_absolutepd_7q','PayNet_absolutepd_8q', 'industry_segment', 'state','PD_status')

