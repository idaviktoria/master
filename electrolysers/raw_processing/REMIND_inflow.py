#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%
file_path_SSP2_L = r'C:\Users\ovid\MasterThesis\master\data\raw\REMIND_generic_C_SMIPv08-L-SSP2-PkPrice400-def-rem-6.mif'
raw_data_SSP2_L = pd.read_csv(file_path_SSP2_L, sep=';')

file_path_SSP2_M = r'C:\Users\ovid\MasterThesis\master\data\raw\REMIND_generic_C_SMIPv08-M-SSP2-NPi2025-def-rem-6.mif'
raw_data_SSP2_M = pd.read_csv(file_path_SSP2_M, sep=';')

file_path_SSP1 = r'C:\Users\ovid\MasterThesis\master\data\raw\REMIND_generic_C_SMIPv08-VLLO-SSP1-PkPrice500-def-rem-6.mif'
raw_data_SSP1 = pd.read_csv(file_path_SSP1, sep=';')

#%% Electrolyser variables

# "Cap|Hydrogen|+|Electricity" - Electrolyzer capacity in GW per year
# "New Cap|Hydrogen|+|Electricity" - New electrolyzer capacity in GW per year

#%% SSP1 data
print(raw_data_SSP1.columns)

#%% Melt SSP1 data to long format
data_long_SSP1 = raw_data_SSP1.melt(
    id_vars=["Model", "Scenario", "Region", "Variable", "Unit"],
    var_name="Year",
    value_name="Value"
)

#%% Check to see which regions are present in the SSP1 data
print(data_long_SSP1["Region"].unique())

#%% Choose EUR to only include European countries in the analysis
data_long_SSP1 = data_long_SSP1[data_long_SSP1["Region"].isin(["EUR", "NEU"])]

print(data_long_SSP1["Region"].unique())

#%% Visualise the data
inflow_SSP1 = data_long_SSP1[data_long_SSP1["Variable"] == "New Cap|Hydrogen|+|Electricity"]

inflow_SSP1.groupby("Year")["Value"].sum().plot()

plt.ylabel("GW")
plt.title("Total New Electrolyzer Capacity")
plt.show()

#%% SSP2-L data
print(raw_data_SSP2_L.columns)

#%% Melt SSP2-L data to long format
data_long_SSP2_L = raw_data_SSP2_L.melt(
    id_vars=["Model", "Scenario", "Region", "Variable", "Unit"],
    var_name="Year",
    value_name="Value"
)

print(data_long_SSP2_L.head())

#%% Check to see which regions are present in the SSP2-L data
print(data_long_SSP2_L["Region"].unique())

#%% Choose EUR to only include European countries in the analysis
data_long_SSP2_L = data_long_SSP2_L[data_long_SSP2_L["Region"].isin(["EUR", "NEU"])]

print(data_long_SSP2_L["Region"].unique())

#%% Visualise the data
inflow_SSP2_L = data_long_SSP2_L[data_long_SSP2_L["Variable"] == "New Cap|Hydrogen|+|Electricity"]

inflow_SSP2_L.groupby("Year")["Value"].sum().plot()

plt.ylabel("GW")
plt.title("Total New Electrolyzer Capacity")
plt.show()

#%% SSP2-M data
print(raw_data_SSP2_M.columns)

#%% Melt SSP2-M data to long format
data_long_SSP2_M = raw_data_SSP2_M.melt(
    id_vars=["Model", "Scenario", "Region", "Variable", "Unit"],
    var_name="Year",
    value_name="Value"
)

print(data_long_SSP2_M.head())

#%% Check to see which regions are present in the SSP2-M data
print(data_long_SSP2_M["Region"].unique())

#%% Choose EUR to only include European countries in the analysis
data_long_SSP2_M = data_long_SSP2_M[data_long_SSP2_M["Region"].isin(["EUR", "NEU"])]

print(data_long_SSP2_M["Region"].unique())

#%% Visualise the data
inflow_SSP2_M = data_long_SSP2_M[data_long_SSP2_M["Variable"] == "New Cap|Hydrogen|+|Electricity"]

inflow_SSP2_M.groupby("Year")["Value"].sum().plot()

plt.ylabel("GW")
plt.title("Total New Electrolyzer Capacity")
plt.show()

#%% Compare the three scenarios
plt.figure(figsize=(10, 6))
inflow_SSP1.groupby("Year")["Value"].sum().plot(label="SSP1-VLLO")
inflow_SSP2_L.groupby("Year")["Value"].sum().plot(label="SSP2-L")
inflow_SSP2_M.groupby("Year")["Value"].sum().plot(label="SSP2-M")
plt.ylabel("GW")
plt.title("Total New Electrolyzer Capacity")
plt.legend()
plt.show()

#%% Save data for later use
REMIND_inflow_SSP1 = inflow_SSP1
REMIND_inflow_SSP2_L = inflow_SSP2_L
REMIND_inflow_SSP2_M = inflow_SSP2_M

print(REMIND_inflow_SSP1)

#%% Save the processed data to CSV files

REMIND_inflow_SSP1.to_csv(r'C:\Users\ovid\MasterThesis\master\data\processed_validation\inflow_SSP1.csv', index=False)
REMIND_inflow_SSP2_L.to_csv(r'C:\Users\ovid\MasterThesis\master\data\processed_validation\inflow_SSP2_L.csv', index=False)
REMIND_inflow_SSP2_M.to_csv(r'C:\Users\ovid\MasterThesis\master\data\processed_validation\inflow_SSP2_M.csv', index=False)
# %%
