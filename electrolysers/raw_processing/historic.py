# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# %%
# Read the excel files
raw_data = r'C:\Users\idapo\master_thesis\master\data\raw\electrolysers\electrolysers.xlsx'
df = pd.read_excel(raw_data)

# %%
# Include relevant columns in a dataframe
electrolyser_df = df[['Date online','Status','Technology', 'Capacity_MWel']].copy()

# Remove rows with missing values in important columns only
electrolyser_df = electrolyser_df.dropna()

# Rename columns
electrolyser_df = electrolyser_df.rename(columns={
    'Date online': 'Year',
    'Capacity_MWel': 'Installed_Capacity_MW'
})

print(electrolyser_df.head())

# %%
# Filter for tech ALK and status operational from 1992 to 2025

ALK_historic_df = electrolyser_df[
    (electrolyser_df['Technology'] == 'ALK') &
    (electrolyser_df['Status'] == 'Operational') &
    (electrolyser_df['Year'] >= 1992) &
    (electrolyser_df['Year'] <= 2025)
]

print(ALK_historic_df.head())

# %%
# Sum capacities by year

ALK_historic_df = ALK_historic_df.groupby('Year', as_index=False).agg({
    'Installed_Capacity_MW': 'sum'
})

print(ALK_historic_df)
#%% 
# Add missing years between 1990 and 2025 with no installed capacity
all_years = pd.DataFrame({'Year': range(1990, 2026)})
ALK_historic_df = pd.merge(all_years, ALK_historic_df, on='Year', how='left')
ALK_historic_df['Installed_Capacity_MW'] = ALK_historic_df['Installed_Capacity_MW'].fillna(0)

print(ALK_historic_df)
#%%
# Plot the historical installed capacity of ALK electrolysers
plt.figure(figsize=(10, 6))
sns.lineplot(data=ALK_historic_df, x='Year', y='Installed_Capacity_MW', marker='o')
plt.title('Inflow - Installed Capacity of ALK Electrolysers (1990-2025)')

#%% cumsum
print(ALK_historic_df['Installed_Capacity_MW'].cumsum())

#%% save as csv
ALK_historic_df.to_csv(r'C:\Users\idapo\master_thesis\master\data\processed_baseline\electrolysers\type_split\ALK_historic_capacity.csv', index=False)

#%%
# Filter for tech PEM and status operational from 1992 to 2025

PEM_historic_df = electrolyser_df[
    (electrolyser_df['Technology'] == 'PEM') &
    (electrolyser_df['Status'] == 'Operational') &
    (electrolyser_df['Year'] >= 1992) &
    (electrolyser_df['Year'] <= 2025)
]

#%%
# Sum capacities by year

PEM_historic_df = PEM_historic_df.groupby('Year', as_index=False).agg({
    'Installed_Capacity_MW': 'sum'
})

print(PEM_historic_df)
#%% 
# Add missing years between 1990 and 2025 with no installed capacity
all_years = pd.DataFrame({'Year': range(1990, 2026)})
PEM_historic_df = pd.merge(all_years, PEM_historic_df, on='Year', how='left')
PEM_historic_df['Installed_Capacity_MW'] = PEM_historic_df['Installed_Capacity_MW'].fillna(0)

print(PEM_historic_df)
#%%
# Plot the historical installed capacity of PEM electrolysers
plt.figure(figsize=(10, 6))
sns.lineplot(data=PEM_historic_df, x='Year', y='Installed_Capacity_MW', marker='o')
plt.title('Inflow - Installed Capacity of PEM Electrolysers (1990-2025)')

#%% save as csv
PEM_historic_df.to_csv(r'C:\Users\idapo\master_thesis\master\data\processed_baseline\electrolysers\type_split\PEM_historic_capacity.csv', index=False)

#%%
# Filter for tech SOEC and status operational from 1992 to 2025

SOEC_historic_df = electrolyser_df[
    (electrolyser_df['Technology'] == 'SOEC') &
    (electrolyser_df['Status'] == 'Operational') &
    (electrolyser_df['Year'] >= 1992) &
    (electrolyser_df['Year'] <= 2025)
]

#%%
# Sum capacities by year

SOEC_historic_df = SOEC_historic_df.groupby('Year', as_index=False).agg({
    'Installed_Capacity_MW': 'sum'
})

print(SOEC_historic_df)
#%% 
# Add missing years between 1990 and 2025 with no installed capacity
all_years = pd.DataFrame({'Year': range(1990, 2026)})
SOEC_historic_df = pd.merge(all_years, SOEC_historic_df, on='Year', how='left')
SOEC_historic_df['Installed_Capacity_MW'] = SOEC_historic_df['Installed_Capacity_MW'].fillna(0)

print(SOEC_historic_df)
#%%
# Plot the historical installed capacity of SOEC electrolysers
plt.figure(figsize=(10, 6))
sns.lineplot(data=SOEC_historic_df, x='Year', y='Installed_Capacity_MW', marker='o')
plt.title('Inflow - Installed Capacity of SOEC Electrolysers (1990-2025)')

#%% save as csv
SOEC_historic_df.to_csv(r'C:\Users\idapo\master_thesis\master\data\processed_baseline\electrolysers\type_split\SOEC_historic_capacity.csv', index=False)

#%%
# Filter for tech AEM and status operational from 1992 to 2025

AEM_historic_df = electrolyser_df[
    (electrolyser_df['Technology'] == 'AEM') &
    (electrolyser_df['Status'] == 'Operational') &
    (electrolyser_df['Year'] >= 1992) &
    (electrolyser_df['Year'] <= 2025)
]

#%%
# Sum capacities by year

AEM_historic_df = AEM_historic_df.groupby('Year', as_index=False).agg({
    'Installed_Capacity_MW': 'sum'
})

print(AEM_historic_df)
#%% 
# Add missing years between 1990 and 2025 with no installed capacity
all_years = pd.DataFrame({'Year': range(1990, 2026)})
AEM_historic_df = pd.merge(all_years, AEM_historic_df, on='Year', how='left')
AEM_historic_df['Installed_Capacity_MW'] = AEM_historic_df['Installed_Capacity_MW'].fillna(0)

print(AEM_historic_df)
#%%
# Plot the historical installed capacity of AEM electrolysers
plt.figure(figsize=(10, 6))
sns.lineplot(data=AEM_historic_df, x='Year', y='Installed_Capacity_MW', marker='o')
plt.title('Inflow - Installed Capacity of AEM Electrolysers (1990-2025)')

#%% save as csv
AEM_historic_df.to_csv(r'C:\Users\idapo\master_thesis\master\data\processed_baseline\electrolysers\type_split\AEM_historic_capacity.csv', index=False)