#%% Import packages
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.optimize import curve_fit
import sys

#%% Read ALK historic data
ALK_historic_data = r'C:\Users\idapo\master_thesis\master\data\processed_baseline\electrolysers\type_split\ALK_historic_capacity.csv'
ALK_historic_df = pd.read_csv(ALK_historic_data)

#%% define logistic function and noise function
def logistic(x:np.ndarray,ti:float,tau:float,C0:float,C1:float) -> np.ndarray:
    """
    General logistic function.
    Arguments:
    - x: np.ndarray of observation points (time)
    - ti: inflection time
    - tau: transition time coefficient
    - C0: start value
    - C1: end value

    Returns:
    - np.ndarray with len(x) number of points
    """
    return (C1 - C0)/(1 + np.exp(-(x - ti) / tau)) + C0  


def noise(start: int, stop: int, lo_time_deltas: list, lo_deviations: list) -> np.ndarray:
    '''
    Generates noise for timeseries for a set of time deltas and 
    deviations.This works by setting random deviations at certain 
    intervals and interpolating the points in between.The noise can then
    simply be added to the smooth timeseries curve to generate the final
    timeseries.
    
    Arguments:
    - start: beginning of the timeseries
    - stop: end of the timeseries
    - lo_time_deltas: list of time deltas which set the points at which 
                      noise trends are set
    - lo_deviations: the respective standard deviation from which the 
                     deviation for each point is drawn.

    Returns:
    - np.ndarray with stop-start+1 values of noise, averaging around 0
    '''
    no_time = stop-start +1 #number of discrete time instances
    final_points = np.zeros(no_time)

    for (time_delta, deviation) in zip(lo_time_deltas, lo_deviations):
        no_points = int((no_time-1)/time_delta)+2 #1 more than necessary to extend series
        end_time = start + (no_points-1)*time_delta
        macro_points = np.random.normal(0, deviation, no_points) 
        macro_point_x = np.linspace(start, end_time,no_points)
        macro_point_x = np.delete(macro_point_x, -1) #delete the extra point here


        extended_macro_points = [macro_points[0]]
        for index, macro_point in enumerate(macro_points[1:]):
            connection = np.linspace(macro_points[index], macro_point, time_delta+1, endpoint=True)
            extended_macro_points.extend(connection[1:])
        extended_macro_points = np.array(extended_macro_points[0:no_time])
        macro_points = np.delete(macro_points, -1)

        final_points = np.add(final_points, extended_macro_points)

    return final_points

#%% Define values for regression
start = 1990
duration = 36  #years 1990-2025 

#the basic values:
oj_years = np.linspace(start, start+duration-1, duration)
ALK_values = ALK_historic_df['Installed_Capacity_MW'].to_numpy()

np.random.seed(seed = 2)
#incomplete values:
#_______________________________________________________________________
# we set indices for what should dissapear - points between 0 and 80% of 
# the range, 35% of them are removed
limit = int(duration*0.8)
removed_share = int(duration*0.35)
removed_indices = [int(i) for i in np.sort(np.random.uniform(0,limit, removed_share))]
incomplete_years = copy.deepcopy(oj_years)
incomplete_values = copy.deepcopy(ALK_values)

remover_counter = 0
for index in removed_indices:
    incomplete_years = np.delete(incomplete_years, index-remover_counter)
    incomplete_values = np.delete(incomplete_values, index-remover_counter)
    remover_counter += 1

#multi noise values:
#_______________________________________________________________________

no_multi_noise_points = 5
multi_noise_level = 2
multi_noisy_years = copy.deepcopy(oj_years)
multi_noisy_values = copy.deepcopy(ALK_values)
for index, (year, value) in enumerate(zip(oj_years, ALK_values)):
    for _ in range(no_multi_noise_points):
        multi_noisy_years = np.insert(multi_noisy_years,index*(no_multi_noise_points+1), year)
        new_value = np.random.normal(0, multi_noise_level,1) + value
        multi_noisy_values = np.insert(multi_noisy_values, index*(no_multi_noise_points+1),new_value)

#noisy values:
#____________________________________________________________________________
single_noise_years = copy.deepcopy(oj_years)
single_noise_distance = [1,8]
single_noise_std = [5, 15]
single_noise = noise(1,duration, single_noise_distance, single_noise_std)
single_noise_values = ALK_values + single_noise


#short data:
#____________________________________________________________________________
short_years = oj_years[0:int(duration*0.6)]
short_values = ALK_values[0:int(duration*0.6)]

labels = ['short', 'incomplete', 'multiple noisy', 'single noisy', 'complete' ]
lo_years = [short_years, incomplete_years, multi_noisy_years, single_noise_years, oj_years]
lo_values = [short_values, incomplete_values, multi_noisy_values, single_noise_values, ALK_values]
lo_symbols = ['v', 'X', 'P','8', 's']
lo_colors = ['blue', 'darkorange', 'forestgreen', 'cyan','black']

#%% Visualize the raw data

plt.figure(figsize=(16,10))
plt.plot(oj_years, ALK_values, 'o', color = 'black', markersize = 5, label = f'original values')
plt.xlabel('Years')
plt.ylabel('Installed Capacity (MW)')
plt.title('Alkaline Electrolyser Installed Capacity (1990-2025)')
plt.show()

#%% Polynomial regression
# Extended values
extended_years = np.arange(1990, 2051)
reg_years = oj_years
ALK_reg_values = ALK_values
reg_predictor_years = extended_years

inputs = reg_years
outputs = ALK_reg_values
pred_inputs = reg_predictor_years

#perform regression:
#set degreee:
degree = 3
#find polynomial
polynomial = np.poly1d(np.polyfit(reg_years, ALK_reg_values, degree))
print(f' the polynomial our fit created is: \n{polynomial}.')

pred_outputs = polynomial(extended_years)

# Plot the results
fig=plt.figure(figsize = (16,8))
plt.plot(reg_years, ALK_reg_values, 's', color = 'black', markersize = 5, label = f'original values')
plt.plot(extended_years, pred_outputs, color = 'crimson', lw = 3, label = f'polynomial regression (degree {degree}) values')
plt.legend(loc = 'best')
plt.xticks(ticks = np.rint(extended_years[::int(len(extended_years)/10)]))
plt.title(f'Alkaline Electrolyser Installed Capacity Polynomial (degree {degree}) regression')
plt.xlabel('Years')
plt.ylabel('Installed Capacity (MW)')
plt.show()

print(pred_outputs)

#%% Logistic regression
# Define x and y
x_data = oj_years
ALK_y_data = ALK_values
ALK_y_data2 = ALK_y_data.cumsum()

# Initial guess for parameters [ti, tau, C0, C1]
# ti ~ midpoint, tau ~ transition width, C0 ~ min, C1 ~ max

# Assume that alkaline will be 20% of the market (moderate) in 2050, and total in Europe is 500GW
p0 = [2028, 3, 0, 100000]

params, _ = curve_fit(
    logistic,
    x_data,
    ALK_y_data2,
    p0=p0,
    bounds=([1990, 0.1, 0, 10000], [2050, 50, 10000, 100000]),
    maxfev=100000
)

# Extract fitted parameters
ti_fit, tau_fit, C0_fit, C1_fit = params

# Predict for extended range
pred_logistic_ALK = logistic(extended_years, ti_fit, tau_fit, C0_fit, C1_fit)

# Ensure predictions stay positive
pred_logistic_ALK = np.maximum(pred_logistic_ALK, 0)

# Validation data for future years 2026-2028
validation_years = np.array([2026, 2027, 2028])
validation_values = np.array([1115.5, 325, 510])  # Announced projects in 2025

# Plot results
plt.figure(figsize=(16, 8))
plt.plot(x_data, ALK_y_data2, 'o', color='black', label='Original data')
plt.plot(extended_years, pred_logistic_ALK, color='green', lw=3, label='Logistic regression')
plt.scatter(validation_years, validation_values, color='red', s=60, marker='X', label='Announced projects (2026-2028)')
plt.title('Logistic Regression Fit - Alkaline Electrolyser Installed Capacity')
plt.xlabel('Year')
plt.ylabel('Installed Capacity (MW)')
plt.legend()
plt.show()

print("Fitted parameters:", params)
print(pred_logistic_ALK)

#%% Logistic regression – Three Scenarios
# Define x and y
x_data = oj_years
ALK_y_data = ALK_values
ALK_y_data2 = ALK_y_data.cumsum()

# Total EU electrolyzer capacity in 2050 (MW)
TOTAL_2050 = 500_000  # 500 GW

# Scenario shares
scenarios = {
    "Conservative (15%)": 0.15,
    "Moderate (20%)": 0.20,
    "High (30%)": 0.30
}

# Storage for predictions
scenario_predictions = {}

plt.figure(figsize=(16, 8))

# Plot historical data
plt.plot(x_data, ALK_y_data2, 'o', color='black', label='Historical data')

for name, share in scenarios.items():
    
    C1_fixed = TOTAL_2050 * share  # saturation level
    C0_fixed = 0
    
    # Define logistic with fixed C0 and C1
    def logistic_fixed(t, ti, tau):
        return logistic(t, ti, tau, C0_fixed, C1_fixed)
    
    # Initial guess for ti and tau
    p0 = [2028, 3]
    
    params, _ = curve_fit(
        logistic_fixed,
        x_data,
        ALK_y_data2,
        p0=p0,
        bounds=([1990, 0.1], [2050, 20]),
        maxfev=100000
    )
    
    ti_fit, tau_fit = params
    
    # Predict
    pred = logistic_fixed(extended_years, ti_fit, tau_fit)
    pred = np.maximum(pred, 0)
    
    scenario_predictions[name] = pred
    
    plt.plot(extended_years, pred, lw=3, label=f'{name}')

plt.title('Alkaline Electrolyser Installed Capacity – Scenario Projections')
plt.xlabel('Year')
plt.ylabel('Installed Capacity (MW)')
plt.legend()
plt.show()

#%%
for name, pred in scenario_predictions.items():
    print(f"\n{name}")
    print("2050 capacity (MW):", pred[-1])

#%%
print(pred_logistic_ALK.cumsum())

#%%
#%% Read PEM historic data
PEM_historic_data = r'C:\Users\idapo\master_thesis\master\data\processed_baseline\electrolysers\type_split\PEM_historic_capacity.csv'
PEM_historic_df = pd.read_csv(PEM_historic_data)
PEM_values = PEM_historic_df['Installed_Capacity_MW'].to_numpy()
#%%
#%% Logistic regression
# Define x and y
PEM_y_data = PEM_values

# Initial guess for parameters [ti, tau, C0, C1]
# ti ~ midpoint, tau ~ transition width, C0 ~ min, C1 ~ max
p0 = [2020, 5, np.min(PEM_y_data), 600000]

params, _ = curve_fit(
    logistic,
    x_data,
    PEM_y_data,
    p0=p0,
    bounds=([1990, 0.1, 0, 10000], [2050, 50, 100000, 1e6]),
    maxfev=100000
)

# Extract fitted parameters
ti_fit, tau_fit, C0_fit, C1_fit = params

# Predict for extended range
pred_logistic_PEM = logistic(extended_years, ti_fit, tau_fit, C0_fit, C1_fit)

# Ensure predictions stay positive
pred_logistic_PEM = np.maximum(pred_logistic_PEM, 0)

# Validation data for future years 2026-2028
#validation_years = np.array([2026, 2027, 2028])
#validation_values = np.array([1115.5, 325, 510])  # Announced projects in 2025

# Plot results
plt.figure(figsize=(16, 8))
plt.plot(x_data, PEM_y_data, 'o', color='black', label='Original data')
plt.plot(extended_years, pred_logistic_PEM, color='green', lw=3, label='Logistic regression')
#plt.scatter(validation_years, validation_values, color='red', s=60, marker='X', label='Announced projects (2026-2028)')
plt.title('Logistic Regression Fit - PEM Electrolyser Installed Capacity')
plt.xlabel('Year')
plt.ylabel('Installed Capacity (MW)')
plt.legend()
plt.show()

print("Fitted parameters:", params)
print(pred_logistic_PEM)

#%% Compare PEM and ALK predictions
plt.figure(figsize=(16, 8))
#plt.plot(x_data, ALK_y_data, 'o', color='black', label='ALK Original data')
#plt.plot(x_data, PEM_y_data, 'o', color='blue', label='PEM Original data')
plt.plot(extended_years, pred_logistic_ALK, color='green', lw=3, label='ALK Logistic regression')
plt.plot(extended_years, pred_logistic_PEM, color='cyan', lw=3, label='PEM Logistic regression')
plt.title('Logistic Regression Fit - Alkaline vs PEM Electrolyser Installed Capacity')
plt.legend()
plt.xlabel('Year')
plt.ylabel('Installed Capacity (MW)')
plt.show()

#%% Cumulative capacity comparison
plt.figure(figsize=(16, 8))
plt.plot(extended_years, pred_logistic_ALK.cumsum(), color='green', lw=3, label='ALK Cumulative Capacity')
plt.plot(extended_years, pred_logistic_PEM.cumsum(), color='cyan', lw=3, label='PEM Cumulative Capacity')
plt.title('Cumulative Installed Capacity - Alkaline vs PEM Electrolyser')
plt.legend()
plt.xlabel('Year')
plt.ylabel('Cumulative Installed Capacity (MW)')
plt.show()

#%%

print(pred_logistic_ALK.cumsum())
print(pred_logistic_PEM.cumsum())