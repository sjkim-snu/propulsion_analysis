import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
import pandas as pd
from scipy.interpolate import splprep, splev

initial_battery_capacity = 2200

# read data 22.5A
Batt_file_path = r"data/battery/Vega Discharge 80A_analysis.csv"
# Batt_file_path = r"data/battery/2.25Ah Discharge Profile.csv" # Old data
df = pd.read_csv(Batt_file_path, on_bad_lines='skip') 
#df = pd.read_csv(Batt_file_path, skiprows=3711, on_bad_lines='skip') # 50A
df.columns = ["Current", "Voltage", "Power", "Time"]

df["Time"] = pd.to_numeric(df["Time"], errors='coerce')
df["Voltage"] = pd.to_numeric(df["Voltage"], errors='coerce')
df["Current"] = pd.to_numeric(df["Current"], errors='coerce')

# SOC(State of Charge)
dt = 1.25 # time interval 1 second
# 30A : 2.1212 / 80A : 1.25
df["SOC (%)"] = 100 - (df["Current"].cumsum() * dt / 3600) / (initial_battery_capacity / 1000) * 100

# SoC must be positive
filtered_df_22 = df[df["SOC (%)"] >= 0]

# Interpolation
SoC2Vol_22 = interp1d(
    filtered_df_22["SOC (%)"],
    filtered_df_22["Voltage"], 
    kind='linear',
    fill_value="extrapolate"
)


"""
# Read Data 50A
df = pd.read_csv(Batt_file_path, skiprows=3711, on_bad_lines='skip') 
df.columns = ["Test", "Time (s)", "Voltage (V)", "Current", "Temp (F)"]

df["Time (s)"] = pd.to_numeric(df["Time (s)"], errors='coerce')
df["Voltage (V)"] = pd.to_numeric(df["Voltage (V)"], errors='coerce')
df["Current"] = pd.to_numeric(df["Current"], errors='coerce')

# SOC(State of Charge)
dt = 1  # time interval 1 second
df["SOC (%)"] = 100 - (df["Current"].cumsum() * dt / 3600) / (initial_battery_capacity / 1000) * 100

# SoC must be positive
filtered_df_50 = df[df["SOC (%)"] >= 0]

# Interpolation
SoC2Vol_50 = interp1d(
    filtered_df_50["SOC (%)"],
    filtered_df_50["Voltage (V)"], 
    kind='linear',
    fill_value="extrapolate"
)

for i in np.arange(0,100.1,0.1):
    voltage = (SoC2Vol_22(i)+SoC2Vol_50(i))*0.5
    print(str(i) + ', '+ str(voltage))
"""


# min_SOC = max(min(filtered_df_22["SOC (%)"]), min(filtered_df_50["SOC (%)"]))
# max_SOC = min(max(filtered_df_22["SOC (%)"]), max(filtered_df_50["SOC (%)"]))
# middle_SOC = (min_SOC + max_SOC) / 2

# # 두 interpolation 함수에서의 전압값 계산
# voltage_22 = SoC2Vol_22(middle_SOC)
# voltage_50 = SoC2Vol_50(middle_SOC)


# Plot SoC vs Voltage
plt.figure(figsize=(7, 5))
plt.plot(filtered_df_22["SOC (%)"], filtered_df_22["Voltage"], label="Voltage vs SoC", color='blue', linewidth=2)

# Add labels and title
plt.xlabel("SOC (%)", fontsize=14)
plt.ylabel("Voltage (V)", fontsize=14)
plt.title("Battery Voltage vs State of Charge (30A)", fontsize=16)

# Add grid and legend
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.show()


for i in range(0,100+0.1,0.1):
    print()
