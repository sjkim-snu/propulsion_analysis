import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
import pandas as pd
from scipy.interpolate import splprep, splev
import os

initial_battery_capacity = 2200

# read data 22.5A
Batt_file_path = r"C:\Users\ksjsms\.vscode\2025\AIAA\propulsion\data\battery\80A.xlsx"
# Batt_file_path = r"D:/Drone/80A
#.excel" # 80A discharge data
df = pd.read_excel(Batt_file_path) 
#df = pd.read_excel(Batt_file_path, skiprows=3711, on_bad_lines='skip') # 80A
df.columns = ["Current", "Voltage","Power", "Thrust"]

df["Current"] = pd.to_numeric(df["Current"], errors='coerce')
df["Voltage"] = pd.to_numeric(df["Voltage"], errors='coerce')
df["Power"] = pd.to_numeric(df["Power"], errors='coerce')

# SOC(State of Charge)
dt = 0.1 # Current interval 1 second
# 80A
# : 2.1212 / 80A : 1.25
df["SOC (%)"] = 100 - (df["Power"].cumsum() * dt / 3600) / (initial_battery_capacity / 1000) * 100

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
# Read Data 80A
df = pd.read_excel(Batt_file_path, skiprows=3711, on_bad_lines='skip') 
df.columns = ["Test", "Current (s)", "Voltage (V)", "Power", "Temp (F)"]

df["Current (s)"] = pd.to_numeric(df["Current (s)"], errors='coerce')
df["Voltage (V)"] = pd.to_numeric(df["Voltage (V)"], errors='coerce')
df["Power"] = pd.to_numeric(df["Power"], errors='coerce')

# SOC(State of Charge)
dt = 1  # Current interval 1 second
df["SOC (%)"] = 100 - (df["Power"].cumsum() * dt / 3600) / (initial_battery_capacity / 1000) * 100

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
    Voltage = (SoC2Vol_22(i)+SoC2Vol_50(i))*0.5
    print(str(i) + ', '+ str(Voltage))
"""


# min_SOC = max(min(filtered_df_22["SOC (%)"]), min(filtered_df_50["SOC (%)"]))
# max_SOC = min(max(filtered_df_22["SOC (%)"]), max(filtered_df_50["SOC (%)"]))
# middle_SOC = (min_SOC + max_SOC) / 2

# # 두 interpolation 함수에서의 전압값 계산
# Voltage_22 = SoC2Vol_22(middle_SOC)
# Voltage_50 = SoC2Vol_50(middle_SOC)


# Plot SoC vs Voltage
plt.figure(figsize=(7, 5))
plt.plot(filtered_df_22["SOC (%)"], filtered_df_22["Voltage"], label="Voltage vs SoC", color='blue', linewidth=2)

# Add labels and title
plt.xlabel("SOC (%)", fontsize=14)
plt.ylabel("Voltage (V)", fontsize=14)
plt.title("Battery Voltage vs State of Charge (80A)", fontsize=16)
plt.ylim([20, 25.2])
plt.xlim([0, 105])
# Add grid and legend
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
# EPS 파일 저장
results_dir = "data/results"
os.makedirs(results_dir, exist_ok=True)
eps_filepath = os.path.join(results_dir, "80A_discharge.eps")
plt.savefig(eps_filepath, format='eps', dpi=300)
print(f"Plot saved as {eps_filepath}")
plt.show()


for i in np.arange(0,100+0.1,0.1):
    print()
