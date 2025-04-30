import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt, pi

# 결과 저장 디렉터리 설정
results_dir = "data/results"
os.makedirs(results_dir, exist_ok=True)  # 디렉터리 없으면 생성

# Get Motor data
file_path = r'data/motor/scorpion_1100kv.xlsx'
data = pd.ExcelFile(file_path)
df = data.parse(data.sheet_names[0])
df_cleaned = df.dropna(subset=['Throttle', 'Voltage', 'Current', 'Torque', 'RPM'])

# Linear Regression
current = df_cleaned['Current'].values
torque = df_cleaned['Torque'].values
X = torque.reshape(-1, 1)
y = current
model = LinearRegression()
model.fit(X, y)
kv = model.coef_[0]
print(kv)

# Optimizing R
R_initial = 0.05
R_step = 0.001  

def calculate_rmse(R_value, kv, df_cleaned):
    throttle = df_cleaned['Throttle'].values
    voltage = df_cleaned['Voltage'].values
    current = df_cleaned['Current'].values
    rpm_measured = df_cleaned['RPM'].values
    
    rpm_predicted = kv * (throttle * voltage - current * R_value) * (30 / pi)
    rmse = sqrt(mean_squared_error(rpm_measured, rpm_predicted))
    return rmse, rpm_predicted

def optimize_R(kv, df_cleaned, R_initial, R_step, tolerance=1e-6):
    R = R_initial
    direction = 1
    best_rmse = float('inf')
    best_R = R_initial
    best_rpm_predicted = None
    
    while True:
        rmse, rpm_predicted = calculate_rmse(R, kv, df_cleaned)
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_R = R
            best_rpm_predicted = rpm_predicted
        else:
            direction *= -1
            R_step /= 2
        
        R += direction * R_step
        if R_step < tolerance:
            break
    
    return best_R, best_rmse, best_rpm_predicted

best_R, best_rmse, best_rpm_predicted = optimize_R(kv, df_cleaned, R_initial, R_step)

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

axs[0].scatter(torque, current, label='Data', alpha=0.7)
axs[0].plot(torque, model.predict(X), color='red', label=f'Linear Regression kv={kv * 60 / (2 * np.pi):.3f}')
axs[0].set_xlabel('Torque (Nm)', fontsize=21)
axs[0].set_ylabel('Current (A)', fontsize=21)
axs[0].set_title('Torque vs Current', fontsize=21)
axs[0].legend(fontsize=16)
axs[0].grid(True, linestyle='--', alpha=0.7)

axs[1].plot(df_cleaned['RPM'].values, label='Measured RPM', color='blue', alpha=0.7)
axs[1].plot(best_rpm_predicted, label=f'Predicted RPM (Best R={best_R:.5f})', color='red', linestyle='--', alpha=0.7)
axs[1].set_xlabel('', fontsize=21)
axs[1].set_ylabel('RPM', fontsize=21)
axs[1].set_title('Measured vs Predicted RPM', fontsize=21)
axs[1].legend(fontsize=16)
axs[1].grid(True, linestyle='--', alpha=0.7)
axs[1].set_xticks([])

# fig.suptitle('Data of Scorpion MII-3011-1100kv', fontsize=16, fontweight='bold')
plt.tight_layout()

# 저장 경로 설정 및 EPS 저장
eps_filepath = os.path.join(results_dir, "scorpion_1100kv_analysis.eps")
plt.savefig(eps_filepath, format='eps', dpi=300)
print(f"Plot saved as {eps_filepath}")

plt.show()

print(f"최적의 R값: {best_R}")
print(f"최소 RMSE: {best_rmse}")
