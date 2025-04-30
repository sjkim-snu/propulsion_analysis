import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import os

# Get analysis data
csvPath = r"data/analysis_withpayload.csv"
analysis_df = pd.read_csv(csvPath)
analysis_df.dropna(how='any',inplace=True)
analysis_df = analysis_df.sort_values(by=['Angle_of_Attack', 'cd', 'cl']).reset_index(drop=True)

aoa_array = analysis_df['Angle_of_Attack'].to_numpy()
cd_array = analysis_df['cd'].to_numpy()
cl_array = analysis_df['cl'].to_numpy()

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(aoa_array, cd_array, label="Drag Coefficient",color='orange')
ax.plot(aoa_array, cl_array, label="Lift Coefficient",color='blue')
ax.set_xlim(-7, 22)
ax.set_ylim(-0.2, 1.6)
ax.set_xlabel("Angle of Attack",fontsize = 14)
ax.set_ylabel("Coefficient Values",fontsize = 14)
ax.set_title(f"AoA vs Lift and Drag Coefficients (Mission 2 & Mission 3)")
ax.grid(True)
ax.legend()

results_dir = "data/results"
os.makedirs(results_dir, exist_ok=True)
eps_filepath = os.path.join(results_dir, "Mission2_3_ansys.eps")
plt.savefig(eps_filepath, format='eps', dpi=300)
print(f"Plot saved as {eps_filepath}")
plt.show()