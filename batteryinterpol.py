import pandas as pd
import numpy as np

# 엑셀 파일 로드
file_path = r"data/battery/배터리 분석결과 (1).xlsx"
xls = pd.ExcelFile(file_path)

# 첫 번째 시트의 데이터 로드
df = pd.read_excel(xls, sheet_name='Sheet1')

# 첫 번째 행 제거 후 유효한 데이터 선택
df_filtered = df.iloc[1:].copy()
df_filtered = df_filtered[['SoC', 'Voltage', 'SoC.1', 'Voltage.1']].astype(float)

# 중간값 계산
df_filtered['SoC_mid'] = (df_filtered['SoC'] + df_filtered['SoC.1']) / 2
df_filtered['Voltage_mid'] = (df_filtered['Voltage'] + df_filtered['Voltage.1']) / 2

# 결과 저장
output_file = "midpoint_data.xlsx"
df_filtered[['SoC_mid', 'Voltage_mid']].to_excel(output_file, index=False)

print(f"중간값 데이터를 {output_file} 파일로 저장했습니다.")
