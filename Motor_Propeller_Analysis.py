import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import os

# Get Propeller data
csvPath = r"data/propDataCSV/PER3_11x45EP.csv"
propeller_df = pd.read_csv(csvPath,skiprows=[1])
propeller_df.dropna(how='any',inplace=True)
propeller_df = propeller_df.sort_values(by=['RPM', 'V(speed) (m/s)']).reset_index(drop=True)

rpm_array = propeller_df['RPM'].to_numpy()
v_speed_array = propeller_df['V(speed) (m/s)'].to_numpy()
torque_array = propeller_df['Torque (N-m)'].to_numpy()
thrust_array = propeller_df['Thrust (kg)'].to_numpy()
propeller_array = np.column_stack((rpm_array, v_speed_array, torque_array, thrust_array))


def thrust_analysis(
    throttle: float,
    speed: float,
    voltage: float,
    Kv: float,
    R: float,
    max_current: float,
    max_power: float,
    propeller_array: np.ndarray,
    graphFlag: bool
):
    expanded_results_array = propeller_fixspeed_data(speed, propeller_array)

    I_list = np.arange(0, min(max_current, max_power / voltage) + 0.5, 1)
    RPM_list = Kv * (voltage * throttle - I_list * R) * 30 / math.pi
    Torque_list = I_list / Kv

    motor_results_array = np.column_stack((I_list, RPM_list, Torque_list))

    if graphFlag:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(expanded_results_array[:, 0], expanded_results_array[:, 1], label="Propeller Torque")
        ax.plot(motor_results_array[:, 1], motor_results_array[:, 2], label="Motor Torque",color='orange')
        ax.set_xlim(0, 25000)
        ax.set_ylim(0, 1.4)
        ax.set_xlabel("RPM",fontsize = 22)
        ax.set_ylabel("Torque (N-m)",fontsize = 22)
        ax.set_title(f"{speed} m/s, Throttle {throttle*100}%", fontsize = 22)
        ax.grid(True)
        ax.legend()
        
        # EPS 파일 저장
        results_dir = "data/results"
        os.makedirs(results_dir, exist_ok=True)
        eps_filepath = os.path.join(results_dir, "20ms_0.8throttle.eps")
        plt.savefig(eps_filepath, format='eps', dpi=300)
        print(f"Plot saved as {eps_filepath}")
        
        plt.show()

    motor_sorted = motor_results_array[motor_results_array[:, 1].argsort()]

    min_rpm = max(motor_sorted[:, 1].min(), expanded_results_array[:, 0].min())
    max_rpm = min(motor_sorted[:, 1].max(), expanded_results_array[:, 0].max())

    # Check for no intersection
    if max_rpm < min_rpm:
        return 0, 0, 0, 0, 0

    rpm_interp = np.linspace(min_rpm, max_rpm, 500)

    torque1 = np.interp(rpm_interp, motor_sorted[:, 1], motor_sorted[:, 2])
    torque2 = np.interp(rpm_interp, expanded_results_array[:, 0], expanded_results_array[:, 1])
    diff = torque1 - torque2

    sign_changes = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(sign_changes) == 0:
        return 0, 0, 0, 0, 0

    idx = sign_changes[0]

    RPM = rpm_interp[idx]
    Torque = torque1[idx]
    motor_torque_sorted = motor_sorted[motor_sorted[:, 2].argsort()]
    I = np.interp(Torque, motor_torque_sorted[:, 2], motor_torque_sorted[:, 0])
    Power = I * voltage
    Thrust = np.interp(RPM, expanded_results_array[:, 0], expanded_results_array[:, 2])

    return RPM, Torque, I, Power, Thrust




def determine_max_thrust(speed:float, voltage:float, Kv:float, R:float, max_current:float, max_power:float, propeller_array:np.ndarray, graphFlag:bool):

    propeller_array_fixspeed = propeller_fixspeed_data(speed,propeller_array)
    
    propeller_sortedby_torque = propeller_array_fixspeed[propeller_array_fixspeed[:, 1].argsort()]  
    
    max_torque = min(max_current,max_power/voltage)/Kv
    propeller_max_rpm = np.interp(max_torque,propeller_sortedby_torque[:, 1],propeller_sortedby_torque[:, 0])
    motor_max_rpm = Kv*(voltage-min(max_current,max_power/voltage)*R)*30/math.pi


    if graphFlag == 1:
        
        I_list = np.arange(0,min(max_current,max_power/voltage)+0.5,1)
        RPM_list = Kv * (voltage - I_list * R) * 30 / math.pi
        Torque_list = I_list / Kv

        motor_results_array = np.column_stack((I_list,RPM_list,Torque_list))
        
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        axs[0].plot( propeller_array_fixspeed[:,0], propeller_array_fixspeed[:,1], label='Propeller')
        axs[0].plot(motor_results_array[:,1],motor_results_array[:,2], label='Motor')

        axs[0].set_xlabel('RPM')
        axs[0].set_ylabel('Torque (N-m)')
        axs[0].set_title('Torque (N-m) vs RPM')
        axs[0].grid(True)
        axs[0].legend()
        
        axs[1].plot(propeller_array_fixspeed[:,0],propeller_array_fixspeed[:,2], label='Propeller')
        axs[1].set_xlabel('RPM')
        axs[1].set_ylabel('Thrust (kg)')
        axs[1].set_title('Thrust (kg) vs RPM')
        axs[1].grid(True)
        axs[1].legend()
        plt.show()
        
        
    if motor_max_rpm >= propeller_max_rpm:
        max_thrust = np.interp(max_torque,propeller_sortedby_torque[:,1],propeller_sortedby_torque[:,2])
        return max_thrust
    else:
        
        ## find intersection
        I_list = np.arange(0,min(max_current,max_power/voltage)+0.5,1)
        RPM_list_fullthrottle = Kv * (voltage - I_list * R) * 30 / math.pi
        Torque_list_fullthrottle = I_list / Kv

        motor_results_array = np.column_stack((I_list,RPM_list_fullthrottle,Torque_list_fullthrottle))       
        motor_sorted = motor_results_array[motor_results_array[:, 1].argsort()] 

        min_rpm = max(motor_sorted[:, 1].min(), propeller_array_fixspeed[:, 0].min())
        max_rpm = min(motor_sorted[:, 1].max(), propeller_array_fixspeed[:, 0].max())

        if max_rpm < min_rpm: # Propeller Windmilling
            return 0

        rpm_interp = np.linspace(min_rpm, max_rpm, 500)

        torque1 = np.interp(rpm_interp, motor_sorted[:, 1], motor_sorted[:, 2]) 
        torque2 = np.interp(rpm_interp, propeller_array_fixspeed[:, 0], propeller_array_fixspeed[:, 1]) 
        diff = torque1 - torque2

        sign_changes = np.where(np.diff(np.sign(diff)) != 0)[0]
        if len(sign_changes) == 0: # Overcurrent
            return np.interp(max_torque,propeller_sortedby_torque[:,1],propeller_sortedby_torque[:,2])

        idx = sign_changes[0]

        RPM = rpm_interp[idx]
        max_thrust = np.interp(RPM,propeller_array_fixspeed[:, 0], propeller_array_fixspeed[:, 2])
        
        return max_thrust
    
    
def propeller_fixspeed_data(speed,propeller_array):
    
    results=[]

    rpm_array = propeller_array[:, 0]      
    v_speed_array = propeller_array[:, 1]    
    torque_array = propeller_array[:, 2]    
    thrust_array = propeller_array[:, 3]     
    
    unique_rpms = sorted(set(rpm_array))
    
    for rpm in unique_rpms:
    
        mask = rpm_array == rpm

        v_subset = v_speed_array[mask]
        torque_subset = torque_array[mask]
        thrust_subset = thrust_array[mask]
    
        min_v = v_subset.min()
        max_v = v_subset.max()

        if min_v <= speed <= max_v:
            torque_at_v = np.interp(speed, v_subset, torque_subset)
            thrust_at_v = np.interp(speed, v_subset, thrust_subset)
            results.append({
            'RPM': rpm,
            'Torque (N-m)': torque_at_v,
            'Thrust (kg)': thrust_at_v
            })
            
    results_array = np.array([(d['RPM'], d['Torque (N-m)'], d['Thrust (kg)']) for d in results])

    rpm_values = results_array[:, 0] 
    torque_values = results_array[:, 1]  
    thrust_values = results_array[:, 2] 
    
    min_rpm = int(rpm_values.min())
    max_rpm = int(rpm_values.max())

    expanded_rpm_values = np.arange(min_rpm, max_rpm + 1, 100)

    torque_interpolated = np.interp(expanded_rpm_values, rpm_values, torque_values)
    thrust_interpolated = np.interp(expanded_rpm_values, rpm_values, thrust_values)
    
    propeller_array_fixspeed = np.column_stack((expanded_rpm_values, torque_interpolated, thrust_interpolated)) 
    return propeller_array_fixspeed

def thrust_reverse_solve(T_desired,speed,voltage, Kv, R, propeller_array):
    
    propeller_array_fixspeed = propeller_fixspeed_data(speed,propeller_array)
    propeller_sortedby_thrust = propeller_array_fixspeed[propeller_array_fixspeed[:, 2].argsort()]
    
    RPM_desired = np.interp(T_desired,propeller_sortedby_thrust[:,2],propeller_sortedby_thrust[:,0])
    torque_desired = np.interp(T_desired,propeller_sortedby_thrust[:,2],propeller_sortedby_thrust[:,1])
    
    I = Kv * torque_desired
    I = max(I,0)
    throttle = ((math.pi/30) * RPM_desired / Kv + I*R)/voltage
    throttle = max(throttle,0)
    Power = voltage * I
    
    return RPM_desired, torque_desired, I, Power, throttle

  
def save_analysis_for_multiple_speeds(
    speeds: list, 
    voltage: float, 
    Kv: float, 
    R: float, 
    max_current: float, 
    max_power: float, 
    propeller_array: np.ndarray, 
    filename: str
):
    # Ensure the results directory exists
    results_dir = "data/results"
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, filename)

    all_data = []  # Store data for all speeds

    for speed in speeds:
        throttle_percentages = np.arange(20, 101, 5)  # Throttle from 20% to 100%
        data = []

        for throttle in throttle_percentages / 100:
            try:
                RPM, Torque, I, Power, Thrust = thrust_analysis(
                    throttle, speed, voltage, Kv, R, max_current, max_power, propeller_array, graphFlag=False
                )
            except Exception:
                # If no valid data, set all values to 0
                RPM, Torque, I, Power, Thrust = 0, 0, 0, 0, 0

            # Append the result
            row = {
                "Speed (m/s)": speed,
                "Throttle (%)": throttle * 100,
                "RPM": round(RPM),
                "Thrust (kg)": round(Thrust, 2),
                "Current (A)": round(I, 2),
                "Power (W)": round(Power, 2),
                "Torque (Nm)": round(Torque, 2),
                "Note": "",  # Default empty note
            }
            data.append(row)

        # Add maximum thrust for this speed
        max_thrust = determine_max_thrust(speed, voltage, Kv, R, max_current, max_power, propeller_array, graphFlag=False)
        data.append({
            "Speed (m/s)": speed,
            "Throttle (%)": "Max Thrust",
            "RPM": "",
            "Thrust (kg)": round(max_thrust, 2),
            "Current (A)": "",
            "Power (W)": "",
            "Torque (Nm)": "",
            "Note": "",  # Empty note for max thrust row
        })

        all_data.extend(data)  # Append the results for this speed

    # Convert to DataFrame
    df = pd.DataFrame(all_data)

    # Ensure 'Current (A)' column is numeric, replacing non-numeric entries with NaN
    df['Current (A)'] = pd.to_numeric(df['Current (A)'], errors='coerce')

    # Add a note for overcurrent
    df.loc[df['Current (A)'] > max_current, 'Note'] = "불가능 (모터 최대 허용 전류 초과)"
    
    # Save to CSV
    df.to_csv(filepath, index=False, encoding='utf-8-sig')
    print(f"Results saved to {filepath}")





def plot_throttle_vs_speed(propeller_array, voltage, Kv, R, max_current, max_power, results_dir="data/results"):
    # 결과 저장 디렉터리 생성
    os.makedirs(results_dir, exist_ok=True)
    
    # EPS 파일 저장 경로 설정
    eps_filepath = os.path.join(results_dir, "540kv_20_13e.eps")

    speed_values = np.arange(0, 41, 5)
    throttle_min = []
    throttle_max = []
    thrust_max_values = []

    for speed in speed_values:
        thrust_range = propeller_array[np.isclose(propeller_array[:, 1], speed, atol=0.5)][:, 3]  # 속도 근사값으로 추출
        if len(thrust_range) > 0:
            min_thrust = thrust_range.min()
            max_thrust = thrust_range.max()
            
            # 최소 및 최대 추력에 해당하는 쓰로틀 계산
            _, _, _, _, throttle_min_val = thrust_reverse_solve(min_thrust, speed, voltage, Kv, R, propeller_array)
            _, _, _, _, throttle_max_val = thrust_reverse_solve(max_thrust, speed, voltage, Kv, R, propeller_array)
            
            throttle_min_val = min(throttle_min_val * 100, 100)
            throttle_max_val = min(throttle_max_val * 100, 100)
            
            if throttle_max_val <= 100:
                throttle_min.append(throttle_min_val)
                throttle_max.append(throttle_max_val)
            else:
                throttle_min.append(None)
                throttle_max.append(None)
        else:
            throttle_min.append(None)
            throttle_max.append(None)
        
        # determine_max_thrust 함수로 최대 추력 계산
        max_thrust = determine_max_thrust(speed, voltage, Kv, R, max_current, max_power, propeller_array, graphFlag=False)
        thrust_max_values.append(max_thrust)

    fig, ax1 = plt.subplots(figsize=(7, 5))

    # 쓰로틀 범위
    ax1.fill_between(speed_values, throttle_min, throttle_max, color='b', alpha=0.3, label='Throttle Range')
    ax1.set_xlabel("Speed (m/s)", fontsize=21)
    ax1.set_ylabel("Throttle (%)", fontsize=21, color='b')
    ax1.set_xlim(0, 40)
    ax1.set_ylim(20, 100)  # 쓰로틀 값 범위 제한
    ax1.grid(True)
    
    # 최대 추력 그래프 추가 (y축 공유)
    ax2 = ax1.twinx()
    ax2.plot(speed_values, thrust_max_values, color='r', linestyle='-', marker='o', label='Max Thrust')
    ax2.set_ylabel("Thrust (kg)", fontsize=21, color='r')
    ax2.set_xlim(0, 40)
    ax2.set_ylim(1, 8)  # 쓰로틀 값 범위 제한
    fig.suptitle("A4220-540Kv & APC 12*45MR", fontsize=16)
    
    # 두 개의 legend를 하나로 합쳐서 표시
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=True, fontsize=16)
    
    # EPS 파일 저장
    plt.savefig(eps_filepath, format='eps', dpi=300)
    print(f"Plot saved as {eps_filepath}")
    
    plt.show()



# #실행 코드 (EPS 파일이 data/results/ 에 저장됨)
# plot_throttle_vs_speed(propeller_array, voltage=23.0, Kv=109.92194, R=0.05933, max_current=60, max_power=1332)


# RPM, Torque, I, Power, Thrust = thrust_analysis(
#     throttle=0.8,     
#     speed=20,     
#     voltage=23.0,  
#     Kv=109.92194,       
#     R=0.05933,       
#     max_current=60,   
#     max_power=1332,   
#     propeller_array=propeller_array, 
#     graphFlag=True    
# )


# Example usage:
save_analysis_for_multiple_speeds(
    speeds=[0, 10, 20, 30, 40],  # Speed values to analyze
    voltage=23.0,  # Battery voltage
    Kv=55.14263,  # Motor Kv
    R=0.079,  # Motor resistance
    max_current=115,  # Motor maximum allowable current
    max_power=2553,  # Motor maximum allowable power
    propeller_array=propeller_array,
    filename="540kv_12_45EP.csv"  # Output file name
)