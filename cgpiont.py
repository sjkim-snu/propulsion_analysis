import serial as arduino
import math

def update_arduino_data(self, dt):
    '''🔥 아두이노에서 두 개의 데이터를 읽어와 UI에 업데이트하는 함수'''
    if arduino is None:
        self.FrontToBodyState_Label = "❌ Arduino Not Connected!"
        self.RightToLeftState_Label = "❌ Arduino Not Connected!"
        return

    try:
        if arduino.in_waiting > 0:  # 데이터가 있는 경우만 읽기
            data = arduino.readline().decode("utf-8").strip()
            values = data.split(",")  # 🔥 `,` 기준으로 데이터 분리
            values = [float(value) for value in values]
            

            if len(values) == 3:  # 🔥 데이터가 3개일 때만 처리

                if 0 < values[0] < 200:
                    values[0] = 0.01698*math.exp(0.02585*values[0])
                elif 200 < values[0] < 240:
                    values[0] = 0.000346*math.exp(0.04510*values[0])    

                if 0 < values[1] < 200:
                    values[1] = 0.01698*math.exp(0.02585*values[1])
                elif 200 < values[1] < 240:
                    values[1] = 0.000346*math.exp(0.04510*values[1])    

                if 0 < values[2] < 200:
                    values[2] = 0.01698*math.exp(0.02585*values[2])
                elif 200 < values[2] < 240:
                    values[2] = 0.000346*math.exp(0.04510*values[2])    

                Front_Gear = values[0]
                Body_Gear_1 = values[1]
                Body_Gear_2 = values[2]

                N_nose_gear = Front_Gear
                N_left_body_gear = Body_Gear_1
                N_right_body_gear = Body_Gear_2
                N_total_body_gear = N_left_body_gear + N_right_body_gear
                N_total = N_nose_gear + N_left_body_gear + N_right_body_gear
                global percent_nose_gear
                global percent_body_gear
                global percent_left_body_gear
                global percent_right_body_gear

                percent_nose_gear = 0
                percent_body_gear = 0
                percent_left_body_gear = 0
                percent_right_body_gear = 0

                if N_total != 0:
                    percent_nose_gear = (N_nose_gear / N_total) * 100
                    percent_body_gear = ((N_left_body_gear + N_right_body_gear) / N_total) * 100

                if N_total_body_gear != 0:
                    percent_left_body_gear = (N_left_body_gear / N_total_body_gear) * 100
                    percent_right_body_gear = (N_right_body_gear / N_total_body_gear) * 100

                self.FrontPercent_Label.text = f"Front Percent: {percent_nose_gear:,.2f}%"
                self.BodyPercent_Label.text = f"Body Percent: {percent_body_gear:,.2f}%"
                #14% 86%   0.25

                if percent_nose_gear > 14.25:
                    self.FrontToBodyState_Label.text = "<Nose Heavy>"
                elif percent_nose_gear < 13.75 and percent_nose_gear > 0:
                    self.FrontToBodyState_Label.text =  "<Tail Heavy>"
                elif percent_nose_gear >= 13.75 and percent_nose_gear <= 14.25:
                    self.FrontToBodyState_Label.text = "<Balanced>"
                elif percent_nose_gear == 0:
                    self.FrontToBodyState_Label.text = "State: Waiting..."
                elif percent_nose_gear == 50.0:
                    self.FrontToBodyState_Label.text = "Lucky Guy!!!"
                
                if percent_right_body_gear > 50.25:
                    self.RightToLeftState_Label.text = "<Right Heavy>"
                elif percent_right_body_gear < 49.75 and percent_body_gear > 0:
                    self.RightToLeftState_Label.text =  "<Left Heavy>"
                elif percent_right_body_gear >= 49.75 and percent_body_gear <= 50.25:
                    self.RightToLeftState_Label.text = "<Balanced>"
                elif percent_right_body_gear == 0:
                    self.RightToLeftState_Label.text = "State: Waiting..."
                elif percent_right_body_gear == 50.0:
                    self.RightToLeftState_Label.text = "Lucky Guy!!!"

                self.RightPercent_Label.text = f"Right Percent: {percent_right_body_gear:,.2f}%"
                self.LeftPercent_Label.text = f"Left Percent: {percent_left_body_gear:,.2f}%"
            else:
                self.RightToLeftState_Label = "❌ Invalid Data Format!"
                self.FrontToBodyState_Label = "❌ Invalid Data Format!"
    except Exception as e:
        print(f"❌ 아두이노 데이터 읽기 오류: {e}")
        self.RightToLeftState_Label = "❌ Error Reading Data!"
        self.FrontToBodyState_Label = "❌ Error Reading Data!"