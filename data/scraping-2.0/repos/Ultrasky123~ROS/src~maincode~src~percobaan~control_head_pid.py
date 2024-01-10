import time
from pymavlink import mavutil
from PyMavlinkin import Guidance
import heading

# jika koneksi langsung komputer/nuc
master = mavutil.mavlink_connection("/dev/ttyACM0", baud=115200)

# koneksi jika pakai companion
# master = mavutil.mavlink_connection('udpin:0.0.0.0:14550')

master.wait_heartbeat()

def main(pm: Guidance):
    # pm.armDisarm()
    pm.arm()

    # target depth

    def set_head():
        while True:
            head_target = 250

            # variabel PID  
            kp = 150
            ki = 0.0
            kd = 30

            # rentang toleransi kedalaman (meter)
            head_tolerance = 0.5

            # membaca nilai kedalaman
            head = heading.get_heading()

            #RUMUS PID
            head_error_prev = 0
            # depth_error_sum = 0

        
            head_error = head_target-head
            if head_error > 180:
                head_error -= 360
            elif head_error < -180:
                head_error += 360

            # hitung turunan dari error Kd
            head_error_diff = head_error - head_error_prev
            head_error_prev = head_error

            # hitung output PID
            pid_output =  (kp * head_error) +(kd * head_error_diff) #+(depth_error_sum * ki)
            # print(depth-depth_target)

            

            # Jika error kedalaman lebih kecil dari toleransi thruster off
            if abs(head - head_target) <= head_tolerance:
                # set kecepatan Motor 1500 = 0
                speed = 1500 
                pm.setRcValue(4, speed)
                print("thruster low")
                print(speed , "head error : ", head_error)

            else:
                #jika dept kurang dari target
                
                
                if head_error < 0:
                    # hitung kecepatan motor
                    speed = 1500-abs(int(pid_output))

                    # batasi kecepatan agar tidak terlalu besar atau kecil
                    
                    if speed >= 1850:
                        speed = 1850
                    elif speed <= 1250:
                        speed = 1250

                    # set kecepatan motor
                    pm.setRcValue(4, speed)
                    print("thruster nyala")
                    print(speed , "head error : ", head_error)
                
                else:
                    # hitung kecepatan motor
                    speed = 1500 + abs(int(pid_output))

                    # batasi kecepatan agar tidak terlalu besar atau kecil
                    if speed >= 1850:
                        speed = 1850
                    elif speed <= 1250:
                        speed = 1250

                    # set kecepatan motor
                    pm.setRcValue(4, speed)
                    print("thruster nyala")
                    print(speed)
                

            # Jika misi sudah selesai, reset kedalaman menjadi 0
            # if mission_complete:
            #     depth_target = 0

            time.sleep(0.1)

    set_head()

    #pm.disarm()

if __name__ == '__main__':
    pymavlink = Guidance(master)
    main(pymavlink)
