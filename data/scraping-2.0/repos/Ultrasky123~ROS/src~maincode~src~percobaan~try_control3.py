import time
from pymavlink import mavutil
from PyMavlinkzz import Guidance
import altitude

# jika koneksi langsung komputer/nuc
# master = mavutil.mavlink_connection("/dev/ttyACM0", baud=115200)

# koneksi jika pakai companion
master = mavutil.mavlink_connection('udpin:0.0.0.0:14550')

master.wait_heartbeat()

# catatan: asumsi robot dalam keadaan default mengambang

def main(pm: Guidance):
    # pm.armDisarm()
    pm.arm()

    # target depth
    depth_target = -0.8

    # variabel PID
    kp = 160
    ki = 0.0
    kd = 35
    
    # rentang toleransi kedalaman (meter)
    depth_tolerance = 0.05

    def depth_hold():
        while True:
            # membaca nilai kedalaman
            depth = altitude.get_altitude()

            #RUMUS PID
            depth_error_prev = 0
            # depth_error_sum = 0

            # hitung error kedalaman 
            depth_error = depth - depth_target

            # hitung turunan dari error Kd
            depth_error_diff = depth_error - depth_error_prev
            depth_error_prev = depth_error

            # hitung output PID
            pid_output =  (kp * depth_error) +(kd * depth_error_diff) #+(depth_error_sum * ki)
            # print(depth-depth_target)

            # Jika error kedalaman lebih kecil dari toleransi thruster off
            if abs(depth - depth_target) <= depth_tolerance:
                # set kecepatan Motor 1500 = 0
                pm.setRcValue(3, 1500)
                print("thruster mati")

            else:
                #jika dept kurang dari target
                if depth < depth_target:
                    # hitung kecepatan motor
                    speed = 1500-abs(int(pid_output))

                    # batasi kecepatan agar tidak terlalu besar atau kecil
                    if speed >= 1850:
                        speed = 1850
                    elif speed <= 1150:
                        speed = 1150

                    # set kecepatan motor
                    pm.setRcValue(3, speed)
                    print("thruster nyala")
                    print(speed)
                
                else:
                    # hitung kecepatan motor
                    speed = 1500 + abs(int(pid_output))

                    # batasi kecepatan agar tidak terlalu besar atau kecil
                    if speed >= 1850:
                        speed = 1850
                    elif speed <= 1150:
                        speed = 1150

                    # set kecepatan motor
                    pm.setRcValue(3, speed)
                    print("thruster nyala")
                    print(speed)
                

            # Jika misi sudah selesai, reset kedalaman menjadi 0
            # if mission_complete:
            #     depth_target = 0

            time.sleep(0.1)

    depth_hold()

    #pm.disarm()

if __name__ == '__main__':
    pymavlink = Guidance(master)
    main(pymavlink)
