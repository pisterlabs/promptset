import time
from pymavlink import mavutil
from PyMavlinkzz import Guidance
import altitude

#jika koneksi langsung komputer/nuc
master = mavutil.mavlink_connection("/dev/ttyACM0", baud=115200)

#koneksi jika pakai companion
# master = mavutil.mavlink_connection('udpin:0.0.0.0:14550')

master.wait_heartbeat()

# CATATAN: ASUMSI ROBOT DALAM KEADAAN DEFAULT MENGAMBANG

def main(pm: Guidance):

    # pm.armDisarm()
    pm.arm()

    # target depth
    depth_target=-0.5

    #variabel PID 
    kp=100
    # ki=0
    kd=50
    
    # Variable error dan variabel integral untuk kontrol PID
    depth_error_prev = 0
    # depth_error_sum = 0

    # Rentang toleransi kedalaman (meter)
    depth_tolerance = 0.1
    
    def pid_control(depth, depth_target):
        global depth_error_prev, depth_error_sum

        # Hitung error kedalaman Kp
        depth_error = depth - depth_target

        # Hitung integral dari error Ki
        # depth_error_sum += depth_error

        # Hitung turunan dari error Kd
        depth_error_diff = depth_error - depth_error_prev
        depth_error_prev = depth_error

        # Hitung output PID
        pid_output = (kp * depth_error) + (kd * depth_error_diff)

        return pid_output

    def depth_hold():
        
       
        while True:
            #membaca nilai kedalaman 
            depth= altitude.get_altitude()

            # memanggil rumus function pid
            pid_output = pid_control(depth, depth_target)

            # Jika kedalaman berada dalam rentang toleransi
            if abs(depth - depth_target) <= depth_tolerance:
                #set kecepatan Motor 1500 = 0
                pm.setRcValue(3,1500)
                print("thruster mati")

            
            else :
        
                # Hitung kecepatan motor
                speed = pid_output
                print(speed)
                # set kecepatan motor  
                pm.setRcValue(3,speed)
                # time.sleep(3)
                # pm.setRcValue(3,1500)
                print("thruster nyala")
            
            
            # Jika misi sudah selesai, reset kedalaman menjadi 0
            # if mission_complete:
            #     depth_target = 0    
            # time.sleep(0.1)

    
    depth_hold()
            
    # pm.disarm()
    
   

if __name__ == '__main__':
    pymavlink = Guidance(master)

    main(pymavlink)