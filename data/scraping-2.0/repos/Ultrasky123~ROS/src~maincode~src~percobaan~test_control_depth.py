import time
import math
# Import mavutil
from pymavlink import mavutil
# Imports for attitude
from pymavlink.quaternion import QuaternionBase
from PyMavlinkin import Guidance


# Create the connection
master = mavutil.mavlink_connection('udpin:0.0.0.0:14550')
#jika koneksi langsung komputer
# master = mavutil.mavlink_connection("/dev/ttyACM0", baud=115200)

boot_time = time.time()
# Wait a heartbeat before sending commands
master.wait_heartbeat()

# 1	Pitch
# 2	Roll
# 3	Throttle
# 4	Yaw
# 5	Forward
# 6	Lateral
# 7	Camera Pan
# 8	Camera Tilt*
# 9	Lights 1 Level
# 10	Lights 2 Level
# 11	Video Switch

def main(pm: Guidance):

    #arm thruster
    pm.arm()

    #set depth hold PID
    # pm.depth_hold(-0.5)
    # print("depth")
    
    roll_angle = pitch_angle = 0
    for yaw_angle in range(0, 500, 10):
        pm.setDepth(-0.5)

    # spin the other way with 3x larger steps
    for yaw_angle in range(0, 60, 10):
        pm.set_target_attitude(roll_angle, pitch_angle, 280)
        time.sleep(1)

    #maju
    pm.setRcValue(5,1600)
    time.sleep(3)
    pm.setRcValue(4,1500)

    #belok kanan 
    # pm.setRcValue(4,1600)
    # time.sleep(1)
    # pm.setRcValue(4,1500)

   



    #maju 
    pm.setRcValue(5,1600)
    time.sleep(3)
    pm.setRcValue(4,1500)

    #belok kanan 
    pm.setRcValue(4,1600)
    time.sleep(1)
    pm.setRcValue(4,1500)

    #maju
    pm.setRcValue(5,1600)
    time.sleep(3)
    pm.setRcValue(5,1500)

    #belok kanan
    pm.setRcValue(4,1600)
    time.sleep(1)
    pm.setRcValue(4,1500)

    #maju
    pm.setRcValue(5,1600)
    time.sleep(3)
    pm.setRcValue(5,1500)


if __name__ == '__main__':
    pymavlink = Guidance(master)

    main(pymavlink)