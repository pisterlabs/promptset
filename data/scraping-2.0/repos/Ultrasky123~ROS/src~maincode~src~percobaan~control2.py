import os
os.environ['MAVLINK20'] = ''
import threading
import math
import time
from pymavlink import mavutil
from pymavlink.quaternion import QuaternionBase
from PyMavlinkin import Guidance
ALT_HOLD_MODE = 2


def main (rov=Guidance):

    #target depth diulang setiap 10 detik
    def set_altitude_loop():
        while True:
            rov.set_target_depthu(-0.5)
            print("loop_depth")
            time.sleep(10)


    # master = mavutil.mavlink_connection('udpin:0.0.0.0:14550')
    master = mavutil.mavlink_connection("/dev/ttyACM0", baud=115200)
    boot_time = time.time()
    # Wait a heartbeat before sending commands
    print(master.wait_heartbeat())

    # Buat thread terpisah untuk menjalankan fungsi set_altitude_loop
    altitude_thread = threading.Thread(target=set_altitude_loop)
    altitude_thread.start()


    # ================================
    # MAIN PROGRAM
    rov.is_armed()
    rov.mode_is()

    while not is_armed(): 
        master.arducopter_arm()

    while not mode_is(ALT_HOLD_MODE):
        master.set_mode('ALT_HOLD')

    pitch = yaw = roll = 0
    # for i in range(500): 
    #heading kanan
    rov.set_target_attitude(roll, pitch, 200)
    print("set_heading2")
    time.sleep(1)

    #heading kiri
    rov.set_target_attitude(roll, pitch, 380)
    print("set_heading2")
    time.sleep(1)

    #heading kearah objek
    rov.set_target_attitude(roll, pitch, 300)
    print("set_heading")
    time.sleep(1)

    #majuu
    rov.setRcValue(5, 1600)
    time.sleep(6)
    print("majuuu")
    rov.setRcValue(5, 1500)
    # maju 

    #heading lurus
    rov.set_target_attitude(roll, pitch, 300)
    print("set_heading")
    time.sleep(1)

    #maju
    rov.setRcValue(5, 1600)
    time.sleep(6)
    print("majuuu")
    rov.setRcValue(5, 1500)



if __name__ == '__main__':
    pymavlink = Guidance(master)
    main(pymavlink)
