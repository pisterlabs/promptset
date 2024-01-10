import time
from pymavlink import mavutil
from PyMavlinkzz import Guidance
from alt_hdg import parameter

#jika koneksi langsung komputer
# master = mavutil.mavlink_connection("/dev/ttyACM0", baud=115200)

#koneksi jika pakai companion
master = mavutil.mavlink_connection('udpin:0.0.0.0:14550')

master.wait_heartbeat()

def main(pm: Guidance):
    

    pm.arm()
    # pm.setHeading(360, 4)
    
    # setmode manual
    pm.setMode('MANUAL')
    pm.setDepth (-0.4) #0.4 meter dibawah
    pm.setMode('ALT_HOLD')
    
    # while true:
    #     param=parameter()
    #     print (param)

    # maju lurus 10 meter
    pm.setRcValue(5, 1700)
    time.sleep(5)
    pm.setRcValue(5, 1500)

    #surface
    pm.setRcValue(3, 1600)
    time.sleep(5)
    pm.setRcValue(3, 1500)
    
    # pm.disarm()
    # misi selesai
   

if __name__ == '__main__':
    pymavlink = Guidance(master)

    main(pymavlink)