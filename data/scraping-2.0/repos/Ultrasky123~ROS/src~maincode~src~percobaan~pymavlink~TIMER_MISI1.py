import time
from pymavlink import mavutil
from PyMavlinkzz import Guidance

#jika koneksi langsung komputer/nuc
master = mavutil.mavlink_connection("/dev/ttyACM1", baud=115200)

#koneksi jika pakai companion
# master = mavutil.mavlink_connection('udpin:0.0.0.0:14550')

master.wait_heartbeat()

# data = [ 1500 ] * 8
# master.mav.rc_channels_override_send(
# master.target_system, master.target_component, *data)

def main(pm: Guidance):

    # pm.armDisarm()

    pm.arm()
    # pm.setHeading(360, 4)

    # setmode manual
    pm.setMode('MANUAL')

    # pm.setDepth (-0.4)

    # pm.setMode('ALT_HOLD')
    

    # maju lurus 10 meter
    pm.setRcValue(3, 1600)
    time.sleep(5)
    pm.setRcValue(3, 1500)
    print("saydwyweh")

    #surface
    # pm.setRcValue(3, 1550)
    # time.sleep(5)
    # pm.setRcValue(3, 1500)
    
    # pm.disarm()
    # misi selesai
   

if __name__ == '__main__':
    pymavlink = Guidance(master)

    main(pymavlink)