import time
from pymavlink import mavutil
from PyMavlink import Guidance

#jika koneksi langsung komputer/nuc
# master = mavutil.mavlink_connection("/dev/ttyACM0", baud=115200)

#koneksi jika pakai companion
master = mavutil.mavlink_connection('udpin:0.0.0.0:14550')

master.wait_heartbeat()

def main(pm: Guidance):
    
    pm.armDisarm()
    
    pm.arm()
   
    pm.setMode('MANUAL')
   
    #(3 = rc inputnya, 1350 = kecepatannya ) note 1500 = netral
    pm.setRcValue(3, 1350) #gerak kebawah on
    time.sleep(1)
    pm.setRcValue(3, 1500) #gerak kebawah off
    
    pm.setMode('ALT_HOLD')

    pm.setHeading(360, 2) #deerajatnya di adjust lagi sesuaikan kompas

    pm.setHeading(315, 2) #putar kiri

    pm.setHeading(360, 2) #kembali ke derajat awal

    pm.setHeading(45, 2) #putar kanan

    pm.setHeading(360, 4) #kembali ke derajat awal


    pm.setRcValue(5, 1720) #gerak lurus ke object 
    time.sleep(10)
    pm.setRcValue(5, 1500) #off 


    pm.setRcValue(3, 1590) #kebawah menempelkan pelampung     
    time.sleep(3)
    pm.setRcValue(3, 1500) #off 


    pm.open_gripper(1000) #buka gripper full
    # pm.close_gripper(1000) #tutup gripper


    pm.setRcValue(5, 1400) #kebawah menempelkan pelampung     
    time.sleep(1)
    pm.setRcValue(5, 1500) #off 

    pm.setMode('MANUAL')

    pm.disarm()



if __name__ == '__main__':
    pymavlink = Guidance(master)

    main(pymavlink)