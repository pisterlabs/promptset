import time
from pymavlink import mavutil
from PyMavlinkin import Guidance
import altitude

# jika koneksi langsung komputer/nuc
# master = mavutil.mavlink_connection("/dev/ttyACM0", baud=115200)

# koneksi jika pakai companion
master = mavutil.mavlink_connection('udpin:0.0.0.0:14550')

master.wait_heartbeat()


def depth_hold(depth_target):
        
        def setRcValue(channel_id, pwm=1500):
  
            if channel_id < 1 or channel_id > 18:
                print("Channel does not exist.")
                return

            # Mavlink 2 supports up to 18 channels:
            # https://mavlink.io/en/messages/common.html#RC_CHANNELS_OVERRIDE
            rc_channel_values = [65535 for _ in range(18)]
            rc_channel_values[channel_id - 1] = pwm
            master.mav.rc_channels_override_send(
                master.target_system,                # target_system
                master.target_component,             # target_component
                *rc_channel_values)                  # RC channel list, in microseconds.



        while True:
            depth_target = -0.5

            # variabel PID
            kp = 150
            ki = 0.0
            kd = 35

            # rentang toleransi kedalaman (meter)
            depth_tolerance = 0.0

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
                speed = 1400 + abs(int(pid_output))
                setRcValue(3, speed)
                print("thruster low")
                print(speed , "depth error : ", depth_error)

            else:
                #jika dept kurang dari target
                if depth < depth_target:
                    # hitung kecepatan motor
                    speed = 1500-abs(int(pid_output))

                    # batasi kecepatan agar tidak terlalu besar atau kecil
                    if speed >= 1650:
                        speed = 1650
                    elif speed <= 1350:
                        speed = 1350

                    # set kecepatan motor
                    setRcValue(3, speed)
                    print("thruster nyala")
                    print(speed , "depth error : ", depth_error)
                
                else:
                    # hitung kecepatan motor
                    speed = 1500 + abs(int(pid_output))

                    # batasi kecepatan agar tidak terlalu besar atau kecil
                    if speed >= 1650:
                        speed = 1650
                    elif speed <= 1350:
                        speed = 1350

                    # set kecepatan motor
                    setRcValue(3, speed)
                    print("thruster nyala")
                    print(speed)
                
            # Jika misi sudah selesai, reset kedalaman menjadi 0
            # if mission_complete:
            #     depth_target = 0
            time.sleep(0.1)

def main(pm: Guidance):
    # pm.armDisarm()
    pm.arm()

    # target depth

    depth_hold(-0.5)

    #pm.disarm()

if __name__ == '__main__':
    pymavlink = Guidance(master)
    main(pymavlink)
