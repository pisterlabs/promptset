import time
from pymavlink import mavutil
from PyMavlinkin import Guidance

lastTime = 0.0
stableDuration = 0.0
depthTarget = -0.5

# master = mavutil.mavlink_connection("/dev/ttyACM0", baud=115200)
master = mavutil.mavlink_connection('udpin:0.0.0.0:14550')

master.wait_heartbeat()

def calculatePwm(depth):
    pwm = 1500

    if depth > depthTarget:
        pwm = int((((depth + 0.5) * 75) / 0.5) + 1500)

        if pwm >= 1575:
            pwm = 1575
        elif pwm <= 1525:
            pwm = 1525

    elif depth < depthTarget:
        pwm = int((((depth + 1) * 75) / 0.5) + 1425)

        if pwm <= 1425:
            pwm = 1425
        elif pwm >= 1475:
            pwm = 1475
    
    return pwm

def main(pm: Guidance):
    startTime = time.time()
    isStable = False

    pm.arm()

    while True:
        currentTime = time.time()
        bootTime = int(currentTime - startTime)
        
        try:
            altitude = pm.getAltitude()
            
            if altitude != None:
                
                pwm = calculatePwm(altitude)
                
                if pwm == None:
                    pwm = 0

                if altitude <= -0.5 and altitude >= -0.1:
                    if not isStable:
                        lastStableTime = currentTime
                    
                    isStable = True
                    
                    stableDuration = int(currentTime - lastStableTime)
                else:
                    isStable = False
                    stableDuration = 0

                pm.setRcValue(3, pwm)

                if stableDuration >= 2 and bootTime <= 3:
                    pm.setRcValue(5, 1600)
        except:
            continue

if __name__ == '__main__':
    pymavlink = Guidance(master)
    main(pymavlink)