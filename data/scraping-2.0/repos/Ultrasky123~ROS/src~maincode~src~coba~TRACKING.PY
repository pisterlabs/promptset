#!/usr/bin/env python3

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from Guidance import control_rov

# Inisialisasi ROS
rospy.init_node('rov_detection_publisher')
image_publisher = rospy.Publisher('rov_detection/image', Image, queue_size=10)

# Inisialisasi OpenCV
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Create window
cv2.namedWindow('ROV')

# Load color threshold values
lower = np.load('/home/lz/Downloads/data1.npy')
upper = np.load('/home/lz/Downloads/data2.npy')

# Background Subtraction model initialization
bs = cv2.createBackgroundSubtractorMOG2()

# Inisialisasi CvBridge
bridge = CvBridge()

def control_rov(rect, frame_width, frame_height):
    centering_zone_width = 320
    centering_zone_height = 240
    centering_zone_x = frame_width / 2 - centering_zone_width / 2
    centering_zone_y = frame_height / 2 - centering_zone_height / 2
    
    center_zone_width = 550
    center_zone_height = 450
    center_zone_x = frame_width / 2 - centering_zone_width / 2
    center_zone_y = frame_height / 2 - centering_zone_height / 2

    target_x, target_y, target_w, target_h = rect
    # jika box tepat di tengah , maju
    if (target_x >= centering_zone_x and target_x + target_w <= centering_zone_x + centering_zone_width and
        target_y >= centering_zone_y and target_y + target_h <= centering_zone_y + centering_zone_height):
        # Stop moving
        print("go maju")
    
    # jika box sudah dekat stop
    elif(target_x >= center_zone_x and target_x + target_w <= center_zone_x + center_zone_width and
        target_y >= center_zone_y and target_y + target_h <= center_zone_y + center_zone_height ):
        print("stop")

    else:
        if target_x < frame_width / 2-20:
            # setRcValue(6,1400)
            print("Move kiri")
        elif target_x > frame_width / 2+1 :
            # setRcValue(6,1600)
            print("Move Kanan")
        # else:
        #     print("stop")



# Main loop
while not rospy.is_shutdown():
    # Read frame from camera
    ret, frame = cap.read()

    if not ret:
        # Exit if no frame is read
        break

    # Apply background subtraction to the frame
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Detect contours
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (based on area) as the target
    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    if max_contour is not None:
        # Calculate the rectangle enclosing the target
        x, y, w, h = cv2.boundingRect(max_contour)

        # Draw the rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Control ROV based on the target rectangle
        control_rov((x, y, w, h), frame.shape[1], frame.shape[0])

    # Display the frame with the detected rectangle
    cv2.imshow('ROV', frame)

    # Convert OpenCV image to ROS image message
    ros_image = bridge.cv2_to_imgmsg(frame, "bgr8")

    # Publish the ROS image message
    image_publisher.publish(ros_image)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
