import cv2
from openai_util.function_call.openaifunc_decorator import openai_func


@openai_func
def get_picture_from_camera():
    """
    Get a picture from the camera and return the path of the picture
    :return: image path
    """
    # Create a VideoCapture object
    cap = cv2.VideoCapture(0)

    # Check if the camera is successfully turned on
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    # read a frame of image
    ret, frame = cap.read()

    # ret should be True if the frame was read correctly
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        exit()

    # save image to file
    cv2.imwrite('temp.jpg', frame)

    # release camera
    cap.release()
    return 'temp.jpg'

# get_picture_from_camera()
