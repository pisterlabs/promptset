import cv2
import numpy as np
from PIL import Image
import PIL
from fractions import Fraction
from geopy.geocoders import Nominatim
import openai

# Set your OpenAI API key
openai.api_key = "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"

def dms_to_decimal(degrees, minutes, seconds, direction):
    decimal_degrees = degrees + minutes/60 + seconds/3600
    if direction in ['S', 'W']:
        decimal_degrees = -decimal_degrees
    return decimal_degrees

def detect_objects(image_path):
    # Load YOLO
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    layer_names = net.getUnconnectedOutLayersNames()

    # Load image using the absolute path
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        exit()

    height, width, _ = image.shape

    # Preprocess image
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(layer_names)

    # Get bounding boxes, confidences, and class ids
    confidences = []
    class_ids = []
    boxes = []
    detected_objects = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                detected_objects.append(classes[class_id])

    # Non-maximum suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Print detected objects
    print("Detected Objects:")
    for i in range(len(boxes)):
        if i in indices:
            label = detected_objects[i]
            confidence = confidences[i]
            print(f"{label} with confidence: {confidence:.2f}")

    # Draw bounding boxes
    for i in range(len(boxes)):
        if i in indices:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the result
    cv2.imshow("Object Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return detected_objects

def get_location_info(image_path):
    img = Image.open(image_path)
    exif = {
        PIL.ExifTags.TAGS[k]: v
        for k, v in img._getexif().items()
        if k in PIL.ExifTags.TAGS
    }

    gps_info = exif.get('GPSInfo')

    if gps_info:
        print("\nGPS Information:\n")
        for tag_id, value in gps_info.items():
            tag_name = PIL.ExifTags.GPSTAGS.get(tag_id, tag_id)
            print(f"{tag_name}: {value}")

            # Convert latitude and longitude to decimal degrees
            if tag_name == 'GPSLatitude':
                latitude = dms_to_decimal(*value, gps_info.get('GPSLatitudeRef', 'N'))
                print(f"Decimal Latitude: {float(latitude)}")
            elif tag_name == 'GPSLongitude':
                longitude = dms_to_decimal(*value, gps_info.get('GPSLongitudeRef', 'E'))
                print(f"Decimal Longitude: {float(longitude)}")

                # Reverse geocode coordinates to obtain address
                geolocator = Nominatim(user_agent="image_metadata_script")
                location = geolocator.reverse((latitude, longitude), language='en')
                address = location.address
                print(f"Address: {address}")

                return address

    else:
        print("No GPS information found in the image.")
        return None

def generate_caption(objects, location):
    prompt = f"Objects detected: {', '.join(objects)}\nLocation: {location}\nGenerate a caption:"

    response = openai.Completion.create(
        engine="text-davinci-003",  # You can choose a different engine if needed
        prompt=prompt,
        max_tokens=100,
        n=1,
        stop=None,
    )

    caption = response.choices[0].text.strip()
    return caption

def main():
    # Assuming you already have these values from your previous code
    image_path = input("Enter the image path: ")
    detected_objects = detect_objects(image_path)
    location_info = get_location_info(image_path)

    # Generate caption
    caption = generate_caption(detected_objects, location_info)

    # Print the generated caption
    print("\nGenerated Caption:")
    print(caption)

if __name__ == "__main__":
    main()