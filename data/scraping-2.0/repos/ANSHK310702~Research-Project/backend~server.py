from flask import Flask, request, render_template, jsonify, send_file
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
from keras.models import load_model
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import openai
import joblib
import mahotas
from sklearn.preprocessing import MinMaxScaler
from osgeo import gdal

openai.api_key = os.getenv('OPENAI_API_KEY')

app = Flask(__name__)
CORS(app)


#################### Leaf Disease Detection ####################
# classes for CNN
leaf_classes = [
    'Target_Spot',
    'Late_blight',
    'Mosaic_virus',
    'Leaf_Mold',
    'Bacterial_spot',
    'Early_blight',
    'Healthy',
    'Yellow_Leaf_Curl_Virus',
    'Two-spotted_spider_mite',
    'Septoria_leaf_spot'
]

# classes for other models
leaf_classes_2 = ['Bacterial_spot', 'Early_blight', 'Late_blight', 'Leaf_Mold', 'Septoria_leaf_spot', 'Spider_mites Two-spotted_spider_mite', 'Target_Spot', 'Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_mosaic_virus', 'Healthy']

# Converting each image to RGB from BGR format
def rgb_bgr(image):
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return rgb_img
# Conversion to HSV image format from RGB
def bgr_hsv(rgb_img):
    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    return hsv_img
# image segmentation
# for extraction of green and brown color
def img_segmentation(rgb_img,hsv_img):
    lower_green = np.array([25,0,20])
    upper_green = np.array([100,255,255])
    healthy_mask = cv2.inRange(hsv_img, lower_green, upper_green)
    result = cv2.bitwise_and(rgb_img,rgb_img, mask=healthy_mask)
    lower_brown = np.array([10,0,10])
    upper_brown = np.array([30,255,255])
    disease_mask = cv2.inRange(hsv_img, lower_brown, upper_brown)
    disease_result = cv2.bitwise_and(rgb_img, rgb_img, mask=disease_mask)
    final_mask = healthy_mask + disease_mask
    final_result = cv2.bitwise_and(rgb_img, rgb_img, mask=final_mask)
    return final_result
# feature-descriptor-1: Hu Moments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature
# feature-descriptor-2: Haralick Texture
def fd_haralick(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick
# feature-descriptor-3: Color Histogram
def fd_histogram(image, mask=None):
    bins = 8
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()
#  pre-process image for dt and rf model
def preprocess_image(img):    # Running Function Bit By Bit
    RGB_BGR       = rgb_bgr(img)
    BGR_HSV       = bgr_hsv(RGB_BGR)
    IMG_SEGMENT   = img_segmentation(RGB_BGR,BGR_HSV)
    # Call for Global Fetaure Descriptors
    fv_hu_moments = fd_hu_moments(IMG_SEGMENT)
    fv_haralick   = fd_haralick(IMG_SEGMENT)
    fv_histogram  = fd_histogram(IMG_SEGMENT)
    # Concatenate 
    processed_img = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])
    return processed_img


def model_predict(img_path, modelType):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    if modelType == "CNN":
        cnn_model = load_model('all_models/model.h5', compile=False)
        cnn_model.compile()
        data = np.array([img])
        result = cnn_model.predict(data)[0]
        predicted = result.argmax()
        pred_answer = leaf_classes[predicted]

        # To get confidence scores for each class with class labels:
        class_confidence_scores = dict(zip(leaf_classes, result))
        # Extract class labels and confidence scores
        class_labels = list(class_confidence_scores.keys())
        confidence_scores = list(class_confidence_scores.values())
    elif modelType == "Random Forest":
        rf_model = joblib.load('all_models/rf_model.pkl')
        img = preprocess_image(img)
        # Predict using the random forest classifier
        pred_answer = rf_model.predict([img])[0]
        pred_answer = pred_answer.split("___")[1]

        confidence_scores = rf_model.predict_proba([img])[0]
    elif modelType == "Decision Tree":
        dt_model = joblib.load('all_models/dt_model.pkl')
        img = preprocess_image(img)
        # Predict using the decision tree classifier
        pred_answer = dt_model.predict([img])[0]
        pred_answer = pred_answer.split("___")[1]

        confidence_scores = dt_model.predict_proba([img])[0]
    elif modelType == "Linear Discriminant Analysis":
        lda_model = joblib.load('all_models/lda_model.pkl')
        img = preprocess_image(img)
        # Predict using the lda classifier
        pred_answer = lda_model.predict([img])[0]
        pred_answer = pred_answer.split("___")[1]

        confidence_scores = lda_model.predict_proba([img])[0]
    elif modelType == "Logistic Regression" or modelType == "Support Vector Machine":
        lr_model = joblib.load('all_models/lr_model.pkl')

        img = preprocess_image(img)
        # Create a MinMaxScaler instance
        scaler = MinMaxScaler(feature_range=(0, 1))
        # Scale the preprocessed image
        scaled_img = scaler.fit_transform([img])        # Predict using the logistic regression classifier
        pred_answer = lr_model.predict(scaled_img)[0]
        pred_answer = pred_answer.split("___")[1]

        confidence_scores = lr_model.predict_proba(scaled_img)[0]

    # Calculate percentages
    total_confidence = sum(confidence_scores)
    percentages = [score / total_confidence * 100 for score in confidence_scores]

    plt.clf()
    # # Create a bar plot
    plt.figure(figsize=(10, 5))
    if modelType == "CNN":
        bars = plt.barh(class_labels, confidence_scores, color='royalblue')
    else:
        bars = plt.barh(leaf_classes_2, confidence_scores, color='royalblue')

    # # Add percentages as text on the bars
    # for bar, percent in zip(bars, percentages):
    #     plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2, f'{percent:.2f}%', va='center')

    plt.xlabel('Confidence Score')  # Update the x-axis label
    plt.title(f'{modelType} Prediction Confidence Scores')
    plt.gca().invert_yaxis()  # Invert the y-axis to have the highest confidence at the top

    # Save the plot to a file (e.g., PNG format)
    plot_filename = 'plot.png'
    plt.savefig(plot_filename, bbox_inches="tight")

    pred_answer = pred_answer.replace("_", " ")
    return pred_answer


def generate_chat_response(prompt):
    if prompt != "Healthy":
        prompts = [{"role": "system", "content": "You are a plant disease expert. You provide the farmer with the solution to their problem."},
                                {"role": "user", "content": f"My crops are having {prompt}. Please provide me a short solution."}]
        
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=prompts)
        # return "hello world"
        return response["choices"][0]["message"]["content"]
    else:
        return "Your plant is healthy. No need to worry."


#################### Process Satellite Image ####################
def ndvi_calc(input_file):
    # open file using GDAL
    ds = gdal.Open(input_file)
    # RGB: read data from raster bands
    red_band = ds.GetRasterBand(1).ReadAsArray().astype(np.uint8)
    green_band = ds.GetRasterBand(2).ReadAsArray().astype(np.uint8)
    blue_band = ds.GetRasterBand(3).ReadAsArray().astype(np.uint8)
    # create RGB imgae
    rgb_image = np.dstack((red_band, green_band, blue_band))
    # NDVI: read data from raster bands
    red_band = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    nir_band = ds.GetRasterBand(2).ReadAsArray().astype(np.float32)
    # prevent zero division error
    nir_band[nir_band == 0] = 1
    # calculate NDVI
    ndvi = (nir_band - red_band) / (nir_band + red_band)
    # specify name of output file
    output_rgb_file = "rgb_image.png"
    output_ndvi_file = "ndvi_image.png"
    output_scaled_ndvi_file = "scaled_ndvi_image.png"
    plt.clf()
    plt.imshow(rgb_image)
    plt.savefig(output_rgb_file)
    # plt.show()

    plt.imshow(ndvi, cmap='RdYlGn')
    plt.colorbar()
    plt.title('NDVI')
    plt.savefig(output_ndvi_file)
    # normalize NDVI values between -1 to 1
    desired_min = -1
    desired_max = 1

    # scale NDVI
    scaled_ndvi = 2 * (ndvi - np.min(ndvi)) / (np.max(ndvi) - np.min(ndvi)) - 1

    # Show scaled NDVI
    fig, ax = plt.subplots()  # Create a figure and axis

    # Plot the scaled NDVI
    im = ax.imshow(scaled_ndvi, cmap='RdYlGn', vmin=desired_min, vmax=desired_max)

    # Create a colorbar for the scaled NDVI
    cbar = plt.colorbar(im, ax=ax)

    # Set the title for the scaled NDVI chart
    ax.set_title('Scaled NDVI')

    # Save the chart
    plt.savefig(output_scaled_ndvi_file)
    # plt.show()

    # obtain maximum and minimum values from NDVI (-1, 1)
    max_ndvi = np.max(scaled_ndvi)
    min_ndvi = np.min(scaled_ndvi)

    # print maximum and minimum values
    print(f"Maximum NDVI: {max_ndvi}")
    print(f"Minimum NDVI: {min_ndvi}")



#################### Flask API ####################
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        imageType = request.form['imageType']
        print("Image type:", imageType)

        # Get the image file from the request
        image = request.files['image']
        # Save the image file
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(image.filename))
        image.save(file_path)

        if imageType == "leaf":
            modelType = request.form['modelType']
            print("Model type:", modelType)

            # Perform prediction and return response
            result = model_predict(file_path, modelType)
            solution = generate_chat_response(result)

            # print(solution)
            response_data = {'prediction': result, 'solution': solution}
            return jsonify(response_data)
        else:
            ndvi_calc(file_path)
            return jsonify({'success': 'NDVI image generated'})

    return None



@app.route('/get_plot', methods=['GET'])
def get_plot():
    # Make sure to provide the correct path to the saved plot file
    plot_filename = 'plot.png'

    # Check if the plot file exists
    if os.path.exists(plot_filename):
        print("Plot sent")
        return send_file(plot_filename, mimetype='image/png')
    else:
        print("Plot not found")
        return jsonify({'error': 'Plot not found'})
    
@app.route('/get_training_history', methods=['GET'])
def get_history():
    # Make sure to provide the correct path to the saved plot file
    plot_filename = 'validation_score/training_plot.png'

    # Check if the plot file exists
    if os.path.exists(plot_filename):
        print("Training history sent")
        return send_file(plot_filename, mimetype='image/png')
    else:
        print("Training history not found")
        return jsonify({'error': 'Training history not found'})
    
@app.route('/get_dt_score', methods=['GET'])
def get_dt_score():
    # Make sure to provide the correct path to the saved plot file
    plot_filename = 'validation_score/dt_report.png'

    # Check if the plot file exists
    if os.path.exists(plot_filename):
        print("Classification report sent")
        return send_file(plot_filename, mimetype='image/png')
    else:
        print("Classification report not found")
        return jsonify({'error': 'Classification report not found'})
@app.route('/get_rf_score', methods=['GET'])
def get_rf_score():
    # Make sure to provide the correct path to the saved plot file
    plot_filename = 'validation_score/rf_report.png'

    # Check if the plot file exists
    if os.path.exists(plot_filename):
        print("Classification report sent")
        return send_file(plot_filename, mimetype='image/png')
    else:
        print("Classification report not found")
        return jsonify({'error': 'Classification report not found'})
@app.route('/get_lr_score', methods=['GET'])
def get_lr_score():
    # Make sure to provide the correct path to the saved plot file
    plot_filename = 'validation_score/lr_report.png'

    # Check if the plot file exists
    if os.path.exists(plot_filename):
        print("Classification report sent")
        return send_file(plot_filename, mimetype='image/png')
    else:
        print("Classification report not found")
        return jsonify({'error': 'Classification report not found'})
@app.route('/get_lda_score', methods=['GET'])
def get_lda_score():
    # Make sure to provide the correct path to the saved plot file
    plot_filename = 'validation_score/lda_report.png'

    # Check if the plot file exists
    if os.path.exists(plot_filename):
        print("Classification report sent")
        return send_file(plot_filename, mimetype='image/png')
    else:
        print("Classification report not found")
        return jsonify({'error': 'Classification report not found'})


@app.route('/get_rgb_image', methods=['GET'])
def get_rgb_image():
    # Make sure to provide the correct path to the saved image file
    img_filename = 'rgb_image.png'

    # Check if the image file exists
    if os.path.exists(img_filename):
        print("RGB image sent")
        return send_file(img_filename, mimetype='image/png')
    else:
        print("RGB image not found")
        return jsonify({'error': 'RGB image not found'})

@app.route('/get_ndvi_image', methods=['GET'])
def get_ndvi_image():
    # Make sure to provide the correct path to the saved image file
    img_filename = 'ndvi_image.png'

    # Check if the image file exists
    if os.path.exists(img_filename):
        print("NDVI image sent")
        return send_file(img_filename, mimetype='image/png')
    else:
        print("NDVI image not found")
        return jsonify({'error': 'NDVI image not found'})

@app.route('/get_scaled_ndvi_image', methods=['GET'])
def get_scaled_ndvi_image():
    # Make sure to provide the correct path to the saved image file
    img_filename = 'scaled_ndvi_image.png'

    # Check if the image file exists
    if os.path.exists(img_filename):
        print("Scaled NDVI image sent")
        return send_file(img_filename, mimetype='image/png')
    else:
        print("Scaled NDVI image not found")
        return jsonify({'error': 'Scaled NDVI image not found'})

if __name__ == '__main__':
    app.run(debug=True)