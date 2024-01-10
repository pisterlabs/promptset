# Import flask and datetime module for showing date and time
from flask import Flask,jsonify,request,Response,make_response
import datetime
from flask_cors import CORS
import cv2
import numpy as np
import openai

x = datetime.datetime.now()
# Initializing flask app
app = Flask(__name__)
CORS(app)



@app.route('/chat', methods=['POST'])
def chat():
	import json
	data = request.get_json()
	print("Json Data")
	data_dict = data
	values = list(data_dict.values())
	openai.api_key = "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"
	# Set up the GPT model
	model_engine = "text-davinci-003"
	prompt = values[0]
	completion = openai.Completion.create(
		engine=model_engine,
		prompt=prompt,
		max_tokens=1024,
		n=1,
		stop=None,
		temperature=0.5,
	)
	response = completion.choices[0].text
	return jsonify({'response': response})


camera = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_faces():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('Face Recognition\\trainer\\trainer.yml')
    cascadePath = "Face Recognition\\haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath);
    font = cv2.FONT_HERSHEY_SIMPLEX

    #iniciate id counter
    id = 0

    # names related to ids: example ==> Marcelo: id=1,  etc
    names = ['None', 'Mihir', 'Sarid', 'Vedant', 'Chintan'] 

    # Initialize and start realtime video capture
    cam = cv2.VideoCapture(0)
    cam.set(3, 640) # set video widht
    cam.set(4, 480) # set video height

    # Define min window size to be recognized as a face
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)

    while True:

        ret, img =cam.read()
        img = cv2.flip(img, 1) # Flip vertically

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale( 
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (int(minW), int(minH)),
           )

        for(x,y,w,h) in faces:

            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

            # Check if confidence is less them 100 ==> "0" is perfect match 
            if (confidence < 100):
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))

            cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
            cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  

        ret, jpeg = cv2.imencode('.jpg', img)
        frame = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        if cv2.waitKey(1) == 27:
            break

    cam.release()
    cv2.destroyAllWindows()

@app.route('/verification')
def verification():
    return Response(detect_faces(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/scorematchprofile', methods=['POST'])
def get_time():
    
    data = request.get_json()
    print(data)
    import json
    import numpy as np
    data_dict = data
    values = list(data_dict.values())
    import numpy as np
    data_list = values
    
    values = np.array([float(val) for val in data_list], dtype='float64')
    values = list(values)
    print(values)
    gender = values[5]
    # data=[1.000000,166.000000,0.000000,0.000000,0.000000,1.000000,0.000000,23.000000]
    import Predict as pr
    result = pr.score(values,gender)
    # print(result)
    # print(type(result))
    json_data = result.to_json(orient='records')
    return json_data

	
# Running app
if __name__ == '__main__':
	app.run(debug=True)
