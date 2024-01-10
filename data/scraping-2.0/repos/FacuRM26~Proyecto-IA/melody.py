from flask import Flask,request, render_template
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import openai
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,_tree
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import csv
import re
import numpy as np
openai.api_key = "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"

respuesta = [] #variable global para guardar la respuesta de la deteccion de sintomas
severityDictionary=dict()
description_list = dict()
precautionDictionary=dict()
symptoms_dict = {}
num_days= 0
present_disease = []



def test ():
    global clf,cols,le,reduced_data
    train=pd.read_csv('./dataset/Training.csv')
    train.head()

    train.prognosis.value_counts()

    train.isna().sum()

    len(train)

    """Split dataset"""
    reduced_data = train.groupby(train['prognosis']).max()
    # Separar las características (X) y la variable objetivo (y)
    cols = train.columns
    cols = cols[:-1]
    x = train[cols]
    y = train['prognosis']

    # Mapear las etiquetas de texto a números
    le = LabelEncoder()
    le.fit(y)
    y = le.transform(y)

    # Dividir el conjunto de datos en entrenamiento, validación y prueba
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.4, random_state=42)
    x_validation, x_test, y_validation, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

    # Entrenar un árbol de decisión en el conjunto de entrenamiento
    clf = DecisionTreeClassifier()
    clf.fit(x_train, y_train)

    # Evaluar el rendimiento del árbol de decisión en el conjunto de validación
    decision_tree_accuracy_validation = clf.score(x_validation, y_validation)
    #print("Accuracy for Decision Tree on validation data:", decision_tree_accuracy_validation)

    # Evaluar el rendimiento del árbol de decisión en el conjunto de prueba
    decision_tree_accuracy_test = clf.score(x_test, y_test)
    #print("Accuracy for Decision Tree on test data:", decision_tree_accuracy_test)



    #test.join(pd.DataFrame(model_rf.predict(Y),columns=["predicted"]))[["prognosis","predicted"]]
def getDescription():
    global description_list
    with open('./dataset/symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _description={row[0]:row[1]}
            description_list.update(_description)

def getSeverityDict():
    global severityDictionary
    with open('./dataset/Symptom_severity.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        try:
            for row in csv_reader:
                _diction={row[0]:int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass

def getprecautionDict():
    global precautionDictionary
    with open('./dataset/symptom_precaution.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _prec={row[0]:[row[1],row[2],row[3],row[4]]}
            precautionDictionary.update(_prec)

def check_pattern(dis_list,inp):
    pred_list=[]
    inp=inp.replace(' ','_')
    patt = f"{inp}"
    regexp = re.compile(patt)
    pred_list=[item for item in dis_list if regexp.search(item)]
    if(len(pred_list)>0):
        return 1,pred_list
    else:
        return 0,[]
def print_disease(node):
    node = node[0]
    val  = node.nonzero()
    disease = le.inverse_transform(val[0])
    return list(map(lambda x:x.strip(),list(disease)))
def sec_predict(symptoms_exp):
    try:
        df = pd.read_csv('./dataset/Testing.csv')
        X = df.iloc[:, :-1]
        y = df['prognosis']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
        rf_clf = DecisionTreeClassifier()
        rf_clf.fit(X_train, y_train)

        symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
        input_vector = np.zeros(len(symptoms_dict))
        for item in symptoms_exp:
            input_vector[[symptoms_dict[item]]] = 1

        return rf_clf.predict([input_vector])
    except:
        socketio.emit('respuesta', "Se produjo un error, al analizar su enfermedad, vuelva a intentarlo")
    
def calc_condition(exp,days):
    sum=0
    global severityDictionary
    try:
        for item in exp:
            sum=sum+severityDictionary[item]
        if((sum*days)/(len(exp)+1)>13):
            socketio.emit('respuesta', "Deberías consultar a un médico. ")
        else:
            socketio.emit('respuesta', "Puede que no sea tan grave, pero deberías tomar precauciones. ")
    except:
        socketio.emit('respuesta', "No se detectaron síntomas en lo que escribiste, por favor vuelve a intentarlo")
test()
getSeverityDict()
getDescription()
getprecautionDict()
tree = clf
feature_names = cols

def chatbot_deteccion(symptoms):
    global tree, feature_names, num_days,tree_,feature_name,disease_input,conf, cnf_dis
    # Create a list to store all the messages for context
    messages = [
        {"role": "system", "content": "You are going to detect the symptoms that the patient writes and return only those symptoms in English separated by commas. If you do not detect any symptoms in what the patient wrote, then return a 'no'"},
        {"role": "assistant", "content": "Use only these symptoms: ['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills', 'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition', 'spotting_ urination', 'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness', 'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation', 'abdominal_pain', 'diarrhea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 'swollen_extremities', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of_urine', 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain', 'abnormal_menstruation', 'dischromic_patches', 'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload.1', 'blood_in_sputum', 'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze']"}
        ]
    user_symptoms = []

    # Keep repeating the following
    while True:
        # Exit program if the user inputs "quit"
        if symptoms.lower() == "quit":
            break

        # Add each new message to the list
        messages.append({"role": "user", "content": symptoms})

        # Request gpt-3.5-turbo for chat completion
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )

        # Print the response and add it to the messages list
        chat_message = response['choices'][0]['message']['content']
        #print(f"Bot: {chat_message}")
        messages.append({"role": "assistant", "content": chat_message})

        # Dividir el mensaje en una lista
        chat_message = chat_message.split(",")
        user_symptoms.append(chat_message)
        respuesta = user_symptoms
        texto = []
        tree_ = tree.tree_

        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]

        chk_dis = ",".join(feature_names).split(",")

        disease_input = user_symptoms[0]

        conf, cnf_dis = check_pattern(chk_dis, disease_input[0])
        if conf == 1:
            return respuesta
        else:
            return [["no"]]
        
        """
            print("Búsquedas relacionadas:")
            for num, it in enumerate(cnf_dis):
                print(num, ")", it)
            if num != 0:
                print(f"Selecciona la que quisiste decir. (0 - {num}):  ", end="")
                socketio.emit('respuesta', "Selecciona la que quisiste decir. (0 - {num}):")
                conf_inp = int(input(""))
            else:
                conf_inp = 0
        """




def calcular():
    global present_disease
    node = 0  # Comenzamos desde el nodo raíz del árbol
    while tree_.feature[node] != _tree.TREE_UNDEFINED:
        name = feature_name[node]
        threshold = tree_.threshold[node]

        if name == disease_input:
            val = 1
        else:
            val = 0
        if val <= threshold:
            node = tree_.children_left[node]
        else:
            symptoms_present.append(name)
            node = tree_.children_right[node]

    present_disease = print_disease(tree_.value[node])
    red_cols = reduced_data.columns
    return red_cols[reduced_data.loc[present_disease].values[0].nonzero()]

    


# Crea una instancia de la aplicación Flask
app = Flask(__name__)
app.config['SECRET_KEY'] = 'vnkdjnfjknfl1232#'
socketio = SocketIO(app, cors_allowed_origins="*",secure=True)
app.template_folder = './'
CORS(app)
cont = 0 #variable global para el indice de que parte de la conversacion va
cont2 = 0 #variable global para contar los sintomas posibles
size_list=0 #variable global para el tamaño de la lista de sintomas posibles
symptoms_exp = []
# Ruta principal que responde con un mensaje
@app.route('/')
def index():
    global cont
    cont = 0
    return render_template('index.html')

@socketio.on('mensaje')
def obtener_mensaje(mensaje):
    global cont, respuesta, num_days, cont2, size_list, syms, symptoms_exp,symptoms_given, present_disease, second_prediction

    if cont == 0:
        socketio.emit('respuesta', "Espere unos segundos, estamos procesando sus sintomas")
        respuesta = chatbot_deteccion(mensaje)
        print(respuesta)
        if respuesta[0][0] == "no":
            socketio.emit('respuesta', "No se detectaron síntomas en lo que escribiste, por favor vuelve a intentarlo")
            return
        else:
            socketio.emit('respuesta', "Por cuántos días ha experimentado esta enfermedad?")
            cont = cont + 1
    elif cont == 1:
        num_days = mensaje
        #comprobacion de que el numero de dias sea un numero
        try:
            num_days = int(num_days)
            print(num_days)
            symptoms_given = calcular()
            size_list = len(symptoms_given)
            
            syms = symptoms_given[cont2]
            if syms in respuesta:
                symptoms_exp.append(syms)
                cont2 += 1
                cont = cont + 1

            else:
                socketio.emit('respuesta', "Estas experimentando alguno de los siguientes sintomas (si/no):")
                socketio.emit('respuesta', (syms + "? : "))  
                cont = cont + 1

                return
        except Exception as e:
            print(e)
            socketio.emit('respuesta', "Por favor ingresa un numero")
            socketio.emit('respuesta', "Digite de nuevo sus sintomas")
            cont = 0
            return
    elif cont == 2:

        if mensaje == "si" or mensaje == "no":
            if mensaje== "si":
                symptoms_exp.append(syms)
            cont2 += 1
        else:
            socketio.emit('respuesta', "Por favor responde con 'si' o 'no'")  
                
        if cont2 < size_list:
            syms = symptoms_given[cont2]
            #si hay un _ remplazarlo por un espacio en blanco
            if syms in respuesta:
                symptoms_exp.append(syms)
            else:
                socketio.emit('respuesta', (syms.replace("_", " ") + "? : "))  
        else:
            try:
                cont = cont + 1
                second_prediction = sec_predict(symptoms_exp)
                calc_condition(symptoms_exp, num_days)
                if present_disease[0] == second_prediction[0]:
                    socketio.emit('respuesta', "Ustede podria tener: " + present_disease[0])
                    socketio.emit('respuesta', description_list[present_disease[0]])
                else:
                    socketio.emit('respuesta', "Posiblemente podria tener " + present_disease[0] )
                    socketio.emit('respuesta', description_list[present_disease[0]])
                    socketio.emit('respuesta', "o Posiblemente podria tener " + second_prediction[0] )
                    socketio.emit('respuesta', description_list[second_prediction[0]])
                precution_list = precautionDictionary[present_disease[0]]
                socketio.emit('respuesta', "Toma las siguientes consideraciones : ")
                for i, j in enumerate(precution_list):
                    socketio.emit('respuesta', f"({i + 1}) {j}")
                socketio.emit('respuesta', "Si tiene otra consulta escriba sus sintomas de nuevo")
                cont = 0
                cont2=0
            except:
                socketio.emit('respuesta', "Se produjo un error, al analizar su enfermedad, vuelva a intentarlo")
                cont = 0
                cont2=0
                return
@socketio.on_error()  # Manejar errores generales
def handle_error(e):
    print(f"Error en Socket.IO: {str(e)}")

@socketio.on_error('/mensaje')  # Manejar errores en el evento 'mensaje'
def handle_message_error(e):
    print(f"Error en el evento 'mensaje': {str(e)}")


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=8080, debug=True)
