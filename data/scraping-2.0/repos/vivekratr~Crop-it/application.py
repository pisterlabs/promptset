from flask import Flask, render_template, request,jsonify
from flask_cors import CORS,cross_origin
application = Flask(__name__, static_folder='static')
app = application

@application.route('/',methods=['GET','POST']) 
@cross_origin() # its purpose is to be available to different countries
def index():
    return render_template("landing.html")
@application.route('/contact') #,methods=['GET','POST'] 
@cross_origin() # its purpose is to be available to different countries
def index10(): 
  
    
    return render_template("contactus.html")

@application.route('/contactuss', methods=['GET','POST']) #,methods=['GET','POST']
@cross_origin() # its purpose is to be available to different countries
def index11():
    import pymongo
    name =request.form['name1']
    email =request.form['email']
    subject=request.form['subject']
    message=request.form['message']
    
    client = pymongo.MongoClient("mongodb+srv://breakratr:breakratr@vivekdb.fwdld9x.mongodb.net/?retryWrites=true&w=majority")
    db = client['Crop']
    collection_1 = db['Contact us']
    dict = {'Name':name,'Email':email,'Subject':subject,'Message':message}
    collection_1.insert_one(dict)
    return render_template("contactus.html")
    
    

@application.route('/crop_sub',methods=['GET','POST'])
@cross_origin() # its purpose is to be available to different countries
def crop_sub():
    nitro= int(request.form['nitro'])
    potass= int(request.form['potass'])
    phospho= int(request.form['phospho'])
    ph= int(request.form['ph'])
    humid= int(request.form['humid'])
    temp= int(request.form['temp'])
    d ={'rice': {'N': {'min': 60, 'max': 99},
  'P': {'min': 35, 'max': 60},
  'K': {'min': 35, 'max': 45},
  'ph': {'min': 5.005306977, 'max': 7.868474653},
  'humidity': {'min': 80.12267476, 'max': 84.96907151},
  'temperature': {'min': 20.0454142, 'max': 26.92995077}},
 'maize': {'N': {'min': 60, 'max': 100},
  'P': {'min': 35, 'max': 60},
  'K': {'min': 15, 'max': 25},
  'ph': {'min': 5.513697923, 'max': 6.995843776},
  'humidity': {'min': 55.28220433, 'max': 74.82913698},
  'temperature': {'min': 18.04185513, 'max': 26.54986394}},
 'chickpea': {'N': {'min': 20, 'max': 60},
  'P': {'min': 55, 'max': 80},
  'K': {'min': 75, 'max': 85},
  'ph': {'min': 5.988992796000002, 'max': 8.868741443},
  'humidity': {'min': 14.25803981, 'max': 19.96978871},
  'temperature': {'min': 17.02498456, 'max': 20.99502153}},
 'kidneybeans': {'N': {'min': 0, 'max': 40},
  'P': {'min': 55, 'max': 80},
  'K': {'min': 15, 'max': 25},
  'ph': {'min': 5.502999119, 'max': 5.99812453},
  'humidity': {'min': 18.09224048, 'max': 24.96969858},
  'temperature': {'min': 15.33042636, 'max': 24.92360104}},
 'pigeonpeas': {'N': {'min': 0, 'max': 40},
  'P': {'min': 55, 'max': 80},
  'K': {'min': 15, 'max': 25},
  'ph': {'min': 4.548202098, 'max': 7.445444882999999},
  'humidity': {'min': 30.40046769, 'max': 69.69141302},
  'temperature': {'min': 18.31910448, 'max': 36.97794384}},
 'mothbeans': {'N': {'min': 0, 'max': 40},
  'P': {'min': 35, 'max': 60},
  'K': {'min': 15, 'max': 25},
  'ph': {'min': 3.504752314, 'max': 9.93509073},
  'humidity': {'min': 40.00933429, 'max': 64.95585424},
  'temperature': {'min': 24.01825377, 'max': 31.99928579}},
 'mungbean': {'N': {'min': 0, 'max': 40},
  'P': {'min': 35, 'max': 60},
  'K': {'min': 15, 'max': 25},
  'ph': {'min': 6.218923893, 'max': 7.199495367999999},
  'humidity': {'min': 80.03499648, 'max': 89.99615558},
  'temperature': {'min': 27.01470397, 'max': 29.914544300000006}},
 'blackgram': {'N': {'min': 20, 'max': 60},
  'P': {'min': 55, 'max': 80},
  'K': {'min': 15, 'max': 25},
  'ph': {'min': 6.500144962, 'max': 7.775306272000001},
  'humidity': {'min': 60.06534859, 'max': 69.96100028},
  'temperature': {'min': 25.09737391, 'max': 34.9466155}},
 'lentil': {'N': {'min': 0, 'max': 40},
  'P': {'min': 55, 'max': 80},
  'K': {'min': 15, 'max': 25},
  'ph': {'min': 5.91645379, 'max': 7.841496029},
  'humidity': {'min': 60.09116626, 'max': 69.92375891},
  'temperature': {'min': 18.06486101, 'max': 29.94413861}},
 'pomegranate': {'N': {'min': 0, 'max': 40},
  'P': {'min': 5, 'max': 30},
  'K': {'min': 35, 'max': 45},
  'ph': {'min': 5.561851831, 'max': 7.199504273},
  'humidity': {'min': 85.12912161, 'max': 94.99897537},
  'temperature': {'min': 18.07132963, 'max': 24.96273236}},
 'banana': {'N': {'min': 80, 'max': 120},
  'P': {'min': 70, 'max': 95},
  'K': {'min': 45, 'max': 55},
  'ph': {'min': 5.505393832999999, 'max': 6.490074429},
  'humidity': {'min': 75.03193255, 'max': 84.97849241},
  'temperature': {'min': 25.01018457, 'max': 29.90888522}},
 'mango': {'N': {'min': 0, 'max': 40},
  'P': {'min': 15, 'max': 40},
  'K': {'min': 25, 'max': 35},
  'ph': {'min': 4.507523551, 'max': 6.9674177660000005},
  'humidity': {'min': 45.02236377, 'max': 54.9640534},
  'temperature': {'min': 27.00315545, 'max': 35.99009679}},
 'grapes': {'N': {'min': 0, 'max': 40},
  'P': {'min': 120, 'max': 145},
  'K': {'min': 195, 'max': 205},
  'ph': {'min': 5.510924848999999, 'max': 6.499604931},
  'humidity': {'min': 80.01639435, 'max': 83.98351748},
  'temperature': {'min': 8.825674745, 'max': 41.94865736}},
 'watermelon': {'N': {'min': 80, 'max': 120},
  'P': {'min': 5, 'max': 30},
  'K': {'min': 45, 'max': 55},
  'ph': {'min': 6.000975617000001, 'max': 6.956508826},
  'humidity': {'min': 80.02621335, 'max': 89.98405233},
  'temperature': {'min': 24.04355803, 'max': 26.98603693}},
 'muskmelon': {'N': {'min': 80, 'max': 120},
  'P': {'min': 5, 'max': 30},
  'K': {'min': 45, 'max': 55},
  'ph': {'min': 6.002927293, 'max': 6.781050372999999},
  'humidity': {'min': 90.01506395, 'max': 94.96218673},
  'temperature': {'min': 27.02415146, 'max': 29.94349168}},
 'apple': {'N': {'min': 0, 'max': 40},
  'P': {'min': 120, 'max': 145},
  'K': {'min': 195, 'max': 205},
  'ph': {'min': 5.514253142, 'max': 6.4992268210000015},
  'humidity': {'min': 90.02575116, 'max': 94.92048112},
  'temperature': {'min': 21.0365275, 'max': 23.99686172}},
 'orange': {'N': {'min': 0, 'max': 40},
  'P': {'min': 5, 'max': 30},
  'K': {'min': 5, 'max': 15},
  'ph': {'min': 6.010391864, 'max': 7.995848977},
  'humidity': {'min': 90.00621688, 'max': 94.96419851},
  'temperature': {'min': 10.01081312, 'max': 34.90665289}},
 'papaya': {'N': {'min': 31, 'max': 70},
  'P': {'min': 46, 'max': 70},
  'K': {'min': 45, 'max': 55},
  'ph': {'min': 6.501521192, 'max': 6.993473247000001},
  'humidity': {'min': 90.03863107, 'max': 94.94482086},
  'temperature': {'min': 23.012401800000006, 'max': 43.67549305}},
 'coconut': {'N': {'min': 0, 'max': 40},
  'P': {'min': 5, 'max': 30},
  'K': {'min': 25, 'max': 35},
  'ph': {'min': 5.50158009, 'max': 6.470465614},
  'humidity': {'min': 90.01734526, 'max': 99.98187601},
  'temperature': {'min': 25.00872392, 'max': 29.8690834}},
 'cotton': {'N': {'min': 100, 'max': 140},
  'P': {'min': 35, 'max': 60},
  'K': {'min': 15, 'max': 25},
  'ph': {'min': 5.801047545, 'max': 7.994679507000001},
  'humidity': {'min': 75.00539324, 'max': 84.87668973},
  'temperature': {'min': 22.00085141, 'max': 25.99237426}},
 'jute': {'N': {'min': 60, 'max': 100},
  'P': {'min': 35, 'max': 60},
  'K': {'min': 35, 'max': 45},
  'ph': {'min': 6.002524871, 'max': 7.4880144039999985},
  'humidity': {'min': 70.88259632, 'max': 89.89106506},
  'temperature': {'min': 23.09433785, 'max': 26.98582182}},
 'coffee': {'N': {'min': 80, 'max': 120},
  'P': {'min': 15, 'max': 40},
  'K': {'min': 25, 'max': 35},'ph': {'min': 6.020947179, 'max': 7.493191968},'humidity': {'min': 50.04557009, 'max': 69.94807345},'temperature': {'min': 23.05951896, 'max': 27.92374437}}}
    cr =[]
    img = []
    dict = {'rice':'../static/rice.jpeg','maize':'../static/maize.png','chickpea':'../static/chickpea.png','kidneybeans':'../static/kidneybeans.png','pigeonpeas':'../static/pigeonpeas.png',
        'mothbeans':'../static/mothbeans.png','mungbean':'../static/mungbean.png','blackgram':'../static/blackgram.png','lentil':'../static/lentil.png','pomegranate':'../static/pomegranate.png',
        'banana':'../static/banana.png','mango':'../static/mango.png','grapes':'../static/grapes.png','watermelon':'../static/watermelon.png','muskmelon':'../static/muskmelon.png',
        'apple':'../static/apple.png','orange':'../static/orange.png','papaya':'../static/papaya.png','coconut':'../static/coconut.png','cotton':'../static/cotton.png',
        'jute':'../static/jute.png','coffee':'../static/coffee.png'}
    for j in d:
        i=d[j]
        # and (potass in range(i['K']['min'],i['K']['max'])) and (ph in range(int(i['ph']['min']),int(i['ph']['max']))  and (humid in range(int(i['humidity']['min']),int(i['humidity']['max']))
        if((nitro in range(i['N']['min'],i['N']['max']))  and (phospho in range(i['P']['min'],i['P']['max']))  ):
            cr.append(j.upper())
            # img.append("../static/cropit1.gif")
            img.append(dict[j])
            
    pancake = []
    for i in range(len(cr)):
        pancake.append({'images':img[i],'crop_name':cr[i]})
    # pancake = [{"images":"../static/cropit1.gif",'crop_name':'Rice'},{"images":"../static/cropit1.gif",'crop_name':'Wheat'}]
    print(f'Nitro: {nitro},Potass : {potass}, Phospho : {phospho}, PH : {ph}, Humid:{humid}, temperature: {temp}')
    print(pancake)
    return render_template("crop_rec_ouput.html",datas =pancake[0:len(pancake)],nitro=nitro,potass=potass,phospho=phospho,ph=ph,humid=humid ,temp=temp)


@application.route('/crop',methods=['GET','POST'])
@cross_origin() # its purpose is to be available to different countries
def index1():
    return render_template("crop_rec_input.html")


@application.route('/fertile',methods=['GET','POST'])
@cross_origin() # its purpose is to be available to different countries
def index5():
    
    return render_template("Fertlilzer_rec.html")

@application.route('/fertile_out',methods=['GET','POST'])
@cross_origin() # its purpose is to be available to different countries
def index2():
    nitro= request.form['nitro']

    potass= int(request.form['potass'])

    phospho= int(request.form['phospho'])

    crop_opt= request.form['crop_opt']

    temp1= int(request.form['temp1'])
    import random

    my_list = ['Urea', 'DAP', '14-35-14', '28-28', '17-17-17', '20-20', '10-26-26']

    random_elements = list(random.sample(my_list, 3))
    dicy = {'Urea':'../static/Urea.png','DAP':'../static/DAP.png','14-35-14':'../static/14-35-14.png','28-28':'../static/28-28.png','17-17-17':'../static/17-17-17.png','20-20':'../static/20-20.png','10-26-26':'../static/10-26-26.png'}
    pancake = []
    for i in random_elements:
        pancake.append({'images':f"../static/{dicy[i]}",'fert_name':i})
    return render_template("fertlilzer_rec_output.html",datas =pancake[0:len(pancake)],nitro=nitro,potass=potass,phospho=phospho,crop_opt=crop_opt,temp1=temp1)

@application.route('/water_in',methods=['GET','POST'])
@cross_origin() # its purpose is to be available to different countries
def index3():
    
    return render_template("cropWaterReq.html")
@application.route('/water_out',methods=['GET','POST'])
@cross_origin() # its purpose is to be available to different countries
def index8():
    nitro= int(request.form['nitro'])
    potass= int(request.form['potass'])
    phospho= int(request.form['phospho'])
    ph= int(request.form['ph'])
    rainf= int(request.form['rainf'])
    cropt= request.form['cropt']
    import pandas as pd
    a = pd.read_csv('dataset/crops1.csv')
    w=a[a['label'] == cropt]['water-availability(liters per year)'].mean()
    w =(w+nitro+potass+phospho+ph+rainf)/48
    v =w /7
    output1=f"Water/Week = {w} litres. \n "
    output2 = f"Water/Day = {v} litres."
    return render_template("crop_water_output.html",nitro=nitro,potass=potass,phospho=phospho,ph=ph,rainf=rainf,cropt=cropt.upper(),result1 =output1,result2 = output2)

@application.route('/cropmap_in',methods=['GET','POST'])
@cross_origin() # its purpose is to be available to different countries
def index4():
    return render_template("crop_mapping_input.html")

@application.route('/mapp',methods=['GET','POST'])
@cross_origin() # its purpose is to be available to different countries
def index7():
    return render_template("my_map.html")

@application.route('/cropmap_out',methods=['GET','POST'])
@cross_origin() # its purpose is to be available to different countries
def index6():
    import pandas as pd
    state_data = {'Punjab': {'Lat': 31.125536166666663, 'Long': 75.49268766666667},
 'Telangana ': {'Lat': 17.930004, 'Long': 78.92139399999999},
 'Tripura': {'Lat': 23.836049, 'Long': 91.279386},
 'Uttar Pradesh': {'Lat': 27.558369, 'Long': 79.53375439285715},
 'Maharashtra': {'Lat': 19.437894555555555, 'Long': 75.37090572222222},
 'Gujarat': {'Lat': 22.172131999999998, 'Long': 71.82683088888889},
 'Mizoram': {'Lat': 23.736701, 'Long': 92.714596},
 'Rajasthan': {'Lat': 26.46041323076923, 'Long': 75.00427223076925},
 'Kerala': {'Lat': 9.611407, 'Long': 76.3798488},
 'West Bengal': {'Lat': 23.729951444444445, 'Long': 88.26926788888889},
 'Haryana': {'Lat': 29.30676490909091, 'Long': 76.58581309090908},
 'Bihar': {'Lat': 25.384672, 'Long': 85.71918683333332},
 'Jammu and Kashmir ': {'Lat': 33.837129000000004, 'Long': 74.61865275},
 'Karnataka': {'Lat': 14.8084356, 'Long': 76.3993348},
 'Chhattisgarh': {'Lat': 21.50752233333333, 'Long': 81.739087},
 'Madhya Pradesh': {'Lat': 23.67546114285714, 'Long': 77.27517714285715},
 'Odisha': {'Lat': 20.797267124999998, 'Long': 85.030559},
 'Chandigarh': {'Lat': 30.736292, 'Long': 76.788398},
 'Tamil Nadu': {'Lat': 10.821000944444442, 'Long': 78.45747294444442},
 'Andhra Pradesh': {'Lat': 16.059975133333335, 'Long': 80.27575453333334},
 'Daman and Diu': {'Lat': 20.564715, 'Long': 71.910156},
 'Uttarakhand': {'Lat': 30.324427, 'Long': 78.033922},
 'Delhi': {'Lat': 28.625976, 'Long': 77.21574749999999},
 'Jharkhand': {'Lat': 23.413133000000002, 'Long': 86.0949815},
 'Assam': {'Lat': 26.334961833333335, 'Long': 93.0454855},
 'Sikkim': {'Lat': 27.325739, 'Long': 88.612155},
 'Manipur': {'Lat': 24.808053, 'Long': 93.944203},
 'Arunachal Pradesh': {'Lat': 27.102349, 'Long': 93.692047},
 'Lakshadweep': {'Lat': 10.566667, 'Long': 72.616667},
 'Nagaland': {'Lat': 25.674673, 'Long': 94.110988},
 'Goa': {'Lat': 15.498289, 'Long': 73.824541},
 'Andaman and Nicobar Islands': {'Lat': 11.666667, 'Long': 92.75},
 'Puducherry': {'Lat': 11.933812, 'Long': 79.829792},
 'Meghalaya': {'Lat': 25.573987, 'Long': 91.896807},
 'Himachal Pradesh': {'Lat': 31.104423, 'Long': 77.166623},
 'Dadra and Nagar Haveli': {'Lat': 20.273855, 'Long': 72.996728}}
    n = request.form.get('crop_opt')
    # print('a',n,'a')
    a =pd.read_csv('dataset/crop_production.csv')
    a =a.dropna()
    b= a[['State_Name','Crop','Area','Production']]
    c= b[b['Crop']==n]
    d =list(c['State_Name'].unique())
    area_mean = c.groupby('State_Name')['Area'].sum()
    prod_mean = c.groupby('State_Name')['Production'].mean()
    import folium
    import os
    import time

    file_path = 'templates/my_map.html'
    try:
        os.remove(file_path)
    except Exception as e:
        pass

    m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)

    m1 = folium.FeatureGroup(name='crop_map')
    for i in d:
        # print(state_data[i])
        if((prod_mean[i])==0):
            continue
        popup_html = f'<table><tr><th>Crop Name:</th><td>{n}</td></tr><tr><th>State Name:</th><td>{i}</td></tr><tr><th>Area:</th><td><br>{int(area_mean[i])} acre</td></tr><tr><th>Production:</th><td><br>{int(prod_mean[i])} pounds</td></tr></table>'
        m1.add_child(folium.Marker(location=[state_data[i]['Lat'],state_data[i]['Long']], popup=popup_html, icon=folium.Icon(color='darkgreen')))
    m.add_child(m1)
#     m.save('templates/my_map.html')
    map_html = m._repr_html_()

#     m.save('templates/my_map.html'+ '?v=' + str(int(time.time())))
    # with open('templates/my_map.html', 'r') as f:
    #     map_html = f.read()
    return render_template("crop_map.html",crop_naam =n,mappa= map_html)
@application.route('/gpt',methods=['GET','POST'])
@cross_origin() # its purpose is to be available to different countries
def index13():
    # import os
    # import openai
    # openai.api_key ="sk-sgJSbbGqlyMl7oUgG86sxw8CDcWCVSlEJU1kBzuqcaEDaFDd"
    input_text = request.form['input-field']
    # completion = openai.ChatCompletion.create(
    #   model="gpt-3.5-turbo",
    #   messages=[
    #     {"role": "user", "content": input_text}
    #   ]
    # )
    import cohere

# Set up a Cohere API client with your API key
    client = cohere.Client("5dPAagesHxYyp8kIg9QIyxknJB7WlAKZJmAmHXwJ")

# Define the text you want to summarize
    text = f'''{input_text}     explain it in minimum 1000 words                                                                                                                                                                                                                                                                                                               .'''

# Call the summarize method with your text
    summary = client.summarize(text) #, num_sentences=2
  
# Print the summary
    output_text = summary.summary
    # output_text=completion.choices[0].message['content']

    return render_template('landing.html',output11=output_text)
# mappu =map_html,

if __name__ == '__main__':
    application.run(debug=True)
