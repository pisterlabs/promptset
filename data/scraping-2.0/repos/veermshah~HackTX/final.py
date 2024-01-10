import streamlit as st
import openai
import cv2
import requests
from pyzbar.pyzbar import decode
import time
import os

def BarcodeReader(image):
    img = cv2.imread(image)

    detectedBarcodes = decode(img)

    if detectedBarcodes:
        for barcode in detectedBarcodes:
            if barcode.data != "": 
                try:
                    #return ["Oojami - Time Is Now (Music CD)", "https://images.barcodelookup.com/449/4498117-1.jpg"]
                    response = requests.get("https://api.barcodelookup.com/v3/products?barcode=" + barcode.data.decode(
                        "utf-8") + "&formatted=y&key=qim1lskytd7i8gj8606javkwjpnw50")
                    response.raise_for_status()
                    data = response.json()
                    return [data['products'][0]['title'], data['products'][0]['images'][0]]
                except requests.exceptions.HTTPError as http_err:
                    print(f"HTTP error occurred: {http_err}")
                    st.write(f"HTTP error occurred: {http_err}")
                except requests.exceptions.RequestException as req_err:
                    print(f"Request exception occurred: {req_err}")
                    st.write(f"Request exception occurred: {req_err}")
                except json.decoder.JSONDecodeError as json_err:
                    print("Error parsing JSON data:", json_err)
                    st.write("Error parsing JSON data:", json_err)


def getImage(cam, curtime, boxes, placeholder):
    if page != "Add Items":
        return
    item = None
    while page == "Add Items":
        
        ret, frame = cam.read()
        if not ret:
            print("\nfailed to grab frame\n")
            break
        img_name = "barcode.png"
        cv2.imwrite(img_name, frame)
        item = BarcodeReader(img_name)
        
        if item is not None and (int(round(time.time())) > curtime + 1):
            break
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        placeholder.image(rgb_frame, use_column_width=True)
    for b in boxes:
            b.empty()
    getData(item, boxes)
    getImage(cam, int(round(time.time())), boxes, placeholder)

def getData(item,boxes):
    global items
    if item is not None:
        print("getting items\n")
        boxes[0].subheader(body="Your item is")
        boxes[1].markdown(
        f'<div style="display: flex; justify-content: center;"><img src="{item[1]}" width="400" /></div>',
        unsafe_allow_html=True)
        boxes[2].subheader(item[0] + "\n")
        with open('inven.txt', 'a') as file:
            file.write(item[0] + "," + item[1] + "\n")
            file.close()
    

def barcode(curtime):
    print("barcode")
    st.title("Add Items")
    placeholder = st.empty()
    boxes = [st.empty(), st.empty(), st.empty()]
    cam = cv2.VideoCapture(0)
    getImage(cam, curtime, boxes, placeholder)
    cam.release()


def inventory():
    st.title("Inventory")
    st.markdown('<hr style="border: 0; height: 1px; background-color: #ccc;">', unsafe_allow_html=True)
    
    # Read the inventory from the file
    with open("inven.txt", "r") as file:
        inventory_lines = file.readlines()
    
    # Create a list to store buttons and their corresponding lines
    buttons_and_lines = []
    
    # Display each item in the inventory along with a "Remove" button
    for i, line in enumerate(inventory_lines):
        item, img_url = line.strip().rsplit(",", 1)
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div style="font-family: 'Lexend Deca', 'Helvetica Neue', Helvetica, Arial, sans-serif; font-size: 24px; line-height: 1.5;">
                    {item}
                </div>
                <div style="float: right;">
                    <img src="{img_url}" alt="Image" style="width: 100px; max-width: 100%; height: auto;">
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Create a "Remove" button and associate it with the line
        if st.button("Remove Item", key=i):
            buttons_and_lines.append((i, line))
        
        st.markdown('<hr style="border: 0; height: 1px; background-color: #ccc;">', unsafe_allow_html=True)

    # Remove items associated with the clicked buttons
    for index, line in buttons_and_lines:
        inventory_lines.pop(index)

    # Update the inventory file with the modified content
    with open("inven.txt", 'w') as filedata:
        filedata.writelines(inventory_lines)


def suggestion():
    st.title("Recipe Suggester")
    if st.button("Suggest a Recipe"):
        API_KEY = st.secrets["apikey"]
        openai.api_key = API_KEY
        
        ingredients = ""
        with open("inven.txt", "r") as file:
            for line in file:
                i = line.strip().split(",", 1)
                ingredients = ingredients + i[0] + ", "
            
        prompt = "Suggest a single recipe using the following ingredients: " + ingredients
        print("Prompt: " + prompt)
        
        messages = [
            {"role": "assistant", "content": prompt}
        ]
    
        # Make the API call
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.9
        )
    
        # Get the generated sentence from the response
        recipe = response['choices'][0]['message']['content']
        st.subheader(recipe)



st.sidebar.title("Pantry Chef")
st.sidebar.image("pantry.jpg")
page = st.sidebar.selectbox("Select Page", ["Add Items", "Inventory", "Recipe Suggester"])

# items = [["Water Bottle", "https://pizzahampstead.com/wp-content/uploads/2016/09/45.jpg"],
#          ["Bread","https://assets.bonappetit.com/photos/5c62e4a3e81bbf522a9579ce/1:1/w_2240,c_limit/milk-bread.jpg"]]

# Display selected page
if page == "Add Items":
    barcode(int(round(time.time())))
elif page == "Inventory":
    inventory()
else:
    suggestion()
