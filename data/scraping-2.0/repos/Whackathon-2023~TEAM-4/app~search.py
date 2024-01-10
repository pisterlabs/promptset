import os
import openai

openai.api_type = "azure"
openai.api_base = "https://whackazureopenai4.openai.azure.com/"
openai.api_version = "2023-07-01-preview"
openai.api_key = "f06743f3d6024b488a38022f476ea726"

ship_data = {
    "vessels": [
        {
            "Vessel Name": "COSCO GENOA",
            "Ports Visited": [
                {
                    "Port Name": "Mesaieed",
                    "Product Type": "Urea",
                    "Quantity": 4500,
                    "FOB Price per Unit in AUD": 300
                },
                {
                    "Port Name": "Donaldsonville",
                    "Product Type": "UAN",
                    "Quantity": 3200,
                    "FOB Price per Unit in AUD": 350
                },
                {
                    "Port Name": "Aqaba",
                    "Product Type": "MOP",
                    "Quantity": 2800,
                    "FOB Price per Unit in AUD": 280
                }
            ]
        },
        {
            "Vessel Name": "YU PENG",
            "Ports Visited": [
                {
                    "Port Name": "Al Jubail",
                    "Product Type": "Urea",
                    "Quantity": 4700,
                    "FOB Price per Unit in AUD": 300
                },
                {
                    "Port Name": "Port Lisas",
                    "Product Type": "UAN",
                    "Quantity": 3500,
                    "FOB Price per Unit in AUD": 350
                },
                {
                    "Port Name": "Ras Al Khair",
                    "Product Type": "MAP",
                    "Quantity": 3100,
                    "FOB Price per Unit in AUD": 280
                }
            ]
        },
        {
            "Vessel Name": "ISLAND EXPRESS",
            "Ports Visited": [
                {
                    "Port Name": "Bontang",
                    "Product Type": "Urea",
                    "Quantity": 4900,
                    "FOB Price per Unit in AUD": 300
                },
                {
                    "Port Name": "Novorossiysk",
                    "Product Type": "UAN",
                    "Quantity": 3400,
                    "FOB Price per Unit in AUD": 350
                },
                {
                    "Port Name": "Portland",
                    "Product Type": "MOP",
                    "Quantity": 2900,
                    "FOB Price per Unit in AUD": 280
                }
            ]
        },
        {
            "Vessel Name": "PALAWAN",
            "Ports Visited": [
                {
                    "Port Name": "Sitra",
                    "Product Type": "Urea",
                    "Quantity": 4600,
                    "FOB Price per Unit in AUD": 300
                },
                {
                    "Port Name": "Sipitang",
                    "Product Type": "UAN",
                    "Quantity": 3300,
                    "FOB Price per Unit in AUD": 350
                },
                {
                    "Port Name": "Ras Al Khair",
                    "Product Type": "MAP",
                    "Quantity": 2900,
                    "FOB Price per Unit in AUD": 280
                }
            ]
        },
        {
            "Vessel Name": "CAPE PILLAR",
            "Ports Visited": [
                {
                    "Port Name": "Sipitang",
                    "Product Type": "Urea",
                    "Quantity": 4800,
                    "FOB Price per Unit in AUD": 300
                },
                {
                    "Port Name": "Aqaba",
                    "Product Type": "UAN",
                    "Quantity": 3200,
                    "FOB Price per Unit in AUD": 350
                },
                {
                    "Port Name": "Ras Al Khair",
                    "Product Type": "MAP",
                    "Quantity": 2700,
                    "FOB Price per Unit in AUD": 280
                }
            ]
        },
        {
            "Vessel Name": "MSC LIDIA",
            "Ports Visited": [
                {
                    "Port Name": "Mesaieed",
                    "Product Type": "Urea",
                    "Quantity": 4700,
                    "FOB Price per Unit in AUD": 300
                },
                {
                    "Port Name": "Donaldsonville",
                    "Product Type": "UAN",
                    "Quantity": 3400,
                    "FOB Price per Unit in AUD": 350
                },
                {
                    "Port Name": "Aqaba",
                    "Product Type": "MOP",
                    "Quantity": 2800,
                    "FOB Price per Unit in AUD": 280
                }
            ]
        },
        {
            "Vessel Name": "APL MEXICO CITY",
            "Ports Visited": [
                {
                    "Port Name": "Mesaieed",
                    "Product Type": "Urea",
                    "Quantity": 4600,
                    "FOB Price per Unit in AUD": 300
                },
                {
                    "Port Name": "Port Lisas",
                    "Product Type": "UAN",
                    "Quantity": 3300,
                    "FOB Price per Unit in AUD": 350
                },
                {
                    "Port Name": "Portland",
                    "Product Type": "MOP",
                    "Quantity": 2900,
                    "FOB Price per Unit in AUD": 280
                }
            ]
        },
        {
            "Vessel Name": "TAHO AFRICA",
            "Ports Visited": [
                {
                    "Port Name": "Sitra",
                    "Product Type": "Urea",
                    "Quantity": 4800,
                    "FOB Price per Unit in AUD": 300
                },
                {
                    "Port Name": "Novorossiysk",
                    "Product Type": "UAN",
                    "Quantity": 3200,
                    "FOB Price per Unit in AUD": 350
                },
                {
                    "Port Name": "Portland",
                    "Product Type": "MOP",
                    "Quantity": 2700,
                    "FOB Price per Unit in AUD": 280
                }
            ]
        },
        {
            "Vessel Name": "HMPNGS GILBERT TOROPO",
            "Ports Visited": [
                {
                    "Port Name": "Mesaieed",
                    "Product Type": "Urea",
                    "Quantity": 4600,
                    "FOB Price per Unit in AUD": 300
                },
                {
                    "Port Name": "Sipitang",
                    "Product Type": "UAN",
                    "Quantity": 3300,
                    "FOB Price per Unit in AUD": 350
                },
                {
                    "Port Name": "Ras Al Khair",
                    "Product Type": "MAP",
                    "Quantity": 2900,
                    "FOB Price per Unit in AUD": 280
                }
            ]
        },
        {
            "Vessel Name": "MITERA",
            "Ports Visited": [
                {
                    "Port Name": "Bontang",
                    "Product Type": "Urea",
                    "Quantity": 4800,
                    "FOB Price per Unit in AUD": 300
                },
                {
                    "Port Name": "Novorossiysk",
                    "Product Type": "UAN",
                    "Quantity": 3200,
                    "FOB Price per Unit in AUD": 350
                },
                {
                    "Port Name": "Portland",
                    "Product Type": "MOP",
                    "Quantity": 2700,
                    "FOB Price per Unit in AUD": 280
                }
            ]
        }
    ],
    "ports": [{
        "Country": "Jordan",
        "Function": "Unknown",
        "LOCODE": "JO AQJ",
        "Name": "Al Aqabah",
        "NameWoDiac": "Al Aqabah",
        "Status": "AI",
        "lat": 29.516667,
        "long": 35.0,
        "outflows": 6340518
    },
    {
        "Country": "Indonesia",
        "Function": "Unknown",
        "LOCODE": "ID BXT",
        "Name": "Bontang Lng Terminal",
        "NameWoDiac": "Bontang Lng Terminal",
        "Status": "AI",
        "lat": 0.1,
        "long": 117.483333,
        "outflows": 6783296
    },
    {
        "Country": "Russia",
        "Function": "Unknown",
        "LOCODE": "RU NVS",
        "Name": "Novorossiysk",
        "NameWoDiac": "Novorossiysk",
        "Status": "AI",
        "lat": 44.716667,
        "long": 37.783333,
        "outflows": 7891578
    },
    {
        "Country": "United States",
        "Function": "Unknown",
        "LOCODE": "US PWM",
        "Name": "Portland",
        "NameWoDiac": "Portland",
        "Status": "AI",
        "lat": 43.666667,
        "long": -70.25,
        "outflows": 7353495
    },
    {
        "Country": "Bahrain",
        "Function": "Unknown",
        "LOCODE": "BH SIT",
        "Name": "Sitrah",
        "NameWoDiac": "Sitrah",
        "Status": "AI",
        "lat": 26.166667,
        "long": 50.666667,
        "outflows": 1436192
    },
    {
        "Country": "Qatar",
        "Function": "1----6--",
        "LOCODE": "QAMES",
        "Name": "Mesaieed",
        "NameWoDiac": "Mesaieed",
        "Status": "RQ",
        "lat": 25.633333333333333,
        "long": 51.916666666666664,
        "outflows": 87230.0
    },
    {
        "Country": "Saudi Arabia",
        "Function": "Unknown",
        "LOCODE": "Unknown",
        "Name": "Al Jubail",
        "NameWoDiac": "Al Jubail",
        "Status": "AI",
        "lat": 27.933333,
        "long": 35.25,
        "outflows": 400071
    },
    {
        "Country": "United States",
        "Function": "Unknown",
        "LOCODE": "Unknown",
        "Name": "Donaldsonville",
        "NameWoDiac": "Donaldsonville",
        "Status": "AI",
        "lat": 40.533333,
        "long": -74.25,
        "outflows": 789083
    },
    {
        "Country": "Trinidad and Tobago",
        "Function": "Unknown",
        "LOCODE": "Unknown",
        "Name": "Port Lisas",
        "NameWoDiac": "Port Lisas",
        "Status": "AI",
        "lat": 10.2,
        "long": -61.7,
        "outflows": 1262443
    },
    {
        "Country": "Saudi Arabia",
        "Function": "Unknown",
        "LOCODE": "Unknown",
        "Name": "Ras Al Khair",
        "NameWoDiac": "Ras Al Khair",
        "Status": "AI",
        "lat": 27.933333,
        "long": 35.25,
        "outflows": 7409830
    },
    {
        "Country": "Malaysia",
        "Function": "Unknown",
        "LOCODE": "Unknown",
        "Name": "Sipitang",
        "NameWoDiac": "Sipitang",
        "Status": "AI",
        "lat": 4.566667,
        "long": 103.466667,
        "outflows": 3928316
    }
    ]
}

def query_ai(query):
    response = openai.ChatCompletion.create(
    engine="team4",
    messages = [{"role":"system","content":"You are an AI assistant that helps people find information."}, {"role": "user", "content": query + " answer this question using the following vessel JSON: " + str(ship_data)}],
    temperature=0.7,
    max_tokens=1000,
    top_p=0.95,
    frequency_penalty=0,
    presence_penalty=0,
    stop=None)

    out = response['choices'][0]['message']['content'].replace("JSON", "database")
    out = out.replace("\n", "<br>")
    return out