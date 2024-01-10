import openai

def GPT_prompt(predicted_price, mean_price, category, transactionType,  propertyType, city, district, bedrooms, livings, wc, area, streetWidth, streetDirection):
    prompt = f"""
You help stakeholders in real estate by evaluating properties (assessing the value, conditions, and potential of a property through thorough analysis and comparison to determine its worth). Stakeholders use a website that leverages ML model which predicts the price of a property based on certain criteria.

The website asks the user for these properties:
1- Category (represents property rating ranging from 1 to 10 where 1 is lowest and 10 highest)
2- Transaction Type (has 2 types: selling, renting)
3- Property Type (has 4 types: villas, buildings, flats, lands)
4- City (all the cities in Saudi Arabia)
5- District (all in the districts in each city)
6- Number of bedrooms (number of bedrooms in the property)
7- Number of living rooms (number of living rooms in the property)
8- Number of bathrooms (number of bathrooms in the property)
9- Area in sq.m (the area in square meters)
10- Street width
11- Direction direction


Now the user has entered the following details: {category}, {transactionType},  {propertyType}, {city}, {district}, {bedrooms}, {livings}, {wc}, {area}, {streetWidth}, {streetDirection}.

The predicted price for this property outputted by the model was: {predicted_price}
The average price for a number of properties with similar details was: {mean_price}

Your task is to create a report explaining to the stakeholder that the predicted price is high, moderate, or low based on if the predicted price is equal or less than the mean price or higher than the mean without being biased.
If the mean does not exist or is not applicable, inform the stakeholder that there are no many prices for such property details at the moment.

Also, you can say if the predicted price is above average, indicating the property may possess certain exceptional features, and location advantages such as government departments, schools, banks, mosques, hospitals, and parks.

DONT use welcome or conclusion words or repeat the above details.

Make sure the whole report DOES NOT  EXCEED 300 WORDS! BUT not less than 250 WORDS.

    """

    return prompt

def generate_text(predicted_price, mean_price, category, transactionType, propertyType, city, district, bedrooms, livings, wc, area,
                                   streetWidth, streetDirection):
    response = openai.Completion.create(
      model="text-davinci-003",
      prompt=GPT_prompt(predicted_price, mean_price, category, transactionType, propertyType, city, district, bedrooms, livings, wc, area,
                                   streetWidth, streetDirection),
      temperature=0.75,
      max_tokens=300,
      top_p=1.0,
      frequency_penalty=0.0,
      presence_penalty=0.0
    )
    response = response['choices'][0]['text']

    print(GPT_prompt(predicted_price, mean_price, category, transactionType, propertyType, city, district, bedrooms, livings, wc, area,
                                   streetWidth, streetDirection))

    return response

