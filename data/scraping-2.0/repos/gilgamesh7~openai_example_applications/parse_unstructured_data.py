import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

unstructured_data_prompt = """
A table summarizing the crime in New Zealand Cities :
How Safe Is New Zealand: New Zealand Crime Rates
Making a move to beautiful New Zealand and unsure of what the New Zealand crime rates are like? You first need to know about the origins of criminal law and the judiciary system in New Zealand. The New Zealand criminal law is rooted in English Criminal law and incorporated in legislation in 1893. The Crimes Act 1961 and its modifications define the majority of felonies that result in imprisonment in New Zealand. This act addressed criminal law, crime statistics, the origin and aspects of crime, prosecution, penalty, and public views of offences.
In terms of enforcing bodies, the New Zealand Police is the principal administrative body. The New Zealand Police Service oversees decreasing crime and improving public security. The land, sea, and air operating workforce trainees are up to western standards, and you can be assured that misbehaviours such as bullying and extortion involving police officers are not primary concerns. The New Zealand Bill of Rights Act 1990 guarantees detainees several rights, including access to fair trial assistance.
Crime rate in New Zealand – an overview
Financial crises, underemployment, natural calamities, additional security, evolving political shifts, greater enforcement, and numerous heritage and cultural changes have all influenced crime rates. A person’s socio-economic life alone influences their ability to commit a crime. Someone going through a financial crisis or suffering from poor mental health is more likely to commit a crime. By building up a reliable socio-economic system in a country, you (kind of) guarantee a crime-free zone for the citizens. New Zealand, a developed country, provides a high quality of life to its people. Thereby, it comes as no surprise that it has been in the top ranking of the global peace index for so many years now.
The Institute for Economics and Peace publishes a study known as the Global Peace Index, which assesses the relative peacefulness of international locations. According to the Global Peace Index, New Zealand is ranked the second safest country worldwide in 2021 and has been part of the top five since 2008.
As per the statistics by Numbeo, New Zealand has a crime index of 43.03 and a safety index of 56.97. Hence, it is generally safe to roam around in broad daylight comparative to at night. Other crimes such as theft, racism, corruption, bribery & assault have low to moderate rate indexes. The crime rate, however, had an exponential increase in the past three years. The data provided comes from individual public experience over the span of three years.
Here are the statistics records from 2020 by the New Zealand Police. Do note that data for 2021 is yet not available.
The most common types of crimes determined by the New Zealand Crime and Safety Survey (NZCASS) are sexual assault, minor violence, and burglary. The murder/ homicide rate has been fluctuating over the past few years. The laws take serious action against such felonies and can be punishable for up to 20 years of imprisonment. The good news, however, is that weapon assault is rare in New Zealand (especially when compared against other developed nations such as the USA) and charges for firearm felonies constitute a minute percentage of the crime rates in New Zealand.
When compared in respect to the total number of crimes reported to the United States, the statistics as given by Nation Master show the following:
Country
Total crime reports
Rank
New Zealand
427,230
25th in the world
United States of America
11.88 million
1st in the world 

(28 times more than New Zealand.)
Statistics based on 2015 average.
The violent crimes, including the rape and homicide index of New Zealand, is low compared to the United States, so we can say it’s safer for women, elders, and children. Being charged for an offence in New Zealand triggers a set of actions and makes you liable in a court of law for further legal proceedings where you submit your plea.
On the bright side, crime levels actually plummeted when New Zealand went into Covid-19 level 4 lockdown in 2020. Many offences remained at lower-than-usual rates by the end of 2020. In January 2020, the number to victims recorded by police was 28,342 – the highest since the data set began in mid 2014. However, this number soon plunged to 12,323 during lockdown in April – the lowest on record.
In case of any emergency, whether police, fire, or Ambulance, dial 111. Calling 111 does not require you to have credit and you can call it for free. Of course, make sure you have a valid emergency for calling!
This article will cover in-depth the crime rate in major cities of New Zealand. It will also give you an idea of if and how these crime rates might affect you. Keep scrolling to read the rest if you are looking forward to booking a flight to New Zealand or deciding on permanent residence.
Let’s cover the statistics for the five major cities in New Zealand, including the Capital, Wellington.
New Zealand crime rates: Wellington

<img decoding="async" src="https://images.unsplash.com/photo-1589871973318-9ca1258faa5d?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&amp;ixlib=rb-1.2.1&amp;auto=format&amp;fit=crop&amp;w=1740&amp;q=80" alt="New zealand sunset city skyline"/>
Photo credit: Sulthan Auliya / Unsplash
Wellington is the capital of New Zealand, and whilst it is a little city, it is a city rich in natural beauty and sights to see, especially given Wellington’s geographical status of being a valley bordered by hillside regions. In fact, all of its best sights are nearby one another!
The Crime Index of Wellington is 28.21, and the safety index is 71.79. The level of crime is low and didn’t increase as much in the past three years. According to Numbeo Wellington, the safety index of walking alone during daylight is 87.11 (Very High), and the safety index of walking alone during the night is 60.78 (High).
The statistic also reveals the overall quality of life to be 194.76, which makes it a suitable place to live as you don’t have to worry as much about theft, assault, or being a victim of hate crimes.
Robbery and shoplifting may be the only noticeable offences in Wellington. Moreover, given that the city is the capital and hub to a diverse population, sexual assault cases mainly focused on the area surrounding the Cuba-Courtenay precinct do occur at late night.
Wellington Police Station Address: 41 Victoria Street, Wellington Central, Wellington 6011, New ZealandTel: 105Opening hours: Open 24 hours (Daily)
New Zealand crime rates: Auckland
Auckland, also known as Tamaki Makaurau, is the most populous city of New Zealand. A blend of natural wonders and urban adventures that sprawls over volcanoes and around twin port cities, it is a multi-cultural melting pot of cuisine, entertainment, history, and artistry.
The Crime Index of Auckland is 45.58, and its safety index is 54.42. One would naturally presume, looking at the statistic given, Auckland loses out on the safety index as compared to Wellington, the country’s capital. According to Numbeo Auckland, the safety index for walking alone during daylight is 71.85 (a relatively high statistic) and the safety index for walking alone during the night is 42.97, which is moderate.
The Auckland Quality of Life Index is 164.19, which may be a little less compared to Wellington but nonetheless on the broader spectrum remains a high standard. The safety and security of Auckland makes this city a wonderful choice to relocate to – as long as you are able to manage the high living costs that come along with it.
Auckland Police Station Address: 13-15 College Hill, Hargreaves Street, Freemans Bay, Auckland 1011, New ZealandTel: +64 9-302 6400Opening hours: Open 24 hours (Daily)
New Zealand crime rates: Christchurch

<img decoding="async" src="https://images.unsplash.com/photo-1460853039135-e25ff9d63405?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&amp;ixlib=rb-1.2.1&amp;auto=format&amp;fit=crop&amp;w=1548&amp;q=80" alt="new zealand scenic ocean view "/>
Photo credit: Edward Manson / Unsplash
Christchurch has its location on the South Island, unlike Wellington and Auckland, which are both found on the North Island. The Christchurch Crime Index is 39.71, and its safety index is 60.29. Its safety index lies somewhere between Wellington and Auckland. Like Auckland, the safety index for walking alone during daylight is 75.21, which is high, and the safety index for walking alone during the night is 44.53, which is moderate. The quality of life index of Christchurch is 186.73, which places it just below Wellington and above Auckland.
Christchurch Police StationAddress: 40 Lichfield Street, Christchurch Central City, Christchurch 8011, New ZealandTel: +64 3-363 7400Opening hours: 7 a.m. to 9 p.m. (Mon. to Fri.); 8:30 a.m. to 6 p.m. (Sat. + Sun.)
New Zealand crime rates: Rotorua
Rotorua is famous for its boiling mud pools, erupting geysers, thermal springs, intriguing pristine lakes, as well as ancient woodlands. Maori culture, which dominates New Zealand society, is prevalent in Rotorua and be seen affecting everything from cuisine to traditions and speech.
According to Numbeo Rotorua, the city has a Crime Index of 50.61 and a safety index of 49.39. Despite being one of the prime tourist attractions in the country, Rotorua has the highest crime rate out of any city in New Zealand. However, in the bigger picture of comparison with other cities worldwide, you can see how safe Rotorua is comparatively. Major cities like Chicago and London all have higher Crime Indexes as compared to Rotorua.
Rotorua Police StationAddress: 1190/1214 Fenton Street, Rotorua 3010, New ZealandTel: +64 7-349 9554Opening hours: 8 a.m. to 4 p.m. (Daily)
New Zealand crime rates: Queenstown

<img decoding="async" src="https://images.unsplash.com/photo-1593755673003-8ca8dbf906c2?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&amp;ixlib=rb-1.2.1&amp;auto=format&amp;fit=crop&amp;w=1740&amp;q=80" alt="new zealand city crowded "/>
Photo credit: Sulthan Auliya / Unsplash
Queenstown is a tourist hotspot in Otago, New Zealand’s South Island, recognised for its enterprise tourism, specifically venture and skiing. Due to the city’s large tourist appeal, it’s a little costly when compared to other cities in New Zealand.
According to Numbeo Queenstown, it has a Crime Index of 20.22 and a safety Index of 79.78. Unfortunately, there has been a definite upswing in sexual violence and assaults over the last year in the city, with an estimate of at least three cases a month in Queenstown. The majority of sexual crimes are being committed when the victim is drunk or using drugs, often after a night out or at kick-ons, when friends put their intoxicated mate to bed. Queenstown police have launched a ‘Don’t Guess the Yes’ campaign, aimed squarely at tackling the perpetrators of sex crimes.
Queenstown Police Station Address: 11 Camp Street, Queenstown 9300, New ZealandTel: +64 3-441 1600Opening hours: 8 a.m. to 5 p.m. (Mon. to Fri.); 9 a.m. to 5 p.m. (Sat. + Sun.)
Feeling ready to make the big move?
Head over here to begin your journeyOr start ticking off your relocation checklist here
Still unsure?
Learn more about New Zealand here Explore more destinations here \n\n
| City | Crime Index | Safety Index | Police Station Address |
"""

response = openai.Completion.create(
  model="text-davinci-003",
  prompt=unstructured_data_prompt,
  temperature=0,
  max_tokens=100,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
)

print(f'{response["choices"][0]["text"]}')
