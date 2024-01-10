# Script for creating xml files of letters as indicated in register of letters
import os
import pandas as pd
from datetime import datetime
import openai
import config
from dateutil.parser import parse

openai.api_key = config.openai_key

""" Analysis of dataset
"""
# import register as pandas df
data = pd.read_csv(os.path.join(os.getcwd(), '..', 'data', 'register', 'enriched_register.csv'), sep=';')

# get an overview on data
print(data.head)

## how many letters were send to Lassberg indicated as 'AN' in data['VON/AN']?
print(data[data['VON/AN'] == 'AN'].shape[0])

## how many letters were send from Lassberg indicated as 'VON' in data['VON/AN']?
print(data[data['VON/AN'] == 'VON'].shape[0])

## How many different correspondents
print(data['Name_voll'].nunique())

## How many different correspondents to Lassberg
print(data[data['VON/AN'] == 'AN']['Name_voll'].nunique())

## How many different correspondents from Lassberg
print(data[data['VON/AN'] == 'VON']['Name_voll'].nunique())

## count unique persons in data and print names and count
print(data['Name_voll'].value_counts())

## count letters send to male and female persons
#print('Geschlecht: \n')
print(data['gender_gnd'].value_counts())


## differentiate between persons that have 'VON' or 'AN' in data['VON/AN'] and print names and count
#print('Empfänger: \n')
print(data[data['VON/AN'] == 'VON']['Name_voll'].value_counts())

#print('Absender: \n')
print(data[data['VON/AN'] == 'AN']['Name_voll'].value_counts())

# get unique list of places in data and count
print(data['Ort'].value_counts())

# differentiate between places that have 'VON' or 'AN' in data['VON/AN'] and print places and count
print('Gesendet von an Lassberg: \n')
print(data[data['VON/AN'] == 'VON']['Ort'].value_counts())

#print('Gesendet nach von Lassberg: \n')
print(data[data['VON/AN'] == 'AN']['Ort'].value_counts())

## use pandas to count persons with the same degree in unique_persons
degree_counts = unique_persons['degrees_gnd'].value_counts()
print(degree_counts)

## get list list of persons ranked according to the number of their publications as indicated in  unique_persons['publications_gnd']
# set '-' in unique_persons['publications_gnd'] to 0
unique_persons['publications_gnd'] = unique_persons['publications_gnd'].replace('-', 0)
# set unique_persons['publications_gnd'] to int
unique_persons['publications_gnd'] = unique_persons['publications_gnd'].astype(int) 
publications_ranking = unique_persons.sort_values(by=['publications_gnd'], ascending=False)
# only display unique_persons['Name_voll'] and unique_persons['publications_gnd']
publications_ranking = publications_ranking[['Name_voll', 'publications_gnd']]
print(publications_ranking)

## get value of occupation_gnd in unique_persons, split by , and print unique values with count
print(unique_persons['occupations_gnd'].str.split(',').explode().value_counts())



## How many letters send each year
## How many letters recieved each year
## Where are those letters from?
## Map of places
## über gnd alter der korrespondierenden abfragen und altersunterschied zu Lassberg
## über gnd Beruf oder Beschäftigung abfragen und analysieren
## über gnd Anzahl der Publikationen abfragen