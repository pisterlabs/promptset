from pymongo import MongoClient
import configparser
import pprint

import re
import subprocess
import os
import requests
import shutil
import sys
import toml 
from langchain.prompts import PromptTemplate
import json
import shutil
from flask import Flask, request, jsonify

pp = pprint.PrettyPrinter(indent=4)

API_URL = os.getenv("PYPIAPI")
MONGO_URI = os.getenv("MONGO")
AITOKEN = os.getenv("OPENAI_API_KEY")

# Create a MongoDB client
client = MongoClient(MONGO_URI)
db = client['pypi-packages']



def registerpypipackage(name, version, dependency, parent):
  parents = []
  parents.append(parent)

  package_data = {
            'name': name,
            'version': version,
            'status': "initial", 
            'dependency': dependency,
            'parents': parents,
            'hotfix': { 'filename': "", 'content': ""},
            'setuppy': False,
            'setuppyhotfix': False,
            'setupcfg': False,
            'prettysetuppy': False,
            'pyprojecttoml': False,
            'sourcedownloaded': False,
            'sourceunpacked': False,
            'specfile': "",
            'specfilecreated': False,
            'rpmbuild': False,
            'rpmfilepublished': False,
            'debbuild': False
        }
  response = requests.post(API_URL, json=package_data)
  if response.status_code == 200:
    print(f'Registered package: {name}')
  else:
    print(f'Error registering package {name}: {response.text}')


def updatepypipackage(name, version, status):
  package_data = {
            'name': name,
            'version': version,
            'status': status
        }
  response = requests.post(API_URL, json=package_data)
  if response.status_code == 200:
    print(f'Updated package: {name}')
  else:
    print(f'Error updating package {name}: {response.text}')
   

def filenames(mypath):
  file_names = []
  files = os.listdir(mypath)
  for file in files:
    file_names.append(file)
  return file_names

def diflist(list1 , list2):
  list_dif = [i for i in list1 + list2 if i not in list1 or i not in list2]
  return list_dif

def downloadpypipackage(name, version):
  download_folder = '/tmp/empty'
  destination_folder = os.getenv('DOWNLOAD_FOLDER', '/tmp')
  try:
    shutil.rmtree(download_folder)	
    os.rmdir(download_folder)
  except:
    pass  
  os.mkdir(download_folder)

  packages = db['pypi_packages']
  downloads = []
  #download_folder = os.getenv('DOWNLOAD_FOLDER', '/tmp')
  try:
    package_name = name + '==' + version
  except:
    return []



  #subprocess.call(["pip", "download", "-d", download_folder, package_name])
  before = filenames(download_folder)
  process = subprocess.run(["pip", "download", '--no-binary' , ':all:',  "-d", download_folder, package_name], capture_output=True, text=True)
  if process.returncode == 0:
    process2 = subprocess.run(["pip", "download", '--no-binary' , ':all:',  "-d", destination_folder, package_name], capture_output=True, text=True)
    if process2.returncode == 0:
      query = {'name': name, 'version': version}
      update = {'$set': {'sourcedownloaded': True, 'status': "Source downloaded"}, }
      packages.update_one(query, update)
      print(f"Downloaded source for {name} {version}")
    else:
      print(f"Error downloading source for {name} {version}")
      print(process.stderr)
      query = {'name': name, 'version': version}
      update = {'$set': {'sourcedownloaded': False, 'status': "Error downloading source"}, }
      packages.update_one(query, update)
    


  after = filenames(download_folder)
  diff = diflist(before, after)
  for i in diff:
    newpackage = i[::-1].split('-', 1)[1][::-1]
    newversion = i[::-1].split('-', 1)[0][::-1].replace('.tar.gz', '').replace('.whl', '').replace('.zip', '')
    if newpackage == name:
      continue
    downloads.append({'filename': i, 'package': newpackage, 'version': newversion})
    parent = name + '==' + version
    for download in downloads:
      query = {'name': download['package'], 'version': download['version']}
      package = packages.find_one(query)
      if package:
        print(f"Package {download['package']} {download['version']} already exists")
        # We should get the parent of this package and add it to the child list
      else:
        registerpypipackage(download['package'], download['version'], True, parent)
        for file in downloads:
          try:
            shutil.copyfile( download_folder + '/' + file['filename'], destination_folder + "/" + file['filename'])
          except:
            pass
  return downloads

print("Starting download")

emptyq = False  
while not emptyq:
  print("Checking for empty packages")

  query = {'sourcedownloaded': False}
  packages = db['pypi_packages']
  packmeta = {}
  queudepth = 0  
  for package in packages.find(query):
    packmeta[package['name']] = package['version']
    print(package['name'])
    print(package['version'])
    print(package['sourcedownloaded'])
    if package['sourcedownloaded'] == False:
      queudepth = queudepth + 1


  if queudepth == 0:
    emptyq = True
  else:
    print("We have %d packages to download" % queudepth)
    for package in packages.find(query):
      downloads = downloadpypipackage(package['name'], package['version'])
      for download in downloads:
        query = {'name': package['name'], 'version': package['version']}
        update = {'$push': {'downloads': download}}
        packages.update_one(query, update)




