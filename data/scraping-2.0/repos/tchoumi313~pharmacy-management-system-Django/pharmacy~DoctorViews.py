from datetime import datetime

import openai
from django.shortcuts import render,redirect
from django.contrib.auth import authenticate,login,logout
from django.contrib import messages
from django.contrib.auth.forms import  UserCreationForm
from openai import OpenAI

from pharm import settings
from .decorators import *

from .forms import *
from .models import *
from .prise_de_decision.main import generer_numero_ordonnance, create_prompt, extract_medical_data


def doctorHome(request):
    prescip = Consultation.objects.all().count()

    context={
        "Prescription_total":prescip

    }
    return render(request,'doctor_templates/doctor_home.html',context)

def doctorProfile(request):
    customuser=CustomUser.objects.get(id=request.user.id)
    staff=Doctor.objects.get(admin=customuser.id)

    form=DoctorForm()
    if request.method == 'POST':
        first_name=request.POST.get('first_name')
        last_name=request.POST.get('last_name')


        customuser=CustomUser.objects.get(id=request.user.id)
        customuser.first_name=first_name
        customuser.last_name=last_name
        customuser.save()

        staff=Doctor.objects.get(admin=customuser.id)
        form =DoctorForm(request.POST,request.FILES,instance=staff)

        staff.save()

        if form.is_valid():
            form.save()

    context={
        "form":form,
        "staff":staff,
        "user":customuser
    }

    return render(request,'doctor_templates/doctor_profile.html',context)

def managePatients(request):
    patients=Patients.objects.all()

    context={
        "patients":patients,

    }
    return render(request,'doctor_templates/manage_patients.html',context)

def addPrescription(request,pk):        
    patient=Patients.objects.get(id=pk)
    print(patient.first_nameé,pk)
    form=ConsultationForm(initial={'patient': patient})
    if request.method == 'POST':
        try:
            form=ConsultationForm(request.POST or None)
            if form.is_valid():
                print(form.instance.patient.first_name)
                form.instance.NUM_ORD = generer_numero_ordonnance(form.instance.patient.first_name,
                                                                  form.instance.patient.last_name)
                form.instance.DATE_ORD = datetime.now().strftime("%Y-%m-%d")

                # Save the form data to the database
                form.save()

                # Update the data_patient dictionary with form instance values
                data_patient = {
                    "NUM_ORD": form.instance.NUM_ORD,
                    "NOM_PAT": form.instance.patient.first_name + " " + form.instance.patient.last_name,
                    "DATE_ORD": form.instance.DATE_ORD,
                    "AGE_PAT": str(form.instance.patient.age) + " ans",
                    "SEXE_PAT": form.instance.patient.gender,
                    "PROFESSION_PAT": form.instance.patient.profession,
                    "ADRESE_PAT": form.instance.patient.address,
                    "TEL_PA": form.instance.patient.phone_number,
                    "SYMP_PAT": form.instance.SYMP_PAT,
                    "ANTECEDENTS_PAT": form.instance.ANTECEDENTS_PAT,
                    "TEMP": form.instance.TEMP + " °C",
                    "FC": form.instance.FC + " bpm",
                    "PA": form.instance.PA + " mmHg",
                    "ALLERGIES": form.instance.ALLERGIES,
                    "HANDICAP": form.instance.HANDICAP,
                    "POIDS": form.instance.POIDS + " kg",
                }
                medicaments_disponibles = Stock.objects.all()

                medicaments = []
                for med in medicaments_disponibles:
                    if med.expired:
                        medicaments.append(f"{med.drug_imprint} - {med.drug_name}")
                # Générer le prompt pour OpenAI
                prompt = create_prompt(data_patient, medicaments)

                # Envoyer le prompt à OpenAI
                client = OpenAI(api_key=settings.OPENAI_API_KEY)

                response = client.completions.create(
                    model="text-davinci-003",
                    prompt=prompt,
                    max_tokens=700
                ).choices[0].text.strip()

                print("Réponse brute d'OpenAI:\n", response)

                # Extraire les données médicales de la réponse d'OpenAI
                data_ordonnance = extract_medical_data(response)

                # Imprimer les données extraites pour le débogage
                print("Données médicales extraites:\n", data_ordonnance)

                # Préparer les données pour le remplissage du PDF
                data_pdf = {**data_patient, **data_ordonnance}
                context = {
                    'response': response,
                    'data_patient': data_patient,
                    'data_ordonnance': data_ordonnance,
                }

                # Render the result_page.html template with the extracted information
                return render(request, 'doctor_templates/result_page.html', context)
        except:
            messages.error(request,'Consultation Not Added')
            return redirect('manage_patient-doctor')


 
    
    context={
        "form":form
    }
    return render(request,'doctor_templates/prescribe_form.html',context)

def patient_personalDetails(request,pk):
    patient=Patients.objects.get(id=pk)
    prescrip=patient.prescription_set.all()

    context={
        "patient":patient,
        "prescription":prescrip

    }
    return render(request,'doctor_templates/patient_personalRecords.html',context)

def deletePrescription(request,pk):
    prescribe=Consultation.objects.get(id=pk)

    if request.method == 'POST':
        try:
            prescribe.delete()
            messages.success(request,'Consultation Deleted successfully')
            return redirect('manage_precrip_doctor')
        except:
            messages.error(request,'Consultation Not Deleted successfully')
            return redirect('manage_precrip_doctor')




    context={
        "patient":prescribe
    }

    return render(request,'doctor_templates/sure_delete.html',context)
    
def managePrescription(request):
    precrip=Consultation.objects.all()

    patient = Patients.objects.all()
    
       
    context={
        "prescrips":precrip,
        "patient":patient
    }
    return render(request,'doctor_templates/manage_prescription.html' ,context)

def editPrescription(request,pk):
    prescribe=Consultation.objects.get(id=pk)
    form=ConsultationForm(instance=prescribe)

    
    if request.method == 'POST':
        form=ConsultationForm(request.POST, instance=prescribe)

        try:
            if form.is_valid():
                form.save()

                messages.success(request,'Consultation Updated successfully')
                return redirect('manage_precrip_doctor')
        except:
            messages.error(request,' Error!! Consultation Not Updated')
            return redirect('manage_precrip_doctor')




    context={
        "patient":prescribe,
        "form":form
    }

    return render(request,'doctor_templates/edit_prescription.html',context)
    