from django.shortcuts import render,redirect
from django.contrib.auth import authenticate,login,logout
from django.contrib import messages
from django.contrib.auth.forms import  UserCreationForm
from .decorators import *

from django.db.models import ExpressionWrapper, Q, BooleanField
from django.utils.timezone import now
from .forms import *
from .models import *

from django.http import JsonResponse
from openai import OpenAI

from django.conf import settings

from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from fpdf import FPDF


# Assuming Django settings are properly configured with OPENAI_API_KEY

medicaments_disponibles = [
    "Paracétamol", "Ibuprofène", "Amoxicilline", "Metformine",
    "Atorvastatine", "Lisinopril", "Amlodipine", "Omeprazole"
    # Add other medications as needed
]

def create_prompt(nom, prenom, age, sexe, poids, temperature, antecedents_medicaux, symptomes, medicaments_disponibles, maladies_symptomes):
    liste_medicaments = ', '.join(medicaments_disponibles)

    # Format the list of diseases and their symptoms
    formatted_maladies_symptomes = "\n".join([f"-- ('{maladie}', '{symptomes}')" for maladie, symptomes in maladies_symptomes])

    prompt = f"""
    Créez une ordonnance médicale détaillée et structurée pour un patient avec les informations suivantes :
    - Nom : {nom}
    - Prénom : {prenom}
    - Âge : {age}
    - Sexe : {sexe}
    - Poids : {poids} kg
    - Température : {temperature} °C
    - Symptômes : {symptomes}
    - Antécédents médicaux : {antecedents_medicaux}
    - Liste des médicaments disponibles en pharmacie : {liste_medicaments}
     Liste des maladies et de leurs symptômes possibles dans notre application :
    {formatted_maladies_symptomes}


    L'ordonnance doit inclure :
    - la potentiel maladie dont le patient souffre(ceci parmis la liste des maladies fournies)
    - Les médicaments appropriés parmi la liste fournie en tenant compte des symptômes, des antécédents (contenant des allergies) et des informations du patient
    - La posologie et la durée du traitement
    - Les recommandations spécifiques pour le patient
    
    NB:
    Aucune reponse insensé, si besoin précise que tu n'as pas de proposition a faire.
    je ne veux aucune reponse hor contexte
    
    Toujours repondre en français
    Ne pas include la liste claire des information venant de la base de donnees.
    Si tu ne sais pas qoui dire, il faut juste repondre par Aucune decision à proposer.
    Veuillez également inclure une mention à la fin de l'ordonnance indiquant que celle-ci doit être revue et approuvée par un médecin avant utilisation.
    Veuillez également ne pas prescrire ce qu'il n'a pas dans la liste des medicaments.
    """
    return prompt

#@csrf_exempt  # Only for demonstration purposes. CSRF protection should be enabled in a real project.
#@require_POST
"""
def get_maladies_symptomes():
    maladies_symptomes = []
    maladies = Maladie.objects.all()

    for maladie in maladies:
        symptomes = Symptome.objects.filter(maladies_associees__pk=maladie.pk)
        symptomes_list = ', '.join([symptome.nom_symptome for symptome in symptomes])
        maladies_symptomes.append((maladie.nom_maladie, symptomes_list))

    return maladies_symptomes

"""
def get_maladies_symptomes():
    maladies_symptomes = []
    maladies = Maladie.objects.all()

    for maladie in maladies:
        symptomes = Symptome.objects.filter(correspondance__pk=maladie.pk)
        symptomes_list = ', '.join([symptome.nom_symptome for symptome in symptomes])
        maladies_symptomes.append((maladie.nom_maladie, symptomes_list))
    print("maladies:")
    liste =[]
    for cor in Correspondance.objects.all():
        print(cor.symptome.nom_symptome)
        print(cor.maladie.nom_maladie)
        liste.append(f'{cor.symptome.nom_symptome}:{cor.maladie.nom_maladie}')
    print(maladies_symptomes)
    return maladies_symptomes

def generate_openai_response(request, patient, prescription):
    print("Entering generate_openai_response function")

    # Get patient information from the patient and prescription parameters
    nom = patient.first_name
    prenom = patient.last_name
    age = patient.age
    sexe = patient.gender
    poids = prescription.poids
    temperature = prescription.temperature
    antecedents_medicaux = prescription.antecedents_medicaux
    symptomes = prescription.symptoms

    print("Before create_pdf_and_get_openai_response")

    # Fetch the list of diseases and symptoms from the database
    maladies_symptomes = get_maladies_symptomes()

    # Create PDF and get OpenAI response
    pdf_filename, openai_response = create_pdf_and_get_openai_response(
        nom, prenom, age, sexe, poids, temperature, antecedents_medicaux, symptomes, maladies_symptomes
    )
    print("After create_pdf_and_get_openai_response")

    # You can do further processing or send the response to the frontend as needed
    return {'pdf_filename': pdf_filename, 'openai_response': openai_response}


def create_pdf_and_get_openai_response(nom, prenom, age, sexe, poids, temperature, antecedents_medicaux, symptomes,maladies_symptomes):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # En-tête
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, "e-Ordonnance Médicale", ln=True, align='C')

    # Corps du document

    stocks = Stock.objects.all().order_by("-id")

    # Filter expired stocks
    expired_stocks = stocks.annotate(
        expired=ExpressionWrapper(Q(valid_to__lt=now()), output_field=BooleanField())
    ).filter(expired=True)

    # Filter non-expired stocks
    non_expired_stocks = stocks.annotate(
        expired=ExpressionWrapper(Q(valid_to__lt=now()), output_field=BooleanField())
    ).filter(expired=False)

    # Extract medication names from both expired and non-expired stocks
    medicament_disponibles_expired = [stock.drug_name  for stock in expired_stocks]
    medicament_disponibles_non_expired = [stock.drug_name for stock in non_expired_stocks]

    ordonnance_info = create_prompt(nom, prenom, age, sexe, poids, temperature, antecedents_medicaux, symptomes,
                                    medicament_disponibles_non_expired, maladies_symptomes)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, ordonnance_info)

    # Note de bas de page
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 10, "Note: Cette ordonnance doit être revue et approuvée par un médecin.",
             ln=True, align='C')

    filename = f"{nom}_{prenom}_ordonnance.pdf"
    pdf.output(filename)

    # Get OpenAI response using the generated prompt

    client = OpenAI(api_key=settings.OPENAI_API_KEY)

    response = client.completions.create(
        model="text-davinci-002",
        prompt=ordonnance_info,
        max_tokens=700
    )
    print("")
    print(response.choices[0].text)
    print("")
    return filename, response.choices[0].text



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


def addConsultation(request, pk):
    patient = Patients.objects.get(id=pk)
    form = ConsultationForm(initial={'patient_id': patient})

    if request.method == 'POST':
        try:
            form = ConsultationForm(request.POST)
            if form.is_valid():
                prescription = form.save()
                print(prescription)
                messages.success(request, 'Consultation added successfully')
                print("here")
                # Call the OpenAI function and pass the necessary parameters
                openai_response = generate_openai_response(request, patient, prescription)
                print("after")
                # Redirect to the prescription result page with the prescription id
                return render(request, 'doctor_templates/prescription_result.html', {'openai_response': openai_response})
        except Exception as e:
            messages.error(request, 'Consultation Not Added')
            print(f'Exception: {str(e)}')
            return redirect('manage_patient_doctor')

    context = {
        "form": form
    }
    return render(request, 'doctor_templates/prescribe_form.html', context)
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
        except Exception as e:
            messages.error(request,'Consultation Not Deleted successfully')
            print(f'Exception: {str(e)}')
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
    