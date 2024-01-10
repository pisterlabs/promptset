import logging
import openai
from decouple import config
from django.db import transaction
import time
from django.conf import settings
import os
import pdfkit
import random
from django.template.loader import get_template
from . import resume_template
from tenacity import retry, stop_after_delay, wait_fixed    
# set up OpenAI API key
openai.api_key = config('OPENAI_API_KEY')


def generate_text_basic(prompt):
    response = None
    while response is None:
        try:
            response = openai.Completion.create(
                engine='text-davinci-003',
                prompt=prompt,
                max_tokens=3000,
                n=1,
                temperature=0.5
            )
        except openai.error.APIError as e:
            if e.status == 429 and 'Please include the request ID' in e.message:
                request_id = e.message.split('Please include the request ID ')[-1].split(' in your message.')[0]
                print(f'Retrying request {request_id}')
                request = openai.api_resources.Request.get(request_id)
                while request.status == 'pending':
                    time.sleep(1)
                    request = openai.api_resources.Request.get(request_id)
                response = openai.api_resources.Completion.get(request.response['id']).choices[0]
            elif e.status == 403:
                print('API key unauthorized')
                return None
            elif e.status == 402:
                print('Ran out of credits')
                return None
            else:
                raise e
    response = response.choices[0].text.strip().replace('\n', '<br>')
    return response

@retry(stop=stop_after_delay(60), wait=wait_fixed(1))
def generate_text(prompt):
  response = None
  while response is None:
    try:
        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
              {"role": "system", "content": "You are a helpful assistant."},
              {"role": "user", "content": prompt},
          ]
        )
    except openai.error.APIError as e:
        if e.status == 429 and 'Please include the request ID' in e.message:
            request_id = e.message.split('Please include the request ID ')[-1].split(' in your message.')[0]
            print(f'Retrying request {request_id}')
            request = openai.api_resources.Request.get(request_id)
            while request.status == 'pending':
                time.sleep(1)
                request = openai.api_resources.Request.get(request_id)
            response = openai.api_resources.ChatCompletion.get(request.response['id']).choices[0]
        elif e.status == 403:
            print('API key unauthorized')
            return None
        elif e.status == 402:
            print('Ran out of credits')
            return None
        else:
            raise e

  response = response.choices[0].message.content.strip().replace('\n', '<br>')
  return response 
    
    
    
def process_business_plan(instance):
    from .models import BusinessPlan
    # Extract fields from content object
    content = instance.content
    executive_summary_x = content.get('executive_summary', '')
    company_description_x = content.get('company_description', '')
    market_analysis_x = content.get('market_analysis', '')
    service_offered_x = content.get('service_offered', '')
    marketing_strategy_x = content.get('marketing_strategy', '')
    management_team_x = content.get('management_team', '')
    swot_analysis_x = content.get('swot_analysis', '')

    # Generate business plan text
    with transaction.atomic():
        try:
            executive_summary_prompt = f"Generate a executive summary for a business plan for the following business using the guidelines provided {executive_summary_x}.\n\n"
            executive_summary = generate_text(executive_summary_prompt)
            if not executive_summary:
                raise ValueError("Executive summary generation failed")

            company_description_prompt = f"Generate a company description for a business plan for the following business using the guidelines provided {company_description_x}. The executive summary is: {executive_summary}\n\n"
            company_description = generate_text(company_description_prompt)
            if not company_description:
                raise ValueError("Company description generation failed")

            market_analysis_prompt = f"Generate a market analysis for a business plan for the following business using the guidelines provided {market_analysis_x}. The company description is: {company_description}\n\n"
            market_analysis = generate_text(market_analysis_prompt)
            if not market_analysis:
                raise ValueError("Market analysis generation failed")

            service_offered_prompt = f"Generate service offered based on {service_offered_x}. The market analysis is: {market_analysis}. Company Description is {company_description}\n\n"
            service_offered = generate_text(service_offered_prompt)
            if not service_offered:
                raise ValueError("Service offered generation failed")

            marketing_strategy_prompt = f"Generate a market analysis for a business plan for the following business using the guidelines provided {marketing_strategy_x}. The service offered is: {service_offered}\n\n"
            marketing_strategy = generate_text(marketing_strategy_prompt)
            if not marketing_strategy:
                raise ValueError("Marketing strategy generation failed")

            management_team_prompt = f"Generate a swot analysis for a business plan for the following business using the guidelines provided. The marketing strategy is: {marketing_strategy}\n\n"
            swot_analysis = generate_text(management_team_prompt)
            if not swot_analysis:
                raise ValueError("Swot Analysis generation failed")

            # Save business plan text to database
            business_plan = BusinessPlan(
                user=instance.user,
                document_session=instance,
                executive_summary=executive_summary,
                company_description=company_description,
                market_analysis=market_analysis,
                service_offered=service_offered,
                marketing_strategy=marketing_strategy,
                swot_analysis=swot_analysis,
            )
            business_plan.save()

        except ValueError as e:
            # Log error and/or send notification
            print(f"Error: {e}")
     

def process_resume(instance):
    from .models import Resume
    # Extract fields from content object
    content = instance.content
    contact_info_x = content.get('contact_info', {})
    full_name_x = contact_info_x.get('full_name', '')
    address_x = contact_info_x.get('address', '')
    phone_number_x = contact_info_x.get('phone_number', '')
    email_x = contact_info_x.get('email', '')
    professional_summary_x = content.get('professional_summary', '')
    work_experience_x = content.get('work_experience', [])
    education_x = content.get('education', [])
    skills_x = content.get('skills', '')
    references_x = content.get('references', [])
    #cover_letter_details = content.get('cover_letter_details', [])
    
    apply_to =content.get('cover_letter').get('apply_to')
    letter_details = content.get('cover_letter').get('letter_details')
    
    # Generate resume text
    with transaction.atomic():
        try:
            professional_summary_prompt = f"Generate professional summary based on {professional_summary_x}.\n\n"
            professional_summary = generate_text(professional_summary_prompt)
            profile=f"Fullname: {full_name_x} address: {address_x} phone number:{phone_number_x}, Email :{email_x}"
            profile+=f"My Professional Summary is :{professional_summary}"
            if not professional_summary:
                raise ValueError("Professional summary generation failed")

            work_experience_prompt = f"Organize the work experiences in a list form: {work_experience_x}.\n\n"
            work_experience = generate_text(work_experience_prompt)
            profile+=f"My Work Experience: {work_experience}"
            if not work_experience:
                raise ValueError("Work experience generation failed")
            
            skills_prompt = f"Organize the skills in a list form: {skills_x}.\n\n"
            skills = generate_text(skills_prompt)
            profile+=f"My skills: {skills}"
            if not skills:
                raise ValueError("Skills generation failed")

            education_prompt = f"Organize education background in a list form: {education_x}.\n\n"
            education = generate_text(education_prompt)
            profile+=f"My Education Background: {education}"
            if not education:
                raise ValueError("Education generation failed")

            references_prompt = f"Organize references in a list form: {references_x}.\n\n"
            references = generate_text(references_prompt)
            profile+=f"My References: {references}"
            
            if not references:
                raise ValueError("References generation failed")
            
            resume_prompt = f"Generate a well detailed resume. Here are my details: {profile}. Add fields that might be required to make it more presentable. Make it as Colorful. Use structure described here {resume_template.resume_instructions()} \n\n"
            resume_x = generate_text(resume_prompt)
                        
            #resume_x = generate_text(f"Fit in {resume_x} into the following resume template {resume_template.resume_template()}")
            
            if not references:
                raise ValueError("Resume generation failed")
            
            cover_letter_prompt = f"Generate Cover Letter based on {profile}. Application details: I'm applying to {apply_to}. Job requirements are {letter_details}.\n\n"
            cover_letter = generate_text(cover_letter_prompt)
            if not cover_letter:
                raise ValueError("Cover letter generation failed")

            # Save resume text to database
            resume = Resume(
                user=instance.user,
                document_session=instance,
                contact_info=f"Fullname: {full_name_x} address: {address_x} phone number:{phone_number_x}, Email :{email_x}",
                professional_summary=professional_summary,
                work_experience=work_experience,
                education=education,
                skills=skills_x,
                references=references,
                resume = resume_x,
                cover_letter=cover_letter
            )
            resume.save()

        except ImportError:
            logging.error("Failed to import Resume model.")
        except ValueError as e:
            logging.error(f"Error: {e}")


def process_project_proposal(instance):
    from .models import ProjectProposal
    # Extract fields from content object
    content = instance.content
    title_x = content.get('title', '')
    description_x = content.get('description', '')
    objectives_x = content.get('objectives', '')
    methodology_x = content.get('methodology', '')
    budget_x = content.get('budget', '')
    timeline_x = content.get('timeline', '')
    conclusion_x = content.get('conclusion', '')

    # Generate project proposal text
    with transaction.atomic():
        try:
            title_prompt = f"Generate title based on {title_x}.\n\n"
            title = generate_text(title_prompt)
            if not title:
                raise ValueError("Title generation failed")

            description_prompt = f"Generate description based on {description_x}. The title is: {title}\n\n"
            description = generate_text(description_prompt)
            if not description:
                raise ValueError("Description generation failed")

            objectives_prompt = f"Generate objectives based on  {objectives_x}. The description is: {description}\n\n"
            objectives = generate_text(objectives_prompt)
            if not objectives:
                raise ValueError("Objectives generation failed")

            methodology_prompt = f"Generate methodology based on {methodology_x}. The objectives are: {objectives}\n\n"
            methodology = generate_text(methodology_prompt)
            if not methodology:
                raise ValueError("Methodology generation failed")

            budget_prompt = f"Generate budget based on {budget_x}. The methodology is: {methodology}\n\n"
            budget = generate_text(budget_prompt)
            if not budget:
                raise ValueError("Budget generation failed")

            timeline_prompt = f"Generate timeline based on {timeline_x}. The budget is: {budget}\n\n"
            timeline = generate_text(timeline_prompt)
            if not timeline:
                raise ValueError("Timeline generation failed")

            conclusion_prompt = f"Generate conclusion based on {conclusion_x}. The timeline is: {timeline}\n\n"
            conclusion = generate_text(conclusion_prompt)
            if not conclusion:
                raise ValueError("Conclusion generation failed")

            # Save project proposal text to database
            project_proposal = ProjectProposal(
                user=instance.user,
                document_session=instance,
                title=title,
                description=description,
                objectives=objectives,
                methodology=methodology,
                budget=budget,
                timeline=timeline,
                conclusion=conclusion,
            )
            project_proposal.save()

        except ValueError as e:
            # Log error and/or send notification
            print(f"Error: {e}")

def process_project_report(instance):
    from .models import ProjectReport
    # Extract fields from content object
    content = instance.content
    title_x = content.get('title', '')
    literature_review_x = content.get('literature_review', '')
    methodology_x = content.get('methodology', '')
    results_x = content.get('results', '')
    discussion_x = content.get('discussion', '')
    conclusion_x = content.get('conclusion', '')
    references_x = content.get('references', '')

    # Generate project report text
    with transaction.atomic():
        try:
            title_prompt = f"Generate title based on {title_x}.\n\n"
            title = generate_text(title_prompt)
            if not title:
                raise ValueError("Title generation failed")

            literature_review_prompt = f"Generate literature review based on {literature_review_x}. The title is: {title}\n\n"
            literature_review = generate_text(literature_review_prompt)
            if not literature_review:
                raise ValueError("Literature review generation failed")

            methodology_prompt = f"Generate methodology based on {methodology_x}. The literature review is: {literature_review}. Title is {title}\n\n"
            methodology = generate_text(methodology_prompt)
            if not methodology:
                raise ValueError("Methodology generation failed")

            results_prompt = f"Generate results based on {results_x}. The methodology is: {methodology}. Literature review is {literature_review}. Title is {title}\n\n"
            results = generate_text(results_prompt)
            if not results:
                raise ValueError("Results generation failed")

            discussion_prompt = f"Generate discussion based on {discussion_x}. The results are: {results}\n\n"
            discussion = generate_text(discussion_prompt)
            if not discussion:
                raise ValueError("Discussion generation failed")

            conclusion_prompt = f"Generate conclusion for {conclusion_x}. The discussion is: {discussion}\n\n"
            conclusion = generate_text(conclusion_prompt)
            if not conclusion:
                raise ValueError("Conclusion generation failed")

            references_prompt = f"Generate references based on {references_x}. The conclusion is: {conclusion}\n\n"
            references = generate_text(references_prompt)
            if not references:
                raise ValueError("References generation failed")

            # Save project report text to database
            project_report = ProjectReport(
                user=instance.user,
                document_session=instance,
                title=title,
                literature_review=literature_review,
                methodology=methodology,
                results=results,
                discussion=discussion,
                conclusion=conclusion,
                references=references,
            )
            project_report.save()

        except ValueError as e:
            # Log error and/or send notification
            print(f"Error: {e}")



#The following tasks are handled in the background by celery
def process_business_plan_pdf(instance):
    '''
    Process the business plan
    '''
    create_pdf('business_plan.html',instance,attachment="business_plan.pdf")
    

def process_project_proposal_pdf(instance):
    '''
    Process the project proposal
    '''
    create_pdf('project_proposal.html',instance,attachment="project_proposal.pdf")
    

def create_resume_pdf(instance):
    '''
    Process the resume
    '''
    create_pdf('resume.html',instance,attachment="resume.pdf")
    create_pdf('cover_letter.html',instance,attachment="cover_letter.pdf")
    

def create_project_report_pdf(instance):
    '''
    Process the project report
    '''
    create_pdf('project_report.html',instance,attachment="project_report.pdf")
    

def create_pdf(template_index,instance,attachment):
    from decouple import config
    from pathlib import Path
    from urllib.parse import urljoin
    from notifications.utils import mail_notification

    # Variables we need
    
    #The name of your PDF file
    filename =f"{random.randint(10000000000000, 99999999999999)}.pdf"

    #HTML FIle to be converted to PDF - inside your Django directory
    template = get_template(template_index)

    #Add any context variables you need to be dynamically rendered in the HTML
    context = {"data":instance}

    #Render the HTML
    html = template.render(context)

    #Options - Very Important [Don't forget this]
    options = {
        'encoding': 'UTF-8',
        'javascript-delay':'1000', #Optional
        'enable-local-file-access': None, #To be able to access CSS
        'page-size': 'A4',
        'custom-header' : [
            ('Accept-Encoding', 'gzip')
        ],
    }
    #Javascript delay is optional

    #Remember that location to wkhtmltopdf
    
    path = r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe" # Windows
    
    pdf_config = pdfkit.configuration(wkhtmltopdf=path)

     #Saving the File
    #file_path = settings.MEDIA_ROOT + '/documents/{}/'.format(instance.user.id)
    file_path = Path(settings.MEDIA_ROOT).joinpath('documents', str(instance.user.id))
    os.makedirs(file_path, exist_ok=True)
    pdf_save_path = "{}/{}".format(file_path,filename)
    #pdf_save_path = urljoin(file_path.as_posix(), filename)

    #Save the PDF
    pdfkit.from_string(html, pdf_save_path,
                       configuration=pdf_config,
                       options=options)
    
    doc_url = 'documents/{}/{}'.format(instance.user.id,filename)
    
    mail_notification(recipient=instance.user.email,
                    subject="File Created Successfully",
                    message="Below Attached is your Document file",
                    attachment_path=pdf_save_path,
                    attachment=attachment
                    )
    
    # Combine base URL and document URL using urljoin
    https_path = urljoin(config('APP_DOMAIN'), doc_url)

    #Return
    return https_path


