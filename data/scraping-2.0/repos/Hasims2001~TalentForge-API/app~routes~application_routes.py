from flask import Blueprint,  request
from app.models import Application,JobSeeker,JobPosting, SkillSet, db
from app import success, fail, successWithData
from openai import OpenAI
import os, json

application_bp = Blueprint('application', __name__)

# apply on job post
@application_bp.route('/apply', methods=['POST'])
def apply_for_job():
    try:
        data = request.get_json()
        user=  request.user
        job_seeker_id = user['id']
        job_posting_id = data.get('job_posting_id')

        if not job_seeker_id or not job_posting_id:
            return fail("Job Seeker ID and Job Posting ID are required"), 400

        job_seeker = JobSeeker.query.get(job_seeker_id)
        job_posting = JobPosting.query.get(job_posting_id)

        if not job_seeker or not job_posting:
            return fail("Job Seeker or Job Posting not found"), 404

        existing_application = Application.query.filter_by(
            job_seeker_id=job_seeker.id,
            job_posting_id=job_posting.id
        ).first()

        if existing_application:
            return fail("Job Seeker has already applied for this job"), 400

        application = Application(
            status=data.get('status', 'Pending'), 
            job_seeker=job_seeker,
            job_posting=job_posting
        )

        db.session.add(application)
        db.session.commit()

        return successWithData("Application submitted successfully!", {"job_id": job_posting_id})

    except Exception as e:
        return fail(str(e)), 500
    
# get all
@application_bp.route('/all', methods=['GET'])
def get_all_applications():
    try:
        applications = Application.query.all()
        result = []

        for application in applications:
            result.append({
                "id": application.id,
                "status": application.status,
                "job_seeker_id": application.job_seeker_id,
                "job_posting_id": application.job_posting_id
            })

        return successWithData("all application",result)

    except Exception as e:
        return fail(str(e)), 401

# get all of single job post
@application_bp.route("/all/<int:jobpost_id>", methods=['GET'])
def get_all_applications_of_job_post(jobpost_id):
    try:
        allApplication = Application.query.filter_by(job_posting_id=jobpost_id).all()
        result = []
        for application in allApplication:
            result.append({
                "id": application.id,
                "status": application.status,
                "job_seeker_id": application.job_seeker_id,
                "job_posting_id": application.job_posting_id
            })

        return successWithData("all application",result)
    except Exception as e:
        return fail(str(e)), 401


#get all of single user(job_seeker)
@application_bp.route("/user/all", methods=['GET'])
def get_all_applications_of_user():
    try:
        job_seeker_id = request.user['id']
        all_application = Application.query.filter_by(job_seeker_id=job_seeker_id).all()
        result = []
        
        for application in all_application:
            result.append({
                "id": application.id,
                "status": application.status,
                "job_seeker_id": application.job_seeker_id,
                "job_posting_id": application.job_posting_id,
                "job_posting": {
                    "id": application.job_posting.id,
                    "job_title": application.job_posting.job_title,
                    "description": application.job_posting.description,
                    "salary": application.job_posting.salary,
                    "graduation":application.job_posting.graduation,
                    "postgraduation":application.job_posting.postgraduation,
                    "location": application.job_posting.location,
                    "role_category": application.job_posting.role_category,
                    "department": application.job_posting.department,
                    "experience": application.job_posting.experience,
                    "required_skills": application.job_posting.required_skills,
                    "prefered_skills": application.job_posting.prefered_skills,
                    "employment_type": application.job_posting.employment_type,
                    "openings": application.job_posting.openings,
                    "recruiter": {
                        "id": application.job_posting.recruiter.id,
                        "name": application.job_posting.recruiter.name,
                        "company_name": application.job_posting.recruiter.company_name,
                        "company_logo": application.job_posting.recruiter.company_logo,
                        "company_description": application.job_posting.recruiter.company_description,
                        "website": application.job_posting.recruiter.website
                    }
                    
                }
            })

        return successWithData(f"all application related to user id {job_seeker_id}",result)
    except Exception as e:
        return fail(str(e)), 401


# update
@application_bp.route('/update/<int:id>', methods=['PATCH', 'PUT'])
def update_application(id):
    try:
        application = Application.query.get(id)
        data = request.get_json()

        if application:
            for key in data:
                setattr(application, key, data[key])

            db.session.commit()
            return successWithData("Job Application updated successfully", application)
        else:
            return fail("Job Application not found!"), 404

    except Exception as e:
        return fail(str(e)), 401

# delete
@application_bp.route('/delete/<int:id>', methods=['DELETE'])
def delete_application(id):
    try:
        application = Application.query.get(id)

        if application:
            db.session.delete(application)
            db.session.commit()
            return success("Job Application deleted successfully")
        else:
            return fail("Job Application not found!"), 404

    except Exception as e:
        return fail(str(e)), 401


openai_api_key = os.getenv("API_KEY") 
client = OpenAI(api_key=openai_api_key)


# recommend applicant
def recommend_job_seeker(jobpostid):
    try:
        print("recommend job post id",jobpostid)
        jobpostid = int(jobpostid)
        jobpost = JobPosting.query.filter_by(id=jobpostid).first()
        if(jobpost):
            desired_skills =[
                jobpost.required_skills.strip(",")
            ]
        
            recommended_jobseekers = JobSeeker.query.filter(
                (JobSeeker.skills.any(SkillSet.skills.in_(desired_skills)))).all()
            result = []
            print('recommended_jobseekers', recommended_jobseekers)
            for each in recommended_jobseekers:
                user_skills = [skill.skills for skill in each.skills]
                user_graduate = [degree.degree for degree in each.graduate]
                user_postgraduate = [degree.degree for degree in each.postgraduate]
                result.append({
                    "id": each.id,
                    "name": each.name,
                    "email": each.email,
                    "graduate": user_graduate,
                    "postgraduate": user_postgraduate,
                    "education": each.education,
                    "skills": user_skills,
                    "experience": each.experience,
                    "city": each.city,
                    "state": each.state,
                    'pincode': each.pincode

                })
            print(result)
            return result
        else:
            return "no job posts found!"
    except Exception as e:
        return str(e)


@application_bp.route("/recommend", methods=['POST'])
def recommend_applicant():
    try:
        messages = request.get_json()
        tools = [
         {
                "type": "function",
                "function": {
                    "name": "recommend_job_seeker",
                    "description": "Recommend applicant using job post id",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "jobpostid": {
                                "type": "number",
                                "description": "job post id number",
                            },
                        },
                        "required": ["jobpostid"],
                    },
                },
            },
        ]
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=messages,
            tools=tools,
            tool_choice="auto", 
        )
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls
        if tool_calls:
            available_functions = {
                "recommend_job_seeker": recommend_job_seeker,
            } 
            messages.append(response_message)  
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_functions[function_name]
                function_args = json.loads(tool_call.function.arguments)
                function_response = function_to_call(
                    jobpostid=function_args.get("jobpostid"),
                )
                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": str(function_response),
                    }
                )  
            # second_response = client.chat.completions.create(
            #     model="gpt-3.5-turbo-1106",
            #     messages=messages,
            # ) 
            
           
            modified_data = {
                "content": function_response,
                "role": "system"
            }
           
            if(type(function_response) == list and len(function_response) > 0):
                return successWithData(msg="here is the some recommendation according to job requirements:", data=modified_data)
            elif(type(function_response) == list):
                return successWithData(msg="sorry, there is no jobseeker available who fullfill your requirements!", data=modified_data)
            else:
                return successWithData(msg="first_output", data=modified_data)
        else:
            result = {
                 "content": response_message.content,
                 "role": response_message.role
            }  
            return successWithData(msg="first_output", data=result)
    except Exception as e:
        return fail(str(e))