from flask import Blueprint, request
from app.models import JobPosting,Application,SkillSet, db
from app import success, fail, successWithData
from openai import OpenAI
from sqlalchemy import or_
import os, json
jobposting_bp = Blueprint('jobposting', __name__)

# create a new 
@jobposting_bp.route('/create', methods=['POST'])
def create_jobposting():
    try:
        data = request.get_json()
        user = request.user
        new_jobposting = JobPosting(
            job_title=data['job_title'],
            description=data['description'],
            salary=data['salary'],
            graduation=data.get('graduation', ""),
            postgraduation=data.get('postgraduation', ""),
            location=data['location'],
            role_category=data['role_category'],
            department=data['department'],
            experience=data['experience'],
            required_skills=data['required_skills'],
            prefered_skills=data['prefered_skills'],
            employment_type=data['employment_type'],
            openings=data['openings'],
            recruiter_id=user['id']
        )

        db.session.add(new_jobposting)
        db.session.commit()
        result = {
            "id": new_jobposting.id,
            "job_title": new_jobposting.job_title,
            "description": new_jobposting.description,
            "salary": new_jobposting.salary,
            "graduation":new_jobposting.graduation,
            "postgraduation":new_jobposting.postgraduation,
            "location": new_jobposting.location,
            "role_category": new_jobposting.role_category,
            "department": new_jobposting.department,
            "experience": new_jobposting.experience,
            "required_skills": new_jobposting.required_skills,
            "prefered_skills": new_jobposting.prefered_skills,
            "employment_type": new_jobposting.employment_type,
            "openings": new_jobposting.openings,
            "recruiter_id": new_jobposting.recruiter_id
        }
        return successWithData("Job post created successfully!", result)

    except Exception as e:
        return fail(str(e)), 401

# get all except applied by jobseeker
@jobposting_bp.route('/all/jobs', methods=['GET'])
def get_all_jobpostings():
    try:
        user_id = request.user['id']
        jobpostings = JobPosting.query.filter(
            ~JobPosting.applications.any(Application.job_seeker_id == user_id)).all()
        result = []
        for jobposting in jobpostings:
            result.append({
                "id": jobposting.id,
                "job_title": jobposting.job_title,
                "description": jobposting.description,
                "salary": jobposting.salary,
                "graduation":jobposting.graduation,
                "postgraduation":jobposting.postgraduation,
                "location": jobposting.location,
                "role_category": jobposting.role_category,
                "department": jobposting.department,
                "experience": jobposting.experience,
                "required_skills": jobposting.required_skills,
                "prefered_skills": jobposting.prefered_skills,
                "employment_type": jobposting.employment_type,
                "openings": jobposting.openings,
                "recruiter_id": jobposting.recruiter_id
            })

        return successWithData("all jobs", result)

    except Exception as e:
        return fail(str(e)), 401

# get all except applied by category
@jobposting_bp.route('/all/jobs/<string:category>', methods=['GET'])
def get_all_jobpostings_by_category(category):
    try:
        user_id = request.user['id']
        jobpostings = JobPosting.query.filter((JobPosting.role_category == category) & ~JobPosting.applications.any(Application.job_seeker_id == user_id)).all()

        result = []
        for jobposting in jobpostings:
            result.append({
                "id": jobposting.id,
                "job_title": jobposting.job_title,
                "description": jobposting.description,
                "salary": jobposting.salary,
                "graduation":jobposting.graduation,
                "postgraduation":jobposting.postgraduation,
                "location": jobposting.location,
                "role_category": jobposting.role_category,
                "department": jobposting.department,
                "experience": jobposting.experience,
                "required_skills": jobposting.required_skills,
                "prefered_skills": jobposting.prefered_skills,
                "employment_type": jobposting.employment_type,
                "openings": jobposting.openings,
                "recruiter_id": jobposting.recruiter_id
            })

        return successWithData("all jobs", result)

    except Exception as e:
        return fail(str(e)), 401


# get all for recruiter
@jobposting_bp.route("/all/<int:id>", methods=['GET'])
def get_all_jobpostings_for_recruiter(id):
    try:
        jobpostings = JobPosting.query.filter_by(recruiter_id=id).all()
        result = []
        for jobposting in jobpostings:
            result.append({
                "id": jobposting.id,
                "job_title": jobposting.job_title,
                "description": jobposting.description,
                "salary": jobposting.salary,
                "graduation":jobposting.graduation,
                "postgraduation":jobposting.postgraduation,
                "location": jobposting.location,
                "role_category": jobposting.role_category,
                "department": jobposting.department,
                "experience": jobposting.experience,
                "required_skills": jobposting.required_skills,
                "prefered_skills": jobposting.prefered_skills,
                "employment_type": jobposting.employment_type,
                "openings": jobposting.openings,
                "recruiter_id": jobposting.recruiter_id
            })

        return successWithData(f"all jobs related to recruiter id {id}", result)
    except Exception as e:
        return fail(str(e)), 401


# get single for job seeker 
@jobposting_bp.route("/<int:id>", methods=['GET'])
def get_jobpostings(id):
    try:
        jobposting = db.session.get(JobPosting, id)
        result = {
                "id": jobposting.id,
                "job_title": jobposting.job_title,
                "description": jobposting.description,
                "salary": jobposting.salary,
                "graduation":jobposting.graduation,
                "postgraduation":jobposting.postgraduation,
                "location": jobposting.location,
                "role_category": jobposting.role_category,
                "department": jobposting.department,
                "experience": jobposting.experience,
                "required_skills": jobposting.required_skills,
                "prefered_skills": jobposting.prefered_skills,
                "employment_type": jobposting.employment_type,
                "openings": jobposting.openings,
                "recruiter_id": jobposting.recruiter_id
        }
        return successWithData(f"job posting id {id}", result)
    except Exception as e:
        return fail(str(e)), 401

# update
@jobposting_bp.route('/update/<int:id>', methods=['PATCH', 'PUT'])
def update_jobposting(id):
    try:
        jobposting = db.session.get(JobPosting, id)
        data = request.get_json()
        if jobposting:
            for key in data:
                setattr(jobposting, key, data[key])

            db.session.commit()
            result = {
                "id": jobposting.id,
                "job_title": jobposting.job_title,
                "description": jobposting.description,
                "salary": jobposting.salary,
                "graduation":jobposting.graduation,
                "postgraduation":jobposting.postgraduation,
                "location": jobposting.location,
                "role_category": jobposting.role_category,
                "department": jobposting.department,
                "experience": jobposting.experience,
                "required_skills": jobposting.required_skills,
                "prefered_skills": jobposting.prefered_skills,
                "employment_type": jobposting.employment_type,
                "openings": jobposting.openings,
                "recruiter_id": jobposting.recruiter_id
            }
            return successWithData('Job post updated successfully!', result)
        else:
            return fail("Job post not found!"), 404

    except Exception as e:
        return fail(str(e)), 401

#  delete 
@jobposting_bp.route('/delete/<int:id>', methods=['DELETE'])
def delete_jobposting(id):
    try:
        jobposting = db.session.get(JobPosting, id)
        if jobposting:
            db.session.delete(jobposting)
            db.session.commit()
            return success("Job Posting deleted successfully")
        else:
            return fail("Job post not found!"), 404

    except Exception as e:
        return fail(str(e)), 401

openai_api_key = os.getenv("API_KEY") 
client = OpenAI(api_key=openai_api_key)
# send recommmended job posts
def send_recommended_job_posts(skills):
    skills = skills.split(",")
    temp = []
    for each in skills:
        temp.append(each.strip())
    skills =  temp

    try:
       
        recommended_jobpostings = JobPosting.query.filter(or_(*[JobPosting.required_skills.like(f"%{skill}%") for skill in skills])).all()

        result = []
        
        if(recommended_jobpostings):
            for jobposting in recommended_jobpostings:
                    result.append({
                        "id": jobposting.id,
                        "job_title": jobposting.job_title,
                        "description": jobposting.description,
                        "salary": jobposting.salary,
                        "graduation":jobposting.graduation,
                        "postgraduation":jobposting.postgraduation,
                        "location": jobposting.location,
                        "role_category": jobposting.role_category,
                        "department": jobposting.department,
                        "experience": jobposting.experience,
                        "required_skills": jobposting.required_skills,
                        "prefered_skills": jobposting.prefered_skills,
                        "employment_type": jobposting.employment_type,
                        "openings": jobposting.openings,
                        "recruiter_id": jobposting.recruiter_id
                    })
            return result
            
        else:
            return "sorry, but there is no any jobs are available!"
    except Exception as e:
        return str(e)

def serialize_choice(choice):
    output = []
    for each in choice.message:
         output.append(serialize_chat_message(each))
   
    return output

def serialize_chat_message(chat_message):
    obj = {
        chat_message[0]: chat_message[1],
    }
    return obj

@jobposting_bp.route("/search/<string:title>", methods=['GET'])
def getJobByTitle(title):
    try:
        filtered_job_posts = JobPosting.query.filter(or_(JobPosting.job_title.ilike(f"%{title}%"))).all()
        result = []
        if(filtered_job_posts):
            for jobposting in filtered_job_posts:
                    result.append({
                        "id": jobposting.id,
                        "job_title": jobposting.job_title,
                        "description": jobposting.description,
                        "salary": jobposting.salary,
                        "graduation":jobposting.graduation,
                        "postgraduation":jobposting.postgraduation,
                        "location": jobposting.location,
                        "role_category": jobposting.role_category,
                        "department": jobposting.department,
                        "experience": jobposting.experience,
                        "required_skills": jobposting.required_skills,
                        "prefered_skills": jobposting.prefered_skills,
                        "employment_type": jobposting.employment_type,
                        "openings": jobposting.openings,
                        "recruiter_id": jobposting.recruiter_id
                    })
            return successWithData(msg=f"{title} jobs", data=result)
            
        else:
            return fail(msg="sorry, but there is no any jobs are available!")

    except Exception as e:
        return fail(msg=str(e))

@jobposting_bp.route("/recommend", methods=['POST'])
def recommend_job():
    try:
        messages = request.get_json()
        tools = [
         {
                "type": "function",
                "function": {
                    "name": "send_recommended_job_posts",
                    "description": "Recommend job posts using skills",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "skills": {
                                "type": "string",
                                "description": "multiple skills in string",
                            },
                        },
                        "required": ["skills"],
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
                "send_recommended_job_posts": send_recommended_job_posts,
            } 
            messages.append(response_message)  
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_functions[function_name]
                function_args = json.loads(tool_call.function.arguments)
                function_response = function_to_call(
                    skills=function_args.get("skills"),
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
           
            if(type(function_response) == list):
                return successWithData(msg="here is the some recommendation according to your skills:", data=modified_data)
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