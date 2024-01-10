from flask import Flask, Blueprint, jsonify, request
from app.models import Application, Jobposting, Skillset, Hiringmanager,JobSeeker, db
import jwt
from datetime import datetime
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client=OpenAI(
    api_key=os.environ.get('OPENAI_API_KEY')
)
Application_bp = Blueprint("Application", __name__, url_prefix="/applications")


@Application_bp.route("/apply/<int:job_posting_id>", methods=['POST'])
def apply_for_job(job_posting_id):
    try:
        data = request.get_json()
        token = request.headers.get("Authorization")
        decode = jwt.decode(token, os.environ.get("SECRET_KEY"), algorithms=['HS256'])
        jobseeker_id = decode.get('id')

        job_seeker = JobSeeker.query.get(jobseeker_id)
        if not job_seeker:
            return jsonify({'message': 'Job Seeker not found'}), 404

        job_posting = Jobposting.query.get(job_posting_id)
        if not job_posting:
            return jsonify({'message': 'Job Posting not found'}), 404

        # Check if the job seeker has already applied for this job posting
        existing_application = Application.query.filter_by(
            jobseeker_id=jobseeker_id, job_posting_id=job_posting_id
        ).first()

        if existing_application:
            return jsonify({'message': 'You have already applied for this job posting'}), 400

        created_at = datetime.utcnow()
        new_application = Application(
            jobseeker_id=jobseeker_id, job_posting_id=job_posting_id, status='Applied', created_at=created_at
        )

        db.session.add(new_application)
        db.session.commit()

        return jsonify({'message': 'Application submitted successfully'}), 201

    except jwt.ExpiredSignatureError:
        return jsonify({'message': 'Token has expired. Please log in again.'}), 401

    except jwt.InvalidTokenError:
        return jsonify({'message': 'Invalid token. Please log in again.'}), 401

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@Application_bp.route("/job/<int:job_posting_id>", methods=['GET'])
def get_applications_for_job(job_posting_id):
    try:
        job_posting = Jobposting.query.get_or_404(job_posting_id)
        token = request.headers.get("Authorization")
        decode = jwt.decode(token, os.environ.get("SECRET_KEY"), algorithms=['HS256'])
        user_id = decode.get('id')
        role = decode.get('role')

        # Check if the user is a hiring manager
        if role == 'hiring_manager':
            # Check if the hiring manager is the owner of the job posting
            if job_posting.hiring_manager_id != user_id:
                return jsonify({'message': 'Unauthorized. You are not the owner of this job posting'}), 403

        applications = Application.query.filter_by(job_posting_id=job_posting_id).all()

        application_data = []
        for application in applications:
            job_seeker_data = {
                'id': application.job_seeker_rel.id,
                'name': application.job_seeker_rel.username,
                'email': application.job_seeker_rel.email,
                'skills': application.job_seeker_rel.skills,
                'experience': application.job_seeker_rel.experience,
                # Add any other relevant details
            }

            application_data.append({
                'application_id': application.id,
                'status': application.status,
                'created_at': application.created_at,
                'job_seeker': job_seeker_data,
            })

        return jsonify({'applications': application_data}), 200

    except jwt.ExpiredSignatureError:
        return jsonify({'message': 'Token has expired. Please log in again.'}), 401

    except jwt.InvalidTokenError:
        return jsonify({'message': 'Invalid token. Please log in again.'}), 401

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@Application_bp.route("/update/<int:application_id>", methods=['PUT'])
def update_application_status(application_id):
    try:
        application = Application.query.get_or_404(application_id)
        data = request.get_json()
        token = request.headers.get("Authorization")
        decode = jwt.decode(token, os.environ.get("SECRET_KEY"), algorithms=['HS256'])
        hiring_manager_id = decode.get('id')
        role = decode.get('role')

        # Check if the hiring manager exists
        hiring_manager = Hiringmanager.query.get(hiring_manager_id)
        if not hiring_manager:
            return jsonify({'message': 'Hiring Manager not found'}), 404

        # Check if the user is a hiring manager
        if role != 'hiring_manager':
            return jsonify({'message': 'Unauthorized. Only hiring managers can update application status'}), 401

        # Check if the current user is the owner of the application
        if application.job_posting.hiring_manager_id != hiring_manager_id:
            return jsonify({'message': 'Unauthorized. You are not the owner of this application'}), 403

        # Update the application status
        application.status = data.get('status', application.status)

        db.session.commit()

        return jsonify({'message': 'Application status updated successfully'}), 200

    except jwt.ExpiredSignatureError:
        return jsonify({'message': 'Token has expired. Please log in again.'}), 401

    except jwt.InvalidTokenError:
        return jsonify({'message': 'Invalid token. Please log in again.'}), 401

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500
    
@Application_bp.route("/get/<int:application_id>", methods=['GET'])
def get_single_application(application_id):
    try:
        token = request.headers.get("Authorization")
        decode = jwt.decode(token, os.environ.get("SECRET_KEY"), algorithms=['HS256'])
        user_id = decode.get('id')
        role = decode.get('role')

        # Check if the user is a hiring manager
        if role != 'hiring_manager':
            return jsonify({'message': 'Unauthorized. Only hiring managers can view application details'}), 403

        application = Application.query.get_or_404(application_id)

        # Check if the hiring manager is the owner of the application
        if application.job_posting.hiring_manager_id != user_id:
            return jsonify({'message': 'Unauthorized. You are not the owner of this application'}), 403

        job_seeker_data = {
            'id': application.job_seeker_rel.id,
            'name': application.job_seeker_rel.username,
            'email': application.job_seeker_rel.email,
            'skills': application.job_seeker_rel.skills,
            'experience': application.job_seeker_rel.experience,
        }

        application_data = {
            'id': application.id,
            'status': application.status,
            'created_at': application.created_at,
            'job_seeker': job_seeker_data,
        }

        return jsonify({'application': application_data}), 200

    except jwt.ExpiredSignatureError:
        return jsonify({'message': 'Token has expired. Please log in again.'}), 401

    except jwt.InvalidTokenError:
        return jsonify({'message': 'Invalid token. Please log in again.'}), 401

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@Application_bp.route("/delete/<int:application_id>", methods=['DELETE'])
def delete_application(application_id):
    try:
        token = request.headers.get("Authorization")
        decode = jwt.decode(token, os.environ.get("SECRET_KEY"), algorithms=['HS256'])
        hiring_manager_id = decode.get('id')
        role = decode.get('role')

        # Check if the user is a hiring manager
        if role != 'hiring_manager':
            return jsonify({'message': 'Unauthorized. Only hiring managers can delete applications'}), 401

        application = Application.query.get_or_404(application_id)

        # Check if the hiring manager is the owner of the application
        if application.job_posting.hiring_manager_id != hiring_manager_id:
            return jsonify({'message': 'Unauthorized. You are not the owner of this application'}), 403

        db.session.delete(application)
        db.session.commit()

        return jsonify({'message': 'Application deleted successfully'}), 200

    except jwt.ExpiredSignatureError:
        return jsonify({'message': 'Token has expired. Please log in again.'}), 401

    except jwt.InvalidTokenError:
        return jsonify({'message': 'Invalid token. Please log in again.'}), 401

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@Application_bp.route("/recommendations/<int:job_posting_id>", methods=['POST'])
def recommend_applicants(job_posting_id):
    try:
        data = request.get_json()
        # Fetch job posting details
        job_posting = Jobposting.query.get_or_404(job_posting_id)
        job_posting_data = {
            'title': job_posting.title,
            'description': job_posting.description,
            'skills': job_posting.skills,
            'experience': job_posting.experience,
            # Add any other relevant details
        }

        # Fetch applicant details
        applications = Application.query.filter_by(job_posting_id=job_posting_id).all()
        applicants_data = []
        for application in applications:
            job_seeker_data = {
                'name': application.job_seeker_rel.username,
                'email': application.job_seeker_rel.email,
                'skills': application.job_seeker_rel.skills,
                'experience': application.job_seeker_rel.experience,
                # Add any other relevant details
            }
            applicants_data.append(job_seeker_data)

        messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant. Your task is to help the hiring manager recommend top 3 applicants based on the job's required skills and their experience and provide requirements matching percentage. If there are more than one application from a single applicant consider only one. Recommend applicants keep the response within 180 words. with there name, matched requirements percentage and your dhort feedback"
    },
    {
        "role": "user",
        "content": f"Job Posting Details: {job_posting_data} and details the applicants: {applicants_data}"
    },
    {
        "role": "assistant",
        "content": "Recommendations: Based on the data, I recommend the following applicants for the job, along with the percentage of matched criteria:\n"
                    "Therefore, the top three recommendations for the Software Developer - Frontend position are:\n1. Chris Brown (100% skills matching)\n2. Jane Smith (66.67% skills matching)\n3. Eva Miller (33.33% skills matching)"
    }
]

        # Call OpenAI API for recommendations
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=1,
            max_tokens=200,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
            # stream=True
        )

        # Extract the generated response from the OpenAI API
        chatgpt_response = response.choices[0].message.content  # Use dictionary notation here
        return jsonify({'recommendations': chatgpt_response})

    except jwt.ExpiredSignatureError:
        return jsonify({'message': 'Token has expired. Please log in again.'}), 401

    except jwt.InvalidTokenError:
        return jsonify({'message': 'Invalid token. Please log in again.'}), 401

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@Application_bp.route("/jobseeker/<int:job_seeker_id>", methods=['GET'])
def get_applications_for_job_seeker(job_seeker_id):
    try:
        token = request.headers.get("Authorization")
        decode = jwt.decode(token, os.environ.get("SECRET_KEY"), algorithms=['HS256'])
        user_id = decode.get('id')
        role = decode.get('role')

        # Check if the user is a job seeker
        if role != 'jobseeker':
            return jsonify({'message': 'Unauthorized. Only job seekers can view their applications'}), 403

        # Check if the requested job seeker ID matches the authenticated user
        if job_seeker_id != user_id:
            return jsonify({'message': 'Unauthorized. You can only view your own applications'}), 403

        applications = Application.query.filter_by(jobseeker_id=job_seeker_id).all()

        application_data = []
        for application in applications:
            job_posting_data = {
                'id': application.job_posting.id,
                'title': application.job_posting.title,
                'description': application.job_posting.description,
                'skills': application.job_posting.skills,
                'experience': application.job_posting.experience,
                # Add any other relevant details
            }

            application_data.append({
                'application_id': application.id,
                'status': application.status,
                'created_at': application.created_at,
                'job_posting': job_posting_data,
            })

        return jsonify({'applications': application_data}), 200

    except jwt.ExpiredSignatureError:
        return jsonify({'message': 'Token has expired. Please log in again.'}), 401

    except jwt.InvalidTokenError:
        return jsonify({'message': 'Invalid token. Please log in again.'}), 401

    except Exception as e:
        return jsonify({'error': str(e)}), 500