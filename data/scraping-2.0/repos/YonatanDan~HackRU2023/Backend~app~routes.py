from typing import List
from flask import Blueprint, jsonify, request
from concurrent.futures import ThreadPoolExecutor
import openai
from flask_cors import CORS
import logging

from .prompts import system_prompt, audiences_prompt, platforms_prompt, generate_post_prompt
from .mocks import get_random_forecast_data
from .models import Audience, Campaign
from .config import ProductionConfig


nicer_api = Blueprint('nicer_api', __name__)
CORS(nicer_api)

executor = ThreadPoolExecutor(20)
results = {}
MODEL = 'gpt-3.5-turbo'


def _target_audience_prompt(campaign: Campaign, api_key: str):
    openai.api_key = api_key

    logging.critical(f'Making the call to get campaign insights for campaign {campaign.name}')

    try:
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=[{'role': 'system', 'content': system_prompt},
                      {'role': 'user', 'content': audiences_prompt.format(desc=campaign.description)}
            ]
        )
    except Exception as e:
        logging.critical(f'Got exception from OpenAI for campaign {campaign.name}')
        logging.critical(e)
        return

    logging.critical(f'Got response from OpenAI for campaign {campaign.name}')

    data = response.choices[0].message.content.split('\n\n')[1:]
    audiences = []
    try:
        for audience in data[:5]:
            audience_name = audience[3:].split(' - ')[0]
            audience_percentage = float(audience.split(' - ')[1].split('%')[0])
            audience_info = audience.split(' - ')[1].split('%')[1][2:]
            logging.critical(f'{audience_name} - {audience_percentage}')
            audiences.append(Audience(audience_name, audience_percentage, audience_info))
    except Exception as e:
        logging.critical(f'Got exception in data processing campaign {campaign.name}')
        logging.critical(e)
        raise

    return audiences


def _platforms_prompt(campaign: Campaign, audiences: List[Audience], api_key: str):
    openai.api_key = api_key

    logging.critical(f'Making the call to get platforms for campaign {campaign.name}')

    try:
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=[{'role': 'system', 'content': system_prompt},
                    {'role': 'user',
                    'content': platforms_prompt.format(
                            desc=campaign.description,
                            audiences=', '.join([audience.name for audience in audiences]))}
            ]
        )
    except Exception as e:
        logging.critical(f'Got exception from OpenAI for campaign {campaign.name}')
        logging.critical(e)
        return

    data = response.choices[0].message.content.split('\n\n')[1].split('\n')
    for i, platform in enumerate(data):
        logging.critical(f'{audiences[i].name} - {platform}')
        audiences[i].platform = platform.split(': ')[1]


def _generate_post(campaign: Campaign, target_index: int, api_key: str):
    openai.api_key = api_key

    logging.critical(f'Making the call to get platforms for campaign {campaign.name}')

    try:
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=[{'role': 'system', 'content': system_prompt},
                    {'role': 'user',
                    'content': generate_post_prompt.format(
                            platform=campaign.audiences[target_index].platform,
                            audience=campaign.audiences[target_index].name,
                            topic=campaign.description)}
            ]
        )
    except Exception as e:
        logging.critical(f'Got exception from OpenAI for campaign {campaign.name}')
        logging.critical(e)
        return

    return response.choices[0].message.content


def openai_get_campaign_insights(campaign : Campaign, api_key: str):
    audiences = _target_audience_prompt(campaign, api_key)
    _platforms_prompt(campaign, audiences, api_key)

    logging.critical('Got everything from OpenAI')

    return audiences


def _generate_result(campaign: Campaign, api_key: str):
    logging.critical(f'Generating result for campaign {campaign.name}')

    audiences = openai_get_campaign_insights(campaign, api_key)
    campaign.audiences = audiences
    results[campaign.uuid] = campaign


def _campaign_create_params_validation(data):
    if not all(field in data for field in ['name', 'start_date', 'end_date',
                                           'history', 'previous_insights', 'skills', 'description']):
        return jsonify({'message': 'Invalid request parameters!'}), 400

    if not data['name']:
        return jsonify({'message': 'Campaign name is required!'}), 400

    if not data['start_date']:
        return jsonify({'message': 'Campaign start date is required!'}), 400

    if not data['end_date']:
        return jsonify({'message': 'Campaign end date is required!'}), 400

    if not data['skills']:
        return jsonify({'message': 'At least one skill is required!'}), 400

    return None


@nicer_api.route('/api/v1/campaign/create', methods=['POST'])
def generate_campaign():
    api_key = ProductionConfig.OPENAI_API_KEY
    data = request.get_json()

    validation_result = _campaign_create_params_validation(data)
    if validation_result:
        return validation_result

    campaign = Campaign(
        name=data['name'],
        start_date=data['start_date'],
        end_date=data['end_date'],
        should_consider_history=data['history'],
        previous_insights=data['previous_insights'],
        skills=data['skills'],
        description=data['description']
    )

    executor.submit(_generate_result, campaign, api_key)

    return jsonify({
        'uuid': campaign.uuid,
        'message': 'Campaign Generated Started',
        'status': 'success'
    })


@nicer_api.route('/api/v1/campaign/get/<campaign_id>', methods=['GET'])
def get_campaign(campaign_id):
    if campaign_id not in results:
        return jsonify({'message': 'Campaign not found!'}), 404

    result = {
        'name': results[campaign_id].name,
        'start_date': results[campaign_id].start_date,
        'end_date': results[campaign_id].end_date,
        'audiences': [audience.name for audience in results[campaign_id].audiences],
    }
    return jsonify(result)


@nicer_api.route('/api/v1/skills/get', methods=['GET'])
def get_skills():
    return jsonify({
        'skills': [
            'Payment Processing',
            'Refunds and Adjustments',
            'Double Booking or Overcharging',
            'Loyalty Program Points or Rewards',
            'Splitting Bills or Group Payments',
            'Check-In and Check-Out',
            'Room Assignments and Upgrades',
            'Reservation Modifications and Cancellations',
            'Lost or Forgotten Items',
            'Room Service and Dining Requests',
            'Hebrew',
            'Arabic',
            'Russian',
            'French',
            'Natural Disasters',
            'Fire Alarms',
            'Medical Emergencies',
            'wi-fi problems',
            'electronic key card',
            'in-room safe'
        ]
    })


@nicer_api.route('/api/v1/campaign/get_pop/<campaign_id>/<index>', methods=['GET'])
def get_pop(campaign_id=None, index='0'):
    if campaign_id not in results:
        return jsonify({'message': 'Campaign not found!'}), 404

    campaign = results[campaign_id]

    if not (index.isdigit() and int(index) >= 0 and int(index) < len(campaign.audiences)):
        return jsonify({'message': 'Invalid index!'}), 400

    result = results[campaign_id]
    audience: Audience = result.audiences[int(index)]

    post = _generate_post(campaign, int(index), ProductionConfig.OPENAI_API_KEY)

    return jsonify({
        'name': audience.name,
        'percentage': audience.percentage,
        'platform': audience.platform,
        'info': audience.info,
        'post': post,
    })