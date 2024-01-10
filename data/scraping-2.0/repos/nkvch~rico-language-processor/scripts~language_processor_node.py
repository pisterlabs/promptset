#!/usr/bin/env python
# encoding: utf8

import rospy
import tiago_msgs.msg
import std_msgs.msg
import os
import actionlib
from std_msgs.msg import String, Bool
from conversation_msgs.srv import GetScenariosIntentsWithParams
from language_processor.srv import DetectIntentAndRetrieveParams, DetectIntentAndRetrieveParamsResponse
from language_processor.srv import InitiateConvBasedOnCtx, InitiateConvBasedOnCtxResponse
from language_processor.srv import GuessActor, GuessActorResponse
from language_processor.srv import DetectIntentDuringTask, DetectIntentDuringTaskResponse
from rico_context.srv import GetContext, GetCurrentScenarioId, IsInTask
from rico_context.msg import HistoryEvent
from task_database.srv import GetScenarioInputs, GetScenarioInputsResponse
from openai_interface import OpenAIInterface
from copy import deepcopy
import json

# convert unicode to utf-8 (\u0119 -> \xc4\x99)
def encode_dict(d):
    return {key.encode('utf-8'): encode_dict(value) if isinstance(value, dict) else value.encode('utf-8') if isinstance(value, unicode) else value for key, value in d.items()}


def main():
    rospy.init_node('language_processor', anonymous=True)
    get_scenarios_intents_with_params = rospy.ServiceProxy(
        'get_scenarios_intents_with_params', GetScenariosIntentsWithParams)
    get_latest_context = rospy.ServiceProxy('/context/get_after_last_scenario', GetContext)
    get_scenario_inputs = rospy.ServiceProxy(
        'get_scenario_inputs', GetScenarioInputs)
    get_current_scenario_id = rospy.ServiceProxy(
        '/context/scenario_id', GetCurrentScenarioId)
    rospy.wait_for_service('get_scenarios_intents_with_params')
    rospy.wait_for_service('/context/get')
    is_in_task = rospy.ServiceProxy('/context/is_in_task', IsInTask)
    pub_context = rospy.Publisher('/context/push', HistoryEvent, queue_size=10)
    pub_cmd = rospy.Publisher('rico_cmd', tiago_msgs.msg.Command, queue_size=10)
    pub_filtered_cmd = rospy.Publisher('/rico_filtered_cmd', tiago_msgs.msg.Command, queue_size=10)

    rico_says_client = actionlib.SimpleActionClient(
            'rico_says', tiago_msgs.msg.SaySentenceAction)

    # conversation_state = ConversationState()

    openai_interface = OpenAIInterface()

    def speakRequest(text):
        goal = tiago_msgs.msg.SaySentenceGoal()
        # goal.sentence = u'niekorzystne warunki pogodowe ' + text
        goal.sentence = u'' + text

        rico_says_client.send_goal(goal)
        rico_says_client.wait_for_result()

    def hear_callback(data):
        is_currently_in_task = is_in_task().result

        print 'hear_callback. is_currently_in_task: ' + str(is_currently_in_task) + ', data.data: ' + data.data

        if is_currently_in_task:
            process_intent_during_task(data.data)
        else:
            process_intent(data.data)


    def initiate_conv_based_on_ctx(req):
        rico_history_events = get_latest_context().events

        filtered_events = []
        for event in rico_history_events:
            if event.actor != 'system':
                filtered_events.append(event)

        response = openai_interface.initiate_conversation(filtered_events)

        return InitiateConvBasedOnCtxResponse(response)
    
    def guess_actor(text):
        rico_history_events = get_latest_context().events

        filtered_events = []
        for event in rico_history_events:
            if event.actor != 'system':
                filtered_events.append(event)

        actor = openai_interface.guess_actor(filtered_events, text)

        return actor

    def process_intent(text):
        last_message_actor = guess_actor(text)
        pub_context.publish(HistoryEvent(
            last_message_actor,
            'say',
            '"%s"' % text,
            ''
        ))

        scenarios_intents_with_params = get_scenarios_intents_with_params().scenarios_intents_with_params

        s_i_with_params_dict = dict(list(map(lambda siwp: (siwp.intent_name, {
            'scenario_id': siwp.scenario_id, 'name': siwp.intent_name, 'parameters': siwp.params}), scenarios_intents_with_params)))

        s_i_with_params_openai = deepcopy(s_i_with_params_dict.values())

        for s_i_with_params in s_i_with_params_openai:
            del s_i_with_params['scenario_id']

        rico_history_events = get_latest_context().events

        conversation_events = []

        for event in rico_history_events:
            if event.action == 'say':
                conversation_events.append(event)
                conversation_events.append(event)

        detect_intent_with_params(s_i_with_params_openai, s_i_with_params_dict, conversation_events)


    def process_last_intent_from_history():
        # same as process_intent, but uses last message from history instead of user input, and doesn't publish to /context/push, and don't use guess_actor
        scenarios_intents_with_params = get_scenarios_intents_with_params().scenarios_intents_with_params

        s_i_with_params_dict = dict(list(map(lambda siwp: (siwp.intent_name, {
            'scenario_id': siwp.scenario_id, 'name': siwp.intent_name, 'parameters': siwp.params}), scenarios_intents_with_params)))
        
        s_i_with_params_openai = deepcopy(s_i_with_params_dict.values())

        for s_i_with_params in s_i_with_params_openai:
            del s_i_with_params['scenario_id']

        rico_history_events = get_latest_context().events

        conversation_events = []

        for event in rico_history_events:
            if event.action == 'say':
                conversation_events.append(event)

        detect_intent_with_params(s_i_with_params_openai, s_i_with_params_dict, conversation_events)

        
    def detect_intent_with_params(s_i_with_params_openai, s_i_with_params_dict, conversation_events):
        conversation_history_string = ''

        for event in conversation_events:
            conversation_history_string += "%s: %s \n" % ('Rico' if event.actor == 'Rico' else 'User', event.complement)

        response = openai_interface.detect_intent_with_params(
            s_i_with_params_openai, conversation_history_string)

        print response

        matched = response is not None

        print 's_i_with_params_dict', s_i_with_params_dict

        if matched:
            try:
                response = encode_dict(response)
            except:
                matched = False

        if matched:
            detected_scenario_intent = s_i_with_params_dict[response['name']]

        scenario_id = detected_scenario_intent['scenario_id'] if matched else -1
        intent_name = detected_scenario_intent['name'] if matched else ''
        retrieved_param_names = []
        retrieved_param_values = []
        unretrieved_param_names = []

        if matched:
            for key, value in response['parameters'].items():
                if value is not None:
                    retrieved_param_names.append(key)
                    retrieved_param_values.append(value)
                else:
                    unretrieved_param_names.append(key)

        all_parameters_present = len(unretrieved_param_names) == 0 if matched else False

        if matched and not all_parameters_present and 'fulfilling_question' not in response:
            fulfilling_question = openai_interface.fallback_get_fulfilling_question(intent_name, unretrieved_param_names)
        else:
            fulfilling_question = response['fulfilling_question'] if matched and not all_parameters_present else ''

        if not matched:
            return speakRequest('I don\'t understand. Could you repeat?')

        if not all_parameters_present:
            return speakRequest(fulfilling_question)

        cmd = tiago_msgs.msg.Command()
        cmd.query_text = ''
        cmd.intent_name = unicode(intent_name, 'utf-8')

        for param_name, param in zip(retrieved_param_names, retrieved_param_values):
            cmd.param_names.append(unicode(param_name, 'utf-8'))
            cmd.param_values.append(unicode(param, 'utf-8'))

        cmd.confidence = 1.0
        cmd.response_text = u'Okej'
        pub_cmd.publish(cmd)


    def process_intent_during_task(text):
        curr_scenario_id = get_current_scenario_id().scenario_id
        scenario_inputs = get_scenario_inputs(curr_scenario_id).inputs
        scenarios_intents_with_params = get_scenarios_intents_with_params(
        ).scenarios_intents_with_params
        s_data_dict = dict(list(map(lambda siwp: (siwp.scenario_id, {
            'scenario_id': siwp.scenario_id, 'name': siwp.intent_name, 'parameters': siwp.params}), scenarios_intents_with_params)))
        
        curr_scenario_data = s_data_dict[curr_scenario_id]
        curr_task_params = curr_scenario_data['parameters']

        in_task_intents = []

        for input in scenario_inputs:
            in_task_intents.append({
                'name': input.intent,
                'description': input.description,
            })

        in_task_intents_dict = dict(list(map(lambda it: (it['name'], it), in_task_intents)))

        events = []

        rico_history_events = get_latest_context().events

        for event in rico_history_events:
            if event.actor != 'system':
                events.append(event)

        last_message_actor = guess_actor(text)

        pub_context.publish(HistoryEvent(
            last_message_actor,
            'say',
            '"%s"' % text,
            ''
        ))

        response = openai_interface.detect_intent_during_task(
            events, in_task_intents, text, curr_task_params, last_message_actor)

        print response

        matched = response is not None

        if not matched:
            detect_unexpected_question_resp = openai_interface.detect_unexpected_question(events, last_message_actor, text)

            if detect_unexpected_question_resp is None:
                return speakRequest('I don\'t understand. Could you repeat?')
            elif isinstance(detect_unexpected_question_resp, dict) and 'answer' in detect_unexpected_question_resp:
                return speakRequest(detect_unexpected_question_resp['answer'])
            elif isinstance(detect_unexpected_question_resp, dict) and 'new_parameter_name' in detect_unexpected_question_resp:
                print 'detect_unexpected_question_resp', detect_unexpected_question_resp
                cmd = tiago_msgs.msg.Command()
                cmd.query_text = ''
                cmd.intent_name = unicode('unexpected_question', 'utf-8')
                cmd.param_names.append(unicode('unexpected_question', 'utf-8'))
                cmd.param_values.append(detect_unexpected_question_resp['new_parameter_name'])
                cmd.confidence = 1.0
                cmd.response_text = u''
                pub_filtered_cmd.publish(cmd)
                return

        else:
            cmd = tiago_msgs.msg.Command()

            cmd.query_text = ''

            cmd.intent_name = response

            cmd.confidence = 1.0
            cmd.response_text = u''
            pub_filtered_cmd.publish(cmd)


    # rospy.Service('detect_intent_and_retrieve_params',
    #               DetectIntentAndRetrieveParams, detect_intent_and_retrieve_params)
    
    # rospy.Service('detect_intent_during_task',
    #               DetectIntentDuringTask, detect_intent_during_task)
    
    rospy.Service('initiate_conv_based_on_ctx',
                  InitiateConvBasedOnCtx, initiate_conv_based_on_ctx)
    
    # rospy.Service('guess_actor',
    #                 GuessActor, guess_actor)
    
    rospy.Subscriber(
        '/rico_hear', String, hear_callback, queue_size=1)
    
    rospy.Subscriber(
        '/rico_process_last_intent_from_history', std_msgs.msg.Empty, lambda _: process_last_intent_from_history(), queue_size=1)

    rospy.spin()


if __name__ == '__main__':
    main()