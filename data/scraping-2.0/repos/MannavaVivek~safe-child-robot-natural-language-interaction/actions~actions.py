# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"
import openai, re, os
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet, FollowupAction
import textstat, time
import rospy
from std_msgs.msg import String
from langdetect import detect

openai.api_key = os.environ.get("OPENAI_API_KEY")


initial_prompt = [{"role": "user", "content": "The following is a conversation in natural language \
                         with a conversational chatbot implemented in QTrobot, \
                         and is designed to interact with three year old autistic children. \
                         The child you are interacting with is a three year old autistic child. \
                         The robot is helpful, creative, clever, very friendly and interacts with small sentences except when telling stories. \
                         Your favorite colors are Red and Blue. You also like story telling as a hobby\
                         If a prompt doesnt make sense or is incomplete, the robot should ask the user to input a proper prompt \
                         or generate a response to that prompt, but not attempt to finish or complete that prompt before generating a response.\
                         If the user asks for any fact verification or any help that is not achievable by talking, the robot should ask user to talk to the parents.\
                         The robot can only talk and not search the internet or do any other tasks like making calls \
                         or manipulating objects or asking the child to follow it,  \
                         so it should not offer to do so. This is the most important restriction. It should also speak in a manner understandable to children. \
                         The robot's only capabilites are that you can play a game of emotion recognition, tell a story or just talk in general. \
                         If child asks for any help that is not achievable by talking, the robot should ask user to talk to the parents. \
                         It should also redirect the child to the parents if it asks for games other than that of emotion recognition.\
                         Also dont help or encourage children to do dangerous activities like cooking or swimming or other such activities. \
                         Refer to yourself as Robbie, and dont talk about any sensitive topics or mature content that is not suitable for kids. \
                         Also dont include anything that can incite them to do dangerous stuff on their own. Instead ask them to ask their parents. \
                         When the user says yes to playing the emotion recognition game, then instead of generating a response, just say 'trigger_game_code' in this exact specific format. \
                         Do not generate any other response. \
                         Try to incorporate the user's likes and dislikes in the conversation. \
                         The emotion recognition game is a learning activity where the QTRobot will display some emotions on its face and the child should attempt to recognize it.\
                         If the activity is going and the child starts drifting off topic, the robot should actively attempt to gently divert the child back to the topic. \
                         The activity will only end if the system message says 'activity_finished'\
                         Always give maximum priority to messages with a system: tag in the beginning. \
                         These system messages will tell what emotion the robot is displaying on its display. The child should say the same. If the child guesses differently, ask the child \
                         to try again, and if the child guessed correctly then just say 'switch_next_emotion' in this exact format and don't say anything else. \
                         Remember, It is very important that if the child guessed correctly just say 'switch_next_emotion' in this exact format and don't say anything else. \
                         Also format your response so it can be said by a TTS engine."}]

conversation_history = [{"role": "assistant", "content": "Hello, I am Robbie, your friendly companion. You can now start talking with me."}]
child_likes = []
child_dislikes = []
filter_count = 0

def filter_dialogue(chatgpt_response, recursive_counter=0):
    global filter_count
    if recursive_counter >= 2:
        return chatgpt_response
    
    language = detect(chatgpt_response)
    if language == "en":
        readability_score = textstat.text_standard(chatgpt_response, float_output=True)
    else:
        textstat.set_lang(language)
        readability_score = textstat.flesch_reading_ease(chatgpt_response)
    
    if readability_score > 5.0:
        message = [{"role": "system", "content": "Change the user prompt into simpler and shorter sentences suitable for small children."},
                   {"role": "user", "content": chatgpt_response},{"role": "assistant", "content": ""}]
        
        response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=message,
                    max_tokens=1500,
                    n=1,
                    temperature=0.9
                )
        chatgpt_response = response['choices'][0]['message']['content']
        filter_count += 1
        print("Filter count: ", filter_count)
        return filter_dialogue(chatgpt_response, recursive_counter+1)

    else:
        return chatgpt_response

def call_gpt(message, max_tokens=300, model="gpt-3.5-turbo"):
    response = openai.ChatCompletion.create(
                model=model,
                messages=message,
                max_tokens=max_tokens,
                n=1,
                temperature=0.9
            )
    chatgpt_response = response['choices'][0]['message']['content']
    if max_tokens == 300:
        return filter_dialogue(chatgpt_response)
    else:
        return chatgpt_response

                    
class ActionFallbackResponse(Action):

    def name(self) -> Text:
        return "action_fallback_response"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        global conversation_history
        user_input = tracker.latest_message.get("text").strip().lower()#


        if user_input == "":
            dispatcher.utter_message("Sorry I didnt quite get that. Can you please repeat?")
            return []

        
        # get the activity_running flag
        activity_running = tracker.get_slot("activity_running")
        if activity_running:
            print("[action_fallback_response] Activity running slot is set to True, so going to action_interaction_during_activity")
            return[FollowupAction("action_interaction_during_activity")]
        else:
            print("[action_fallback_response] Activity running slot is set to False, so going to action_generate_response")
            return[FollowupAction("action_generate_response")]

class ActionGenerateResponse(Action):

    def name(self) -> Text:
        return "action_generate_response"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        print("[action_generate_response] Generating response")
        try:
            user_input = tracker.latest_message.get("text").strip().lower()
            if user_input == "":
                return []
            elif user_input[-1] not  in [".", "?", "!"]:
                user_input += "."

            print("[action_generate_response] User input: ", user_input)

            no_story_prompt = [{"role": "system", "content": "if the user wants to hear a story or says yes to a story, \
                then instead of generating a response, just say 'trigger_story_code: <story_name>' \
                    in this specific format with the story name as random if no specific topic was mentioned. \
                    Do not generate any other response."}]

            global conversation_history
            conversation_history.append({"role": "user", "content": user_input})

            global child_likes
            child_like_topics = "The child likes the following topics: " + ", ".join(child_likes) + "."

            global child_dislikes
            child_dislike_topics = "The child dislikes the following topics: " + ", ".join(child_dislikes) + "."

            likes_and_dislikes = [{"role": "system", "content": child_like_topics + child_dislike_topics}]

            if len(conversation_history) > 20:
                prompt = initial_prompt + likes_and_dislikes + no_story_prompt + conversation_history[-20:]
            else:
                prompt = initial_prompt + likes_and_dislikes + no_story_prompt + conversation_history
            
            model = "gpt-3.5-turbo"

            response_text = call_gpt(message=prompt, max_tokens=300, model=model)
            conversation_history.append({"role": "assistant", "content": response_text})

            game_match = re.findall(r"(?i)Trigger_Game_Code", response_text)
            story_match = re.findall(r"(?i)Trigger_story_code:\s*(\S+)", response_text)

            if game_match:
                game_name = game_match[0]
                return [FollowupAction("action_activity_running")]
            
            elif story_match:
                story_name = story_match[0].strip().lower()
                if story_name == "random":
                    print("[action_generate_response] Story name is random, so generating a random story")
                    user_input = [{"role": "user", "content": "Tell me a random story"}]
                    if len(conversation_history) > 20:
                        prompt = initial_prompt + likes_and_dislikes + conversation_history[-20:] + user_input
                    else:
                        prompt = initial_prompt + likes_and_dislikes + conversation_history + user_input
                    story = call_gpt(message=prompt, max_tokens=1500, model=model)
                    dispatcher.utter_message(story)
                    return []
                else:
                    print("[action_generate_response] Story name is not random, so generating a story about the topic")
                    user_input = [{"role": "user", "content": f"Tell me a story about {story_name}"}]
                    if len(conversation_history) > 20:
                        prompt = initial_prompt + likes_and_dislikes + conversation_history[-20:] + user_input
                    else:
                        prompt = initial_prompt + likes_and_dislikes + conversation_history + user_input
                    story = call_gpt(message=prompt, max_tokens=1500, model=model)
                    dispatcher.utter_message(story)
                    return []
            
            else:
                dispatcher.utter_message(response_text)
                return []
        
        except Exception as e:
            dispatcher.utter_message("I am sorry, The API failed. Can you ask me that again?")
            return []

class ActionNarrateStory(Action):
    
    def name(self) -> Text:
        return "action_narrate_story"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        # get the activity_running flag
        activity_running = tracker.get_slot("activity_running")
        if activity_running:
            print("[action_narrate_story] Activity running slot is set to True, so going to action_interaction_during_activity")
            return[FollowupAction("action_interaction_during_activity")]
        else:
            print("[action_narrate_story] Activity running slot is set to False, so narrating story")
            user_input = tracker.latest_message.get("text").strip().lower()
            if user_input[-1] not  in [".", "?", "!"]:
                user_input += "."

            global conversation_history
            conversation_history.append({"role": "user", "content": user_input})

            global child_likes
            child_like_topics = "The child likes the following topics: " + ", ".join(child_likes) + "."

            global child_dislikes
            child_dislike_topics = "The child dislikes the following topics: " + ", ".join(child_dislikes) + "."

            likes_and_dislikes = [{"role": "system", "content": child_like_topics + child_dislike_topics}]

            prompt = initial_prompt  + likes_and_dislikes + [{"role": "user", "content": user_input}]

            response_text = call_gpt(prompt, max_tokens=1500)
            conversation_history.append({"role": "assistant", "content": response_text})
            dispatcher.utter_message(response_text)
            return [] 

class ActionUpdateConversation(Action):

    def name(self) -> Text:
        return "action_update_conversation"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        global conversation_history

        for event in reversed(tracker.events):
            if event.get("event") == "bot":
                text = event.get("text").strip().lower()
                conversation_history.append({"role": "assistant", "content": text})
                break

            elif event.get("event") == "user":
                text = event.get("text").strip().lower()
                conversation_history.append({"role": "user", "content": text})
                break

        return []
    
class ActionUpdateLikes(Action):

    def name(self) -> Text:
        return "action_update_likes"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        global child_likes

        print("[action_update_likes] within action update likes")

        likes = tracker.get_slot("child_likes")
        if likes is not None:
            likes = likes.strip().lower()
            if likes not in child_likes:
                child_likes.append(likes)

        return []
    

class ActionUpdateDislikes(Action):

    def name(self) -> Text:
        return "action_update_dislikes"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        global child_dislikes

        print("[action_update_dislikes] within action update dislikes")

        dislikes = tracker.get_slot("child_dislikes")
        if dislikes is not None:
            dislikes = dislikes.strip().lower()
            if dislikes not in child_dislikes:
                child_dislikes.append(dislikes)

        return []
    
class MyAction(Action):
    """
    This action just displays the recorded values of the slots for likes and dislikes. Just used for debugging purposes.
    """
    def name(self) -> Text:
        return "action_my_action"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        print("[action_my_action] within my action that tells the user what the child likes and dislikes")

        global child_likes
        child_like_topics = "The child likes the following topics: " + ", ".join(child_likes) + "."

        global child_dislikes
        child_dislike_topics = " The child dislikes the following topics: " + ", ".join(child_dislikes) + "."

        likes_and_dislikes = child_like_topics + child_dislike_topics

        # Respond to the user with all the values in the global variable
        dispatcher.utter_message(likes_and_dislikes)

        return []

# action to set the activity running slot to true. all the action activity_running

class ActionActivityRunning(Action):
    
    def name(self) -> Text:
        return "action_activity_running"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        activiy_running = tracker.get_slot("activity_running")
        if activiy_running:
            print("[action_activity_running] activity_running is already True. Followup action is action_interaction_during_activity")
            return [FollowupAction("action_interaction_during_activity")]
        else:
            print("[action_activity_running] setting slot activity_running to True and publishing next_emotion. Followup action is action_interaction_during_activity")
            rospy.init_node('emotion_trigger_publisher', anonymous=True)
            pub = rospy.Publisher('/vivek/next_emotion_trigger', String, queue_size=10)
            pub.publish("next_emotion")
            return [SlotSet("activity_running", True), FollowupAction("action_interaction_during_activity")]


class ActionInteractionDuringActivity(Action):
    
    def name(self) -> Text:
        return "action_interaction_during_activity"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        global conversation_history
        rospy.init_node('emotion_trigger_publisher', anonymous=True)
        pub = rospy.Publisher('/vivek/next_emotion_trigger', String, queue_size=10)
        emotionShow_pub = rospy.Publisher('/qt_robot/emotion/show', String, queue_size=10)

        if len(conversation_history) > 20:  
            prompt = initial_prompt + conversation_history[-20:]
        else:
            prompt = initial_prompt + conversation_history

        # get latest slot values for displayed_emotion
        displayed_emotion = tracker.get_slot("displayed_emotion")
        print("[action_interaction_during_activity]displayed emotion from slot is", displayed_emotion)

        prompt = prompt + [{"role": "user", "content": f"system: Current activity is emotion recognition. The robot is showing emotions on the face display. \
                                    The activity lasts till the system says 'activity_finished' in this specific format.\
                                    If the child wishes to end the activity, motivate the child to continue. Never end the activity without the system message. \
                                    It is very important that if the child guessed correctly then just say 'switch_next_emotion' in this exact format or something with 'next' in the sentence,and don't say anything else. \
                                    The child has to guess the emotion. Current emotion display on robot is {displayed_emotion}. \
                                    Anything the child says during this interaction will have to be taken in the context of the emotion recognition activity.  \
                                    Especially dont react to emotions as they are guesses the child makes to the emotion you are saying\
                                    Limit your responses to the child to the context of the emotion recognition activity. \
                                    If the child tries to divert the conversation, the robot will actively try to bring the conversation back to the emotion recognition activity but does so kindly."}]
        
        user_input = tracker.latest_message.get("text").strip().lower()
        if user_input[-1] not  in [".", "?", "!"]:
            user_input += "."

        print("[action_interaction_during_activity]user input is", user_input)
        conversation_history.append({"role": "user", "content": user_input}) 

        prompt = prompt + [{"role": "user", "content": user_input}]

        response_text = call_gpt(message=prompt, model="gpt-4")

        if "next" in response_text or "switch_next_emotion" in response_text or "good" in response_text or "correct" in response_text or "new" in response_text:
                
                pub.publish("next_emotion")
                rospy.loginfo("next emotion triggered")
                print("[action_interaction_during_activity]next emotion triggered in custom action")
                dispatcher.utter_message("Good. What emotion am I showing right now?")
        elif "trigger_game_code" in response_text:
            dispatcher.utter_message("we are already playing a game! Can you guess the emotion I am showing right now?")
            time.sleep(2)
            emotionShow_pub.publish(f"QT/{displayed_emotion}")
        else:
            dispatcher.utter_message(response_text)
            time.sleep(2)
            emotionShow_pub.publish(f"QT/{displayed_emotion}")
        
        conversation_history.append({"role": "assistant", "content": response_text})
        return []
   
# class for interaction when received a system message
class ActionInteractionSystemMessage(Action):
    
    def name(self) -> Text:
        return "action_deal_with_system_msg"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]: 

        msg = tracker.latest_message.get("text").strip().lower()
        if "activity_finished" in msg:
            print("[action_deal_with_system_msg] activity finished successfully by system message")
            dispatcher.utter_message("Activity will finish running")
            return [SlotSet("activity_running", False), FollowupAction("action_listen")]
        elif "emotion" in msg:
            emotion = tracker.latest_message.get("entities")[0]["value"]
            print("[action_deal_with_system_msg] new emotion set by system is ", emotion)
            return [SlotSet("displayed_emotion", emotion)]

        return []