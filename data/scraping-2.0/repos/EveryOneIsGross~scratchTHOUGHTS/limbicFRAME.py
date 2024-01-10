'''
### Cognitive Science Viewpoint:

1. **Emotion Recognition and Processing**:
   
   Cognitive science has long been interested in how humans recognize and process emotions. Emotions play a crucial role in human decision-making, social interactions, and general well-being. The `EmotionRecognition` class in the code mirrors this human capability, recognizing emotions from textual inputs using linguistic cues. This is analogous to how humans might infer someone's emotional state from their speech or writing.
   
2. **Emotional Memory**:
   
   Just as humans have episodic memories associated with emotions (e.g., remembering how you felt on your graduation day), the `EmotionalMemory` class serves as a repository to store and retrieve emotion-related data. This allows the system to "recall" past emotional data, possibly influencing future interactions, much like how our past emotional experiences shape our future reactions and decisions.

3. **Decision Making**:

   Cognitive science posits that our emotions greatly influence our decision-making. The `DecisionMaking` class simulates this by making decisions based on the emotional context of the input. It takes into account the sentiment and emotional state to produce a contextually appropriate response.

4. **Mood Modulation**:

   Our mood, a longer-lasting cousin of emotions, affects our perceptions, interactions, and decisions. The `MoodModulation` class keeps a mood score, adjusting it based on recognized emotions. This is reminiscent of how a series of happy events might uplift our mood for an extended period.

5. **Social Interaction Simulation**:

   Emotions play a foundational role in human social interactions. The `SocialInteraction` class simulates human-like social interactions based on the provided input, aiming to reproduce the kind of responses a human might give, colored by their emotional state.

6. **Interface Node (Pre & Post Processing)**:

   This is analogous to the human cognitive process of perception (input) and action (output). Before we process information (input), we filter and preprocess it based on various factors, and when we act (output), it's often fine-tuned by our current emotional and cognitive state. The `InterfaceNode` class does something similar for the system, refining inputs and outputs.

7. **Action Suggestion**:

   Emotions often drive us to take certain actions. Feeling sad might drive someone to listen to music or call a friend. The `ActionSuggestion` class is a nod to this phenomenon, suggesting actions based on the identified emotional state.

8. **Emotion Vector Representation**:

   In cognitive neuroscience, there's an interest in representing mental states (like emotions) in some sort of quantifiable manner, often in the form of neural activation patterns. Here, emotions are represented as vectors, quantifying them in a form that the system can process and store.

9. **Topic and Sentiment Extraction**:

   These are linguistic tools to extract meaningful information from text, mirroring how humans can understand the topic of a conversation and the sentiment behind it, both crucial for emotional processing.

10. **User Feedback and Adaptation**:

   Just as humans learn from feedback, adapting and refining their behaviors, the system collects feedback through the `UserFeedback` class and adjusts its future actions accordingly. This is reminiscent of reinforcement learning in cognitive science, where behaviors are strengthened or weakened based on received rewards or penalties.

In essence, this system is an attempt to computationally model and simulate some aspects of human emotional processing and decision-making. By integrating various cognitive and emotional elements, it seeks to generate human-like responses to given inputs.
'''

# WORK IN PROGRESS, THIS IS A DRAFT THAT ISN'T YET WORKING RIGHT, TBH I WAS V HIGH. LOVE MY EMO VECTORS IDEA THO EVEN IF NOT IMPLEMENTED RIGHT 🤪

import pickle
import os
from typing import Dict, Any, Tuple
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.sentiment import SentimentIntensityAnalyzer
import openai
from gpt4all import GPT4All, Embed4All
import numpy as np
import string
import re

openai.api_base = "http://localhost:4892/v1"
openai.api_key = "null"

# Define constants
model = "mistral trismegistus"
OPENAI_ENGINE = "model"

# Ensure necessary nltk resources are downloaded
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Set OpenAI API key
openai.api_key = "YOUR_OPENAI_API_KEY"

# Set threshold value for emotion vector magnitude
THRESHOLD_VALUE = 0.5
MAX_TOKENS=100

def embed(input_data: str) -> Any:
    """Function to generate embeddings for given text."""
    embedder = Embed4All()
    return embedder.embed(input_data)


def sanitize_filename(filename):
    # Remove invalid characters
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    sanitized = ''.join(c for c in filename if c in valid_chars)
    # Truncate long filenames
    return sanitized[:50]


def get_temperature_and_chunk_size(sentiment: str, sentiment_score: float) -> Tuple[float, int]:
    """Determine temperature and chunk size based on sentiment."""
    if sentiment == "positive":
        temperature = min(0.7 + abs(sentiment_score)/2, 0.9)  # range: [0.7, 0.9]
        chunk_size = 150  # smaller chunk for positive sentiment
    elif sentiment == "negative":
        temperature = max(0.5 - abs(sentiment_score)/2, 0.3)  # range: [0.3, 0.5]
        chunk_size = 100  # even smaller chunk for negative sentiment
    else:  # neutral
        temperature = 0.7
        chunk_size = 200  # larger chunk for neutral sentiment
    return temperature, chunk_size

class EmotionRecognition:
    def recognize(self, input_data: str) -> str:
        sentiment, sentiment_score = SentimentExtraction().extract_sentiment(input_data)
        temperature, _ = get_temperature_and_chunk_size(sentiment, sentiment_score)
        response = openai.Completion.create(
            model=model, 
            prompt=f"Identify the emotion from the following text: {input_data}",
            max_tokens=MAX_TOKENS,
            temperature=temperature
        )
        
        return response.choices[0].text.strip()



class EmotionalMemory:
    def store(self, emotional_state, data):
        # Create a valid filename by removing non-alphanumeric characters and limiting the length
        valid_filename = re.sub(r'[^a-zA-Z0-9]', '', emotional_state)[:15] + '.pickle'
        
        with open(valid_filename, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    def retrieve(self, emotion: str) -> Any:
        """Retrieve stored data associated with an emotion from a pickle file."""
        if os.path.exists(f'{emotion}.pickle'):
            with open(f'{emotion}.pickle', 'rb') as handle:
                return pickle.load(handle)

class DecisionMaking:
    def decide(self, input_data: str) -> str:
        sentiment, sentiment_score = SentimentExtraction().extract_sentiment(input_data)
        temperature, chunk_size = get_temperature_and_chunk_size(sentiment, sentiment_score)
        
        # If input_data is too long, chunk it based on determined chunk_size
        chunked_input = input_data[:chunk_size]
        prompt = f"Given the context: '{chunked_input}', what would be the best course of action?"
        
        response = openai.Completion.create(
            model=model, 
            prompt=prompt,
            max_tokens=MAX_TOKENS,
            temperature=temperature
        )
        return response.choices[0].text.strip()

class MotivationReward:
    def evaluate(self, action: str, user_feedback: int) -> float:
        """Evaluate the success of actions based on user feedback."""
        return user_feedback  # Assuming feedback is a score between -1 (bad) to 1 (good)


class MoodModulation:
    mood_score = 0  # Starting at a neutral mood

    def adjust_mood(self, emotion: str, intensity: str) -> None:
        """Adjust the system's mood based on detected emotion and its intensity."""
        if emotion == "happy" and intensity == "strong":
            self.mood_score += 1
        elif emotion == "sad" and intensity == "strong":
            self.mood_score -= 1
        # Additional mood adjustments for other emotions can be added


class SocialInteraction:
    def process(self, input_data: str) -> str:
        """Generate human-like social interaction based on the input data using GPT-3."""
        prompt = f"Generate a human-like social interaction based on: '{input_data}'"
        response = openai.Completion.create(model=model, prompt=prompt, max_tokens=MAX_TOKENS)
        return response.choices[0].text.strip()

class InterfaceNode:
    def preprocess_input(self, input_data: str) -> str:
        """Clean and normalize user input."""
        #input_data = InterfaceNode().preprocess_input(input_data)

        return input_data.strip()

    def postprocess_output(self, output_data: str) -> str:
        """Format and potentially add politeness or emojis based on mood."""
        if MoodModulation.mood_score > 0:
            return output_data + " 😊"
        elif MoodModulation.mood_score < 0:
            return output_data + " 😔"
        return output_data
    
    def generate_response(self, input_data: str) -> str:
        """Generate a response based on mood and user input using GPT-3."""
        prompt = f"Given a mood score of {MoodModulation.mood_score}, how should I respond to: '{input_data}'?"
        response = openai.Completion.create(model=model, prompt=prompt, max_tokens=MAX_TOKENS)
        return response.choices[0].text.strip()


class ActionSuggestion:
    def suggest(self, emotional_state: str) -> str:
        """Suggest actions based on emotional state using GPT-3."""
        prompt = f"Given that someone is feeling {emotional_state}, what would be a good course of action or suggestion for them?"
        response = openai.Completion.create(model=model, prompt=prompt)
        suggested_action = response.choices[0].text.strip()
        return suggested_action

class VectorStorage:
    def store_vector(self, emotion: str, vector: Any):
        """Store emotion vector to a pickle file."""
        sanitized_emotion = sanitize_filename(emotion)

        with open(f'vector_{sanitized_emotion}.pickle', 'wb') as handle:

            pickle.dump(vector, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def retrieve_vector(self, emotion: str) -> Any:
        """Retrieve stored emotion vector from a pickle file."""
        if os.path.exists(f'vector_{emotion}.pickle'):
            with open(f'vector_{emotion}.pickle', 'rb') as handle:
                return pickle.load(handle)
            
    def fetch_related_emotions(self, emotion_vector: Any) -> list:
        """Retrieve emotions related to the provided vector."""
        # Placeholder logic to fetch related emotions based on cosine similarity or other measures
        related_emotions = []
        return related_emotions

class TopicExtraction:
    def extract(self, input_data: str) -> list:
        """Extract up to 5 main topics from input data using nltk."""
        
        # Tokenize and part-of-speech tag the input data
        tokens = word_tokenize(input_data)
        tagged_tokens = pos_tag(tokens)
        
        # Extract nouns and noun phrases as potential topics
        topics = [word for word, pos in tagged_tokens if pos in ['NN', 'NNS', 'NNP', 'NNPS']]
        
        # Return up to 5 topics
        return topics[:5]

class SentimentExtraction:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def extract_sentiment(self, input_data: str) -> Tuple[str, float]:
        """Determine sentiment of the input data using nltk."""
        scores = self.sia.polarity_scores(input_data)
        sentiment = "neutral"
        if scores['compound'] > 0.05:
            sentiment = "positive"
        elif scores['compound'] < -0.05:
            sentiment = "negative"
        return sentiment, scores['compound']



class EmotionVectorGeneration:
    def generate(self, emotion: str) -> Tuple[Any, str]:
        """Generate emotion vectors and determine its intensity."""
        
        emotion_vector = embed(emotion)
        
        # Determine the intensity of the emotion based on the vector's magnitude
        magnitude = np.linalg.norm(emotion_vector)
        intensity = "strong" if magnitude > THRESHOLD_VALUE else "weak"
        
        return emotion_vector, intensity

class EmotionVectorIntegration:
    def __init__(self):
        self.vector_store = VectorStorage()  # Assuming VectorStorage has methods to fetch related emotions

    def integrate(self, emotion_vector: Any, intensity: str) -> list:
        """Integrate emotion vectors into decisions and other processes."""
        
        companion_emotions = []
        
        if intensity == "strong":
            # Fetch related emotions from the vector store to reinforce the main emotion
            companion_emotions = self.vector_store.fetch_related_emotions(emotion_vector)
        
        # The companion emotions can now be used to influence other parts of the system
        return companion_emotions

class UserFeedback:
    def collect_feedback(self) -> int:
        """Collect feedback from the user regarding the system's response."""
        # Placeholder for feedback collection, for instance, asking the user to rate the response
        feedback_score = 0
        return feedback_score

class UserProfile:
    profiles = {}  # Dictionary to store user-specific data

    def store_user_data(self, user_id: str, data: Any):
        """Store user-specific data."""
        self.profiles[user_id] = data

    def fetch_user_data(self, user_id: str) -> Any:
        """Retrieve user-specific data."""
        return self.profiles.get(user_id, {})


def process_input(input_data: str) -> Dict[str, Any]:
    """Main function to process input and generate the JSON output."""
    
    # Step 2: Interface Node (Preprocessing)
    input_data = InterfaceNode().preprocess_input(input_data)
    
    # Step 3: Emotion Recognition
    er = EmotionRecognition()
    emotional_state = er.recognize(input_data)
    
    # Step 4: Topic Extraction
    topics = TopicExtraction().extract(input_data)
    
    # Step 5: Sentiment Extraction
    sentiment, sentiment_score = SentimentExtraction().extract_sentiment(input_data)
    
    # Step 6: Emotion Vector Generation
    emotion_vector, intensity = EmotionVectorGeneration().generate(emotional_state)
    
    # Step 7: Mood Modulation
    MoodModulation().adjust_mood(emotional_state, intensity)
    
    # Step 8: Vector Storage & Emotion Vector Integration
    VectorStorage().store_vector(emotional_state, emotion_vector)
    companion_emotions = EmotionVectorIntegration().integrate(emotion_vector, intensity)
    
    # Step 9: Decision Making
    decision = DecisionMaking().decide(input_data)
    
    # Step 10: Action Suggestion
    asg = ActionSuggestion()
    suggested_action = asg.suggest(emotional_state)
    
    # Step 11: Social Interaction
    interaction = SocialInteraction().process(input_data)
    
    # Step 12: Feedback and Motivation/Reward
    feedback = UserFeedback().collect_feedback()
    reward = MotivationReward().evaluate(suggested_action, feedback)
    
    # Step 13: Emotional Memory
    EmotionalMemory().store(emotional_state, {'input': input_data, 'response': interaction, 'reward': reward})
    
    # Step 14: Interface Node (Post-Processing)
    final_response = InterfaceNode().postprocess_output(interaction)
    
    return {
        "emotional_state": emotional_state,
        "topics": topics,
        "sentiment": sentiment,
        "decision": decision,
        "suggested_action": suggested_action,
        "interaction": interaction,
        "final_response": final_response,
        "description": f"Description of the {emotional_state} state"
    }

if __name__ == "__main__":
    # Wait for user input
    input_data = input("Please enter your input: ")
    result = process_input(input_data)
    print(result)
