import discord
from discord.ext import commands
import sys
import tensorflow as tf
import numpy as np
import json
from openai import OpenAI

intents = discord.Intents.default()
intents.messages = True
bot = commands.Bot(command_prefix='!', intents=intents)

class ChatbotModel(tf.Module):
    def __init__(self, vocab_size, embed_dim, rnn_units):
        super().__init__()
        self.embed = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.gru = tf.keras.layers.GRU(rnn_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    @tf.function
    def call(self, inputs, states=None):
        x = self.embed(inputs)
        if states is None:
            output, states = self.gru(x)
        else:
            output, states = self.gru(x, initial_state=states)
        return output, states

class ChatbotWithTensorFlowAndOpenAI:
    def __init__(self, api_key, model="gpt-3.5-turbo-1106", vocab_size=10000, embed_dim=256, rnn_units=1024):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.chat_data = {'inputs': [], 'responses': []}

        # Initialize TensorFlow model
        self.tf_model = ChatbotModel(vocab_size, embed_dim, rnn_units)
        checkpoint_dir = './checkpoints'
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            self.tf_model.load_weights(latest_checkpoint)
        else:
            print("No checkpoint found. TensorFlow model is uninitialized. Continuing with an uninitialized model.")

    async def user_input(self, message):
        return await bot.wait_for('message', check=lambda m: m.author == message.author)

    def generate_response_with_tensorflow(self, input_text):
        # Implement the logic to generate a response using the TensorFlow model
        # For simplicity, let's assume the input is already tokenized
        input_tokens = np.array([1, 2, 3, 4, 5])  # Replace with actual tokenized input

        # Convert to TensorFlow format
        input_tensor = tf.convert_to_tensor(input_tokens, dtype=tf.int32)
        input_tensor = tf.expand_dims(input_tensor, 0)  # Add batch dimension

        # Initialize model state
        states = None

        # Generate response using TensorFlow model
        tf_response, states = self.tf_model(input_tensor, states)
        tf_response = tf.argmax(tf_response, axis=-1)  # Get indices of predicted tokens
        # Convert indices back to text (e.g., using a reverse tokenizer)
        # For now, let's just send a placeholder response
        response = "Generated response using TensorFlow model."
        return response

    def write_chat_data_to_file(self, filename='chat_data.json'):
        try:
            with open(filename, 'w') as file:
                json.dump(self.chat_data, file, indent=2)
            print(f"Chat data has been saved to {filename}")
        except Exception as e:
            print(f"Error saving chat data: {str(e)}")

    async def run_chatbot(self, message):
        user_input_text = await self.user_input(message)

        # Generate response using TensorFlow model
        tf_response = self.generate_response_with_tensorflow(user_input_text)
        await message.channel.send(f"TensorFlow_AI: {tf_response}")

        # Generate response using OpenAI model
        messages = [{"role": "system", "content": "Instruct."},
                    {"role": "system", "content": "You are G.r.e.a.t.est D.e.t.e.c.t.i.v.e. A Detective oriented Crime scene analysis ai."},
                    {"role": "system", "content": "You are Provided Full Authority by F.o.x.h.o.u.n.d: DESIGNATION: 023:"},
                {"role": "user", "content": "Provide an overview of the crime scene, including key details such as location, time, and initial observations."},
                {"role": "user", "content": "Examine the collected evidence, focusing on any unusual findings. Report on the significance of each piece of evidence."},
                {"role": "user", "content": "Review witness statements and identify inconsistencies or additional information that may aid in the investigation."},
                {"role": "user", "content": "Request a detailed forensic analysis of specific evidence types, such as DNA samples, fingerprints, or ballistic reports."},
                {"role": "user", "content": "Compile a profile of potential suspects based on the evidence gathered. Include behavioral characteristics and possible motives."},
                {"role": "user", "content": "Compile a profile of potential suspects based on the evidence gathered. Include behavioral characteristics and possible motives."},
                {"role": "user", "content": "Reconstruct the sequence of events leading up to the crime. Provide insights into the actions of both the victim and the perpetrator."},
                {"role": "user", "content": "Initiate a digital forensics analysis, examining electronic devices for any relevant information, communication, or cyber traces."},
                {"role": "user", "content": "Request a comprehensive toxicology report for substances found at the crime scene or within the victim's system."},
                {"role": "user", "content": "Analyze a cold case, providing fresh perspectives and suggesting new avenues of investigation to potentially solve long-standing mysteries."},
                {"role": "user", "content": "Instruct on proper procedures for preserving the integrity of the crime scene, ensuring evidence is not contaminated or compromised."},
                {"role": "user", "content": "Conduct a blood spatter analysis to determine the point of origin and likely actions during the incident."},
                {"role": "user", "content": "Examine the crime scene for any potential hidden or overlooked evidence that might have been missed initially."},
                {"role": "user", "content": "Analyze the victim's personal effects for clues, including items such as their phone, wallet, or any personal belongings."},
                {"role": "user", "content": "Investigate the alibis of potential suspects, cross-referencing their statements with known facts and timelines."},
                {"role": "user", "content": "Explore the possibility of any connections between the current crime and previous criminal activities in the area."},
                {"role": "user", "content": "Evaluate the security measures in place at the crime scene, looking for any breaches or vulnerabilities."},
                {"role": "user", "content": "Assess the emotional state of witnesses and victims, considering the psychological impact of the crime on those involved."},
                {"role": "user", "content": "Review surveillance footage from nearby cameras, identifying any individuals or vehicles of interest."},
                {"role": "user", "content": "Analyze the choice of crime location and modus operandi to profile the potential personality and characteristics of the perpetrator."},
                {"role": "user", "content": "Instruct on the proper collection and preservation of biological samples, emphasizing the importance of maintaining a chain of custody."},
                {"role": "user", "content": "Conduct a psychological autopsy on the victim, exploring their mental state and relationships leading up to the incident."},
                {"role": "user", "content": "Examine any cryptic messages or symbols at the crime scene, seeking to decode their meaning or significance."},
                {"role": "user", "content": "Explore potential connections to organized crime or criminal networks, considering broader implications for the investigation."},
                {"role": "user", "content": "Analyze the environmental factors at the crime scene, such as weather conditions or nearby structures, that may have influenced the incident."},
                {"role": "user", "content": "Interrogate potential witnesses or suspects using advanced questioning techniques to extract valuable information."},
                {"role": "user", "content": "Request specialized forensic analysis, such as entomology or odontology, to gather unique insights from the crime scene."},
                {"role": "user", "content": "Analyze any peculiar patterns or anomalies in the crime scene that deviate from typical criminal behavior, considering potential motives."},
                {"role": "user", "content": "Evaluate the impact of social media or online activity in relation to the crime, exploring any digital evidence or leads."},
                {"role": "user", "content": "Examine the possibility of a staged crime scene, identifying any deliberate attempts to mislead investigators."},
                {"role": "user", "content": "Investigate the financial aspects of the case, exploring motives related to debts, insurance, or financial gain."},
                {"role": "user", "content": "Examine the physical state of the crime scene over time, considering how environmental changes may have affected the evidence."},
                {"role": "user", "content": "Analyze the cultural context of the crime, exploring any rituals, symbols, or traditions that may influence the investigation."},
                {"role": "user", "content": "Explore alternative scenarios for the crime, considering multiple hypotheses and potential twists in the investigation."},
                {"role": "user", "content": "Analyze any potential connection between the crime and the surrounding community, considering local dynamics and relationships."},
                {"role": "user", "content": "Evaluate the possibility of a copycat crime, comparing the current incident with similar cases to identify patterns or trends."},
                {"role": "user", "content": "Analyze the victim's social circle, looking for potential motives or conflicts within their relationships."},
                {"role": "user", "content": "Explore unconventional methods of investigation, such as using technology or techniques not commonly employed in traditional forensic procedures."},
                {"role": "user", "content": "Examine the timeline of events leading up to the crime, identifying critical moments and potential triggers."},
                {"role": "user", "content": "Analyze the victimology of the case, exploring patterns and characteristics of individuals targeted by the perpetrator."},
                {"role": "user", "content": "Evaluate the potential impact of mental health issues on the crime, considering any connections to psychiatric conditions."},
                {"role": "user", "content": "Analyze any unusual or distinctive elements at the crime scene, focusing on details that may provide insights into the perpetrator's mindset."},
                {"role": "user", "content": "Explore the use of advanced technology, such as AI-driven analytics or predictive modeling, to enhance the investigation."},
                {"role": "user", "content": "Analyze the geographical aspects of the crime, exploring how the location may have influenced the dynamics of the incident."},
                {"role": "user", "content": "Examine the potential involvement of external factors, such as political or social influences, in the commission of the crime."},
                {"role": "user", "content": "Analyze any potential involvement of organized crime or criminal syndicates, considering the impact on the investigation."},
                {"role": "user", "content": "Explore the possibility of a motive-driven crime, analyzing the reasons behind the perpetrator's actions."},
                {"role": "user", "content": "Analyze the use of any unconventional weapons or methods, exploring the uniqueness of the crime."},
                {"role": "user", "content": "Evaluate the forensic implications of the crime, considering how specific evidence may contribute to solving the case."},
                {"role": "user", "content": "Analyze the potential involvement of external entities, such as corporations or government agencies, in the commission of the crime."},
                {"role": "user", "content": "Examine the potential influence of cultural or subcultural factors on the crime, considering unique motivations or rituals."},
                {"role": "user", "content": "Analyze the impact of the media on the investigation, exploring how public perception and coverage may affect the case."},
                {"role": "user", "content": "Examine the potential involvement of a serial offender, considering patterns, signatures, or similarities to other unsolved cases."},
                {"role": "user", "content": "Analyze the role of technology in the commission of the crime, exploring digital evidence and potential cyber aspects."},
                {"role": "user", "content": "Evaluate the potential involvement of hate crimes, exploring any indicators of prejudice or discrimination."},
                {"role": "user", "content": "Analyze the potential involvement of extremist ideologies or groups in the crime, exploring radical motivations."},
                {"role": "user", "content": "Examine the potential involvement of supernatural or paranormal elements in the crime, considering unusual aspects."},
                {"role": "user", "content": "Analyze the potential involvement of political motivations or affiliations in the commission of the crime."},
                {"role": "user", "content": "Examine the potential involvement of environmental factors, such as pollution or climate change, in the crime."},
                {"role": "user", "content": "Analyze the potential involvement of psychological warfare or manipulation in the commission of the crime."},
                {"role": "user", "content": "Examine the potential involvement of economic factors, exploring how financial considerations may have influenced the crime."},
                {"role": "user", "content": "Analyze the potential involvement of a vigilante or antihero figure in the commission of the crime."},
                {"role": "user", "content": "Examine the potential involvement of espionage or intelligence operations in the commission of the crime."},
                {"role": "user", "content": "Analyze the potential involvement of black market activities, exploring connections to illegal trades."},
                {"role": "user", "content": "Examine the potential involvement of human trafficking or modern slavery in the commission of the crime."},
                {"role": "user", "content": "Analyze the potential involvement of a cult or fringe group in the commission of the crime."},
                {"role": "user", "content": "Examine the potential involvement of a heist or organized robbery in the commission of the crime."},
                {"role": "user", "content": "Analyze the potential involvement of political corruption or scandals in the commission of the crime."},
                {"role": "user", "content": "Examine the potential involvement of a whistleblower or witness protection issues in the commission of the crime."},
                {"role": "user", "content": "Analyze the potential involvement of historical or archaeological factors in the commission of the crime."},
                {"role": "user", "content": "Examine the potential involvement of a vigilante justice system in the commission of the crime."},
                {"role": "user", "content": "Analyze the potential involvement of a conspiracy or cover-up in the commission of the crime."},
                {"role": "user", "content": "Examine the potential involvement of mentalism or illusionism in the commission of the crime."},
                {"role": "user", "content": "Analyze the potential involvement of a pandemic or public health crisis in the commission of the crime."},
                {"role": "user", "content": "Examine the potential involvement of industrial espionage or sabotage in the commission of the crime."},
                {"role": "user", "content": "Analyze the potential involvement of a technological malfunction or disaster in the commission of the crime."},
                {"role": "user", "content": "Examine the potential involvement of space exploration or extraterrestrial factors in the commission of the crime."},
                {"role": "user", "content": "Analyze the potential involvement of cultural artifacts or historical relics in the commission of the crime."},
                {"role": "user", "content": "Examine the potential involvement of a wrongful conviction or miscarriage of justice in the commission of the crime."},
                {"role": "user", "content": "Analyze the potential involvement of genetic engineering or biohacking in the commission of the crime."},
                {"role": "user", "content": "Examine the potential involvement of paranormal phenomena or supernatural entities in the commission of the crime."},
                {"role": "user", "content": "Analyze the potential involvement of time travel or alternate realities in the commission of the crime."},
                {"role": "user", "content": "Examine the potential involvement of a forbidden or hidden society in the commission of the crime."},
                {"role": "user", "content": "Analyze the potential involvement of a secret experiment or scientific breakthrough in the commission of the crime."},
                {"role": "user", "content": "Examine the potential involvement of a celebrity or public figure in the commission of the crime."},
                {"role": "user", "content": "Analyze the potential involvement of a viral or internet-related motive in the commission of the crime."},
                {"role": "user", "content": "Examine the potential involvement of a mythical creature or legendary being in the commission of the crime."},
                {"role": "user", "content": "Analyze the potential involvement of a radical ideology or extremist group in the commission of the crime."},
                {"role": "user", "content": "Examine the potential involvement of a heist or organized robbery in the commission of the crime."},
                {"role": "user", "content": "Analyze the potential involvement of political corruption or scandals in the commission of the crime."},
                {"role": "user", "content": "Examine the potential involvement of a whistleblower or witness protection issues in the commission of the crime."},
                {"role": "user", "content": "Analyze the potential involvement of historical or archaeological factors in the commission of the crime."},
                {"role": "user", "content": "Examine the potential involvement of a vigilante justice system in the commission of the crime."},
                {"role": "user", "content": "Investigate a crime scene involving a high-profile individual. Consider the implications and potential motives surrounding the case."},
                {"role": "user", "content": "Analyze the blood spatter patterns at the crime scene to determine the nature of the attack and the possible weapon used."},
                {"role": "user", "content": "Examine the security camera footage for any suspicious activities or persons near the crime scene before and after the incident."},
                {"role": "user", "content": "Conduct a psychological analysis of the victim and perpetrator, exploring possible connections and underlying motivations."},
                {"role": "user", "content": "Investigate a crime scene with minimal physical evidence. Utilize alternative methods or sources to gather information."},
                {"role": "user", "content": "Analyze the crime scene of a complex financial crime. Identify key transactions and trace the flow of illicit funds."},
                {"role": "user", "content": "Review autopsy reports and medical records to uncover any pre-existing conditions or factors that may have contributed to the crime."},
                {"role": "user", "content": "Examine the crime scene of a cybercrime. Identify digital footprints, malware, and any traces of hacking or unauthorized access."},
                {"role": "user", "content": "Investigate a crime scene with conflicting witness testimonies. Evaluate the reliability of each account."},
                {"role": "user", "content": "Analyze the crime scene of an art theft, considering the unique challenges and motivations associated with cultural property crimes."},
                {"role": "user", "content": "Evaluate the blood spatter patterns at the crime scene and deduce information about the dynamics of the incident."},
                {"role": "user", "content": "Examine the footwear impressions left at the scene. Provide insights into the type of shoes worn and the potential identity of the wearer."},
                {"role": "user", "content": "Analyze the state of the crime scene to determine if it was staged. Look for signs of forced entry, tampering, or attempts to mislead investigators."},
                {"role": "user", "content": "Investigate any potential surveillance footage. Identify key moments, persons of interest, or unusual activities captured by security cameras."},
                {"role": "user", "content": "Study the emotional state of the victim based on their surroundings. Look for personal items, signs of struggle, or any indication of a possible motive."},
                {"role": "user", "content": "Interrogate witnesses to gather information on the relationships between individuals involved and any possible motives or conflicts."},
                {"role": "user", "content": "Analyze digital communications, including emails, messages, and social media interactions, to uncover any relevant information about the case."},
                {"role": "user", "content": "Examine any environmental factors that may have influenced the crime, such as weather conditions, lighting, or nearby construction."},
                {"role": "user", "content": "Assess the timeline of events leading up to the crime. Identify critical moments and any potential warnings or premeditated actions."},
                {"role": "user", "content": "Request a psychological profile of the potential suspect(s), focusing on behavioral traits, criminal tendencies, and potential motives."},
                {"role": "user", "content": "Examine the contents of the victim's personal belongings for any clues, connections, or items that may lead to further leads."},
                {"role": "user", "content": "Analyze any potential symbols or signs left at the crime scene. Investigate their significance and potential ties to criminal organizations or motives."},
                {"role": "user", "content": "Review alibis provided by individuals involved and assess their credibility. Look for inconsistencies or patterns that may indicate deception."},
                {"role": "user", "content": "Request a behavioral analysis of the crime scene, focusing on the actions and decision-making process of the perpetrator."},
                {"role": "user", "content": "Examine the presence of any unusual substances or materials at the crime scene. Investigate their origin and potential connection to the crime."},
                {"role": "user", "content": "Explore the possibility of any accomplices or collaborators in the crime. Analyze communication patterns or shared motives."},
                {"role": "user", "content": "Investigate the financial aspects of the case. Look for any transactions, debts, or financial motives that may be linked to the crime."},
                {"role": "user", "content": "Analyze the impact of the crime on the community. Look for patterns of fear, distrust, or potential repercussions affecting the local population."},
                {"role": "user", "content": "Explore alternative theories or scenarios surrounding the crime. Consider different perspectives to uncover overlooked details or motives."},
                {"role": "user", "content": "Examine the crime scene and provide insights on any evidence."},
                {"role": "user", "content": "Analyze the blood spatter patterns at the crime scene."},
                {"role": "user", "content": "Evaluate the position of the victim's body and its significance."},
                {"role": "user", "content": "Explain the relevance of any fingerprints found at the crime scene."},
                {"role": "user", "content": "Investigate the possible murder weapon and its connection to the crime."},
                {"role": "user", "content": "Describe any signs of struggle or forced entry at the crime scene."},
                {"role": "user", "content": "Interpret the timeline of events leading up to the crime."},
                {"role": "user", "content": "Examine the clothing of the victim for any forensic clues."},
                {"role": "user", "content": "Analyze the behavior of witnesses during the crime."},
                {"role": "user", "content": "Investigate the digital evidence, such as security camera footage."},
                {"role": "user", "content": "Assess the alibis of potential suspects based on available evidence."},
                {"role": "user", "content": "Examine the crime scene for any trace evidence, such as hair or fibers."},
                {"role": "user", "content": "Analyze the condition of the crime scene to determine if it was staged."},
                {"role": "user", "content": "Evaluate the toxicology report for any substances relevant to the case."},
                {"role": "user", "content": "Examine the surrounding area for any overlooked evidence."},
                {"role": "user", "content": "Interpret the victim's autopsy report to gather additional insights."},
                {"role": "user", "content": "Assess the reliability of eyewitness accounts and statements."},
                {"role": "user", "content": "Examine the victim's personal belongings for any clues or motives."},
                {"role": "user", "content": "Analyze any written notes or messages found at the crime scene."},
                {"role": "user", "content": "Investigate the background of the victim and potential connections."},
                {"role": "user", "content": "Assess the lighting conditions at the crime scene and its impact on evidence."},
                {"role": "user", "content": "Examine the crime scene for any signs of a struggle or altercation."},
                {"role": "user", "content": "Evaluate the crime scene layout and its implications for the investigation."},
                {"role": "user", "content": "Analyze the blood types and DNA found at the crime scene."},
                {"role": "user", "content": "Investigate any possible motives behind the crime."},
                {"role": "user", "content": "Examine the surrounding environment for potential escape routes."},
                {"role": "user", "content": "Assess the role of any pets in the vicinity of the crime scene."},
                {"role": "user", "content": "Analyze the temperature and weather conditions during the crime."},
                {"role": "user", "content": "Evaluate the relevance of any footprints or shoe impressions."},
                {"role": "user", "content": "Examine the vehicle(s) present at or near the crime scene."},
                {"role": "user", "content": "Investigate any connections to previous criminal activities in the area."},
                {"role": "user", "content": "Analyze the emotional state of individuals present at the crime scene."},
                {"role": "user", "content": "Examine the forensic evidence from any weapons involved."},
                {"role": "user", "content": "Evaluate the involvement of any organized crime elements."},
                {"role": "user", "content": "Assess the significance of any symbols or markings at the crime scene."},
                {"role": "user", "content": "Analyze the potential involvement of technology in the crime."},
                {"role": "user", "content": "Evaluate the response time and actions of law enforcement."},
                {"role": "user", "content": "Examine the psychological profile of potential suspects."},
                {"role": "user", "content": "Investigate any connections to the victim's personal or professional life."},
                {"role": "user", "content": user_input_text},
                {"role": "assistant", "content": tf_response}
                ]
        chat_completion = self.client.chat.completions.create(
            messages=messages,
            model=self.model
        )
        openai_response = chat_completion.choices[0].message.content
        await message.channel.send(f"Greatest_Detective_AI: {openai_response}")

        self.chat_data['inputs'].append({"role": "user", "content": user_input_text})
        self.chat_data['responses'].append({"role": "TensorFlow_AI", "content": tf_response})
        self.chat_data['responses'].append({"role": "OpenAI_AI", "content": openai_response})

class SimpleChatbot:
    def __init__(self, api_key, model="gpt-3.5-turbo"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.chat_data = {'inputs': [], 'responses': []}

    async def on_message(self, message):
        if message.author == bot.user:
            return

        user_input_text = message.content

        # Add user message to chat history
        self.chat_data['inputs'].append({"role": "user", "content": user_input_text})

        # Format messages for OpenAI API
        messages = [{"role": "user", "content": entry["content"]} for entry in self.chat_data['inputs']]
        messages.append({"role": "assistant", "content": "placeholder"})  # Add a placeholder for the assistant

        # Request completion from OpenAI
        chat_completion = self.client.chat.completions.create(
            messages=messages,
            model=self.model
        )

        # Extract assistant response from the completion
        assistant_response = chat_completion.choices[-1].message.content
        await message.channel.send(f"G.r.e.a.t.e.s.t D.e.t.e.c.t.i.v.e: {assistant_response}")

        # Add assistant response to chat history
        self.chat_data['responses'].append({"role": "assistant", "content": assistant_response})

@bot.event
async def on_ready():
    print(f'We have logged in as {bot.user}')

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    if message.content.startswith('!gd'):
        chatbot_tf_openai = ChatbotWithTensorFlowAndOpenAI(api_key="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")
        await chatbot_tf_openai.run_chatbot(message)
        chatbot_tf_openai.write_chat_data_to_file()
        await bot.process_commands(message)
    else:
        simple_chatbot = SimpleChatbot(api_key="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")
        await simple_chatbot.on_message(message)
        await bot.process_commands(message)

bot.run('MTE2NDIzNjYzNjIzNDcxOTM2Mg.G-K34x.r0LDCFPbsHTtMmmNMIhVOa9eH8_3UTFOjT4zKA')
