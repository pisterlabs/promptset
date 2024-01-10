import os
from dotenv import load_dotenv, find_dotenv
from requests_oauthlib import OAuth1Session
import openai

load_dotenv(find_dotenv(), verbose=True)

SEMANTIC_SCHOLAR_GRAPH_API_URL = os.getenv("SEMANTIC_SCHOLAR_GRAPH_API_URL")
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
SEMANTIC_SCHOLAR_DATASETS_API_URL = os.getenv(
    "SEMANTIC_SCHOLAR_DATASETS_API_URL")

base_model = "gpt-3.5-turbo"
openai.api_key = os.getenv("OPEN_AI_API_KEY")

ss_headers = {'x-api-key': SEMANTIC_SCHOLAR_API_KEY}

base_prompt = "You are a translator—you translate neuroscience to developmental biology. Deep symmetries exist between these fields. Use them to infer novel, grounded and plausible hypotheses in developmental biology. " + \
    "Follow these instructions carefully. Each translation from neuroscience to developmental biology should:" + \
    "- Read as developmental biology. Neuroscience concepts like 'hippocampus' translated to most relevant/related developmental biology term." + \
    "- Use real terms from developmental biology literature." + \
    "- Don't include any neuroscience words, like a part of the brain. Do the best you can to find the most relevant translation." + \
    "- Be compelling. No fanciful language just be scientifically novel and plausible, given what is known in science." + \
    "- Unless necessary to prove a point, the translation should be structurally similar to the input. " + \
    "For example, here are some terms and plausible translations ('N:' is neuroscience and 'D:' is Developmental Biology):" + \
    "N:Neuron D:Cell" + \
    "N:Behavior D:Morphogenesis" + \
    "N:Millisecond D:Minute" + \
    "N:Memory D:Pattern Memory" + \
    "N:Brain D:Body" + \
    "N:Retina D:Epithelium" + \
    "N:Synapse D:Gap junction" + \
    "N:Neuroscience D:Developmental biology" + \
    "N:Navigating D:Navigating in morphospace" + \
    "N:Lateralization D:Left-right asymmetry" + \
    "N:Mental illness D:Birth defects" + \
    "N:Psychiatry D:Developmental teratology" + \
    "N:Senses D:Receptors" + \
    "N:Action potential D:Change of vmem" + \
    "N:Consciousness D:Somatic consciousness" + \
    "N:Neuroimaging D:Body tissue imaging" + \
    "N:Synaptic D:Electrical-synaptic" + \
    "N:Cognitive D:Proto-cognitive" + \
    "N:Psychiatry D:Developmental teratology" + \
    "N:Space D:Anatomical morphospace" + \
    "N:Animal D:Tissue" + \
    "N:Goals D:Target morphologies" + \
    "N:Muscle contractions D:Cell behavior" + \
    "N:Behavioral space D:Morphospace" + \
    "N:Pattern completion D:Regeneration" + \
    "N:Behavior D:Morphogenesis" + \
    "N:Think D:Regenerate" + \
    "N:Intelligence D:Ability to regenerate" + \
    "N:Event-related potentials D:Bioelectrical signals" + \
    "N:Transcranial D:Optogenetic" + \
    "N:Down the axon D:Across the cell surface" + \
    "N:Action potential movement within an axon D:Differential patterns of Vmem across single cells’ surface" + \
    "N:Neurogenesis D:Cell proliferation" + \
    "N:Neuromodulation D:Developmental signaling" + \
    "N:Critical plasticity periods D:Competency windows for developmental induction events" + \
    "N:What are the goals of hedgehogs D:What are the target morphologies of hedgehogs" + \
    "N:On brains. Retina, behavioral plasticity, muscle, synaptic activity and lateralization D:On bodies. Epithelium, regenerative capacity, cell, cell-signaling activity  and left-right asymmetry" \
    "[Examples done]"

minimal_prompt = "You are a translator—you translate neuroscience to developmental biology. Deep symmetries exist between these fields. Use them to infer novel, grounded and plausible hypotheses in developmental biology. " + \
    "Follow these instructions carefully. Each translation from neuroscience to developmental biology should:" + \
    "- Read as developmental biology. Neuroscience concepts like 'hippocampus' translated to most relevant/related developmental biology term." + \
    "- Use real terms from developmental biology literature." + \
    "- Don't include any neuroscience words, like a part of the brain. Do the best you can to find the most relevant translation." + \
    "- Be compelling. No fanciful language just be scientifically novel and plausible, given what is known in science." + \
    "For example, here are some terms and plausible translations ('N:' is neuroscience and 'D:' is Developmental Biology):" + \
    "N:Neuron D:Cell" + \
    "N:Behavior D:Morphogenesis" + \
    "N:Millisecond D:Minute" + \
    "N:Memory D:Pattern Memory" + \
    "N:Brain D:Body" + \
    "N:Neuroscience D:Developmental biology" + \
    "N:Navigating D:Navigating in morphospace" + \
    "N:Lateralization D:Left-right asymmetry" + \
    "N:Mental illness D:Birth defects" + \
    "N:Psychiatry D:Developmental teratology" + \
    "N:What are the goals of hedgehogs D:What are the target morphologies of hedgehogs" + \
    "N:On brains. Retina, behavioral plasticity, muscle, synaptic activity and lateralization D:On bodies. Epithelium, regenerative capacity, cell, cell-signaling activity  and left-right asymmetry" \
    "[Examples done]"
