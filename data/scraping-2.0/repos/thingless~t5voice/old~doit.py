import torch
import soundfile as sf
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech
from transformers import SpeechT5HifiGan
from datasets import load_dataset
from langchain.text_splitter import CharacterTextSplitter
import numpy as np
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

from util import context_timer

DEVICE = 'cpu'
DEVICE = 'cuda:0'
#VOICE_INDEX = 6799
VOICE_INDEX = 6800
#VOICE_INDEX = 7306
TOKEN_LIMIT = 500
TEXT = """
The following transcript of President Joe Biden’s State of the Union speech was released by the White House after he delivered before Congress, Tuesday February 7, 2023.

THE PRESIDENT:  Mr. Speaker — (applause) —

(Turns to audience members.)  Thank you.  You can smile.  It’s okay.

Thank you, thank you, thank you.  (Applause.)  Thank you.  Please.

Mr. Speaker, Madam Vice President, our First Lady and Second Gentleman — good to see you guys up there — (applause) — members of Congress —

And, by the way, Chief Justice, I may need a court order.  She gets to go to the game tomorr- — next week.  I have to stay home.  (Laughter.)  We got to work something out here.

Members of the Cabinet, leaders of our military, Chief Justice, Associate Justices, and retired Justices of the Supreme Court, and to you, my fellow Americans:

    You know, I start tonight by congratulating the 118th Congress and the new Speaker of the House, Kevin McCarthy.  (Applause.)

Speaker, I don’t want to ruin your reputation, but I look forward to working with you.  (Laughter.)

And I want to congratulate the new Leader of the House Democrats, the first African American Minority Leader in history, Hakeem Jeffries.  (Applause.)

He won despite the fact I campaigned for him.  (Laughter.)

Congratulations to the longest-serving Leader in the history of the United States Senate, Mitch McConnell.  Where are you, Mitch?  (Applause.)

And congratulations to Chuck Schumer, another — you know, another term as Senate Minority [Majority] Leader.  You know, I think you — only this time you have a slightly bigger majority, Mr. Leader.  And you’re the Majority Leader.  About that much bigger?  (Laughter.)  Yeah.

Well, I tell you what — I want to give specolec- — special recognition to someone who I think is going to be considered the greatest Speaker in the history of the House of Representatives: Nancy Pelosi.  (Applause.)

Folks, the story of America is a story of progress and resilience, of always moving forward, of never, ever giving up.  It’s a story unique among all nations.

We’re the only country that has emerged from every crisis we’ve ever entered stronger than we got into it.

Look, folks, that’s what we’re doing again.

Two years ago, the economy was reeling.  I stand here tonight, after we’ve created, with the help of many people in this room, 12 million new jobs — more jobs created in two years than any President has created in four years — because of you all, because of the American people.  (Applause.)

Two years ago — and two years ago, COVID had shut down — our businesses were closed, our schools were robbed of so much.  And today, COVID no longer controls our lives.

And two years ago, our democracy faced its greatest threat since the Civil War.  And today, though bruised, our democracy remains unbowed and unbroken.  (Applause.)

As we gather here tonight, we’re writing the next chapter
in the great American story — a story of progress and resilience.

When world leaders ask me to define America — and they do, believe it or not — I say I can define it in one word, and I mean this: possibilities.  We don’t think anything is beyond our capacity.  Everything is a possibility.

You know, we’re often told that Democrats and Republicans can’t work together.  But over the past two years, we proved the cynics and naysayers wrong.

Yes, we disagreed plenty.  And yes, there were times when Democrats went alone.

But time and again, Democrats and Republicans came together.  Came together to defend a stronger and safer Europe.  You came together to pass one in a gen- — one-in-a-generation — once-in-a-generation infrastructure law building bridges connecting our nation and our people.  We came together to pass one the most significant law ever helping victims exposed to toxic burn pits.  And, in fact — (applause) — it’s important.
""".strip()

with context_timer('load_models'):
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    model.to(DEVICE)
    vocoder.to(DEVICE)

#with context_timer('load_dataset'):
#    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
#    speaker_embeddings = torch.tensor(embeddings_dataset[VOICE_INDEX]["xvector"]).unsqueeze(0)

with context_timer('load_my_embedding'):
    speaker_embeddings = torch.tensor(np.load('embed.npy')).unsqueeze(0).to(DEVICE)

text_splitter = CharacterTextSplitter(chunk_size=TOKEN_LIMIT, chunk_overlap=0)
batches = text_splitter.split_text(TEXT)

with context_timer('processor'):
    inputses = [
        processor(text=txt, return_tensors="pt")
        for txt in batches
    ]
    for d in inputses:
        d['input_ids'] = d['input_ids'].to(DEVICE)

with context_timer('generate'):
    # slow step
    with torch.no_grad():
        speeches = [
            model.generate_speech(
                input_ids=inputs["input_ids"],
                speaker_embeddings=speaker_embeddings,
                vocoder=vocoder,
            )
            for inputs in tqdm(inputses)
        ]

with context_timer('write'):
    out = torch.concat(speeches).cpu().numpy()
    sf.write("tts_example.wav", out, samplerate=16000)
