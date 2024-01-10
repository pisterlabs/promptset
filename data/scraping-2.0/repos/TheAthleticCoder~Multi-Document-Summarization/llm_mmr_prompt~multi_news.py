import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from torch import cuda
import torch
from tqdm import tqdm
import pandas as pd
import time
from langchain import HuggingFacePipeline, PromptTemplate, LLMChain
import csv

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Generate summaries using a language model.")
parser.add_argument("--model_id", type=str, help="Hugging Face model ID")
parser.add_argument("--file_path", type=str, help="Path to the input CSV file")
parser.add_argument("--new_file_save_path", type=str, help="Path to save the generated CSV file")
args = parser.parse_args()

# device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
device = 'cuda:1'
print(device)

model_id = args.model_id
hf_auth = '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'
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# print(torch.cuda.current_device())

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, use_auth_token=hf_auth)
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_auth)
print(f"Model size: {model.get_memory_footprint():,} bytes")

DEFAULT_SYSTEM_PROMPT = """\
You are an agent who generates a summary using key ideas and concepts. You have to ensure that the summaries are coherent, fluent, relevant and consistent.
"""
instruction = """
"""
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
"""
DO NOT CHANGE ANY OF THE BELOW FUNCTIONS
"""
def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS #system prompt: Default instruction to be given to the model
    prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST # Final Template: takes in instruction as well. Here it would take in the summary and the source
    return prompt_template
#Function to remove the prompt from the final generated answer
def cut_off_text(text, prompt):
    cutoff_phrase = prompt
    index = text.find(cutoff_phrase)
    if index != -1:
        return text[:index]
    else:
        return text
def remove_substring(string, substring):
    return string.replace(substring, "")
### What torch.autocast is: casts tensors to a smaller memory footprint
def generate(text):
    prompt = get_prompt(text)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs,
                             max_new_tokens=512,
                             eos_token_id=tokenizer.eos_token_id,
                             pad_token_id=tokenizer.eos_token_id,
                             )
    final_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    final_outputs = cut_off_text(final_outputs, '</s>')
    final_outputs = remove_substring(final_outputs, prompt)

    return final_outputs#, outputs
def parse_text(text):
        wrapped_text = textwrap.fill(text, width=100)
        print(wrapped_text +'\n\n')
        # return wrapped_text
from langchain import HuggingFacePipeline
from langchain import PromptTemplate,  LLMChain
from transformers import pipeline
pipe = pipeline("text-generation",
                model=model,
                tokenizer= tokenizer,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                # max_new_tokens = 1024,
                max_new_tokens = 512,
                #do_sample=True,
                #top_k=30,
                #num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                # device=1,
                )
llm = HuggingFacePipeline(pipeline=pipe, model_kwargs = {'temperature':0})

system_prompt = """
You are an agent who is tasked with creating cohesive and relevant text that integrates key ideas and concepts provided in a list format of an output of MMR. Your goal is to produce summaries that are fluent, coherent, and consistent.

Example 1:
MMR: ['By By Kim I. Hartman May 22, 2010 in Entertainment Tampa - Hulk Hogan is back in federal court in Florida, this time over a cartoon commercial. Hulk claims his reputation was harmed in a Cocoa Pebbles ad featuring Bamm-Bamm tossing " Hulk Boulder" into the air and winning the match. Hogan, whose real name is Terry Bollea, is suing the maker of In the "Cocoa Smashdown" commercial, a cartoon character resembling Hogan easily beats Fred and Barney inside the ring. But then Bamm-Bamm steps in and pounds the blond-haired, mustachioed wrestler to bits. Hulk, the federal lawsuit states, "is shown humiliated and cracked into pieces with broken teeth, with the closing banner, \'Little Pieces…BIG TASTE!\'" The commercial character goes by the name "Hulk Boulder," which Hogan\'s lawsuit says is a name he used early in his career until wrestling promoter Vince McMahon decided he should have an Irish name. The lawsuit says Post Foods never sought or received Hogan\'s permission to use his likeness to promote the cereal. Hogan says he raised his objections with Post in August, but the ads continued. The wrestler contends he has been harmed by, among other things, "the unauthorized and degrading depictions in the Cocoa Smashdown advertisements." Hogan has recently been in the news when he Hogan blames Wells Fargo and claims he was inadequately insured when his teenage son, Nick Bollea, got into a Hulk Hogan is going to the mat against The Flintstones, reports Tampa Bay Online. Hogan, whose real name is Terry Bollea, is suing the maker of Cocoa Pebbles, Post Cereal , accusing the company of appropriating his image in commercials for the cereal.In the "Cocoa Smashdown" commercial, a cartoon character resembling Hogan easily beats Fred and Barney inside the ring. But then Bamm-Bamm steps in and pounds the blond-haired, mustachioed wrestler to bits.Hulk, the federal lawsuit states, "is shown humiliated and cracked into pieces with broken teeth, with the closing banner, \'Little Pieces…BIG TASTE!\'"The commercial character goes by the name "Hulk Boulder," which Hogan\'s lawsuit says is a name he used early in his career until wrestling promoter Vince McMahon decided he should have an Irish name.The lawsuit says Post Foods never sought or received Hogan\'s permission to use his likeness to promote the cereal. Hogan says he raised his objections with Post in August, but the ads continued.The wrestler contends he has been harmed by, among other things, "the unauthorized and degrading depictions in the Cocoa Smashdown advertisements."Hogan has recently been in the news when he filed suit in state court , that was later moved to Federal Court against Wells Fargo Insurance for failing to upgrade his coverage when his exposure to risk grew.Hogan blames Wells Fargo and claims he was inadequately insured when his teenage son, Nick Bollea, got into a wreck that grievously injured passenger John Graziano . Hogan settled a lawsuit with the seriously injured victim Graziano in February. More about Hulk, Hogan, Bollea, Wwf, Wrestling More news from hulk hogan bollea wwf wrestling wrestler car accident divorce flintstones cocoa pebbles federal court florida legal terry terrybollea cocoasmashdown smashdown bammbamm bamm post cereal wells fargo insurance TBO > News > Breaking News',
'Reporter Elaine Silvestrini can be reached at (813) 259-7837.',
'Share this:']

Summary: – Hulk Hogan is back in federal court in Florida, this time over a cartoon commercial. Hulk claims his reputation was harmed in a "Cocoa Smashdown" featuring Hulk Boulder vs Fred and Barney when Bamm-Bamm steps in, tosses him into the air, and wins the match. Hulk is suing Post cereal over the use of the look-a-like character, reports the Tampa Tribune. More details at Digital Journal here.
"""

# Example 1:
# MMR: ['Alexis and Serena met the way two people do in the best love stories: by chance. Actually, it runs a little deeper than that because, let’s face it, Alexis was initially considered by Serena and the others she was with to be an irritant they were hoping would just get the hint and go away. The location was the Cavalieri hotel, in Rome, on May 12, 2015. That night Serena was about to play her first match in the Italian Open. She is not a morning person and usually doesn’t eat breakfast, but the buffet offering at the Cavalieri was beyond extravagant and Jessica was champing at the bit, so they went to try it along with longtime agent Jill Smoller, of William Morris Endeavor Entertainment, and Zane Haupt, who handles some business-development opportunities for Serena. The buffet had closed down five minutes before the group got there, so their only recourse was to go to the pool area and sit at a table for four and order breakfast. Other people on Serena’s team were expected at an adjoining table. The night before, Alexis had stayed up until one or two in the morning drinking at a café with Kristen Wiig and friends—Wiig was in Rome shooting Zoolander 2, and he knew her cousin, so he introduced himself. He passed out when he got back to the hotel, where he was staying for the Festival of Media Global conference, and was slightly hungover when he came down to breakfast. He too headed out to the pool area. Which is when he decided without thinking about it to sit at the table next to Serena, his only interest to get coffee and food and put on his headphones and work on his laptop. Which struck Serena and the others as a pain in the neck, since Alexis had a choice of other empty tables. “I knew it was coming,” she says of the proposal. “I was like, ‘Serena, you’re ready. This is what you want.’ ” “This big guy comes and he just plops down at the table next to us, and I’m like, ‘Huh! All these tables and he’s sitting here?,’ ” Serena remembered. Alexis recalled that the pool area was “not quite so empty.” Then came the quintessential Australian accent of Zane Haupt. “Aye, mate! There’s a rat. There’s a rat by your table. You don’t want to sit there.” Serena started laughing. “We were trying to get him to move and get out of there,” said Serena. “He kind of refuses and he looks at us. And he’s like, ‘Is there really a rat here?’ ” At which point Serena remembers the first words she ever said to him. “No, we just don’t want you sitting there. We’re going to use that table.” “I’m from Brooklyn. I see rats all the time.” “Oh, you’re not afraid of rats?” “No.” Which is when Serena suggested a compromise and invited Alexis to join them. Which is when Alexis became “98 percent sure” that the person asking about his rat tolerance was Serena Williams. He knew generally about her accomplishments on the court. But Alexis, an avid pro-football-and-basketball fan, had “never watched a match on television or in real life. It was literally the sport—even if ESPN was announcing tennis updates, I would just zone out. . . . I really had no respect for tennis.” He did keep this to himself. Serena asked about the tech conference and whom Alexis had come to hear speak. He later described the question as a “softball lobbed over the plate” that even he could hit out of the park. “Actually, I’m here to speak.” Alexis told her about Reddit. Serena knew nothing about it but acted as if she did, and said she had been on it earlier in the morning. To which Alexis asked, “Oh, were you? What do you like about it?” To which Serena gave a very long “Wellllll . . . ” and was saved by Jessica and Jill chiming in. Serena Williams has long been queen of the tennis court, but her success also extends to business, fashion, and philanthropy. Watch the video below to see some of her biggest career achievements. Serena started asking him about her Web site and if she should have an app. Alexis thought, “This is an interesting, charming, beautiful woman.” But he had just come out of a five-year relationship and was still slightly hungover and “I was not thinking beyond ‘Yeah sure, I can give you some feedback on your Web site.’ ” Serena thought he was interested in Jessica. But she did give him her number—she later said it was only because she might have more tech-related questions. He was eminently likable, and Jill, after finding out he was a client of WME for his speaking gigs, invited him to the match that night. Serena had an injury and did not play well but still won. Afterward she and her team got on a van to head back to the hotel. Alexis was on board as well and Serena freaked out a little bit. “I see this super-tall guy get in our [van], and I was like, ‘Oh my God, Jill. Tell me what’s wrong. Do I have another stalker? Why is Rome sending personal security with me. . . . And she’s like, ‘No, that’s Alexis.’ I remembered his name because it was a unique name. I was like, ‘Oh, I remember.’ ” After recognizing him, she invited Alexis to join her team for dinner that night. It didn’t work out. But something was in the air, and as our love story continues, there’s only one place to find out just what.',
# 'NEW YORK (AP) — Serena Williams is showing off her pregnancy with a nude photo on the cover of the August issue of Vanity Fair.',
# 'The tennis superstar is seen in profile with her right arm covering her breasts and her pregnant stomach prominently on display. The magazine unveiled the cover Tuesday.',
# 'Serena announced her pregnancy with Reddit co-founder Alexis Olhanian in April. The magazine reports the couple will be married in the fall after the baby is born. Williams tells the magazine she "did a double take" and her heart "dropped" when she saw a positive test because it came just before the Australian Open, which she ended up winning.']

# Summary: – Serena Williams is showing off her pregnancy with a nude photo on the cover of the August issue of Vanity Fair, the AP reports. The tennis superstar is seen in profile with her right arm covering her breasts and her pregnant stomach prominently on display. The magazine unveiled the cover Tuesday. Williams announced her pregnancy with Reddit co-founder Alexis Ohanian in April. The magazine reports the couple will be married in the fall after the baby is born. Williams tells the magazine she "did a double take" and her heart "dropped" when she saw a positive test because it came just before the Australian Open, which she ended up winning. Click to read the Vanity Fair piece, which the magazine says is the "full love story" between Williams and Ohanian.' 



# Example 3:
# MMR: ['The good news reached Maria Helena Pambo as she stood in line in St. Peter\'s Square to pray at the tomb of Pope John Paul II On a gloriously sunny afternoon, Pambo, 34, heard that the former pontiff is to be officially beatified this spring, barely six years after his death — the quickest anyone has been bestowed the honor in modern times. The Vatican announced Friday that his successor, Pope Benedict XVI , had approved the move."It\'s a day of joy and happiness," said Pambo, a nun from Peru. "I never met John Paul II, but now that I live in Rome, every two or three days when I have some time off, I come to pray at his tomb. I ask him for help."Tens of thousands of her fellow devotees are expected to converge on the square May 1, the first Sunday after Easter, for the beatification ceremony. Replete with religious pomp and fervor, the event is expected to be a morale booster for an institution beleaguered by accusations of silence and duplicity in its handling of thousands of allegations of priestly abuse.John Paul\'s elevation was set after Benedict certified the findings of a panel charged with verifying a miracle ascribed to the late pontiff, a prerequisite for beatification, which is an intermediate step toward sainthood.Church-appointed investigators concluded that a French nun was miraculously cured of Parkinson\'s disease after praying to John Paul within weeks of his death on April 2, 2005. He had suffered from the same ailment.The popular Polish-born pontiff is now one step closer to being declared a saint. To qualify for that, a second miracle must be determined to have occurred at his posthumous intervention.He was launched on the road to sainthood much sooner than usual: The Vatican\'s rules decree that a person must be dead for at least five years before the process leading to canonization can begin.But soon after John Paul\'s death, Benedict declared that he would waive the waiting period and initiate the process immediately, perhaps in response to the throngs of fervent followers who crowded St. Peter\'s Square for the pope\'s funeral, waving signs demanding, "Sainthood right now!"Such a "fast track" to sainthood is unusual, but not unprecedented. In fact, John Paul did the same for Mother Teresa, who died in 1997 and was beatified in 2003. His own elevation will beat out hers for the title of quickest by a matter of days.Marco Tosatti, a veteran Vatican-watcher, said the public outpouring of adulation for John Paul almost certainly had an effect on Benedict\'s decision, aside from any personal reverence he might feel toward his predecessor."If the Vatican had said \'No, we don\'t have enough reasons to say he\'s a saint,\' Catholic people would consider him a saint in spite of it. They don\'t need the Vatican seal of approval," said Tosatti, who writes for the Italian newspaper La Stampa. "The people considered him a saint even when he was still alive."For millions around the world, the late pope was an inspiring figure because of his unwavering opposition to communism during the Cold War, his resilience after being seriously wounded in a 1981 assassination attempt and a common touch that endeared him to devotees during his frequent trips around the world to promote his church.But there are critics, too, who note his rigid stance in opposing female priests, contraception and gay rights, and the fact that many of the cases of sexual molestation and physical abuse of minors by priests, nuns and other Catholic workers occurred during his 27-year papacy."The church hierarchy can avoid rubbing more salt into these wounds by slowing down their hasty drive to confer sainthood on the pontiff under whose reign most of the widely documented clergy sex crimes and cover-ups took place," Barbara Dorris, spokeswoman for a victims rights group, Survivors Network of those Abused by Priests, said in a statement from St. Louis. "We urge Vatican officials to move cautiously in their haste to honor Pope John Paul II."Last year, some Vatican officials reportedly had doubts about the diagnosis and healing of the Parkinson\'s disease that afflicted Sister Marie Simon-Pierre, a nun who works in a hospital in Arles, France. She had told church authorities that after other nuns had prayed to John Paul for her and after she herself had written down his name on a piece of paper, she awoke one morning in June 2005 free of the disabling symptoms that had made normal life impossible.But the Vatican\'s Congregation for the Causes of Saints said Friday that their medical investigators had scrutinized the case carefully and concluded that the nun\'s recovery from the degenerative disease had no scientific explanation, in other words, that the miracle was genuine.Interviewed Friday by French and Italian television, Sister Marie Simon-Pierre said John Paul "hasn\'t left me; he won\'t leave me until the end of my life."Tosatti said that more than 300 miracles already have been attributed to John Paul since his death; and though the nun\'s case might not have been the strongest, church authorities apparently decided to stick with it.Despite the misgivings of some people about the fast-track process, Tosatti said the late pope made a smart candidate because so much of his life had been subjected to the glare of the international news media and no unsavory new details were likely to be unearthed.At the end of 2009, Benedict gave formal recognition of John Paul\'s "heroic virtues" and granted him the title of "venerable." After his beatification, the late pontiff will be known as "blessed."In Joan Millan\'s eyes, he already is."John Paul II always demonstrated intelligence and strong character," said Millan, 49, who was visiting St. Peter\'s Square on Friday with his wife, Patricia Rangel, from their home in Barcelona, Spain. "He also made a huge effort to unify all the churches. He was a man a step ahead of the others."This doesn\'t make him a saint," Millan said. "But it definitely helped." Today, Pope John Paul II is one step closer to becoming a saint. John Paul\'s beatification -- a stage in which an individual is known as "blessed," and one of the final steps before obtaining sainthood -- has been scheduled for May 1. He reached that stage of the Catholic Church\'s sainthood process after Pope Benedict XVI officially agreed there was evidence that John Paul had performed a miracle.The miracle supposedly occurred after a French nun, Sister Marie Simon-Pierre, prayed to the late pope and subsequently recovered from Parkinson\'s disease.So how, exactly, does the sainthood process work? Surge Desk breaks it down.This is a stringent process, so the person who dies must have lived an exemplary life, such as Pope John Paul II apparently did.The process of becoming a saint, or canonization, is long, involved and even quite expensive, so the people who are pushing for someone\'s sainthood must organize themselves. They will work with the Catholic Church and collect evidence, information and money, as necessary, during the proceedings.Someone must officially nominate the deceased person for sainthood.The bishop of the diocese of the person who died initiates an investigation into the nomination. This investigation looks into the life the person led, including whether he or she lived a virtuous life and whether there is any preliminary evidence of miracles performed. The investigation will also look closely at the individual\'s writings to see what they say about the person\'s faith in God and relationship with the Catholic Church.The findings from the investigation go to the Congregation of Rites in Rome, the Vatican organization that handles sainthood nominations. From here, the Congregation of Rites conducts its own investigation into the person\'s background, writings, alleged miracles, etc. The members of the congregation hold a debate, twice a month, over the person\'s qualifications for sainthood. If they decide to move forward, they issue a decree announcing so.The candidate for sainthood must be credited with performing at least one miracle (unless he or she was a martyr, in which case this process is a bit different). The Catholic Church investigates whether the miracle took place, with pretty stringent guidelines on what constitutes a miracle. The pope must sign off on the miracle.If a miracle is certified, the person becomes beatified. Now, he or she is known as "blessed."To move forward in the process, another miracle is required, with the same steps as before.Once there is proof of a second miracle, the person can be canonized, officially becoming a saint.']

# Summary: – Pope John Paul II will be beatified May 1, the Vatican announced today. The news follows Pope Benedict's determination that his predecessor performed a miracle when a nun was cured of Parkinson’s disease after praying to him, the Los Angeles Times reports. It’s been a speedy road toward sainthood for the late pope; Benedict opened the process weeks after his predecessor died nearly six years ago. A second verified miracle is required before he can be canonized; click to see the entire process of becoming a saint.

instruction = """
Generate the required text using the list of key ideas and concepts provided below:
MMR: {text}

Summary: 
"""
#Loading and setting the system prompt
template = get_prompt(instruction, system_prompt)
prompt = PromptTemplate(template=template, input_variables=["text"])
llm_chain = LLMChain(prompt=prompt, llm=llm)
print(template)

df = pd.read_csv(args.file_path)
df['mmr'] = df['mmr'].apply(eval)
# df['mmr_length'] = df['mmr'].apply(len)
# N = 50
# filtered_df = df[df['mmr_length'] > N]
# print(filtered_df.head())

# df = filtered_df
df['generated_text'] = ""  # Adding a new column to the dataframe

with open(args.new_file_save_path, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)    
    csvwriter.writerow(df.columns)
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Rows"):
        try:
            generated_output = llm_chain.run(row['mmr'])
        except:
            generated_output = ""

        row['generated_text'] = generated_output
        csvwriter.writerow(row)
    
#python generate_summary.py --model_id "mistralai/Mistral-7B-Instruct-v0.1" --file_path "sample_test_0.2_mmr.csv" --new_file_save_path "sample_test_0.2_mmr_generated.csv"


# hey = "
