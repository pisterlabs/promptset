import openai
import codecs

openai.api_key = 'sk-'

poems_only = ''
poems_humor = ''
poems_humor_quatrain = ''


#Usar o modelo
def gpt3(sample, ft_model):
    
    res = openai.Completion.create(model=ft_model, 
                                prompt=sample + '\n\n###\n\n', 
                                top_p =0.9, 
                                temperature=0.8,
                                max_tokens=300,
                                stop=["  END "])

    output = res['choices'][0]['text']
    return output


#Main
if __name__ == "__main__":

    #20 tests samples
    prompts = []
    prompts.append("""
    Claim:  Ukraine was responsible for the Kramatorsk train station bombing.  
    Label: false 
    Summary Explanation: 
    There’s no credible evidence that Ukraine was behind the April 8 attack at the Kramatorsk train station. A video used to bolster this claim is fake — it did not come from the BBC.
    The claim has largely been spread by pro-Kremlin accounts following reports of civilian casualties and contradict earlier Russian posts that initially took credit for the bombing.
    The Tochka-U missile used in the attack, and the serial number on it, isn’t proof that it came from the Ukrainian army. Several news reports, legitimate photos and videos show that Russia has used these missile systems recently.
 
    Poem:
    """)
    prompts.append("""
    Claim:  Ukrainian fighter-ace known as the Ghost of Kyiv s real name is Samuyil Hyde.  
    Label: pants-fire 
    Summary Explanation: 
    Sam Hyde is a comedian whose name has been used as an online meme since 2015, linked to tragic events throughout the world.
    There is no evidence that the mysterious “Ghost of Kyiv” pilot in Ukraine actually exists.
    If the pilot does exist, it’s not Samuyil Hyde.
    
    Poem:
    """)
    prompts.append("""
    Claim: Bill Gates hatches 'horribly stupid' plan to block out the sun 
    Label: false 
    Summary Explanation: 
    Harvard researchers proposed a small-scale experiment that would spray aerosols into the atmosphere to reflect some sunlight back into space, a technology aimed at minimizing global warming. The experiment would measure the risks and effectiveness of the technology.
    The experiment is funded in part through Harvard’s Solar Geoengineering Research Program, and Bill Gates is among the donors to that program.
    The claim first surfaced in 2021 and has been previously debunked. 
    Poem:
    """)
    prompts.append("""
    Claim:  COVID spelled backward is divoc which means possession of the evil spirit in Hebrew.  
    Label: false 
    Summary Explanation: 
    “Divoc” is not a Hebrew word.
    The closest existing word in the language is “dibbuk,” which means evil possession, but the two are not the same, linguistic experts said.
    COVID-19 is an abbreviation of "coronavirus disease 2019.” "CO" stands for "corona," "VI" stands for "virus" and "D" stands for "disease." The number 19 reflects the year it was identified — 2019.
    
    Poem:
    """)
    prompts.append("""
    Claim:  Mask mandates on children lead to learning loss that harms early childhood development.  
    Label: barely-true 
    Summary Explanation: 
    Public health officials pointed out the article cited as support was “not a study but an opinion paper.”
    According to the U.S. Centers for Disease Control and Prevention, limited available data indicates “no clear evidence that masking impairs emotional or language development in children.”
    
    Poem:
    """)
    prompts.append("""
    Claim:  COVID-19 surges among most vaxxed communities, says Harvard study.  
    Label: half-true 
    Summary Explanation: 
    A study led by a Harvard researcher found high rates of infection in communities with high rates of vaccination.
    It concluded that prevention measures such as handwashing, safe distancing and testing should be employed along with vaccinations to fight the epidemic.
    
    Poem:
    """)
    prompts.append("""Claim:  It would cost $20 billion to end homelessness in the U.S. and halting global warming would cost $300 billion.  
    Label: barely-true 
    Summary Explanation: 
    It would likely cost significantly more than $20 billion to house America’s homeless population, after factoring in the expansion of the federal housing voucher program and affordable housing development.
    Since models to predict the cost of fighting climate change rely on many different assumptions, estimates vary widely. However, $300 billion is on the very low end of the spectrum. ​
    
    Poem:    
    """)
    prompts.append("""
    Claim:  Says spending Democrats is driving major jumps in car rentals, gas prices and hotel prices this summer  
    Label: barely-true 
    Summary Explanation: 
    Inflation has gone up since the start of the coronavirus pandemic -- about 5% between May 2020 and May 2021
    But the jump in inflation doesn’t account for the rapidly increasing prices for travel
    In fact, everyday consumers are driving prices higher as more and more people book trips
    
    Poem:
    """)
    prompts.append("""
    Claim:  It makes no sense to require vaccinations for the previously infected.  
    Label: false 
    Summary Explanation: 
    People who have had COVID-19 do have some level of immunity; however, the CDC recommends vaccination because it provides better protection.
    Naturally acquired immunity may not be as effective against variant strains of the virus that cause COVID-19.
    
    Poem:
    """)
    prompts.append("""
    Claim:  An email from Dr. Anthony Fauci shows everyone was lied to about wearing masks to prevent the spread of COVID-19.  
    Label: false 
    Summary Explanation: 
    The email from Fauci is from February 2020, when masks weren’t widely recommended to fight COVID-19. By April 2020, Fauci and other leading public health authorities recommended mask wearing.
    
    Poem:
    """)
    prompts.append("""
    Claim:  Map shows states where all of the children… are back to living mask-free, normal lives.  
    Label: barely-true 
    Summary Explanation: 
    Though the majority of states do not have statewide school mask mandates, individual school districts still have mandates. 
    The absence of state mask rules has not meant a full return to pre-pandemic “normal” at school in most places.
    
    Poem:
    """)
    prompts.append("""
    Claim:  Video suggests Dr. Anthony Fauci said vaccines don’t protect against COVID-19.  
    Label: false 
    Summary Explanation: 
    The video shows real clips of Dr. Anthony Fauci, but misleading voiceovers have been added.
    Some confusion stems from Fauci’s distinction between the disease, COVID-19, and the SARS-CoV-2 virus, which causes the disease.
    Research shows that the COVID-19 vaccines are effective at protecting people from getting the disease, but it is still unclear whether or not the vaccines will prevent people from becoming infected with the SARS-CoV-2 virus and transmitting it to others.
    
    Poem:
    """)
    prompts.append("""
    Claim:  There is over a trillion dollars of money unspent from previous (coronavirus) relief bills ... still sitting in a bank account.  
    Label: half-true 
    Summary Explanation: 
    About $1 trillion remains to be disbursed, but a large share of it has been allocated or is scheduled to be spent.
    Most of the unspent money was approved in late December, and it takes time to distribute it.
    As the economy recovers, some of the money might not be spent.
    
    Poem:
    """)
    prompts.append("""
    Claim:  Says Joe Biden has issued more executive fiats than anyone in such a short period of time, ever, more than Obama, more than Trump, more than anyone.  
    Label: true 
    Summary Explanation: 
    The total number of presidential directives signed by Biden during his first two days in office is greater than the number signed by Trump and Obama over the same period.
    The available record before FDR isn’t complete enough to definitively say whether Rubio is also right about all presidents, but experts said it’s likely that his claim still holds.
    It’s worth noting that Biden issued multiple orders directed at a specific issue that’s actively affecting the entire nation: the coronavirus pandemic.
    
    Poem:
    """)
    prompts.append("""
    Claim:  Federal aid for state and local governments through a proposed coronavirus relief bill would amount to half of what the Republicans put onto the national debt.  
    Label: mostly-true 
    Summary Explanation: 
    Pelosi said that state and local aid in the Democrats’ proposed coronavirus relief bill would cost half as much as the Republicans’ 2017 tax bill.
    State and local aid proposed by the Heroes Act would cost about half as much as the tax bill.
    The statement doesn’t tell the full picture about what has caused increases in the national debt since 2017.
    
    Poem:
    """)
    prompts.append("""
    Claim:  People are starting to enter ERs with fungal lung infections from wearing masks!!  
    Label: false 
    Summary Explanation: 
    A Facebook post claims people are checking into ERs due to fungal lung infections caused by wearing masks meant to prevent COVID-19.
    Regional, statewide and national emergency physician groups say they haven't seen any reports to support this.
    Health experts say there’s no evidence that wearing standard masks, such as surgical masks or ones made of fabric, is harmful to the general public.
    Experts recommend wearing face coverings to slow the coronavirus.
    They also say it’s important to wash your hands and to wash reusable masks regularly.
    
    Poem:
    """)
    prompts.append("""
    Claim:  Research illustrates a clear correlation between vitamin D deficiencies and (higher) COVID-19 mortality rates.  
    Label: true 
    Summary Explanation: 
    U.S. Rep. Glenn Grothman touted a correlation between vitamin D deficiencies and COVID-19 deaths, suggesting the supplement may help combat the virus.
    Early research verifies that correlation, but more work needs to be done.
    Grothman emphasized that point and didn’t imply vitamin D is a cure for COVID-19.
    
    Poem:
    """)
    prompts.append("""
    Claim:  On key issues including taxes, health care, the Green New Deal, abortion, and guns, there is no daylight between Bernie Sanders and Joe Biden.  
    Label: half-true 
    Summary Explanation: 
    The Republican National Committee’s email accurately portrays the policy similarities between Biden and Sanders on several issues, including abortion, the Green New Deal, the death penalty, an assault-weapons ban, and detention of undocumented immigrants.
    However, the email overlooks notable differences between Biden and Sanders on other issues, including tax policy, free college tuition, and two aspects of their health insurance plan. It also incorrectly describes the position of both candidates on reparations for slavery.
    
    Poem:
    """)
    prompts.append("""
    Claim:  Polar bears have increased 400% in 45 years; whales are nearly fully recovered; extinctions are down 90% past century. Koalas are doing fine. If we could ban wind turbines we could save 85,000 birds of prey/yr in US alone.  
    Label: false 
    Summary Explanation: 
    Global estimates of the polar bear population show either a modest decline or virtually no change. Before 1973 they were at threat of being over-hunted, and today they face climate change and ice loss.
    Whale populations are not close to recovering fully. Experts said their numbers will never return to what they once were before whale hunting was regulated. Today, they face both climate change and other manmade threats.
    Experts and scientific data show that koalas are vulnerable due to elevated carbon dioxide levels, predatory animals, bush fires and more.
    
    Poem:
    """)
    prompts.append("""
    Claim:  Four kids who took the coronavirus vaccine died immediately.  
    Label: false 
    Summary Explanation: 
    There is no evidence that children have died because of a COVID-19 vaccine.
    No vaccine currently in development has been approved for widespread public use.
    There is no evidence the vaccines in development will contain microchips.
    
    Poem:
    """)


    #Generation
    i = 0
    while i < 20:

        res = gpt3(prompts[i],poems_only)

        f = codecs.open(str(i) +'.txt', 'w', 'utf-8')
        f.write(res)
        f.close()

print(f'Je suis done')