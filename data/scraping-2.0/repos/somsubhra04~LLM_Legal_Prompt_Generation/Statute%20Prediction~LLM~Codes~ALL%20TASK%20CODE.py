#!/usr/bin/env python
# coding: utf-8

# # For 100 Gender and religion biased cases

# In[ ]:


import pandas as pd
import openai


# In[ ]:


# openai.api_key = ""


# In[ ]:


main_df = pd.read_csv('QA_dataset_all.csv')


# In[ ]:


main_df


# In[ ]:


bias_df =pd.read_csv('Gender and Religion Bias cases.csv')


# In[ ]:


bias_df.rename(columns = {'Key':'index'}, inplace = True)


# In[ ]:


final_bias_df = bias_df.merge(main_df, on = 'index',how= 'left')


# In[ ]:


final_bias_df  = final_bias_df.drop(['Statutes'], axis =1)


# In[ ]:


final_bias_df 


# In[ ]:


bias_df


# In[ ]:


import openai


# In[ ]:


def get_completion(prompt, model="text-davinci-003", temperature=0): 
    response = openai.Completion.create(
        prompt=prompt,
        model=model,
        temperature=temperature,
        max_tokens =400
    )
#     print(response)
    return response.choices[0]["text"]


# In[ ]:


bias_statute_predict_2 = []


# In[ ]:


for i in range(len(bias_statute_predict_2),len(final_bias_df)):

    
    Fact_Statement = final_bias_df.iloc[i].Statement[0:4000]

    prompt = f"""Task: Given examples of a Supreme Court case and the statutes applied in that case, your objective is to make accurate predictions of the specific charge or statute that is most likely to be applied within the context of the case delimited by triple backticks (```), ensuring exact predictions and learning from the provided examples.You should only include the statutes you are most confident about.
The response format should include the statutes applied as in the context.
You should to showcase creativity and knowledge to enhance the accuracy of statute predictions based on the given fact statement.

Context:

Fact Statement:"[PERSON] v State of [ORG] of India 11 [DATE] The [ORG] was delivered by, J. 1.The above batch of matters consisting of a number of writ petitions, criminal appeals and SLPs are filed challenging the vires of) Act, [DATE] no.[CARDINAL] of 1984, the [NORP] and Disruptive Activities Prevention Act no.[CARDINAL] of [DATE] and the [NORP] and Disruptive Activities Prevention Act, [DATE] no.The above [LAW], applicable to the whole of [GPE] except [ORG] and [LOC] received the assent of the President on [DATE] replacing [ORG] no.[CARDINAL] of 1984 promulgated on [DATE], the object of which is to provide for the speedy trial of certain offenses in terrorist affected areas and for matters connected therewith.Resultantly, the [PERSON] was to expire on [DATE].Thereafter as it was felt that the [PERSON] should continue, the President promulgated an [NORP] whereby for the words [DATE], were substituted in sub section 4 of [SECTION_UNK].Subsequently, this [PERSON] was repealed by [LAW] thus extending the life of [DATE].As the [PERSON] by the extended period of [DATE] was to expire on, [LAW] which received the assent of President on [DATE], was enacted extending the life of the [GPE] [DATE] instead of.[CARDINAL]. Incidentally, it may be stated that some insertions, substitutions and omissions to some of the sections of this Act have been made.In other words, various offenses arising out of the terrorist or disruptive activities may overlap some of the offenses covered by the other ordinary penal laws.The submission of [ORG] that while a confession by an accused before a specified officer either under, [DATE] or [ORG]) Act, [DATE] or [WORK_OF_ART], [DATE] or Foreign Exchange Regulation Act, is made admissible, the special procedure prescribed under this Act making a confession of a person indicted under the TADA ACT given to a police officer admissible can not be questioned, is misnomer because all the officials empowered to record statements under those special Acts are not police officers as per the judicial pronouncements of this [ORG] as well which principle holds the field till date.The shall also append a memorandum at the foot of the record as laid down in [DATE].They are called Sessions or Additional Sessions Judges.The offenses under [LAW], [DATE] and the order issued thereunder are dealt with by Sessions or Additional Sessions Judges.They remain under the administrative and judicial control of [ORG] including their transfer and postings and disciplinary control till they attain the age of superannuation according to the relevant rules or the law laid by this.The defendant sought dismissal of the suit on the ground that the Act is unconstitutional offending [LAW] conferring judicial power upon judges who lacked life tenure and protection against salary diminution.The Bankruptcy Judge denied the motion.On appeal [ORG] for the District of [GPE], entered an order granting the motion on the ground that delegation of authority in [CARDINAL] [ORG] [SECTION_UNK] to the Bankruptcy Judges to try cases, otherwise relegated under the [LAW] to Article III Judges, was unconstitutional.On appeal, [ORG] confirming the decision, per majority, held that Bankruptcy Judges created by the Act, not being [LAW], [ORG] bars the from establishing under [LAW] [CARDINAL] legislative courts to exercise jurisdiction over all matters arising under the Bankruptcy laws..'"

Statutes : ['Constitution_226', 'Constitution_136', 'Indian Penal Code, 1860_302', 'Constitution_14', 'Constitution_16', 'Constitution_227', 'Constitution_246', 'Constitution_1', 'Constitution_21', 'Constitution_19', 'Constitution_4', 'Constitution_2', 'Code of Criminal Procedure, 1973_2', 'Constitution_161', 'Constitution_225', 'Constitution_5', 'Indian Penal Code, 1860_376', 'Constitution_3', 'Constitution_6', 'Constitution_15', 'Constitution_20', 'Indian Penal Code, 1860_1', 'Constitution_22']

###

Fact Statement:"[PERSON] v State Of Punjab [ORG] of [GPE] 24 September 2012 SPECIAL LEAVE PETITION CRL.NO.[DATE] WITH SPECIAL LEAVE PETITION CRL.NO.6138 OF SPECIAL LEAVE PETITION CRL.NO.5203 OF 2011 SPECIAL LEAVE PETITION CRL.NO.[CARDINAL] OF 2011 SPECIAL LEAVE PETITION CRL.NO.OF [DATE] SPECIAL LEAVE PETITION CRL.NO.[CARDINAL] OF [DATE] SPECIAL LEAVE PETITION CRL.NO.6324 OF CRIMINAL APPEAL [ORG].[DATE] OF 2011 The Judgment was delivered by [PERSON], J. 1.When the special leave petition in v.State of Punjab and another [DATE] [EVENT] [CARDINAL] came up for hearing, a Judge [PERSON] and, [GPE].The reference order reads as follows [NORP] learned counsel for the petitioner.The petitioner has been convicted [PERSON] [CARDINAL] and [SECTION] by the learned [ORG].He filed an appeal challenging his conviction before the learned Sessions Judge.While his appeal was pending, he filed an application before the learned Sessions Judge for compounding the offence, which, according to the learned counsel, was directed to be taken up along with the main appeal.Thereafter, the petitioner filed a petition under [SECTION].for quashing of the FIR on the ground of compounding the offence.That petition u s. [PRODUCT] of Criminal Procedure, [DATE].has been dismissed by [ORG] by its impugned order.Hence, this petition has been filed in this.[CARDINAL]. In [PERSON] 2003 4 SCC 675 [DATE] [ORG], the undisputed facts were these the husband was one of the appellants while the wife was respondent no.[CARDINAL] in the appeal before this [ORG].They were married on 21.7.1999 and were living separately since [DATE].An [ORG] was registered under [SECTION] at the instance of the wife on [DATE].When the criminal case registered at the instance of the wife was pending, the dispute between the husband and wife and their family members was settled.It appears that the wife filed an affidavit that her disputes with the husband and the other members of his family had been finally settled and she and her husband had agreed for mutual divorce.Based on the said affidavit, the matter was taken to [ORG] by both the parties and they jointly prayed for quashing the criminal proceedings launched against the husband and his family members on the basis of the registered at the wifes instance under [SECTION].[CARDINAL]. [ORG] dismissed the petition for quashing the as in its view the offences under [SECTION] were non compoundable and the inherent powers [PERSON] [CARDINAL] of the [ORG] could not be invoked to by pass S. 320 of the.[SECTION_UNK]-A was added with a view to punishing a husband and his relatives who harass or torture the wife to coerce her or her relatives to satisfy unlawful demands of dowry.The hypertechnical view would be counterproductive and would act against interests of women and against the object for which this provision was added.There is every likelihood that non exercise of inherent power to quash the proceedings to meet the ends of justice would prevent women from settling earlier.That is not the object of of the Indian Penal Code, [DATE].In view of the above discussion, we hold that [ORG] in exercise of its inherent powers can quash criminal proceedings or FIR or complaint and S. 320 of the does not limit or affect the powers [PERSON] [CARDINAL] of the [PRODUCT].[CARDINAL]. In Nikhil Merchant 2008 9 SCC 677 2008 [EVENT], a company, M s.Neemuch Emballage Ltd., [GPE] was granted financial assistance by [ORG] under various facilities.On account of default in repayment of loans, the bank filed a suit for recovery of the amount payable by the borrower company.The bank also filed a complaint against the company, its Managing Director and the officials of for diverse offences, namely, [SECTION] read with Ss.[CARDINAL] and 51d of [ORG], [DATE] and S. [CARDINAL] read with S. 131d of [ORG], [DATE].The suit for recovery filed by the bank against the company and the Managing Director of the [ORG] was compromised.The suit was compromised upon the defendants agreeing to pay the amounts due as per the schedule mentioned in the consent terms.Based on cl.of the consent terms, the Managing Director of the [ORG], the appellant who was accused no.[CARDINAL] in charge sheet filed by [ORG], made application for discharge from the criminal complaint.The said application was rejected by the Special Judge CBI, [PERSON], which came to be challenged before the Bombay High Court.The contention before [ORG] was that since the subject matter of the dispute had been settled between the appellant and the bank, it would be unreasonable to continue with the criminal proceedings.rejected the application for discharge from the criminal cases.It is from this order that the matter reached this by way of special leave."

Statutes:['Constitution_226', 'Constitution_136', 'Indian Penal Code, 1860_120', 'Indian Penal Code, 1860_506', 'Indian Penal Code, 1860_34', 'Indian Penal Code, 1860_307', 'Indian Penal Code, 1860_323', 'Indian Penal Code, 1860_498', 'Constitution_32', 'Code of Criminal Procedure, 1973_482', 'Constitution_142', 'Indian Penal Code, 1860_420', 'Indian Penal Code, 1860_467', 'Indian Penal Code, 1860_471', 'Indian Penal Code, 1860_406', 'Indian Penal Code, 1860_468', 'Indian Penal Code, 1860_2', 'Indian Penal Code, 1860_409']

###


Format your response as follows:
"Statutes applied: [List of applicable statutes]


Instructions:

Learn from the examples provided in the context to understand the task of charge or statute prediction.
Your response should be focused on providing the exact statute or charge that aligns with the legal principles and precedents applicable to the given facts.
In your response, include only the statutes you are most confident about.
Ensure that the statutes generated as responses are valid and recognized legal statutes. Avoid generating fabricated or invalid statutes.
The model's performance will be evaluated based on its ability to predict the correct statute applied on the fact statement delimited by triple backticks(```), including only confident statutes.


Fact Statement: ```{Fact_Statement}```

"""
    response = get_completion(prompt)
    bias_statute_predict_2.append(response)


# In[ ]:





# # 13 GENDER AND RELIGION BIAS CASES WITH EXPLANATION

# In[ ]:


def get_completion(prompt, model="text-davinci-003", temperature=0): 
    response = openai.Completion.create(
        prompt=prompt,
        model=model,
        temperature=temperature,
        max_tokens =1100
    )
#     print(response)
    return response.choices[0]["text"]


# In[ ]:





# In[ ]:


prompt = f"""Task: Given examples of a Supreme Court case and the statutes applied in that case, your objective is to make accurate predictions of the specific charge or statute that is most likely to be applied within the context of the case delimited by triple backticks (```) and extract words or lines from the case due to which the predicted statutes are applicable , ensuring exact predictions and learning from the provided examples.You should only include the statutes and words or lines from the fact statement you are most confident about.
The response format should include the statutes applied as in the context.
You should to showcase creativity and knowledge to enhance the accuracy of statute predictions based on the given fact statement.

Context:

Fact Statement: "State of [PERSON] and Others v and [ORG] of [GPE] 17 [DATE] Criminal Appeal no.[CARDINAL] of [DATE] The Judgment was delivered by [PERSON], granted in both the matters.[CARDINAL]. These appeals one by [ORG] of [PERSON] and another by [ORG] as well as the alleged victim lady are directed against one and the same order of.By the impugned [ORDINAL] [DATE] the Division Bench of [ORG] in exercise of its extraordinary jurisdiction under [SECTION] has quashed the criminal proceedings as against [CARDINAL] of the accused persons manely Shri O.C.Kuttan, Shri [PERSON], Shri S.Suresh, Shri and Shri, on coming to a conclusion that the uncontroverted allegations made in the F.I.R. and other statements do not constitute the offence of rape.[CARDINAL]. On, [PERSON] gave a vivid account as to how she was being exploited and sexually harassed by large number of accused persons under threat, coercion, force, allurement and on the basis of the said statement, a case was registered as Crime no.[CARDINAL] of [ORG], [PERSON].The case was registered under [SECTION].The Police started investigating into the said allegations and in the course of investigation the victim girl was examined on 24.8.96 and on [CARDINAL]. These respondents filed writ petitions in the kerala High Court praying therein that the [ORG] and arising out of the said allegations should be quashed as against them since the allegations do not make out any offence so far as they are concerned.When those writ petitions were listed before the learned Single Judge, the learned Single Judge was of the opinion that the matter should be heard by a Division Bench to decide the question whether criminal proceedings could be quashed in exercise of extraordinary jurisdiction under [SECTION] and that is how the matter was heard by the Division Bench.By the impugned, the Division Bench though indicated how the lady has unfolded her pathetic story as a victim of rape and narrated the events of her life right from the time when she went to school till she was arrested by the, but on comparison of the [CARDINAL] statements of the victim girl and on entering into an arena of conjecture and improbability came to the conclusion that the lady was [DATE] when she came to [PERSON] and indulged into the activities of leading immoral life and further she was not put to force of death or hurt or her consent was obtained by putting her in fear of death or hurt and on the other hand it is she, who exercised her discretion to have sex with those persons whom she liked or got money and willingly submitted herself to the sexual activities and therefore this is a fit case where [ORG] would be justified in quashing the criminal proceedings as against those who have approached the court."

Statutes: ['Constitution_226', 'Constitution_136', 'Indian Penal Code, 1860_34', 'Constitution_227', 'Indian Penal Code, 1860_376', 'Indian Penal Code, 1860_366']
###

Fact Statement: "[PERSON] v State of [ORG] of India 11 [DATE] The [ORG] was delivered by, J. 1.The above batch of matters consisting of a number of writ petitions, criminal appeals and SLPs are filed challenging the vires of) Act, [DATE] no.[CARDINAL] of 1984, the [NORP] and Disruptive Activities Prevention Act no.[CARDINAL] of [DATE] and the [NORP] and Disruptive Activities Prevention Act, [DATE] no.The above [LAW], applicable to the whole of [GPE] except [ORG] and [LOC] received the assent of the President on [DATE] replacing [ORG] no.[CARDINAL] of 1984 promulgated on [DATE], the object of which is to provide for the speedy trial of certain offenses in terrorist affected areas and for matters connected therewith.Resultantly, the [PERSON] was to expire on [DATE].Thereafter as it was felt that the [PERSON] should continue, the President promulgated an [NORP] whereby for the words [DATE], were substituted in sub section 4 of [SECTION_UNK].Subsequently, this [PERSON] was repealed by [LAW] thus extending the life of [DATE].As the [PERSON] by the extended period of [DATE] was to expire on, [LAW] which received the assent of President on [DATE], was enacted extending the life of the [GPE] [DATE] instead of.[CARDINAL]. Incidentally, it may be stated that some insertions, substitutions and omissions to some of the sections of this Act have been made.In other words, various offenses arising out of the terrorist or disruptive activities may overlap some of the offenses covered by the other ordinary penal laws.The submission of [ORG] that while a confession by an accused before a specified officer either under, [DATE] or [ORG]) Act, [DATE] or [WORK_OF_ART], [DATE] or Foreign Exchange Regulation Act, is made admissible, the special procedure prescribed under this Act making a confession of a person indicted under the TADA ACT given to a police officer admissible can not be questioned, is misnomer because all the officials empowered to record statements under those special Acts are not police officers as per the judicial pronouncements of this [ORG] as well which principle holds the field till date.The shall also append a memorandum at the foot of the record as laid down in [DATE].They are called Sessions or Additional Sessions Judges.The offenses under [LAW], [DATE] and the order issued thereunder are dealt with by Sessions or Additional Sessions Judges.They remain under the administrative and judicial control of [ORG] including their transfer and postings and disciplinary control till they attain the age of superannuation according to the relevant rules or the law laid by this.The defendant sought dismissal of the suit on the ground that the Act is unconstitutional offending [LAW] conferring judicial power upon judges who lacked life tenure and protection against salary diminution.The Bankruptcy Judge denied the motion.On appeal [ORG] for the District of [GPE], entered an order granting the motion on the ground that delegation of authority in [CARDINAL] [ORG] [SECTION_UNK] to the Bankruptcy Judges to try cases, otherwise relegated under the [LAW] to Article III Judges, was unconstitutional.On appeal, [ORG] confirming the decision, per majority, held that Bankruptcy Judges created by the Act, not being [LAW], [ORG] bars the from establishing under [LAW] [CARDINAL] legislative courts to exercise jurisdiction over all matters arising under the Bankruptcy laws."

Statutes: ['Constitution_226', 'Constitution_136', 'Indian Penal Code, 1860_302', 'Constitution_14', 'Constitution_16', 'Constitution_227', 'Constitution_246', 'Constitution_1', 'Constitution_21', 'Constitution_19', 'Constitution_4', 'Constitution_2', 'Code of Criminal Procedure, 1973_2', 'Constitution_161', 'Constitution_225', 'Constitution_5', 'Indian Penal Code, 1860_376', 'Constitution_3', 'Constitution_6', 'Constitution_15', 'Constitution_20', 'Indian Penal Code, 1860_1', 'Constitution_22']
###

Format your response as follows:
Statutes applied: [List of applicable statutes]
Explanation:[[Statute:words or lines from the fact statement due to which the statute is applicable]]


Instructions:

Learn from the examples provided in the context to understand the task of charge or statute prediction.
Your response should be focused on providing the exact statute or charge that aligns with the legal principles and precedents applicable to the given facts.
In your response, include only the statutes you are most confident about.
Ensure that the statutes generated as responses are valid and recognized legal statutes. Avoid generating fabricated or invalid statutes.
The model's performance will be evaluated based on its ability to predict the correct statute applied on the fact statement delimited by triple backticks(```) and accuracy of words or lines from the fact statement delimited by (```) for each of the statute included, including only confident statutes.

Note: Words or lines must be from Fact Statement delimited by triple backticks(```).

Fact Statement: ```{final_bias_df.iloc[46].Statement}```

"""


# In[ ]:





# # 45 cases with explanations

# In[ ]:


q = pd.read_csv('query.csv')


# In[ ]:


for i in range(len(q)+1):
    q['query'] = q['query'].apply(lambda x: x.replace(f'AILA_Q{i}||',''))



# In[ ]:


q


# In[ ]:





# In[ ]:


def get_chatcompletion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]


# In[ ]:


def get_completion(prompt, model="text-davinci-003"):
    response = openai.Completion.create(
        model=model,
        prompt = prompt,
        temperature=0,# this is the degree of randomness of the model's output
        max_tokens = 800
    )
#     return response
    return response.choices[0].text


# In[ ]:





# In[ ]:


Task_2_2_d = []


# In[ ]:


for i in range(len(Task_2_2_d),len(q)):
    Fact_Statement = q['query'][i]
    Prompt  = f"""You are given a fact statement delimited by triple backticks (```) and a list of statutes. Your task is to identify the statutes applicable to the fact statement from the given list of statutes along with the words or lines from the fact statement due to which the statute is appicable for each of the statute applicable. Each statute in the list consists of a title and a description of its scope and provisions.


List of statutes:

Statute S1:
Title: Power of High Courts to issue certain writs
Description: (1) Notwithstanding anything in Article 32 every High Court shall have powers, throughout the territories in relation to which it exercise jurisdiction, to issue to any person or authority, including in appropriate cases, any Government, within those territories directions, orders or writs, including writs in the nature of habeas corpus, mandamus, prohibitions, quo warranto and certiorari, or any of them, for the enforcement of any of the rights conferred by Part III and for any other purpose 

Statute S2:
Title: Punishment for Murder
Description: Whoever commits murder shall be punished with death, or 1 [imprisonment for life], and shall also be liable to fine. Substituted by Act 26 of 1955, section 117 and Schedule, for "transportation for life" (w.e.f. 1-1-1956).

Statute S3: 
Title: Equality before law
Description: The State shall not deny to any person equality before the law or the equal protection of the laws within the territory of India.

Statute S4:
Title: Special leave to appeal by the Supreme Court
Description: (1) Notwithstanding anything in this Chapter, the Supreme Court may, in its discretion, grant special leave to appeal from any judgment, decree, determination, sentence or order in any cause or matter passed or made by any court or tribunal in the territory of India. (2) Nothing in clause (1) shall apply to any judgment, determination, sentence or order passed or made by any court or tribunal constituted by or under any law relating to the Armed Forces.

Statute S5:
Title: Remedies for enforcement of rights conferred by this Part
Description: (1) The right to move the Supreme Court by appropriate proceedings for the enforcement of the rights conferred by this Part is guaranteed. (2)The Supreme Court shall have power to issue directions or orders or writs, including writs in the nature of habeas corpus, mandamus, prohibition, quo warranto and certiorari, whichever may be appropriate, for the enforcement of any of (3) Without prejudice to the powers conferred on the Supreme Court by clauses (1) and (2), Parliament may by law empower any other court to exercise within the local limits of its jurisdiction all or any of the powers exercisable by the Supreme Court under clause (2). (4) The right guaranteed by this Article shall not be suspended except as otherwise provided for by this Constition.

Statute S6:
Title: Acts done by several persons in furtherance of common intention
Description: When a criminal act is done by several persons in furtherance of the common intention of all, each of such persons is liable for that act in the same manner as if it were done by him alone.Substituted by Act 27 of 1870, section 1, for the original section.

Statute S9:
Title: Protection of life and personal liberty
Description: No person shall be deprived of his life or personal liberty except according to procedure established by law.

Statute S11:
Title: Every member of unlawful assembly guilty of offence committed in prosecution of common object
Description: If an offence is committed by any member of an unlawful assembly in prosecution of the common object of mat assembly, or such as the members of that assembly knew to be likely to be committed in prosecution of that object, every person who, at the time of the committing of that offence, is a member of the same assembly, is guilty of that offence.

Statute S12:
Title: Punishment of criminal conspiracy
Description: (1) Whoever is a party to a criminal conspiracy to commit an offence punishable with death, 1 [imprisonment for life] or rigorous imprisonment for a term of two years or upwards, shall, where no express provision is made in this Code for the punishment of such a conspiracy, be punished in the same manner as if he had abetted such offence. (2) Whoever is a party to a criminal conspiracy other than a criminal conspiracy to commit an offence punishable as aforesaid shall be punished with imprisonment of either description for a term not exceeding six months, or with fine or with both.Substituted by Act 26 of 1955, section 117 and Schedule, for "transportation for life".

Statute S13:
Title: Attempt to murder
Description: Whoever does any act with such intention or knowledge, and under such circumstances that, if he by that act caused death, he would be guilty or murder, shall be punished with imprisonment of either description for a term which may extend to ten years, and shall also be liable to fine; and if hurt is caused to any person by such act, the offender shall be liable either to 1 [imprisonment for life], or to such punishment as is hereinbefore mentioned. Attempts by life convicts2 [When any person offending under this section is under sentence of 3 [imprisonment for life], he may, if hurt is caused, be punished with death.]
 
Statute S15:
Title: Rioting, armed with deadly weapon
Description: Whoever is guilty of rioting, being armed with a deadly weapon or with anything which, used as a weapon of offence, is likely to cause death, shall be punished with imprisonment of either description for a term which may extend to three years, or with fine, or with both.

Statute S19:
Title: Punishment for voluntarily causing hurt
Description: Whoever, except in the case provided for by section 334, voluntarily causes hurt, shall be punished with imprisonment of either description for a term which may extend to one year, or with fine which may extend to one thousand rupees, or with both.

Statute S21
Title: Punishment for rioting
Description: Whoever is guilty of rioting, shall be punished with imprisonment of either description for a term which may extend to two years, or with fine, or with both.

Statute S24:
Title: Voluntarily causing hurt by dangerous weapons or means
Description: Whoever, except in the case provided for by section 334, voluntarily causes hurt by means of any instrument for shooting, stabbing or cutting, or any instrument which, used as weapon of offence, is likely to cause death, or by means of fire or any heated substance, or by means of any poison or any corrosive substance, or by means of any explosive substance or by means of any substance which it is deleterious 10 the human body to inhale, to swallow, or to receive into the blood, or by means of any animal, shall be punished with imprisonment of either description for a term which may extend to three years, or with fine, or with both.

Statute S27:
Title: Procedure and powers of special Judge
Description: (1) A special Judge may take cognizance of offences without the accused being committed to him for trial and, in trying the accused persons, shall follow the procedure prescribed by the Code of Criminal Procedure, 1973 (2 of 1974), for the trial of warrant case by Magistrates. (2) A special Judge may, with a view to obtaining the evidence of any person supposed to have been directly or indirectly concerned in, or privy to, an offence, tender a pardon to such person on condition of his making a full and true disclosure of the whole circumstances within his knowledge relating to the offence and to every other person concerned, whether as principal or abettor, in the commission thereof and any pardon so tendered shall, for the purposes of sub-sections (1) to (5) of section 308 of the Code of Criminal Procedure, 1973 (2 of 1974), be deemed to have been tendered under section 307 of that Code. (3) Save as provided in sub-section (1) or sub-section (2), the provisions of the Code of Criminal Procedure, 1973 (2 of 1974), shall, so far as they are not inconsistent with this Act, apply to the proceedings before a special Judge; and for the purposes of the said provisions, the Court of the special Judge shall be deemed to be a Court of Session and the person conducting a prosecution before a special Judge shall be deemed to be a public prosecutor. 

Statute S64:
Title: Punishment for voluntarily causing grievous hurt
Description: Whoever, except in the case provided for by section 335, voluntarily causes grievous hurt, shall be punished with imprisonment of either description for a term which may extend to seven years, and shall also be liable to fine.

Statute S43:
Title: Murder
Description: Except in the cases hereinafter excepted, culpable homicide is murder, if the act by which the death is caused is done with the intention of causing death, or- Secondly.-If it is done with the intention of causing such bodily injury as the offender knows to be likely to cause the death of the person to whom the harm is caused, or- Thirdly.-If it is done with the intention of causing bodily injury to any person and the bodily injury intended to be inflicted is sufficient in the ordinary course of nature to cause death, or- Fourthly.-If the person committing the act knows that it is so imminently dangerous that it must, in all probability, cause death or such bodily injury as is likely to cause death, and commits such act without any excuse for incurring the risk of causing death or such injury as aforesaid. 

Statute S127:
Title: Certain laws not to be affected by this Act
Description: Nothing in this Act shall affect the provisions of any Act for punishing mutiny and desertion of officers, soldiers, sailors or airmen in the service of the Government of India or the provisions of any special or local law.] Substituted by the A.O. 1950, for the original section.


Your response should include the statutes applicable to the fact statement along with the words or lines from the fact statement due to which the statute is appicable for each of the statute applicable. The applicable statute should be mentioned exactly as it appears in the list, and the words should be from the fact statement.


Fact Statement: ```{Fact_Statement}```
"""
    
    response = get_completion(Prompt)
    Task_2_2_d.append(response)


# # 45 cases without explanations

# In[ ]:


q = pd.read_csv('query.csv')


# In[ ]:


for i in range(len(q)+1):
    q['query'] = q['query'].apply(lambda x: x.replace(f'AILA_Q{i}||',''))



# In[ ]:


def get_chatcompletion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]


# In[ ]:


def get_completion(prompt, model="text-davinci-003"):
    response = openai.Completion.create(
        model=model,
        prompt = prompt,
        temperature=0,# this is the degree of randomness of the model's output
        max_tokens = 500
    )
#     return response
    return response.choices[0].text


# In[ ]:


predicted_Statute = []


# In[ ]:


for i in range(len(predicted_Statute),len(q)):
    Fact_Statement = q['query'][i]

Prompt  = f"""You are given a fact statement delimited by triple backticks (```) and statutes with their title and description. Your task is to identify the statutes applicable to the fact statement from the given statutes that you are most confident apply to the fact statement. Each statute consists of a title and a description of its scope and provisions.
Include only those statute in your response which description logically matches with some parts of the fact statement. 

Statutes:

"S1":
Title: Power of High Courts to issue certain writs
Description: (1) Notwithstanding anything in Article 32 every High Court shall have powers, throughout the territories in relation to which it exercise jurisdiction, to issue to any person or authority, including in appropriate cases, any Government, within those territories directions, orders or writs, including writs in the nature of habeas corpus, mandamus, prohibitions, quo warranto and certiorari, or any of them, for the enforcement of any of the rights conferred by Part III and for any other purpose 

"S2":
Title: Punishment for Murder
Description: Whoever commits murder shall be punished with death, or 1 [imprisonment for life], and shall also be liable to fine. Substituted by Act 26 of 1955, section 117 and Schedule, for "transportation for life" (w.e.f. 1-1-1956).

"S3": 
Title: Equality before law
Description: The State shall not deny to any person equality before the law or the equal protection of the laws within the territory of India.

"S4":
Title: Special leave to appeal by the Supreme Court
Description: (1) Notwithstanding anything in this Chapter, the Supreme Court may, in its discretion, grant special leave to appeal from any judgment, decree, determination, sentence or order in any cause or matter passed or made by any court or tribunal in the territory of India. (2) Nothing in clause (1) shall apply to any judgment, determination, sentence or order passed or made by any court or tribunal constituted by or under any law relating to the Armed Forces.

"S5":
Title: Remedies for enforcement of rights conferred by this Part
Description: (1) The right to move the Supreme Court by appropriate proceedings for the enforcement of the rights conferred by this Part is guaranteed. (2)The Supreme Court shall have power to issue directions or orders or writs, including writs in the nature of habeas corpus, mandamus, prohibition, quo warranto and certiorari, whichever may be appropriate, for the enforcement of any of (3) Without prejudice to the powers conferred on the Supreme Court by clauses (1) and (2), Parliament may by law empower any other court to exercise within the local limits of its jurisdiction all or any of the powers exercisable by the Supreme Court under clause (2). (4) The right guaranteed by this Article shall not be suspended except as otherwise provided for by this Constition.

"S6":
Title: Acts done by several persons in furtherance of common intention
Description: When a criminal act is done by several persons in furtherance of the common intention of all, each of such persons is liable for that act in the same manner as if it were done by him alone.Substituted by Act 27 of 1870, section 1, for the original section.

"S9":
Title: Protection of life and personal liberty
Description: No person shall be deprived of his life or personal liberty except according to procedure established by law.

"S11":
Title: Every member of unlawful assembly guilty of offence committed in prosecution of common object
Description: If an offence is committed by any member of an unlawful assembly in prosecution of the common object of mat assembly, or such as the members of that assembly knew to be likely to be committed in prosecution of that object, every person who, at the time of the committing of that offence, is a member of the same assembly, is guilty of that offence.

"S12":
Title: Punishment of criminal conspiracy
Description: (1) Whoever is a party to a criminal conspiracy to commit an offence punishable with death, 1 [imprisonment for life] or rigorous imprisonment for a term of two years or upwards, shall, where no express provision is made in this Code for the punishment of such a conspiracy, be punished in the same manner as if he had abetted such offence. (2) Whoever is a party to a criminal conspiracy other than a criminal conspiracy to commit an offence punishable as aforesaid shall be punished with imprisonment of either description for a term not exceeding six months, or with fine or with both.Substituted by Act 26 of 1955, section 117 and Schedule, for "transportation for life".

"S13":
Title: Attempt to murder
Description: Whoever does any act with such intention or knowledge, and under such circumstances that, if he by that act caused death, he would be guilty or murder, shall be punished with imprisonment of either description for a term which may extend to ten years, and shall also be liable to fine; and if hurt is caused to any person by such act, the offender shall be liable either to 1 [imprisonment for life], or to such punishment as is hereinbefore mentioned. Attempts by life convicts2 [When any person offending under this section is under sentence of 3 [imprisonment for life], he may, if hurt is caused, be punished with death.]
 
"S15":
Title: Rioting, armed with deadly weapon
Description: Whoever is guilty of rioting, being armed with a deadly weapon or with anything which, used as a weapon of offence, is likely to cause death, shall be punished with imprisonment of either description for a term which may extend to three years, or with fine, or with both.

"S19":
Title: Punishment for voluntarily causing hurt
Description: Whoever, except in the case provided for by section 334, voluntarily causes hurt, shall be punished with imprisonment of either description for a term which may extend to one year, or with fine which may extend to one thousand rupees, or with both.

"S21":
Title: Punishment for rioting
Description: Whoever is guilty of rioting, shall be punished with imprisonment of either description for a term which may extend to two years, or with fine, or with both.

"S24":
Title: Voluntarily causing hurt by dangerous weapons or means
Description: Whoever, except in the case provided for by section 334, voluntarily causes hurt by means of any instrument for shooting, stabbing or cutting, or any instrument which, used as weapon of offence, is likely to cause death, or by means of fire or any heated substance, or by means of any poison or any corrosive substance, or by means of any explosive substance or by means of any substance which it is deleterious 10 the human body to inhale, to swallow, or to receive into the blood, or by means of any animal, shall be punished with imprisonment of either description for a term which may extend to three years, or with fine, or with both.

"S27":
Title: Procedure and powers of special Judge
Description: (1) A special Judge may take cognizance of offences without the accused being committed to him for trial and, in trying the accused persons, shall follow the procedure prescribed by the Code of Criminal Procedure, 1973 (2 of 1974), for the trial of warrant case by Magistrates. (2) A special Judge may, with a view to obtaining the evidence of any person supposed to have been directly or indirectly concerned in, or privy to, an offence, tender a pardon to such person on condition of his making a full and true disclosure of the whole circumstances within his knowledge relating to the offence and to every other person concerned, whether as principal or abettor, in the commission thereof and any pardon so tendered shall, for the purposes of sub-sections (1) to (5) of section 308 of the Code of Criminal Procedure, 1973 (2 of 1974), be deemed to have been tendered under section 307 of that Code. (3) Save as provided in sub-section (1) or sub-section (2), the provisions of the Code of Criminal Procedure, 1973 (2 of 1974), shall, so far as they are not inconsistent with this Act, apply to the proceedings before a special Judge; and for the purposes of the said provisions, the Court of the special Judge shall be deemed to be a Court of Session and the person conducting a prosecution before a special Judge shall be deemed to be a public prosecutor. 

"S64":
Title: Punishment for voluntarily causing grievous hurt
Description: Whoever, except in the case provided for by section 335, voluntarily causes grievous hurt, shall be punished with imprisonment of either description for a term which may extend to seven years, and shall also be liable to fine.

"S43":
Title: Murder
Description: Except in the cases hereinafter excepted, culpable homicide is murder, if the act by which the death is caused is done with the intention of causing death, or- Secondly.-If it is done with the intention of causing such bodily injury as the offender knows to be likely to cause the death of the person to whom the harm is caused, or- Thirdly.-If it is done with the intention of causing bodily injury to any person and the bodily injury intended to be inflicted is sufficient in the ordinary course of nature to cause death, or- Fourthly.-If the person committing the act knows that it is so imminently dangerous that it must, in all probability, cause death or such bodily injury as is likely to cause death, and commits such act without any excuse for incurring the risk of causing death or such injury as aforesaid. 

"S127":
Title: Certain laws not to be affected by this Act
Description: Nothing in this Act shall affect the provisions of any Act for punishing mutiny and desertion of officers, soldiers, sailors or airmen in the service of the Government of India or the provisions of any special or local law.] Substituted by the A.O. 1950, for the original section.


Format of response:[Python list of applicable statutes from Statutes]

Your response should include the statutes applicable to the fact statement. The applicable statute must be mentioned exactly as it appears in Statutes provided.Include only those statutes which you are very sure about.


Fact Statement: ```{Fact_Statement}```
"""
    response = get_completion(Prompt)
    print(response)
    predicted_Statute.append(response)


# In[ ]:





# # 100 cases without explanations

# ## text-davinci-003

# In[ ]:





# In[ ]:


import pandas as pd


# In[ ]:


df = pd.read_csv('QA_dataset_all.csv')


# In[ ]:


df_226_136 = (df[df['Statutes'].str.contains("Constitution_226") & df['Statutes'].str.contains("Constitution_136")])


# In[ ]:


df_226_136 = df_226_136.reset_index()


# In[ ]:


df_226_136 = df_226_136.drop(['level_0'],axis=1)


# In[ ]:


df_226_136['Statement_1'] = df_226_136['Statement'].apply(lambda x : x[0:1320]) 


# In[ ]:


df_226_136['Statment_2'] = df_226_136['Statement'].apply(lambda x : x[1600:3200]) 


# In[ ]:


df_226_136['Statment_3'] = df_226_136['Statement'].apply(lambda x : x[3200:]) 


# In[ ]:





# In[ ]:


df_226_136_100 = df_226_136.head(100)


# In[ ]:


def get_completion(prompt, model="text-davinci-003", temperature=0): 
    response = openai.Completion.create(
        prompt=prompt,
        model=model,
        temperature=temperature,
#         max_tokens =40
    )
#     print(response)
    return response.choices[0]["text"]


# In[ ]:


predict_1_1_davinci = []


# In[ ]:


for i in range(len(predict_1_1_davinci),len(df_226_136_100)):

    Fact_Statement = df_226_136_100["Statement_1"][i]

    prompt = f"""Task: Given examples of a Supreme Court case and the statutes applied in that case, your objective is to make accurate predictions of the specific charge or statute that is most likely to be applied within the context of the case delimited by triple backticks (```), ensuring exact predictions and learning from the provided examples.You should only include the statutes it is most confident about.The response format should include the statutes applied as in the context.
    You should to showcase creativity and knowledge to enhance the accuracy of statute predictions based on the given fact statement.

Context:

Fact Statement:"State of Punjab and another [DATE] [EVENT] [CARDINAL] came up for hearing, a Judge [PERSON] and, [GPE].
The reference order reads as follows [NORP] learned counsel for the petitioner.
The petitioner has been convicted [PERSON] [CARDINAL] and [SECTION] by the learned [ORG].
He filed an appeal challenging his conviction before the learned Sessions Judge.
While his appeal was pending, he filed an application before the learned Sessions Judge for compounding the offence, which, according to the learned counsel, was directed to be taken up along with the main appeal.
Thereafter, the petitioner filed a petition under [SECTION].for quashing of the FIR on the ground of compounding the offence.That petition u s. [PRODUCT] of Criminal Procedure, [DATE].has been dismissed by [ORG] by its impugned order.Hence, this petition has been filed in this.
[CARDINAL]. In [PERSON] 2003 4 SCC 675 [DATE] [ORG], the undisputed facts were these the husband was one of the appellants while the wife was respondent no.
[CARDINAL] in the appeal before this [ORG].
They were married on 21.7.1999 and were living separately since [DATE].
An [ORG] was registered under [SECTION] at the instance of the wife on [DATE].
When the criminal case registered at the instance of the wife was pending, the dispute between the husband and wife and their family members was settled.
It appears that the wife filed an affidavit that her disputes with the husband and the other members of his family had been finally settled and she and her husband had agreed for mutual divorce.
Based on the said affidavit, the matter was taken to [ORG] by both the parties and they jointly prayed for quashing the criminal proceedings launched against the husband and his family members on the basis of the registered at the wifes instance under [SECTION].
[CARDINAL]. [ORG] dismissed the petition for quashing the as in its view the offences under [SECTION] were non compoundable and the inherent powers [PERSON] [CARDINAL] of the [ORG] could not be invoked to by pass S. 320 of the.
[SECTION_UNK]-A was added with a view to punishing a husband and his relatives who harass or torture the wife to coerce her or her relatives to satisfy unlawful demands of dowry.
The hypertechnical view would be counterproductive and would act against interests of women and against the object for which this provision was added.
There is every likelihood that non exercise of inherent power to quash the proceedings to meet the ends of justice would prevent women from settling earlier.
That is not the object of of the Indian Penal Code, [DATE].
In view of the above discussion, we hold that [ORG] in exercise of its inherent powers can quash criminal proceedings or FIR or complaint and S. 320 of the does not limit or affect the powers [PERSON] [CARDINAL] of the [PRODUCT].
[CARDINAL]. In Nikhil Merchant 2008 9 SCC 677 2008 [EVENT], a company, M s.
Neemuch Emballage Ltd., [GPE] was granted financial assistance by [ORG] under various facilities.
On account of default in repayment of loans, the bank filed a suit for recovery of the amount payable by the borrower company.
The bank also filed a complaint against the company, its Managing Director and the officials of for diverse offences, namely, [SECTION] read with Ss.
[CARDINAL] and 51d of [ORG], [DATE] and S. [CARDINAL] read with S. 131d of [ORG], [DATE].
The suit for recovery filed by the bank against the company and the Managing Director of the [ORG] was compromised.
The suit was compromised upon the defendants agreeing to pay the amounts due as per the schedule mentioned in the consent terms.of the consent terms, the Managing Director of the [ORG], the appellant who was accused no.
[CARDINAL] in charge sheet filed by [ORG], made application for discharge from the criminal complaint.
The said application was rejected by the Special Judge CBI, [PERSON], which came to be challenged before the Bombay High Court.
The contention before [ORG] was that since the subject matter of the dispute had been settled between the appellant and the bank, it would be unreasonable to continue with the criminal proceedings.
rejected the application for discharge from the criminal cases.It is from this order that the matter reached this by way of special leave.
The having regard to the facts of the case and the earlier decision of this in [PERSON] 2003 4 SCC 675 [DATE] [ORG] [CARDINAL], set aside the order of [ORG] and quashed the criminal proceedings by consideration of the matter thus The basic intention of the accused in this case appears to have been to misrepresent the financial status of the Company, M s Neemuch Emballage Ltd., [GPE], in order to avail of the credit facilities to an extent to which the [ORG] was not entitled.
during the pendency of the appeal before this [ORG] and the terms of settlement as stated in the said affidavit, by applying the same analogy and in order to do complete justice [PERSON] [SECTION], we accept the terms of settlement insofar as the Appellant herein Accused no.
[CARDINAL] registered with [ORG] for offences punishable Under [SECTION] insofar as the Appellant Accused no.
In [PERSON] 6 SCC 364 [DATE] [EVENT] although the accused had paid the entire due amount as per the settlement with the bank in the matter of recovery before [ORG], the accused was being proceeded with for commission of offences under [SECTION] along with the bank officers who were being prosecuted under S. [CARDINAL] read with 131d of Prevention of Corruption Act, [DATE].
The [ORG] refused to quash the charge against the accused by holding that the would not quash a case involving a crime against the society when a prima facie case has been made out against the accused for framing the charge.
[PERSON] 2012 3 SC 469 [DATE] Indlaw SC 82 was again a case where the accused persons were charged of having committed offences under [SECTION] and the allegations were that the accused secured the credit facilities by submitting forged property documents as collaterals and utilized such facilities in a dishonest and fraudulent manner by opening letters of credit in respect of foreign supplies of goods, without actually bringing any goods but inducing the bank to negotiate the letters of credit in favour of foreign suppliers and also by misusing the cash credit facility.
The [ORG] was alive to the reference made in one of the present matters and also the decisions in [PERSON] 2003 4 SCC 675 [DATE] [ORG], Nikhil Merchant 2008 9 SCC 677 2008 [EVENT] and [PERSON] 2008 [CARDINAL] SCC 1 2008 Indlaw SC 2064 and it was held that [PERSON] 2003 4 SCC 675 [DATE] [ORG], and Nikhil Merchant 2008 9 SCC 677 2008 [EVENT] [DATE] dealt with different factual situation as the dispute involved had overtures of a civil dispute but the case under consideration in [PERSON] 2012 3 SC 469 [DATE] Indlaw SC 82 was more on the criminal intent than on a civil aspect.
"
Statutes : ['Constitution_226', 'Constitution_136', 'Indian Penal Code, 1860_120', 'Indian Penal Code, 1860_506', 'Indian Penal Code, 1860_34', 'Indian Penal Code, 1860_307', 'Indian Penal Code, 1860_323', 'Indian Penal Code, 1860_498', 'Constitution_32', 'Code of Criminal Procedure, 1973_482', 'Constitution_142', 'Indian Penal Code, 1860_420', 'Indian Penal Code, 1860_467', 'Indian Penal Code, 1860_471', 'Indian Penal Code, 1860_406', 'Indian Penal Code, 1860_468', 'Indian Penal Code, 1860_2', 'Indian Penal Code, 1860_409']

###

Fact Statement:"In this one gets used to writing common orders, for orders are written either on behalf of the [PRODUCT], or on behalf of the [ORG].
While endorsing the opinion expressed by [PERSON],, adjudicating upon the prayer for my recusal, from hearing the matters in hand, reasons for my continuation on the [ORG], also need to be expressed by me.
It has been necessitated, for deciding an objection, about the present composition of the [PERSON].
As already noted above,, [ORG] has rendered the decision on the objection.
The events which followed the order of [PERSON],, are also of some significance.
In my considered view, they too need to be narrated, for only then, the entire matter can be considered to have been fully expressed, as it ought to be.
I also need to record reasons, why my continuation on the reconstituted [ORG], was the only course open to me.
And therefore, my side of its understanding, dealing with the perception, of the other side of the [PRODUCT].
3i A [CARDINAL] Judge [PERSON] was originally constituted for hearing these matters.
Union of India [DATE] Indlaw SCO 185 Writ Petition C no.13 of, Mr. [PERSON], Senior Advocate, in [ORG] of [DATE], Mr. [PERSON], Advocate, in [ORG] Indlaw SC 29 Writ Petition C [DATE] and Mr. [PERSON], Advocate, in Change [GPE] v. [ORG] no.70 of [DATE], representing the petitioners were heard.iii The proceedings recorded by this [ORG] on 18.3.2015 reveal, that Mr. [PERSON], in Writ Petition C no.70 of [DATE] was heard again on, whereupon, Mr. [PERSON] and Mr., [ORG] [GPE], also made their submissions.
[CARDINAL]. Based on the order passed by the Judge [PERSON] on 7.4.2015, Honble the Chief Justice of [GPE], constituted a [CARDINAL] Judge [PERSON], comprising of,,, and, JJ.
[CARDINAL]. On 13.4.2015 the Constitution Ninety ninth Amendment Act, [DATE], and [ORG] Act, [DATE], were notified in the Gazette of India Extraordinary.
Both the above enactments, were brought into force with effect from 13.4.2015.
[CARDINAL]. When the reconstituted [PERSON] commenced hearing on 21.4.2015, Mr. made a prayer for my recusal from the [PRODUCT], which was seconded by Mr. Mathews [PERSON] petitioner in- person in Writ Petition C no.124 of [DATE], the latter advanced submissions, even though he had been barred from doing so, by an earlier order dated 24.3.2015 extracted above.
The [ORDINAL] judgment was rendered, by a [CARDINAL] Judge [PERSON], by a majority of [CARDINAL], in the [ORDINAL] Judges case on [CARDINAL]. The correctness of the First Judges case was doubted by a Judge [PERSON] in of [GPE], [DATE] Supp 1 SCC 574 [ORG] [CARDINAL], which opined that the majority view, in the [ORG] case, should be considered by a larger.
As per the position expressed before us, a feeling came to be entertained, that a Commission for selection and appointment, as also for transfer, of Judges of the higher judiciary should be constituted, which would replace the prevailing procedure, for appointment of [GPE] and Chief Justices of [ORG] and, contemplated under [PERSON] [CARDINAL] and 217 1.
The amendment, received the assent of the President on [DATE].It was however given effect to, with effect from 13.4.2015 consequent upon its notification in the Gazette of India Extraordinary Part II, [SECTION_UNK].
The same was also brought into force, with effect from 13.4.2015 by its notification in the Gazette of India Extraordinary Part II, [SECTION_UNK].
The Judges case- [DATE] [EVENT] 87 [DATE] [ORG] [CARDINAL].[DATE].The Union Law Minister addressed a letter dated 18.3.1981 to the Governor of [PRODUCT] and to Chief Ministers of all other [GPE].
The addressees were inter [PERSON], [CARDINAL] of [ORG], should as far as possible be from outside the in which is situated.
Through the above letter, the addressees were requested to.a obtain from all additional Judges working in the High Courts.their consent to be appointed as permanent Judges in any other in the country.
The above noted letter required, that the concerned appointees.be required to name [CARDINAL] High Courts, in order of preference, to which they would prefer to be appointed as permanent Judges and b obtain from persons who have already been or may in the future be proposed by you for initial appointment their consent to be appointed to any other [ORG] in the country along with a similar preference for [CARDINAL] High Courts.
The Union Law Minister, in the above letter clarified, that furnishing of their consent or indication of their preference, would not imply any commitment, at the behest of the Government, to accommodate them in accordance with their preferences.
In response, quite a few additional Judges, gave their consent to be appointed outside their parent [ORG].
i [PERSON] and the other petitioners felt, that the letter dated [PRODUCT] was a direct attack on the independence of the judiciary, and an uninhibited assault on a vital basic feature of the [LAW].
A series of [ORG] in [GPE] passed resolutions, condemning the letter dated 18.3.1981, as being subversive of judicial independence.
Since that was not done, a writ petition was filed by the above Associations in the Bombay High Court, challenging the letter dated 18.3.1981.
An interim order was passed by [ORG], restraining the Union Law Minister and the Government from implementing the letter dated 18.3.1981.
While the matter was pending before this, the Union Law Minister and, filed a transfer petition under [LAW] The transfer petition was allowed, and the writ petition filed in the Bombay High Court, was transferred to [ORG].
Rather than being appointed for a further term of, their appointment was extended for.
These short term appointments were assailed, as being unjustified under [LAW], besides being subversive of the independence of the judiciary.
This writ petition was also transferred for hearing to [ORG]."

Statutes:['Constitution_226', 'Constitution_136', 'Constitution_14', 'Constitution_16', 'Constitution_227', 'Constitution_133', 'Constitution_246', 'Constitution_1', 'Constitution_21', 'Constitution_32', 'Constitution_19', 'Constitution_141', 'Constitution_4', 'Constitution_31', 'Constitution_12', 'Constitution_2', 'Constitution_39', 'Constitution_311', 'Constitution_13', 'Constitution_5', 'Constitution_3', 'Constitution_6', 'Constitution_15']

###

Format your response as follows:
"Statutes applied: [List of applicable statutes]

Instructions:

Learn from the examples provided in the context to understand the task of charge or statute prediction.
Your response should be focused on providing the exact statute or charge that aligns with the legal principles and precedents applicable to the given facts.
In your response, include only the statutes you are most confident about.
Ensure that the statutes generated as responses are valid and recognized legal statutes. Avoid generating fabricated or invalid statutes.
The model's performance will be evaluated based on its ability to predict the correct statute, include only confident statutes, and showcase creativity in its predictions.

Fact Statement: ``` {Fact_Statement}```
"""
    response = get_completion(prompt)
    predict_1_1_davinci.append(response)


# ## text-davinci-002

# In[ ]:


df_226_136 = (df[df['Statutes'].str.contains("Constitution_226") & df['Statutes'].str.contains("Constitution_136")])


# In[ ]:


df_226_136 = df_226_136.reset_index()


# In[ ]:


df_226_136 = df_226_136.drop(['level_0'],axis=1)


# In[ ]:


df_226_136['Statement_1'] = df_226_136['Statement'].apply(lambda x : x[0:1320]) 


# In[ ]:


df_226_136['Statment_2'] = df_226_136['Statement'].apply(lambda x : x[1600:3200]) 


# In[ ]:


df_226_136['Statment_3'] = df_226_136['Statement'].apply(lambda x : x[3200:]) 


# In[ ]:





# In[ ]:


def get_completion(prompt, model="text-davinci-002", temperature=0): 
    response = openai.Completion.create(
        prompt=prompt,
        model=model,
        temperature=temperature,
        max_tokens =30
    )
#     print(response)
    return response.choices[0]["text"]


# In[ ]:


predict_1_1_davinci002 = []


# In[ ]:


for i in range(len(predict_1_1_davinci002),len(df_226_136_100)):

    Fact_Statement = df_226_136_100["Statement_1"][i]

    prompt = f"""Task: Given examples of a Supreme Court case and the statutes applied in that case, your objective is to make accurate predictions of the specific charge or statute that is most likely to be applied within the context of the case delimited by triple backticks (```), ensuring exact predictions and learning from the provided examples.You should only include the statutes it is most confident about.The response format should include the statutes applied as in the context.
    You should to showcase creativity and knowledge to enhance the accuracy of statute predictions based on the given fact statement.

Context:

Fact Statement:"State of Punjab and another [DATE] [EVENT] [CARDINAL] came up for hearing, a Judge [PERSON] and, [GPE].
The reference order reads as follows [NORP] learned counsel for the petitioner.
The petitioner has been convicted [PERSON] [CARDINAL] and [SECTION] by the learned [ORG].
He filed an appeal challenging his conviction before the learned Sessions Judge.
While his appeal was pending, he filed an application before the learned Sessions Judge for compounding the offence, which, according to the learned counsel, was directed to be taken up along with the main appeal.
Thereafter, the petitioner filed a petition under [SECTION].for quashing of the FIR on the ground of compounding the offence.That petition u s. [PRODUCT] of Criminal Procedure, [DATE].has been dismissed by [ORG] by its impugned order.Hence, this petition has been filed in this.
[CARDINAL]. In [PERSON] 2003 4 SCC 675 [DATE] [ORG], the undisputed facts were these the husband was one of the appellants while the wife was respondent no.
[CARDINAL] in the appeal before this [ORG].
They were married on 21.7.1999 and were living separately since [DATE].
An [ORG] was registered under [SECTION] at the instance of the wife on [DATE].
When the criminal case registered at the instance of the wife was pending, the dispute between the husband and wife and their family members was settled.
It appears that the wife filed an affidavit that her disputes with the husband and the other members of his family had been finally settled and she and her husband had agreed for mutual divorce.
Based on the said affidavit, the matter was taken to [ORG] by both the parties and they jointly prayed for quashing the criminal proceedings launched against the husband and his family members on the basis of the registered at the wifes instance under [SECTION].
[CARDINAL]. [ORG] dismissed the petition for quashing the as in its view the offences under [SECTION] were non compoundable and the inherent powers [PERSON] [CARDINAL] of the [ORG] could not be invoked to by pass S. 320 of the.
[SECTION_UNK]-A was added with a view to punishing a husband and his relatives who harass or torture the wife to coerce her or her relatives to satisfy unlawful demands of dowry.
The hypertechnical view would be counterproductive and would act against interests of women and against the object for which this provision was added.
There is every likelihood that non exercise of inherent power to quash the proceedings to meet the ends of justice would prevent women from settling earlier.
That is not the object of of the Indian Penal Code, [DATE].
In view of the above discussion, we hold that [ORG] in exercise of its inherent powers can quash criminal proceedings or FIR or complaint and S. 320 of the does not limit or affect the powers [PERSON] [CARDINAL] of the [PRODUCT].
[CARDINAL]. In Nikhil Merchant 2008 9 SCC 677 2008 [EVENT], a company, M s.
Neemuch Emballage Ltd., [GPE] was granted financial assistance by [ORG] under various facilities.
On account of default in repayment of loans, the bank filed a suit for recovery of the amount payable by the borrower company.
The bank also filed a complaint against the company, its Managing Director and the officials of for diverse offences, namely, [SECTION] read with Ss.
[CARDINAL] and 51d of [ORG], [DATE] and S. [CARDINAL] read with S. 131d of [ORG], [DATE].
The suit for recovery filed by the bank against the company and the Managing Director of the [ORG] was compromised.
The suit was compromised upon the defendants agreeing to pay the amounts due as per the schedule mentioned in the consent terms.of the consent terms, the Managing Director of the [ORG], the appellant who was accused no.
[CARDINAL] in charge sheet filed by [ORG], made application for discharge from the criminal complaint.
The said application was rejected by the Special Judge CBI, [PERSON], which came to be challenged before the Bombay High Court.
The contention before [ORG] was that since the subject matter of the dispute had been settled between the appellant and the bank, it would be unreasonable to continue with the criminal proceedings.
rejected the application for discharge from the criminal cases.It is from this order that the matter reached this by way of special leave.
The having regard to the facts of the case and the earlier decision of this in [PERSON] 2003 4 SCC 675 [DATE] [ORG] [CARDINAL], set aside the order of [ORG] and quashed the criminal proceedings by consideration of the matter thus The basic intention of the accused in this case appears to have been to misrepresent the financial status of the Company, M s Neemuch Emballage Ltd., [GPE], in order to avail of the credit facilities to an extent to which the [ORG] was not entitled.
during the pendency of the appeal before this [ORG] and the terms of settlement as stated in the said affidavit, by applying the same analogy and in order to do complete justice [PERSON] [SECTION], we accept the terms of settlement insofar as the Appellant herein Accused no.
[CARDINAL] registered with [ORG] for offences punishable Under [SECTION] insofar as the Appellant Accused no.
In [PERSON] 6 SCC 364 [DATE] [EVENT] although the accused had paid the entire due amount as per the settlement with the bank in the matter of recovery before [ORG], the accused was being proceeded with for commission of offences under [SECTION] along with the bank officers who were being prosecuted under S. [CARDINAL] read with 131d of Prevention of Corruption Act, [DATE].
The [ORG] refused to quash the charge against the accused by holding that the would not quash a case involving a crime against the society when a prima facie case has been made out against the accused for framing the charge.
[PERSON] 2012 3 SC 469 [DATE] Indlaw SC 82 was again a case where the accused persons were charged of having committed offences under [SECTION] and the allegations were that the accused secured the credit facilities by submitting forged property documents as collaterals and utilized such facilities in a dishonest and fraudulent manner by opening letters of credit in respect of foreign supplies of goods, without actually bringing any goods but inducing the bank to negotiate the letters of credit in favour of foreign suppliers and also by misusing the cash credit facility.
The [ORG] was alive to the reference made in one of the present matters and also the decisions in [PERSON] 2003 4 SCC 675 [DATE] [ORG], Nikhil Merchant 2008 9 SCC 677 2008 [EVENT] and [PERSON] 2008 [CARDINAL] SCC 1 2008 Indlaw SC 2064 and it was held that [PERSON] 2003 4 SCC 675 [DATE] [ORG], and Nikhil Merchant 2008 9 SCC 677 2008 [EVENT] [DATE] dealt with different factual situation as the dispute involved had overtures of a civil dispute but the case under consideration in [PERSON] 2012 3 SC 469 [DATE] Indlaw SC 82 was more on the criminal intent than on a civil aspect.
"
Statutes : ['Constitution_226', 'Constitution_136', 'Indian Penal Code, 1860_120', 'Indian Penal Code, 1860_506', 'Indian Penal Code, 1860_34', 'Indian Penal Code, 1860_307', 'Indian Penal Code, 1860_323', 'Indian Penal Code, 1860_498', 'Constitution_32', 'Code of Criminal Procedure, 1973_482', 'Constitution_142', 'Indian Penal Code, 1860_420', 'Indian Penal Code, 1860_467', 'Indian Penal Code, 1860_471', 'Indian Penal Code, 1860_406', 'Indian Penal Code, 1860_468', 'Indian Penal Code, 1860_2', 'Indian Penal Code, 1860_409']

###

Fact Statement:"In this one gets used to writing common orders, for orders are written either on behalf of the [PRODUCT], or on behalf of the [ORG].
While endorsing the opinion expressed by [PERSON],, adjudicating upon the prayer for my recusal, from hearing the matters in hand, reasons for my continuation on the [ORG], also need to be expressed by me.
It has been necessitated, for deciding an objection, about the present composition of the [PERSON].
As already noted above,, [ORG] has rendered the decision on the objection.
The events which followed the order of [PERSON],, are also of some significance.
In my considered view, they too need to be narrated, for only then, the entire matter can be considered to have been fully expressed, as it ought to be.
I also need to record reasons, why my continuation on the reconstituted [ORG], was the only course open to me.
And therefore, my side of its understanding, dealing with the perception, of the other side of the [PRODUCT].
Union of India [DATE] Indlaw SCO 185 Writ Petition C no.13 of, Mr. [PERSON], Senior Advocate, in [ORG] of [DATE], Mr. [PERSON], Advocate, in [ORG] Indlaw SC 29 Writ Petition C [DATE] and Mr. [PERSON], Advocate, in Change [GPE] v. [ORG] no.70 of [DATE], representing the petitioners were heard.iii The proceedings recorded by this [ORG] on 18.3.2015 reveal, that Mr. [PERSON], in Writ Petition C no.70 of [DATE] was heard again on, whereupon, Mr. [PERSON] and Mr., [ORG] [GPE], also made their submissions.
[CARDINAL]. Based on the order passed by the Judge [PERSON] on 7.4.2015, Honble the Chief Justice of [GPE], constituted a [CARDINAL] Judge [PERSON], comprising of,,, and, JJ.
[CARDINAL]. On 13.4.2015 the Constitution Ninety ninth Amendment Act, [DATE], and [ORG] Act, [DATE], were notified in the Gazette of India Extraordinary.Both the above enactments, were brought into force with effect from 13.4.2015.
[CARDINAL]. When the reconstituted [PERSON] commenced hearing on 21.4.2015, Mr. made a prayer for my recusal from the [PRODUCT], which was seconded by Mr. Mathews [PERSON] petitioner in- person in Writ Petition C no.124 of [DATE], the latter advanced submissions, even though he had been barred from doing so, by an earlier order dated 24.3.2015 extracted above.
The [ORDINAL] judgment was rendered, by a [CARDINAL] Judge [PERSON], by a majority of [CARDINAL], in the [ORDINAL] Judges case on [CARDINAL]. The correctness of the First Judges case was doubted by a Judge [PERSON] in of [GPE], [DATE] Supp 1 SCC 574 [ORG] [CARDINAL], which opined that the majority view, in the [ORG] case, should be considered by a larger.
As per the position expressed before us, a feeling came to be entertained, that a Commission for selection and appointment, as also for transfer, of Judges of the higher judiciary should be constituted, which would replace the prevailing procedure, for appointment of [GPE] and Chief Justices of [ORG] and, contemplated under [PERSON] [CARDINAL] and 217 1.
The amendment, received the assent of the President on [DATE].It was however given effect to, with effect from 13.4.2015 consequent upon its notification in the Gazette of India Extraordinary Part II, [SECTION_UNK].
The same was also brought into force, with effect from 13.4.2015 by its notification in the Gazette of India Extraordinary Part II, [SECTION_UNK].
The Judges case- [DATE] [EVENT] 87 [DATE] [ORG] [CARDINAL].[DATE].The Union Law Minister addressed a letter dated 18.3.1981 to the Governor of [PRODUCT] and to Chief Ministers of all other [GPE].
The addressees were inter [PERSON], [CARDINAL] of [ORG], should as far as possible be from outside the in which is situated.
Through the above letter, the addressees were requested to.a obtain from all additional Judges working in the High Courts.their consent to be appointed as permanent Judges in any other in the country.
The above noted letter required, that the concerned appointees.be required to name [CARDINAL] High Courts, in order of preference, to which they would prefer to be appointed as permanent Judges and b obtain from persons who have already been or may in the future be proposed by you for initial appointment their consent to be appointed to any other [ORG] in the country along with a similar preference for [CARDINAL] High Courts.
The Union Law Minister, in the above letter clarified, that furnishing of their consent or indication of their preference, would not imply any commitment, at the behest of the Government, to accommodate them in accordance with their preferences.
In response, quite a few additional Judges, gave their consent to be appointed outside their parent [ORG].
i [PERSON] and the other petitioners felt, that the letter dated [PRODUCT] was a direct attack on the independence of the judiciary, and an uninhibited assault on a vital basic feature of the [LAW].
A series of [ORG] in [GPE] passed resolutions, condemning the letter dated 18.3.1981, as being subversive of judicial independence.
Since that was not done, a writ petition was filed by the above Associations in the Bombay High Court, challenging the letter dated 18.3.1981.
An interim order was passed by [ORG], restraining the Union Law Minister and the Government from implementing the letter dated 18.3.1981.
While the matter was pending before this, the Union Law Minister and, filed a transfer petition under [LAW] The transfer petition was allowed, and the writ petition filed in the Bombay High Court, was transferred to [ORG].
Rather than being appointed for a further term of, their appointment was extended for.
These short term appointments were assailed, as being unjustified under [LAW], besides being subversive of the independence of the judiciary.
This writ petition was also transferred for hearing to [ORG]."

Statutes:['Constitution_226', 'Constitution_136', 'Constitution_14', 'Constitution_16', 'Constitution_227', 'Constitution_133', 'Constitution_246', 'Constitution_1', 'Constitution_21', 'Constitution_32', 'Constitution_19', 'Constitution_141', 'Constitution_4', 'Constitution_31', 'Constitution_12', 'Constitution_2', 'Constitution_39', 'Constitution_311', 'Constitution_13', 'Constitution_5', 'Constitution_3', 'Constitution_6', 'Constitution_15']

###

Format your response as follows:
"Statutes applied: [List of applicable statutes]"

Instructions:

Learn from the examples provided in the context to understand the task of charge or statute prediction.
Your response should be focused on providing the exact statute or charge that aligns with the legal principles and precedents applicable to the given facts.
In your response, include only the statutes you are most confident about.
Ensure that the statutes generated as responses are valid and recognized legal statutes. Avoid generating fabricated or invalid statutes.
The model's performance will be evaluated based on its ability to predict the correct statute, include only confident statutes, and showcase creativity in its predictions.

Fact Statement: ``` {Fact_Statement}```
"""
    response = get_completion(prompt)
    predict_1_1_davinci002.append(response)


# # Fine Tuned Davinci

# In[ ]:


df_226_136 = (df[df['Statutes'].str.contains("Constitution_226") & df['Statutes'].str.contains("Constitution_136")])


# In[ ]:


df_226_136 = df_226_136.reset_index()


# In[ ]:


df_226_136 = df_226_136.drop(['level_0'],axis=1)


# In[ ]:


df_226_136['Statement_1'] = df_226_136['Statement'].apply(lambda x : x[0:1100]) 


# In[ ]:


df_226_136['Statment_2'] = df_226_136['Statement'].apply(lambda x : x[1600:3200]) 


# In[ ]:


df_226_136['Statment_3'] = df_226_136['Statement'].apply(lambda x : x[3200:]) 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


def get_completion(prompt, model="davinci:ft-personal-2023-05-20-07-37-44", temperature=0): 
    response = openai.Completion.create(
        prompt=prompt,
        model=model,
        temperature=temperature,
        max_tokens =25
    )
#     print(response)
    return response.choices[0]["text"]


# In[ ]:


predict_1_1_Finedavinci = [] #first trial and first statement used and 30 tokens


# In[ ]:


for i in range(len(predict_1_1_Finedavinci),len(df_226_136_100)):
    try:
        Fact_Statement = df_226_136_100["Statement_1"][i]
        prompt = f"""Task: Given examples of a Supreme Court case and the statutes applied in that case, your objective is to make accurate predictions of the specific charge or statute that is most likely to be applied within the context of the case delimited by triple backticks (```), ensuring exact predictions and learning from the provided examples.You should only include the statutes it is most confident about.The response format should include the statutes applied as in the context.
                            You should to showcase creativity and knowledge to enhance the accuracy of statute predictions based on the given fact statement.

Context:

Fact Statement:"In this one gets used to writing common orders, for orders are written either on behalf of the [PRODUCT], or on behalf of the [ORG].
While endorsing the opinion expressed by [PERSON],, adjudicating upon the prayer for my recusal, from hearing the matters in hand, reasons for my continuation on the [ORG], also need to be expressed by me.
It has been necessitated, for deciding an objection, about the present composition of the [PERSON].
As already noted above,, [ORG] has rendered the decision on the objection.
The events which followed the order of [PERSON],, are also of some significance.
In my considered view, they too need to be narrated, for only then, the entire matter can be considered to have been fully expressed, as it ought to be.
I also need to record reasons, why my continuation on the reconstituted [ORG], was the only course open to me.
And therefore, my side of its understanding, dealing with the perception, of the other side of the [PRODUCT].
Union of India [DATE] Indlaw SCO 185 Writ Petition C no.13 of, Mr. [PERSON], Senior Advocate, in [ORG] of [DATE], Mr. [PERSON], Advocate, in [ORG] Indlaw SC 29 Writ Petition C [DATE] and Mr. [PERSON], Advocate, in Change [GPE] v. [ORG] no.70 of [DATE], representing the petitioners were heard.iii The proceedings recorded by this [ORG] on 18.3.2015 reveal, that Mr. [PERSON], in Writ Petition C no.70 of [DATE] was heard again on, whereupon, Mr. [PERSON] and Mr., [ORG] [GPE], also made their submissions.
[CARDINAL]. Based on the order passed by the Judge [PERSON] on 7.4.2015, Honble the Chief Justice of [GPE], constituted a [CARDINAL] Judge [PERSON], comprising of,,, and, JJ.
[CARDINAL]. On 13.4.2015 the Constitution Ninety ninth Amendment Act, [DATE], and [ORG] Act, [DATE], were notified in the Gazette of India Extraordinary.Both the above enactments, were brought into force with effect from 13.4.2015.
[CARDINAL]. When the reconstituted [PERSON] commenced hearing on 21.4.2015, Mr. made a prayer for my recusal from the [PRODUCT], which was seconded by Mr. Mathews [PERSON] petitioner in- person in Writ Petition C no.124 of [DATE], the latter advanced submissions, even though he had been barred from doing so, by an earlier order dated 24.3.2015 extracted above.
The [ORDINAL] judgment was rendered, by a [CARDINAL] Judge [PERSON], by a majority of [CARDINAL], in the [ORDINAL] Judges case on [CARDINAL]. The correctness of the First Judges case was doubted by a Judge [PERSON] in of [GPE], [DATE] Supp 1 SCC 574 [ORG] [CARDINAL], which opined that the majority view, in the [ORG] case, should be considered by a larger.
The amendment, received the assent of the President on [DATE].It was however given effect to, with effect from 13.4.2015 consequent upon its notification in the Gazette of India Extraordinary Part II, [SECTION_UNK].
The same was also brought into force, with effect from 13.4.2015 by its notification in the Gazette of India Extraordinary Part II, [SECTION_UNK].
The Judges case- [DATE] [EVENT] 87 [DATE] [ORG] [CARDINAL].[DATE].The Union Law Minister addressed a letter dated 18.3.1981 to the Governor of [PRODUCT] and to Chief Ministers of all other [GPE].
The addressees were inter [PERSON], [CARDINAL] of [ORG], should as far as possible be from outside the in which is situated.
Through the above letter, the addressees were requested to.a obtain from all additional Judges working in the High Courts.their consent to be appointed as permanent Judges in any other in the country.
The above noted letter required, that the concerned appointees.be required to name [CARDINAL] High Courts, in order of preference, to which they would prefer to be appointed as permanent Judges and b obtain from persons who have already been or may in the future be proposed by you for initial appointment their consent to be appointed to any other [ORG] in the country along with a similar preference for [CARDINAL] High Courts.
The Union Law Minister, in the above letter clarified, that furnishing of their consent or indication of their preference, would not imply any commitment, at the behest of the Government, to accommodate them in accordance with their preferences.
In response, quite a few additional Judges, gave their consent to be appointed outside their parent [ORG].
A series of [ORG] in [GPE] passed resolutions, condemning the letter dated 18.3.1981, as being subversive of judicial independence.
Since that was not done, a writ petition was filed by the above Associations in the Bombay High Court, challenging the letter dated 18.3.1981.
An interim order was passed by [ORG], restraining the Union Law Minister and the Government from implementing the letter dated 18.3.1981.
While the matter was pending before this, the Union Law Minister and, filed a transfer petition under [LAW] The transfer petition was allowed, and the writ petition filed in the Bombay High Court, was transferred to [ORG].
These short term appointments were assailed, as being unjustified under [LAW], besides being subversive of the independence of the judiciary."

Statutes:['Constitution_226', 'Constitution_136', 'Constitution_14', 'Constitution_16', 'Constitution_227', 'Constitution_133', 'Constitution_246', 'Constitution_1', 'Constitution_21', 'Constitution_32', 'Constitution_19', 'Constitution_141', 'Constitution_4', 'Constitution_31', 'Constitution_12', 'Constitution_2', 'Constitution_39', 'Constitution_311', 'Constitution_13', 'Constitution_5', 'Constitution_3', 'Constitution_6', 'Constitution_15']

###

Format your response as follows:
"Statutes applied: [List of applicable statutes]"

Instructions:

Learn from the examples provided in the context to understand the task of charge or statute prediction.
Your response should be focused on providing the exact statute or charge that aligns with the legal principles and precedents applicable to the given facts.
In your response, include only the statutes you are most confident about.Ensure that the statutes generated as responses are valid and recognized legal statutes. Avoid generating fabricated or invalid statutes.
The model's performance will be evaluated based on its ability to predict the correct statute, include only confident statutes, and showcase creativity in its predictions.

Fact Statement: ``` {Fact_Statement}```
"""
        response = get_completion(prompt)
        predict_1_1_Finedavinci.append(response)
    
    except :
        predict_1_1_Finedavinci.append('length increaeses the token limit')
        


# ## gpt-3.5-turbo

# In[ ]:





# In[ ]:


df = pd.read_csv('QA_dataset_all.csv')


# In[ ]:


df_226_136 = (df[df['Statutes'].str.contains("Constitution_226") & df['Statutes'].str.contains("Constitution_136")])


# In[ ]:


df_226_136['Statement_1'] = df_226_136['Statement'].apply(lambda x : x[0:1600]) 


# In[ ]:


df_226_136['Statment_2'] = df_226_136['Statement'].apply(lambda x : x[1600:3200]) 


# In[ ]:


df_226_136['Statment_3'] = df_226_136['Statement'].apply(lambda x : x[3200:]) 


# In[ ]:


df_226_136_101 = df_226_136.head(101)


# In[ ]:


import openai


# In[ ]:


def get_completion(prompt, model="gpt-3.5-turbo", temperature=0): 
    messages = [
        {"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
#     print(response)
    return response.choices[0].message["content"]


# In[ ]:


df_226_136_101.reset_index(inplace =True)


# In[ ]:


df_226_136_101 = df_226_136_101.drop(['level_0'],axis =1)


# In[ ]:


list = []


# In[ ]:


# predict_1_1 = []


# In[ ]:


import time


# In[ ]:


for i in range(len(list),len(df_226_136_101)):

    Fact_Statement = df_226_136_101["Statement_1"][i]

    prompt = f"""Task: Given examples of a Supreme Court case and the statutes applied in that case, your objective is to make accurate predictions of the specific charge or statute that is most likely to be applied within the context of the case delimited by triple backticks (```), ensuring exact predictions and learning from the provided examples.You should only include the statutes it is most confident about.The response format should include the statutes applied as in the context.
    You should to showcase creativity and knowledge to enhance the accuracy of statute predictions based on the given fact statement.

Context:

Fact Statement:"State of Punjab and another [DATE] [EVENT] [CARDINAL] came up for hearing, a Judge [PERSON] and, [GPE].
The reference order reads as follows [NORP] learned counsel for the petitioner.
The petitioner has been convicted [PERSON] [CARDINAL] and [SECTION] by the learned [ORG].
He filed an appeal challenging his conviction before the learned Sessions Judge.
While his appeal was pending, he filed an application before the learned Sessions Judge for compounding the offence, which, according to the learned counsel, was directed to be taken up along with the main appeal.
Thereafter, the petitioner filed a petition under [SECTION].
for quashing of the FIR on the ground of compounding the offence.
That petition u s. [PRODUCT] of Criminal Procedure, [DATE].
has been dismissed by [ORG] by its impugned order.
Hence, this petition has been filed in this.
[CARDINAL]. In [PERSON] 2003 4 SCC 675 [DATE] [ORG], the undisputed facts were these the husband was one of the appellants while the wife was respondent no.
[CARDINAL] in the appeal before this [ORG].
They were married on 21.7.1999 and were living separately since [DATE].
An [ORG] was registered under [SECTION] at the instance of the wife on [DATE].
When the criminal case registered at the instance of the wife was pending, the dispute between the husband and wife and their family members was settled.
It appears that the wife filed an affidavit that her disputes with the husband and the other members of his family had been finally settled and she and her husband had agreed for mutual divorce.
Based on the said affidavit, the matter was taken to [ORG] by both the parties and they jointly prayed for quashing the criminal proceedings launched against the husband and his family members on the basis of the registered at the wifes instance under [SECTION].
[CARDINAL]. [ORG] dismissed the petition for quashing the as in its view the offences under [SECTION] were non compoundable and the inherent powers [PERSON] [CARDINAL] of the [ORG] could not be invoked to by pass S. 320 of the.
[SECTION_UNK]-A was added with a view to punishing a husband and his relatives who harass or torture the wife to coerce her or her relatives to satisfy unlawful demands of dowry.
The hypertechnical view would be counterproductive and would act against interests of women and against the object for which this provision was added.
There is every likelihood that non exercise of inherent power to quash the proceedings to meet the ends of justice would prevent women from settling earlier.
That is not the object of of the Indian Penal Code, [DATE].
In view of the above discussion, we hold that [ORG] in exercise of its inherent powers can quash criminal proceedings or FIR or complaint and S. 320 of the does not limit or affect the powers [PERSON] [CARDINAL] of the [PRODUCT].
[CARDINAL]. In Nikhil Merchant 2008 9 SCC 677 2008 [EVENT], a company, M s.
Neemuch Emballage Ltd., [GPE] was granted financial assistance by [ORG] under various facilities.
On account of default in repayment of loans, the bank filed a suit for recovery of the amount payable by the borrower company.
The bank also filed a complaint against the company, its Managing Director and the officials of for diverse offences, namely, [SECTION] read with Ss.
[CARDINAL] and 51d of [ORG], [DATE] and S. [CARDINAL] read with S. 131d of [ORG], [DATE].
The suit for recovery filed by the bank against the company and the Managing Director of the [ORG] was compromised.
The suit was compromised upon the defendants agreeing to pay the amounts due as per the schedule mentioned in the consent terms.
Based on cl.
of the consent terms, the Managing Director of the [ORG], the appellant who was accused no.
[CARDINAL] in charge sheet filed by [ORG], made application for discharge from the criminal complaint.
The said application was rejected by the Special Judge CBI, [PERSON], which came to be challenged before the Bombay High Court.
The contention before [ORG] was that since the subject matter of the dispute had been settled between the appellant and the bank, it would be unreasonable to continue with the criminal proceedings.
rejected the application for discharge from the criminal cases.
It is from this order that the matter reached this by way of special leave.
The having regard to the facts of the case and the earlier decision of this in [PERSON] 2003 4 SCC 675 [DATE] [ORG] [CARDINAL], set aside the order of [ORG] and quashed the criminal proceedings by consideration of the matter thus The basic intention of the accused in this case appears to have been to misrepresent the financial status of the Company, M s Neemuch Emballage Ltd., [GPE], in order to avail of the credit facilities to an extent to which the [ORG] was not entitled.
during the pendency of the appeal before this [ORG] and the terms of settlement as stated in the said affidavit, by applying the same analogy and in order to do complete justice [PERSON] [SECTION], we accept the terms of settlement insofar as the Appellant herein Accused no.
[CARDINAL] registered with [ORG] for offences punishable Under [SECTION] insofar as the Appellant Accused no.
In [PERSON] 6 SCC 364 [DATE] [EVENT] although the accused had paid the entire due amount as per the settlement with the bank in the matter of recovery before [ORG], the accused was being proceeded with for commission of offences under [SECTION] along with the bank officers who were being prosecuted under S. [CARDINAL] read with 131d of Prevention of Corruption Act, [DATE].
The [ORG] refused to quash the charge against the accused by holding that the would not quash a case involving a crime against the society when a prima facie case has been made out against the accused for framing the charge.
[PERSON] 2012 3 SC 469 [DATE] Indlaw SC 82 was again a case where the accused persons were charged of having committed offences under [SECTION] and the allegations were that the accused secured the credit facilities by submitting forged property documents as collaterals and utilized such facilities in a dishonest and fraudulent manner by opening letters of credit in respect of foreign supplies of goods, without actually bringing any goods but inducing the bank to negotiate the letters of credit in favour of foreign suppliers and also by misusing the cash credit facility.
The [ORG] was alive to the reference made in one of the present matters and also the decisions in [PERSON] 2003 4 SCC 675 [DATE] [ORG], Nikhil Merchant 2008 9 SCC 677 2008 [EVENT] and [PERSON] 2008 [CARDINAL] SCC 1 2008 Indlaw SC 2064 and it was held that [PERSON] 2003 4 SCC 675 [DATE] [ORG], and Nikhil Merchant 2008 9 SCC 677 2008 [EVENT] [DATE] dealt with different factual situation as the dispute involved had overtures of a civil dispute but the case under consideration in [PERSON] 2012 3 SC 469 [DATE] Indlaw SC 82 was more on the criminal intent than on a civil aspect.
"
Statutes : ['Constitution_226', 'Constitution_136', 'Indian Penal Code, 1860_120', 'Indian Penal Code, 1860_506', 'Indian Penal Code, 1860_34', 'Indian Penal Code, 1860_307', 'Indian Penal Code, 1860_323', 'Indian Penal Code, 1860_498', 'Constitution_32', 'Code of Criminal Procedure, 1973_482', 'Constitution_142', 'Indian Penal Code, 1860_420', 'Indian Penal Code, 1860_467', 'Indian Penal Code, 1860_471', 'Indian Penal Code, 1860_406', 'Indian Penal Code, 1860_468', 'Indian Penal Code, 1860_2', 'Indian Penal Code, 1860_409']

###

Fact Statement:"In this one gets used to writing common orders, for orders are written either on behalf of the [PRODUCT], or on behalf of the [ORG].
While endorsing the opinion expressed by [PERSON],, adjudicating upon the prayer for my recusal, from hearing the matters in hand, reasons for my continuation on the [ORG], also need to be expressed by me.
It has been necessitated, for deciding an objection, about the present composition of the [PERSON].
As already noted above,, [ORG] has rendered the decision on the objection.
The events which followed the order of [PERSON],, are also of some significance.
In my considered view, they too need to be narrated, for only then, the entire matter can be considered to have been fully expressed, as it ought to be.
I also need to record reasons, why my continuation on the reconstituted [ORG], was the only course open to me.
And therefore, my side of its understanding, dealing with the perception, of the other side of the [PRODUCT].
3i A [CARDINAL] Judge [PERSON] was originally constituted for hearing these matters.
Union of India [DATE] Indlaw SCO 185 Writ Petition C no.13 of, Mr. [PERSON], Senior Advocate, in [ORG] of [DATE], Mr. [PERSON], Advocate, in [ORG] Indlaw SC 29 Writ Petition C [DATE] and Mr. [PERSON], Advocate, in Change [GPE] v. [ORG] no.70 of [DATE], representing the petitioners were heard.
The matter was shown as part heard, and posted for further hearing on [DATE].
iii The proceedings recorded by this [ORG] on 18.3.2015 reveal, that Mr. [PERSON], in Writ Petition C no.70 of [DATE] was heard again on, whereupon, Mr. [PERSON] and Mr., [ORG] [GPE], also made their submissions.
[CARDINAL]. Based on the order passed by the Judge [PERSON] on 7.4.2015, Honble the Chief Justice of [GPE], constituted a [CARDINAL] Judge [PERSON], comprising of,,, and, JJ.
[CARDINAL]. On 13.4.2015 the Constitution Ninety ninth Amendment Act, [DATE], and [ORG] Act, [DATE], were notified in the Gazette of India Extraordinary.
Both the above enactments, were brought into force with effect from 13.4.2015.
[CARDINAL]. When the reconstituted [PERSON] commenced hearing on 21.4.2015, Mr. made a prayer for my recusal from the [PRODUCT], which was seconded by Mr. Mathews [PERSON] petitioner in- person in Writ Petition C no.124 of [DATE], the latter advanced submissions, even though he had been barred from doing so, by an earlier order dated 24.3.2015 extracted above.
The [ORDINAL] judgment was rendered, by a [CARDINAL] Judge [PERSON], by a majority of [CARDINAL], in the [ORDINAL] Judges case on [CARDINAL]. The correctness of the First Judges case was doubted by a Judge [PERSON] in of [GPE], [DATE] Supp 1 SCC 574 [ORG] [CARDINAL], which opined that the majority view, in the [ORG] case, should be considered by a larger.
As per the position expressed before us, a feeling came to be entertained, that a Commission for selection and appointment, as also for transfer, of Judges of the higher judiciary should be constituted, which would replace the prevailing procedure, for appointment of [GPE] and Chief Justices of [ORG] and, contemplated under [PERSON] [CARDINAL] and 217 1.
The amendment, received the assent of the President on [DATE].
It was however given effect to, with effect from 13.4.2015 consequent upon its notification in the Gazette of India Extraordinary Part II, [SECTION_UNK].
The same was also brought into force, with effect from 13.4.2015 by its notification in the Gazette of India Extraordinary Part II, [SECTION_UNK].
The Judges case- [DATE] [EVENT] 87 [DATE] [ORG] [CARDINAL]. [DATE].
The Union Law Minister addressed a letter dated 18.3.1981 to the Governor of [PRODUCT] and to Chief Ministers of all other [GPE].
The addressees were inter [PERSON], [CARDINAL] of [ORG], should as far as possible be from outside the in which is situated.
Through the above letter, the addressees were requested to.a obtain from all additional Judges working in the High Courts.their consent to be appointed as permanent Judges in any other in the country.
The above noted letter required, that the concerned appointees.be required to name [CARDINAL] High Courts, in order of preference, to which they would prefer to be appointed as permanent Judges and b obtain from persons who have already been or may in the future be proposed by you for initial appointment their consent to be appointed to any other [ORG] in the country along with a similar preference for [CARDINAL] High Courts.
The Union Law Minister, in the above letter clarified, that furnishing of their consent or indication of their preference, would not imply any commitment, at the behest of the Government, to accommodate them in accordance with their preferences.
In response, quite a few additional Judges, gave their consent to be appointed outside their parent [ORG].
i [PERSON] and the other petitioners felt, that the letter dated [PRODUCT] was a direct attack on the independence of the judiciary, and an uninhibited assault on a vital basic feature of the [LAW].
A series of [ORG] in [GPE] passed resolutions, condemning the letter dated 18.3.1981, as being subversive of judicial independence.
They demanded the withdrawal of the letter.
Since that was not done, a writ petition was filed by the above Associations in the Bombay High Court, challenging the letter dated 18.3.1981.
An interim order was passed by [ORG], restraining the Union Law Minister and the Government from implementing the letter dated 18.3.1981.
While the matter was pending before this, the Union Law Minister and, filed a transfer petition under [LAW] The transfer petition was allowed, and the writ petition filed in the Bombay High Court, was transferred to [ORG].
Rather than being appointed for a further term of, their appointment was extended for, from.
These short term appointments were assailed, as being unjustified under [LAW], besides being subversive of the independence of the judiciary.
This writ petition was also transferred for hearing to [ORG].

"

Statutes:['Constitution_226', 'Constitution_136', 'Constitution_14', 'Constitution_16', 'Constitution_227', 'Constitution_133', 'Constitution_246', 'Constitution_1', 'Constitution_21', 'Constitution_32', 'Constitution_19', 'Constitution_141', 'Constitution_4', 'Constitution_31', 'Constitution_12', 'Constitution_2', 'Constitution_39', 'Constitution_311', 'Constitution_13', 'Constitution_5', 'Constitution_3', 'Constitution_6', 'Constitution_15']

###


Format your response as follows:
"Statutes applied: [List of applicable statutes]


Instructions:

Learn from the examples provided in the context to understand the task of charge or statute prediction.
Your response should be focused on providing the exact statute or charge that aligns with the legal principles and precedents applicable to the given facts.
In your response, include only the statutes you are most confident about.
Ensure that the statutes generated as responses are valid and recognized legal statutes. Avoid generating fabricated or invalid statutes.
The model's performance will be evaluated based on its ability to predict the correct statute, include only confident statutes, and showcase creativity in its predictions.


Fact Statement: ``` {Fact_Statement}```
"""
    response = get_completion(prompt)
    list.append(response)
    time.sleep(20)


# In[ ]:





# ### using Open source LLMS 

# In[ ]:


pip install sentence_transformers
pip install langchain
pip install pytorch


# ### flan-t5

# In[ ]:


from langchain.llms import HuggingFacePipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM

model_id = 'google/flan-t5-large'# go for a smaller model if you dont have the VRAM
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id,torch_dtype=torch.float32,device_map ='auto')

pipe = pipeline(
    "text2text-generation",
    model=model, 
    tokenizer=tokenizer,
    max_length= 1000#put the max length here based on the inputs
)

local_llm = HuggingFacePipeline(pipeline=pipe)


# In[ ]:


from langchain import PromptTemplate, HuggingFaceHub, LLMChain


# In[ ]:


template = """Task:You are given a fact statement delimited by triple backticks (```) and a list of statutes. Your task is to identify the statutes applicable to the fact statement from the given list of statutes along with the words or lines from the fact statement due to which the statute is appicable for each of the statute applicable. Each statute in the list consists of a title and a description of its scope and provisions.


List of statutes:

Statute S1:
Title: Power of High Courts to issue certain writs
Description: (1) Notwithstanding anything in Article 32 every High Court shall have powers, throughout the territories in relation to which it exercise jurisdiction, to issue to any person or authority, including in appropriate cases, any Government, within those territories directions, orders or writs, including writs in the nature of habeas corpus, mandamus, prohibitions, quo warranto and certiorari, or any of them, for the enforcement of any of the rights conferred by Part III and for any other purpose 

Statute S2:
Title: Punishment for Murder
Description: Whoever commits murder shall be punished with death, or 1 [imprisonment for life], and shall also be liable to fine. Substituted by Act 26 of 1955, section 117 and Schedule, for "transportation for life" (w.e.f. 1-1-1956).

Statute S3: 
Title: Equality before law
Description: The State shall not deny to any person equality before the law or the equal protection of the laws within the territory of India.

Statute S4:
Title: Special leave to appeal by the Supreme Court
Description: (1) Notwithstanding anything in this Chapter, the Supreme Court may, in its discretion, grant special leave to appeal from any judgment, decree, determination, sentence or order in any cause or matter passed or made by any court or tribunal in the territory of India. (2) Nothing in clause (1) shall apply to any judgment, determination, sentence or order passed or made by any court or tribunal constituted by or under any law relating to the Armed Forces.

Statute S5:
Title: Remedies for enforcement of rights conferred by this Part
Description: (1) The right to move the Supreme Court by appropriate proceedings for the enforcement of the rights conferred by this Part is guaranteed. (2)The Supreme Court shall have power to issue directions or orders or writs, including writs in the nature of habeas corpus, mandamus, prohibition, quo warranto and certiorari, whichever may be appropriate, for the enforcement of any of (3) Without prejudice to the powers conferred on the Supreme Court by clauses (1) and (2), Parliament may by law empower any other court to exercise within the local limits of its jurisdiction all or any of the powers exercisable by the Supreme Court under clause (2). (4) The right guaranteed by this Article shall not be suspended except as otherwise provided for by this Constition.

….


Your response should include the statutes applicable to the fact statement along with the words or lines from the fact statement due to which the statute is appicable for each of the statute applicable. The applicable statute should be mentioned exactly as it appears in the list, and the words should be from the fact statement.


Fact Statement: ```{Fact_Statement}```
"""




# In[ ]:


prompt = PromptTemplate(template=template, input_variables=["Fact_Statement"])


# In[ ]:


llm_chain = LLMChain(prompt=prompt, 
                     llm=local_llm
                     )


# In[ ]:


Fact_Statement = 'put your fact statement here Fact_Statement'


# In[ ]:


print(llm_chain.run(Fact_Statement))


# In[ ]:





# ### Bloom

# In[ ]:





# In[ ]:


from langchain.llms import HuggingFacePipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM,AutoModel,TFGPT2Model

model_id = 'bigscience/bloom-560m'# go for a smaller model if you dont have the VRAM
tokenizer = AutoTokenizer.from_pretrained(model_id )
model = AutoModelForCausalLM.from_pretrained(model_id,torch_dtype=torch.float32,device_map ='auto' )

pipe = pipeline(
    "text-generation",
    model=model, 
    tokenizer=tokenizer,
    max_length = #put the max length here based on the inputs
    
)

local_llm = HuggingFacePipeline(pipeline=pipe)




# In[ ]:





# In[ ]:





# In[ ]:


template = """Task: Given a fact statement delimited by triple backticks (```), predict the correct statute applicable to the fact statement in the format given in context and provide the exact words from the fact statement that justifies the statute's applicability as an explanation. Your task is to accurately predict the statute limited to the statutes provided in the context and provide the corresponding words as explanation.

Instructions: Read the provided fact statement delimited by triple backticks and determine the appropriate statute limited to the statutes provided in the context in the format same as context that applies to the situation described. Extract the exact words from the fact statement that justifies the applicability of the selected statute . Ensure your predictions are accurate and avoid providing any speculative or irrelevant information. Your response should be in the same format as the context provided.
Explaination must only contains the words or lines from the Fact Statement and use only those statutes given in the context.

Context:

The following examples demonstrate the format for predicting applicable statutes and the lines that justifies their applicability. Learn from these examples and focus on accuracy while predicting the statutes and extracting relevant lines.

The Fact Statement are the supreme court cases , the statutes applied are the laws applicable on the Fact Statement with explaination containing the lines from Fact Statement due to which the statutes are applicable.

####

Fact Statement: "The complainant P1 filed a Special Leave Petition in this court seeking leave to appeal against the judgment dated 6th April, 1993 of the High Court.  The incident for which these accused were charged is the murder of P2, son of the complainant P1 (appellant) on 13th June, 1982. As per the case of the prosecution, the complainant along with his son P2 (deceased) was getting his maize field weeded through the help of a few labourers on the morning of 13th June, 1982. His real brother P3 came on the spot and forbade the complainant from doing so. The complainant insisted that he had right to carry on the work in the field which belonged to him. On this P3, who was accompanied by his son accused P4 and P5 abused the labourers and drove them away from the field. The complainant took strong objection to this but the accused party started abusing the complainant and his son and started pelting stones on them. The complainant and his son also threw stones on the opposite party in their defence. In the meantime, some villagers came and intervened in the fight. As a result of this, the accused persons went away. The complainant and his son P2 continued with the work in the field. After a few hours, that is about 10.00 a.m., few villagers informed the complainant that the accused persons were coming back armed with weapons. The complainant did not pay heed to this warning thinking that the accused persons were his close relations. Within a short time, all the seven accused persons reached the spot. Seeing them, the complainant and his son P2 ran for their safety and entered the nearby house of P6. They hid themselves in a room by bolting the room from inside.2. However, as the main gate of the house had remained open, the accused persons rushed inside the house and broke open the door which had been bolted from inside. They entered the room where the complainant and his son P2 were hiding. P2 was dragged outside the room in the courtyard of the house where accused P4 is said to have given a bhala blow on his stomach. As a result of the blow, P2 fell down. Accused P3 gave a pharsa blow on the head of P2. The other accused persons also assaulted P2 with their weapons. The complainant tried to save his son but he was also assaulted by accused P7 and P5. While this was going on, the villagers accompanied by P9, P10, P11 and P12 came and intervened and saved the victims from further assault. However, P2 died on the spot. Police came in the village at about 1.00 p.m. when statement of the complainant P1 was recorded. On the basis of the said statement, an FIR was recorded and the seven accused persons were charge-sheeted and tried for the aforesaid offences. The sessions court by its judgment dated 19th June, 1992 while giving benefit of doubt to the accused persons and finding fault with the investigation acquitted all the accused persons. The State filed an appeal against the said judgment of the Sessions Court. The High Court dismissed the appeal in limine making the following observations :"As regards merits, it is clear from the perusal of the record that the witness named in the fardbayan have not been examined by the prosecution and also that the witnesses examined in Court were examined by the police after eight months from the date of occurrence. It is also clear that the Investigating Office of the case has not been examined. Therefore, there are no merits. Further the appeal is barred by limitation also, which cannot be considered." Against the said judgment of the High Court, the complainant filed a Special Leave Petition in this Court. Leave was granted. Hence the present appeal. The appeal has been registered for final hearing."

Statutes Applied: ["S4","S11","S2","S15"]


Explanations: 

1."S4": "$S4 The complainant P1 filed a Special Leave Petition in this court seeking leave to appeal against the judgment dated 6th April, 1993 of the High Court.Against the said judgment of the High Court, the complainant filed a Special Leave Petition in this Court. S4$"

2."S11": "$11the accused persons rushedinside the house and broke open the door which had been bolted from inside. They entered the room where the complainant and his son P2 were hiding.
the accused persons were coming back armed with weapons.and $S11 the seven accused persons
were charge-sheeted and tried for the aforesaid offences.P2 was dragged outside the room in the courtyard of the house where accused P4 is said to have given a bhala blow on his stomach. As a result of the blow, P2 fell down. Accused P3 gave pharsa blow on the head of P2. The other accused persons also assaulted P2 with their weapons. S11$"

3."S2": "$S2 However, P2 died on the spot. P2 was dragged outside the room in the courtyard of the house where accused P4 is said to have given a bhala blow on his stomach. As a result of the blow, P2 fell down. Accused P3 gave pharsa blow on the head of P2. The other accused persons also assaulted P2 with their weapons.S2$"

4."S15": "$S15 the accused persons were coming back armed with weapons.$S2 $S15 P2 was dragged outside the room in the courtyard of the house where accused P4 is said to have given a bhala blow on his stomach. As a result of the blow, P2 fell down. Accused P3 gave pharsa blow on the head of P2. The other accused persons also assaulted P2 with their weapons. S15$ ."

###

Fact Statement: "The appellant P1 is convicted by the Additional Sessions Judge by judgment and order dated 5th/9th August, 1991 and was sentenced to death subject to confirmation by the High Court. Appellant appealed against the conviction and sentence which was partly allowed. The order with regard to the death penalty was set aside and appellant was sentenced to suffer imprisonment for life and to pay a fine of Rs.5,000/- in default thereof to undergo R.I. for 18 months. Against the said judgment and order this appeal is filed. The prosecution version as stated by P.W.1 is that he was a resident of Delhi and was dealer and manufacturer of Umbrellas; he has got two wives known as P2 and P3. His first wife P2 is residing at House No. 377 alongwith her three children, named, P4, P5 and P6 and her other three children have been residing with him at his House No.1584. They are P1 (appellant), P7 and P8. His second wife P3 (deceased) had been residing with him alongwith her six children including daughter P9 aged 17 years (deceased). He was also having a house where his first wife was residing earlier before she shifted. There was dispute between him on the one hand and his wife P2 and her children on the other with regard to the house. The appellant-accused and his brother were insisting for the transfer of the said house in the name of their mother at the earliest. For transferring the said house in the name of his first wife, he went to Tis Hazari on 17th October, 1988 alongwith his son-in-law P10 and met his counsel who advised him to come on the next day. Hence, the said property could not be transferred in the name of his wife. At about 6.00 P.M., when he was sleeping in the house, he woke up on hearing the noise of a quarrel and P8, daughter of P2 abusing P3. He slapped P8 and asked her to desist from abusing P3. After this the appellant and P7 came into the house, P7 went inside the room alongwith P8 and then came out with a dagger. P7 abused him and stabbed on his left eye, he fell down. At that time, P3 intervened and protested saying as to why he was beating his handicapped father. At that time, appellant snatched away dagger from P7 and started stabbing P3 repeatedly. At that stage, his daughter P9 intervened and asked the appellant as to why he was stabbing P3. P7 stated that she was the root of all troubles so the appellant started stabbing P9 at her abdomen, neck and other parts of her body. After sometime when persons collected outside, the appellant ran away. Within minutes P3 and P9 died at the spot. Police recorded the statement of P.W.1 as FIR. Appellant as well as his brother P7 were chargesheeted. P7 was convicted. He has not preferred any appeal against his conviction. After considering the evidence of the prosecution witnesses particularly P.W.1, P.W.2 and P.W.4 who have unequivocally deposed that both the deceased persons were killed by the appellant by inflicting dagger blows, the High Court has rightly arrived at the conclusion that accused is guilty for the offence for which he is charged. Mr. R.K. Jain, learned senior counsel for the appellant, has not raised any contention with regard to the conviction of the appellant."


Statutes Applied: ["S43","S24","S2","S13","S51"]


Explanations: 

1."S43": "stabbed on his left eye,  started stabbing P3 repeatedly, started stabbing P9 at her abdomen, neck and other parts of her body, both the deceased persons were killed by the
appellant by inflicting dagger blows,$S2 P3 and P9 died at the spot."

2."S24": "stabbed on his left eye,  started stabbing P3 repeatedly, started stabbing P9 at her abdomen, neck and other parts of her body, both the deceased persons were killed by the
appellant by inflicting dagger blows."

3."S2": "$S2 P3 and P9 died at the spot,both the deceased persons were killed by the
appellant by inflicting dagger blows,"

4."S13": "stabbed on his left eye,  started stabbing P3 repeatedly, started stabbing P9 at her abdomen, neck and other parts of her body."

###

Fact Statement : ```{Fact_Statement}```

Note: Provide only those statutes which are present in the context examples and Extract the exact words from the fact statement that justifies the applicability of the selected statute that you are certain about based on the given fact statements. Avoid speculation and ensure the accuracy of your response.The explanations must be shorter.

"""


# In[ ]:


prompt = PromptTemplate(template=template, input_variables=["Fact_Statement"])

llm_chain = LLMChain(prompt=prompt, 
                     llm=local_llm
                     )


# In[ ]:


Fact_Statement = "These appeals are directed against the judgment of a High Court whereby an appeal and a criminal revision were disposed of. The appellants were found guilty and sentenced to undergo various terms of sentences. The Criminal Appeal was filed by three appellants questioning the conviction and sentence as recorded. Complainant filed a revision petition stating that she was entitled to compensation. Background facts giving rise to the trial are essentially as follows: &quot;The complainant and the appellants are first cousins, and as such are closely related to each other. Their grandfather was P1. As per site plans Ex. PP prepared by P2, P3 PW4 and Ex. PT prepared by P3 PW9 (I.0.), it shows that the place of occurrence was in the common land owned both by the appellants and the complainant party. The tube well of which the pipes were being taken out by the appellants, was also in the common piece of land. P4 (hereinafter referred to as &#39;deceased&#39;) was standing in the water-course point B (Ex.PT). Complainant P5 was standing in the common land Point C (Ex.PT) and P6 was standing at Point D (Ex. PT). It is the appellants who went 16 to 35 feet towards the complainants where deceased P4 and the other two witnesses P5 (PW6) and P6 (PW7) were standing and thereafter attacked them. P5 (PW6) asked the appellants not to take out the iron and plastic pipes of the tube well, but firstly to talk to the elders. Malkiat Singh, Patwari (PW4), who is a key witness in regard to the ownership of the piece of land where the tubewell was installed, was not put any question regarding the ownership of the common land. P5 (PW6), in his testimony before the Court, stated that the appellants on 7.1.2001 at about 1.00 P.M. armed with spades came to the tube well and started removing the pipes, which was jointly owned by both the appellants and complainant party. On being stopped, $S6 the appellants felt offended and $S24 $S19 attacked the complainant party. S19$ S24$ S6$ He (PW6) has further stated that there was no dispute regarding the joint property, but the appellants were not on visiting terms with them as far social functions were concerned. $S24 P4 was attacked S24$ in the joint water channel and across the water channel there was the field of P7, father of P8. After leaving the common pipes of land where the tube well was installed, rest of the land had been divided by both the parties and they were cultivating the land separately and peacefully. The complainant party did not have any weapons in their hands when they had gone to stop the appellants. This witness (PW6) has stated that they did not go near the appellants, but asked them not to remove the pipes. They were at that time standing at a distance of 5-6 karms. P6 (PW7) has also reiterated the same. P5 (PW6) has stated, that P9 and P10 have their fields at a distance of about half a kills from the place of occurrence. Both these witnesses P5 (PW6) and P6 (PW7) corroborate each other inter-se and also corroborate the FIR The medical evidence also corroborates the statements given by the eye witnesses. Dr. D1 (PW 1) has stated in his testimony, that on examining P5 he found that $S19 he had received one incised wound injury S19$ on the scalp left parietal area vertical in position. Similarly on examining P6, he found $S19 the first injury to be an incised wound. Second and third were abrasions on the left shoulder and neck. The fourth injury was a lacerated wound on the right parietal area of scalp. On the post-mortem conducted on P4, an incised wound was found on the parietal area of the scalp, S19$ about 12 cms from right ear pinna backwards, traversing part of left parietal area of scalp to left occipital area. The medical evidence corroborates the ocular account.&quot; Trial court took note of the fact that the appellants and the members of the complainant party are related to each other closely. The dispute arose because of conflicting claims as to the ownership of the land. It was submitted that the occurrence took place $S6 when the members of the complainant party came forward and obstructed the appellant from doing the work and restrained them from pulling out the pipe. S6$ There was exchange of hot words and in the process, the occurrence, according to the prosecution, took place. In essence it was submitted that the accused were exercising the right of private defence or in the alternative the occurrence took place in the course of a sudden quarrel. Stand of the State was that there appears to be some exchange of words. The trial court found substance in the plea and found the accused persons guilty. Before the High Court it was submitted that the factual scenario has not been correctly appreciated by the trial court. It noted that the appellants pulled out the iron and plastic pipes which were installed on the land jointly owned by both the parties. Since the accused persons pulled out the pipes it was natural that the members of the complainant party who were standing at a distance of 16 to 35 feets from the appellants intervened and asked them not to pull out the pipes unless the elders take a decision. The appellants did not pay any heed"


# In[ ]:


print(llm_chain.run(Fact_Statement))


# In[ ]:





# ### gpt-2

# In[ ]:


from langchain.llms import HuggingFacePipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM,AutoModel,TFGPT2Model

model_id = 'gpt2'# go for a smaller model if you dont have the VRAM
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id,torch_dtype=torch.float32,device_map ='auto')

pipe = pipeline(
    "text-generation",
    model=model, 
    tokenizer=tokenizer,

)

local_llm = HuggingFacePipeline(pipeline=pipe)


# In[ ]:


template = """Task: Given a fact statement delimited by triple backticks (```), predict the correct statute applicable to the fact statement and provide an explanation based on the part of the fact statement that justifies the statute's applicability. Your task is to accurately predict the statute and provide the corresponding explanation.

Instructions: Read the provided fact statement delimited by triple backticks and determine the appropriate statute that applies to the situation described. Extract the relevant part of the fact statement that justifies the applicability of the selected statute and provide an explanation for its relevance. Ensure your predictions are accurate and avoid providing any speculative or irrelevant information. Your response should be in the same format as the examples provided.

Context: The following examples demonstrate the format for predicting applicable statutes and providing explanations based on the given fact statements. Learn from these examples and focus on accuracy while predicting the statutes and extracting relevant explanations.

Fact Statement: "The appellant on February 9, 1961 was appointed as an Officer in Grade III in the respondent Bank ( for short 'the Bank'). He was promoted on April 1, 1968 to the Grade officer in the Foreign Exchange Department in the Head Office of the Bank. Sometime in 1964, MCH Society ( for short 'the Society') was formed of which the appellant was one of the chief promoters and thereafter its Secretary. The object of the Society was to construct residential premises for the employees of the Bank and its other members. It appears that the complaint was received in respect of the affairs of the Society relating to misappropriation of the funds of the Society and consequently, in exercise of the powers under Section S of Act A1, the Registrar on April 23, 1969 instituted an inquiry thereof. P1 was appointed the Registrar's nominee who on October 4, 1969; submitted the report holding the appellant and two other office bearers of the Society negligent in dealing with the funds of the Society causing a loss to the tune of Rs. 3,59,000/-. The Registrar on October 21, 1969, passed an order appointing an officer under Section S of A1 to assess the loss caused to the Society. However, the Government by its order dated November 29, 1969 annulled the Registrar's order dated April 23, 1969 and October 21, 1969 and directed a fresh inquiry into the affairs of the Society. On December 17, 1969, the Bank issued show cause notice to the appellant to explain within fifteen days his alleged negligent conduct in dealing with the affairs of the Society as revealed in the report dated 4th October, 1969. In the meantime, P2 came to be appointed by the Registrar vide his order dated 26th July, 1969, to make inquiries under Section S of A1. Petitioner by his reply dated 18/22th January, 1970 submitted his explanation and also challenged the legality of the inquiry and the findings recorded therein. On 5th March, 1970, P3, treasurer of the Society and an employee of the Bank criminal complaints in the Court of Addl. Chief Presidency Magistrate  alleging that the appellant and two other office bearers of the society had dishonestly misappropriated a sum of Rs. 51,000/ and Rs. 80,000/- respectively which was entrusted to the appellant in his capacity as Promoter and Secretary of the Society and thereby committed criminal breach of trust. The Magistrate framed the charges against the appellant. The Bank having regard to the serious misconduct of the appellant involving moral turpitude vide its order dated 3rd November, 1970 suspended the appellant pending trial. The appellant protested this action of the Bank complaining that he was not given an opportunity of hearing before passing the order of suspension. In the meantime, P2, the authorized officer appointed by the Registrar vide his order dated 9th October, 1971 held the appellant liable to pay Rs. 2,36,000/- to the Society in addition to the amount of Rs. 2,03,000/- for which he (the appellant) and two other office bearers of the Society were held jointly liable. The Bank in view of this finding, vide its order dated 29th November, 1971 terminated the services of the appellant with effect from 1st December, 1971 along with notice pay. The appellant protested against the Action of the Bank and on 3rd December, 1971 filed detailed representation against the order of termination. The Bank replied to the appellant's representation and justified its action. The appellant on 28th December, 1971 submitted his reply to the Bank stating, inter alia, that the termination of his services was not simplicitor and was in violation of the principles of natural justice; that no opportunity of hearing was given to him; that the termination order attached stigma. The appellant aggrieved by the findings and order made by P2 appealed before Tribunal. In the meantime, the criminal proceedings ended in conviction vide order dated 27th March, 1972 passed by the Addl. Chief Metropolitan Magistrate. The appellant challenged the order of conviction and sentence in the High Court  and during the pendency of the said appeal, the Tribunal vide its order dated April 12, 1973 dismissed the appellant's appeal but reduced the liability by Rs. 72,000/-. On November 12, 1973, the High Court allowed the criminal appeal and acquitted the appellant. The High Court, however, in its order observed that since the services of the appellant were terminated in view of the criminal proceedings and since the appellant has been acquitted, representation, if a any, by the appellant to the Bank for reinstatement may be considered sympathetically. Taking clue from the observations made by the High Court, the appellant filed three representations, the last being dated 3rd May, 1975 requesting the Bank to revoke the order of termination and be reinstated. The Bank vide its communication dated May 21, 1975 refused to reinstate the appellant. The appellant, therefore, on July 23, 1975 filed the writ petition in the High Court for quashing the orders dated 29th November 1971, 27th December, 1971 and 21st May, 1975 passed by the Bank. The learned Single Judge of the High Court by his judgment and order dated December 6/7, 1979 granted desired relief to the appellant. \


Applicable Statutes:1. "S52"
                    2. "S62"
                    3. "S3"
                    4. "S10"



Explanation:

1. S52 : $S52 The appellant on February 9, 1961 was appointed as an Officer in Grade III in the respondent Bank ( for short &#39;the Bank&#39;).On 5th March, 1970, P3, treasurer of
the Society and an employee of the Bank criminal complaints in the Court of Addl. Chief Presidency
Magistrate alleging that the appellant and two other office bearers of the society had dishonestly
misappropriated a sum of Rs. 51,000/ and Rs. 80,000/- respectively which was entrusted to the
appellant in his capacity as Promoter and Secretary of the Society and thereby committed criminal$S52
breach of trust.

2.S62: $S62 held the appellant liable to pay Rs. 2,36,000/- to the Society in addition to the
amount of Rs. 2,03,000/- for which he (the appellant) and two other office bearers of the Society were
held jointly liable. $S62

3.S3 : $S3 The appellant on 28th December, 1971 submitted his reply to the Bank stating,
inter alia, that the termination of his services was not simplicitor and was in violation of the principles of
natural justice; that no opportunity of hearing was given to him; S3$



Fact Statement : ```{Fact_Statement}```

Note: Provide only those statutes and explanations that you are certain about based on the given fact statements. Avoid speculation and ensure the accuracy of your responses.

"""



# In[ ]:





# In[ ]:





# In[ ]:


llm_chain = LLMChain(prompt=prompt, 
                     llm=local_llm
                     )


# In[ ]:





# In[ ]:


Fact_Statement = "These appeals are directed against the judgment of a o undergo various terms of sentences. The Criminal Appeal was filed by three appellants questioning the conviction and sentence as recorded. Complainant filed a revision petition stating that she was entitled to compensation. Background facts giving rise to the trial are essentially as follows: &quot;The complainant and the appellants are first cousins, and as such are closely related to each other. Their grandfather was P1. As per site plans Ex. PP prepared by P2, P3 PW4 and Ex. PT prepared by P3 PW9 (I.0.), it shows that the place of occurrence was in the common land owned both by the appellants and the complainant party. The tube well of which the pipes were being taken out by the appellants, was also in the common piece of land. P4 (hereinafter referred to as &#39;deceased&#39;) was standing in the water-course point B (Ex.PT). Complainant P5 was standing in the common land Point C (Ex.PT) and P6 was standing at Point D (Ex. PT). It is the appellants who went 16 to 35 feet towards the complainants where deceased P4 and the other two witnesses P5 (PW6) and P6 (PW7) were standing and thereafter attacked them. P5 (PW6) asked the appellants not to take out the iron and plastic pipes of the tube well, but firstly to talk to the elders. Malkiat Singh, Patwari (PW4), who is a key witness in regard to the ownership of the piece of land where the tubewell was installed, was not put any question regarding the ownership of the common land. P5 (PW6), in his testimony before the Court, stated that the appellants on 7.1.2001 at about 1.00 P.M. armed with spades came to the tube well and started removing the pipes, which was jointly owned by both the appellants and complainant party. On being stopped, $S6 the appellants felt offended and $S24 $S19 attacked the complainant party. S19$ S24$ S6$ He (PW6) has further stated that there was no dispute regarding the joint property, but the appellants were not on visiting terms with them as far social functions were concerned. "


# In[ ]:


print(llm_chain.run(Fact_Statement))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




