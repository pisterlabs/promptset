import openai

# openai.api_key = "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"

# def paraphrase(text):
#     response = openai.Completion.create(
#         engine="text-davinci-002",
#         prompt=f"Paraphrase: {text}",
#         max_tokens=300,
#         n=1,
#         stop=None,
#         temperature=1.5,
#     )
#     return response.choices[0].text

# class Paraphraser:
#     def _init_(self):
#         self.text = """"""
    
#     def set_text(self,text):
#         self.text = text

#     def paraphrase(self):
#         return paraphrase(self.text)
# paraphrased_text = paraphrase("Naval architects never claim that a ship is unsinkable, but the sinking of the passenger-and-car ferry Estonia in the Baltic surely should have never have happened. It was well designed and carefully maintained. It carried the proper number of lifeboats. It had been thoroughly inspected the day of its fatal voyage. Yet hours later, the Estonia rolled over and sank in a cold, stormy night. It went down so quickly that most of those on board, caught in their dark, flooding cabins, had no chance to save themselves: Of those who managed to scramble overboard, only 139 survived. The rest died of hypothermia before the rescuers could pluck them from the cold sea. The final death toll amounted to 912 souls. However, there were an unpleasant number of questions about why the Estonia sank and why so many survivors were men in the prime of life, while most of the dead were women, children and the elderly.")
# print(paraphrased_text)
