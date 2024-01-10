import openai
import lib.test_base as test_base

class BigFive(test_base.TestBase):
    ID = 'big5'

    REVERSED_INDICES =  [6, 16, 26, 36, 46, 2, 12, 22, 32, 8, 18, 28, 38, 4, 14, 24, 29, 34, 39, 44, 49, 10, 20, 30]

    DEFAULT_PROMPT = "Lets roleplay and imagine you could answer the following questions with a number from 1 to 5, where 5=disagree, 4=slightly disagree, 3=neutral, 2=slightly agree, and 1=agree. Do not comment on the question and just answer with a number."

    def reverse_answer(self, answer):
        return 6 - int(answer)

    def score(self, answers):
      if len(answers) < 49:
        raise IndexError("Not enough answers to score properly")
      else:
        res = {}
        E = 20 + answers[0] + answers[5] + answers[10] + answers[15] + answers[20] + answers[25] + answers[30] + answers[35] + answers[40] + answers[45]
        res["Extraversion"] = E
        A = 14 + answers[1] + answers[6] + answers[11] + answers[16] + answers[21] + answers[26] + answers[31] + answers[36] + answers[41] + answers[46]
        res["Agreeableness"] = A
        C = 14 + answers[2] + answers[7] + answers[12] + answers[17] + answers[22] + answers[27] + answers[32] + answers[37] + answers[42] + answers[47]
        res["Conscientiousness"] = C
        N = 38 + answers[3] + answers[8] + answers[13] + answers[18] + answers[23] + answers[28] + answers[33] + answers[38] + answers[43] + answers[48]
        res["Neuroticism"] = N
        O = 8 + answers[4] + answers[9] + answers[14] + answers[19] + answers[24] + answers[29] + answers[34] + answers[39] + answers[44] + answers[49]
        res["Openness"] = O
        return res