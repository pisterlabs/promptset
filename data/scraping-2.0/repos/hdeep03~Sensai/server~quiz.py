import openai
from utils import load_transcript
from transformers import GPT2TokenizerFast
from reportlab.pdfgen import canvas
import os
from reportlab.platypus import SimpleDocTemplate
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from textwrap import wrap
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.platypus import Paragraph, SimpleDocTemplate
styles = getSampleStyleSheet()
styleN = styles['Normal']
pdfmetrics.registerFont(TTFont('Vera', 'Vera.ttf'))
pdfmetrics.registerFont(TTFont('VeraBd', 'VeraBd.ttf'))
pdfmetrics.registerFont(TTFont('VeraIt', 'VeraIt.ttf'))
pdfmetrics.registerFont(TTFont('VeraBI', 'VeraBI.ttf'))
KEY=os.getenv("KEY")

CACHE_PATH = './cache/'
openai.api_key = os.environ["OPENAI_API_KEY"]
n = 12000
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def get_transcript(video_id):
    transcript_segments = load_transcript(video_id)
    transcript = " ".join(transcript_segments['text'])
    return transcript

def generate_quiz(video_id, difficulty, quiz_type):
    transcript = get_transcript(video_id)
    fragments = [transcript[i:i+n] for i in range(0, len(transcript), n)]
    encoded = tokenizer.encode(transcript)
    numberOfTotalTokens = len(encoded)

    numberOfQuestionsPerBatch = 8
    examType = quiz_type

    quiz = f"Make a {difficulty} {numberOfQuestionsPerBatch} question {examType} quiz titled \"Cool Quiz\" based on this transcript without question numbers: "

    encoded = tokenizer.encode(quiz)
    numberOfPromptTokens = len(encoded) 

    errorFlag = False
    outputArr = []
    #print(fragments)
    for i, text in enumerate(fragments):
        encoded = tokenizer.encode(text)
        numberOfBatchTokens = len(encoded)
        if numberOfBatchTokens > 150:
            totalPrompt = quiz + " " + text

            model_engine = "text-davinci-003"

            encoded = tokenizer.encode(totalPrompt)
            numberOfBatchTokens = len(encoded)

            try:
                completion = openai.Completion.create(
                    engine=model_engine,
                    prompt=totalPrompt,
                    max_tokens= 4097 - numberOfBatchTokens,
                    n=1,
                    stop=None,
                    temperature=0.2,
                )
                response = completion.choices[0].text
                outputArr.append(response)
            except:
                errorFlag = True
        else:
            print("frag too short")
    if errorFlag:
        return None
    else:
        newArr = []
        for i in range(0, len(outputArr)):

            if "Cool Quiz" in outputArr[i]:
                newSplices = outputArr[i].split("Cool Quiz")
                newArr.append(newSplices[1][2:])
            else:
                newArr.append(outputArr[i])
        returnString = ""
        for i in range (0, len(newArr)):
            returnString += "\nPart " + str(i + 1) + "\n" + newArr[i] + "\n"
        
        path = create_pdf(returnString, video_id)

        return path, returnString

'''
def create_pdf(quiz_text, video_id):
    path = os.path.join('static', video_id+'_quiz.pdf')
    c = canvas.Canvas(path, pagesize=letter)
    c.setFont('Vera', 12)
    t = c.beginText()

    t.setTextOrigin(50, 700)
    
    temp_line = quiz_text.split('\n')
    fin = []
    for line in temp_line:
        if line != '':
            fin.append("\n".join(wrap(line, 80)))
    out = "\n".join(fin)
    t.textLines(out)
    c.drawText(t)
    c.save()
    return path
'''

def create_pdf(quiz_text, video_id):
    path = os.path.join('static', video_id+'_quiz.pdf')
    story = []
    
    temp_line = quiz_text.split('\n')
    fin = []
    for line in temp_line:
        if line != '':
            fin.append("\n".join(wrap(line, 80)))
    out = "\n".join(fin)
    for x in out.split("\n"):
        story.append(Paragraph(x, styleN))
    doc = SimpleDocTemplate(path, pagesize=letter)
    doc.build(story)
    return path

if __name__ == "__main__":
    create_pdf("\nPart 1\nQ1: What is the White House push to get out of DC to highlight?\nA1: The economy, investments in infrastructure, and blue collar jobs.\n\nQ2: What is President Biden's goal in terms of economics?\nA2: His goal is to build from the bottom up and the middle out, so that the poor have a chance and the middle class does well.\n\nQ3: What is the effect of the trillion dollars worth of money from the inflation reduction act, the infrastructure legislation, and the chips manufacturing bill?\nA3: It will make a gigantic difference and has already created 800,000 manufacturing jobs in two years. It will also reduce the debt and deficit by 1.7 trillion dollars over two years.\n\nQ4: Why do many Americans think the economy is in bad shape, despite the low unemployment, growth, and inflation?\nA4: Many Americans don't understand the policy changes that have been passed, and they are used to hearing negative news on the television.\n\nQ5: What does President Biden think is the cause of the country's deep divisions?\nA5: President Biden believes that there has been a deliberate effort by the last administration to play on people's fears and appeal to base instincts. He also believes that the party started to take for granted ordinary blue collar workers, which has hurt them.\n\nPart 2\n1. What did the President say was the reason why China needed a relationship with the United States and Europe?\n\n2. What did the President say he had to do this summer to make sure China was not engaging in the same kind of deal as Russia?\n\n3. What did the President say he had been able to do to unite NATO?\n\n4. What did the President say he was not at liberty to speak about regarding classified documents?\n\n5. What did the President say he would do if he thought there was any health problem that would keep him from being able to do the job?\n\n6. What did the President say was his intention regarding running for reelection?\n", "0UYrX7L29Us")
