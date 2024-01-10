from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import os
import time
from pydub import AudioSegment

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

story = """

دو چُوہے تھے جو ایک دُوسرے کے بہت گہرے دوست تھے۔ ایک چُوہا شہر کی ایک حویلی میں بِل بنا کر رہتا تھا اور دُوسرا پہاڑوں کے درمیان ایک گاؤں میں رہتا تھا۔ گاؤں اور شہر میں فاصلہ بہت تھا، اِس لیے وہ کبھی کبھار ہی ایک دُوسرے سے مِلتے تھے۔
ایک دِن جو مُلاقات ہوئی تو گاؤں کے چُوہے نے اپنے دوست شہری چُوہے سے کہا، ”بھائی! ہم دونوں ایک دُوسرے کے گہرے دوست ہیں۔

کِسی دِن میرے گھر تو آئیے اور ساتھ کھانا کھائیے۔“

شہری چُوہے نے اِس کی دعوت قبول کرلی اور مُقررہ دِن وہاں پہنچ گیا۔ گاؤں کا چُوہا بہت عِزت سے پیش آیا اور اپنے دوست کی خاطر داری میں کوئی کسر اُٹھا نہ رکھی۔ کھانے میں مٹر، گوشت کے ٹُکڑے، آٹا، پنیر اور میٹھے میں پکے ہوئے سیب کے تازہ ٹُکڑے اِس کے سامنے لا کر رَکھے۔

شہری چُوہا کھاتا رہا اور وہ خود اُس کے پاس بیٹھا میٹھی میٹھی باتیں کرتا رہا۔

اِس اندیشے سے کہ کہیں مِہمان کو کھانا کم نہ پڑ جائے، وہ خود گیہوں کی بالی مُنہ میں لے کر آہستہ آہستہ چباتا رہا۔
جب شہری چُوہا کھانا کھا چُکا تو اُس نے کہا، "ارے یار جانی! اگر اِجازت ہو تو میں کُچھ کہوں؟"
گاؤں کے چُوہے نے کہا، "کہو بھائی! ایسی کیا بات ہے؟"
شہری چُوہے نے کہا، "تم ایسے خراب اور گندے بِل میں کیوں رہتے ہو؟ اِس جگہ میں نہ صفائی ہے اور نہ رونق۔
چاروں طرف پہاڑ، ندی اور نالے ہیں۔ دُور دُور تک کوئی نظر نہیں آتا۔ تم کیوں نہ شہر میں چل کر رہو۔ وہاں بڑی بڑی عمارتیں ہیں۔ سرکار دربار ہیں۔ صاف ستھری روشن سڑکیں ہیں۔ کھانے کے لیے طرح طرح کی چیزیں ہیں۔ آخر یہ دو دِن کی زندگی ہے۔ جو وقت ہنسی خوشی اور آرام سے گُزر جائے وہ غنیمت ہے۔ بس اب تم میرے ساتھ چلو۔ دونوں پاس رہیں گے۔ باقی زِندگی آرام سے گُزرے گی۔"
گاؤں کے چُوہے کو اپنے دوست کی باتیں اچھی لَگیں اور وہ شہر چلنے پر راضی ہو گیا۔ شام کے وقت چل کر دونوں دوست آدھی رات کے قریب شہر کی اُس حویلی میں جا پہُنچے جہاں شہری چُوہے کا بِل تھا۔ حویلی میں ایک ہی دِن پہلے بڑی دعوت ہوئی تھی جِس میں بڑے بڑے افسر، تاجر، زمیندار، وڈیرے اور وزیر شریک ہوئے تھے۔ وہاں پہنچے تو دیکھا کہ حویلی کے نوکروں نے اچھے اچھے کھانے کھِڑکیوں کے پیچھے چُھپا رکھے ہیں۔
شہری چُوہے نے اپنے دوست، گاؤں کے چُوہے کو ریشمی یرانی قالین پر بِٹھایا اور کھِڑکیوں کے پیچھے چُھپے ہوئے کھانوں میں سے طرح طرح کے کھانے اُس کے سامنے لا کر رَکھے۔ مِہمان چُوہا کھاتا جاتا اور خُوش ہو کر کہتا جاتا، "واہ یار! کیا مزیدار کھانے ہیں۔ ایسے کھانے تو میں نے خواب میں بھی نہیں دیکھے تھے۔"
ابھی وہ دونوں قالین پر بیٹھے کھانے کے مزے لُوٹ ہی رہے تھے کہ یکایک کِسی نے دروازہ کھولا۔
دروازے کے کھلنے کی آواز پر دونوں دوست گھبرا گئے اور جان بَچانے کے لیے اِدھر اُدھر بھاگنے لگے۔ اِتنے میں دو کُتے بھی زور زور سے بھونکنے لگے۔ یہ آواز سُن کر گاؤں کا چُوہا ایسا گھبرایا کہ اُس کے ہوش و ہواس اُڑ گئے۔ ابھی وہ دونوں ایک کونے میں دُبکے ہوئے تھے کہ بِلیوں کے غُزانے کی آواز سُنائی دی۔ گاؤں کے چُوہے نے گھبرا کر اپنے دوست شہری چُوہے سے کہا، "اے بھائی! اگر شہر میں ایسا مزہ اور یہ زِندگی ہے تو یہ تُم کو مُبارک ہو۔ میں تو باز آیا۔ ایسی خُوشی سے تو مُجھے اپنا گاؤں، اپنا گندا بِل اور مٹر کے دانے ہی خُوب ہیں۔"

"""

# Split the story into sentences
sentences = story.split("۔")

# Initialize an empty AudioSegment for concatenation
combined = AudioSegment.empty()

speech_file_path = Path(__file__).parent / "dou-chuhey-concat.mp3"

# Loop through sentences and generate speech for each
for i, sentence in enumerate(sentences):

    # Add a check to sleep after every 2 calls for 1 minute - to overcome API call limit
    if i % 2 == 0 and i > 0:
        
        time.sleep(61)

    # Add the full stop back to each sentence
    sentence_with_full_stop = sentence + "۔" if i < len(sentences) - 1 else sentence
    response = client.audio.speech.create(
        model="tts-1-hd",
        voice="alloy",
        input=sentence_with_full_stop,
    )
    
    temp_speech_path = f"temp_speech_{i}.mp3"
    response.stream_to_file(temp_speech_path)
    
    # Append the audio file to the combined AudioSegment
    speech_audio = AudioSegment.from_file(temp_speech_path)
    combined += speech_audio

# Export the combined audio to a single file
combined.export(speech_file_path, format="mp3")

# Repair story - individual audio file

# speech_file_path = Path(__file__).parent / "temp_speech_34.mp3"
# response = client.audio.speech.create(
#   model="tts-1-hd",
#   voice="alloy",
#   input= "ابھی وہ دونوں ایک کونے میں دُبْکے ہوئے تھے کہ بِلِّیوں کے گُزَرنے کی آواز سُنائی دی۔"

# )

# response.stream_to_file(speech_file_path)