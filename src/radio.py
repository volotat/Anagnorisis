from TTS.api import TTS
import sounddevice as sd
from unidecode import unidecode

text =  '''"Как на войне" is a song by the Russian band "Агата Кристи" from their album "Позорная звезда". The song was released in 1992 and is a post-punk/gothic rock track. The lyrics are about a relationship that is on the brink of collapse, with the narrator comparing the situation to a war. The song is driven by a pounding drum beat and a haunting guitar riff, creating a sense of tension and anxiety.
The lyrics of the song are full of imagery and metaphor, with the narrator comparing the relationship to a war. The song is a powerful and emotional track that captures the complexities of love and relationships.
The song has been covered by several artists and has been featured in several films and television shows. It is a classic track that has stood the test of time and continues to be a favorite among fans of post-punk and gothic rock.'''


tts = TTS('tts_models/en/ljspeech/vits')

# Split the text into sentences or paragraphs, depending on your needs
sentences = text.split('\n')

# Romanize each sentence and join them back together
romanized_text = "\n".join([unidecode(sentence) for sentence in sentences])


#romanized_text = translit(text, 'ru', reversed=True)
wav = tts.tts(romanized_text)
sd.play(wav, 22050)
sd.wait()



def gen_prompt():

  prompt = f'''Imagine you are a dj at the radio. Create some interesting fact about the song "Как на войне" by "Агата Кристи" band from the album "Позорная звезда" as a short cohesive story that could be told before the songs.  Mediadata: Year of release - "1992", Ganre - "Post-Punk/Gothic Rock".

Do not print any introduction. If you don't know much about this song, do not try to made stuff up, tell about something relative, the band, the ganre or anything else.. Be creative, do not use general endings like "So sit back, relax, and let the haunting melodies of the song transport you to a world of dark beauty and introspection.". You do not need to use all the information about the song provided.'''