import speech_recognition as sr

# initialize the recognizer
r = sr.Recognizer()

# read the audio file
with sr.AudioFile('pgbi5n.wav') as source:
    audio_data = r.record(source)

# recognize the speech in the audio data
transcript = r.recognize_google(audio_data)

# print the transcript
print(transcript)