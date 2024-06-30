from dotenv import load_dotenv
from openai import OpenAI
from elevenlabs.client import ElevenLabs
from elevenlabs import play
from elevenlabs import Voice
from elevenlabs import VoiceSettings

import speech_recognition as sr

# For GUI and Audio recording
import pyaudio
import os
import wave
import time
import threading
import tkinter as tk

# Training my voice with neural network
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import librosa
import numpy as np

# Define GPT-3.5 API key
client = OpenAI(
    api_key = os.getenv('openai_api_key')
)

# Define Elevenlabs API Key
client2 = ElevenLabs(
  api_key = os.getenv('elevenlabs_api_key') # Defaults to ELEVEN_API_KEY
)

def configure():
    load_dotenv()

configure()

def AI_Assistant(input):
    system_data = [
        {"role": "system", "content": os.getenv('instructions')},
        {"role": "user", "content": input}
    ]

    # Make a client.chat.completions.create() API call and set the model and messages.
    response = client.chat.completions.create(
        model = "gpt-3.5-turbo",
        messages = system_data
    )

    # Extract the AI's response from the API call and set the value to assistant_response.
    assistant_response = response.choices[0].message.content

    # Create a new dictionary to define the "assistant" role and the assistant_response as the content value, and add the dictionary to the system_data list.
    system_data.append({"role": "assistant", "content": assistant_response})

    # Print the assistant's response.
    print(assistant_response)
    audio = client2.generate(
        text = assistant_response,
        voice = Voice (
            voice_id = 'Pe3wyeqR0uxsfAqSJXXf',
            settings = VoiceSettings(stability = 0.80, similarity_boost = 0.39, style = 0.89, use_speaker_boost = True),
        ),
        model = "eleven_multilingual_v2"
        )   
    play(audio)



class Speech:
    def __init__(self):
        # Initialize the recognizer
        recognizer = sr.Recognizer()

        # Load an audio file
        audio_file = "new_voice_sample.wav"

        # Open the audio file
        with sr.AudioFile(audio_file) as source:
            # Record the audio data
            audio_data = recognizer.record(source)

            try:
                # Recognize the speech
                text = recognizer.recognize_google(audio_data)
                print("Recognized speech: ", text)
                AI_Assistant(text)
            except sr.UnknownValueError:
                print("Speech recognition could not understand the audio.")
            except sr.RequestError as e:
                print(f"Could not request results from service; {e}")


def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)


# Example dataset with your voice samples and samples from other people
X_my_voice = np.array([extract_features(file) for file in ['my_voice.wav', 'my_voice2.wav', 'my_voice3.wav', 'my_voice4.wav', 'my_voice5.wav', 'my_voice6.wav', 'my_voice7.wav', 'my_voice8.wav', 'my_voice9.wav', 'my_voice10.wav', 'my_voice11.wav', 'my_voice12.wav', 'my_voice13.wav']])
X_other_voice = np.array([extract_features(file) for file in ['other_voice1.wav', 'other_voice2.wav', 'other_voice3.wav', 'other_voice4.wav', 'other_voice5.wav', 'other_voice6.wav', 'other_voice7.wav', 'other_voice8.wav']])

# Labels
y_my_voice = np.array(['my_voice'] * len(X_my_voice))
y_other_voice = np.array(['other_voice'] * len(X_other_voice))

# Combine datasets and labels
X = np.concatenate((X_my_voice, X_other_voice), axis=0)
y = np.concatenate((y_my_voice, y_other_voice), axis=0)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Reshape features for Conv1D input
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

model = Sequential()
model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(X.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Adding dropout for regularization
model.add(Dense(2, activation='softmax'))  # Two classes: my_voice and other_voice

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=300, batch_size=16, validation_data=(X_test, y_test))

# Evaluate model performance
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')


def predict_voice(file_path):
    features = extract_features(file_path).reshape(1, X.shape[1], 1)
    prediction = model.predict(features)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
    confidence = np.max(prediction)
    print(f"Prediction: {predicted_label[0]}, Confidence: {confidence}")
    return predicted_label[0], confidence


def PredictThis():
    predicted_label, confidence = predict_voice('new_voice_sample.wav')
    if predicted_label == 'my_voice' and confidence > 0.95:  # Adjust confidence threshold as needed
        Speech()
    #elif predicted_label == 'other_voice' and not confidence == 1.0:
    #    Speech()
    else:
        audio2 = client2.generate(
        text = os.getenv('denied'),
        voice = Voice (
            voice_id = 'Pe3wyeqR0uxsfAqSJXXf',
            settings = VoiceSettings(stability = 0.80, similarity_boost = 0.39, style = 0.89, use_speaker_boost = True),
        ),
        model = "eleven_multilingual_v2"
        )   
    play(audio2)


class VoiceRecorder:
    def __init__(self):
        self.root = tk.Tk()
        self.root.resizable(False, False)
        self.button = tk.Button(text="🎤", font=("Arial", 120, "bold"),
                                command=self.click_handler)
        self.button.pack()
        self.label = tk.Label(text="00:00:00", font=("Arial", 24, "bold"))
        self.label.pack()
        self.recording = False
        self.root.mainloop()

    def click_handler(self):
        if self.recording:
            self.recording = False
            self.button.config(fg="black")
        else:
            self.recording = True
            self.button.config(fg="red")
            threading.Thread(target=self.record).start()

    def record(self):
        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paInt16, channels=1, rate=44100,
                            input=True, frames_per_buffer=1024)
        frames = []

        start = time.time()

        while self.recording:
            data = stream.read(1024)
            frames.append(data)

            passed = time.time() - start
            secs = passed % 60
            mins = passed // 60
            hours = mins // 60
            self.label.config(text=f"{int(hours):02d}:{int(mins):02d}:{int(secs):02d}")

        stream.stop_stream()
        stream.close()
        audio.terminate()

        # Save recorded audio to file
        exists = True
        i = 1
        while exists:
            if os.remove(f"new_voice_sample.wav"):
                i += 1
            else:
                exists = False

        sound_file = wave.open(f"new_voice_sample.wav", "wb")
        sound_file.setnchannels(1)
        sound_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        sound_file.setframerate(44100)
        sound_file.writeframes(b"".join(frames))
        sound_file.close()

        # Perform voice prediction and action
        PredictThis()


# Initialize voice recorder
VoiceRecorder()