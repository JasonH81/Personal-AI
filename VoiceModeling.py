# Training my voice with neural network
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import librosa
import numpy as np

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

model.save('my_model.keras')