import os
import random
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
import tensorflow as tf
from pydub import AudioSegment
import matplotlib.pyplot as plt
import noisereduce as nr


segment_length = 3000  # audio length in ms
metadata = []

# def classify(result):
#     if result[0][0] >= 0.75:
#         return "Not Parkinson's"
#     elif result[0][1] >= 0.75:
#         return "Parkinson's"
#     else:
#         return "Uncertain"

def spectral_gating(audio, rate):
    audio_np_array = np.array(audio.get_array_of_samples())

    reduced_audio_array = nr.reduce_noise(y=audio_np_array, sr=rate, stationary=True, prop_decrease=0.8)

    reduced_audio_segment = AudioSegment(reduced_audio_array.tobytes(), frame_rate=rate, sample_width=audio.sample_width, channels=audio.channels)
    
    return reduced_audio_segment

def extract_mfcc_feature(file_name):
    audio_data, sample_rate = librosa.load(file_name)
    mfcc_feature = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40, n_fft=1918)
    return np.mean(mfcc_feature.T, axis=0)


def cut_audio_and_create_csv():
    # cut audio files into 3 second slices.
    for directory in ["data\\pd", "data\\npd"]:
        print("\ndirectory: ", directory, "...")
        files_in_dir = os.listdir(directory)
        for file in files_in_dir[1:]:
            
            audio = AudioSegment.from_wav(os.path.join(directory, file))

            # do spectral gating here, to remove noise in audio...
            print("audio segment created")
            audio = spectral_gating(audio, audio.frame_rate)
            print("spectral gating done")
            cropped_audio = audio[1000:-1000]
            segments = [cropped_audio[i:i + segment_length] for i in range(0, len(cropped_audio), segment_length)]

            for i, segment in enumerate(segments):
                output_file = f"{directory}\\split\\{file[:-4]}_segment_{i + 1}.wav"
                # print(segment.frame_rate)
                print("exporting segment")
                segment.export(output_file, format="wav")

            print("segments exported")
            
    print("\nloading audio clips into memory...")
    for directory in ["data\\pd\\split", "data\\npd\\split"]:
        files_in_dir = os.listdir(directory)
        for file in files_in_dir:
            metadata.append([os.path.join(directory, file), directory.split("\\")[1]])


# def predict_audio(audio_file, model):
#     pre_mfcc_feature = extract_mfcc_feature(audio_file)
#     pre_mfcc_feature = pre_mfcc_feature.reshape(1, -1)

#     pred = model.predict(pre_mfcc_feature)

#     return pred

print("preporcess audio files...")

cut_audio_and_create_csv()

print("\npreprocessing done.")

dataset = pd.DataFrame([[extract_mfcc_feature(row[0]), row[1]] for row in metadata], columns=["feature", "class"])

print("\ntest train split..")

x_train, x_test, y_train, y_test = train_test_split(dataset["feature"], dataset["class"], test_size=0.3,
                                                    random_state=random.randint(0, 100), shuffle=True)

x_train = np.array([x for x in x_train])
x_test = np.array([x for x in x_test])


label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

num_classes = len(label_encoder.classes_)
y_train_onehot = tf.keras.utils.to_categorical(y_train_encoded, num_classes=num_classes)
y_test_onehot = tf.keras.utils.to_categorical(y_test_encoded, num_classes=num_classes)

model = Sequential()

model.add(Dense(100, input_shape=(40, )))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy", metrics=['accuracy'], optimizer='adam')

history = model.fit(x_train, y_train_onehot, epochs=20, validation_data=(x_test, y_test_onehot))

test_accuracy = model.evaluate(x_test, y_test_onehot, verbose=0)
print("test accuracy:", test_accuracy)

model_path = "pmodel.keras"

model.save(model_path)

plt.figure(figsize=(12, 6))

# pprint.pprint(history.history["loss"])
# pprint.pprint(history.history["val_loss"])
# pprint.pprint(history.history["accuracy"])
# pprint.pprint(history.history["val_accuracy"])
# exit(1)

# Plot training & validation loss values
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')

# Plot training & validation accuracy values
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='lower right')

plt.tight_layout()
plt.show()