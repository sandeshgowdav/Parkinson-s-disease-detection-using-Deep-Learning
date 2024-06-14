import os
import pprint
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
import tqdm
import datetime
import streamlit as st
from scipy.signal import wiener


segment_length = 3000  # audio length in ms
# metadata = []

def cut_audio(audio):
    audio_segment = AudioSegment(audio)
    cropped_audio = audio_segment[1000:-1000]
    segments = [cropped_audio[i:i + segment_length] for i in range(0, len(cropped_audio), segment_length)]
    count = 1
    for segment in segments:
        segment.export("temp\\" + str(count) + ".wav", format="wav")
        count += 1
    return

def classify(result):
    if result[0][0] >= 0.75:
        return "Not Parkinson's"
    elif result[0][1] >= 0.75:
        return "Parkinson's"
    else:
        return "Uncertain"

def extract_mfcc_feature(file_name):
    audio_data, sample_rate = librosa.load(file_name)
    mfcc_feature = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40, n_fft=1919)
    return np.mean(mfcc_feature.T, axis=0)
    #return mfcc_feature

def apply_wiener_filter(audio_data):
    return wiener(audio_data)

# def cut_audio_and_create_csv():
#     # cut audio files into 3 second slices.
#     for directory in ["data\\pd", "data\\npd"]:
#         print("\ndirectory: ", directory, "...")
#         files_in_dir = os.listdir(directory)
#         for file in tqdm.tqdm(files_in_dir[1:]):
#             audio = AudioSegment.from_wav(os.path.join(directory, file))
#             cropped_audio = audio[1000:-1000]
#             segments = [cropped_audio[i:i + segment_length] for i in range(0, len(cropped_audio), segment_length)]

#             for i, segment in enumerate(segments):
#                 output_file = f"{directory}\\split\\{file[:-4]}_segment_{i + 1}.wav"
#                 segment.export(output_file, format="wav")

#     print("\ncreating metadate csv file")
#     for directory in ["data\\pd\\split", "data\\npd\\split"]:
#         files_in_dir = os.listdir(directory)
#         for file in tqdm.tqdm(files_in_dir):
#             metadata.append([os.path.join(directory, file), directory.split("\\")[1]])


def predict_audio(audio_file, model):
    pieces = cut_audio(audio_file)

    pred = []

    for piece in os.listdir("temp"):
        pre_mfcc_feature = extract_mfcc_feature(os.path.join("temp", piece))
        pre_mfcc_feature = pre_mfcc_feature.reshape(1, -1)

        pred.append(classify(model.predict(pre_mfcc_feature)))
    

    p = pred.count("Parkinson's")
    np = pred.count("Not Parkinson's")

    if p > np:
        return "Parkinson's"
    else:
        return "Not Parkinson's"
    
# print("preporcess audio files...")

# cut_audio_and_create_csv()

# print("\npreprocessing done.")

# dataset = pd.DataFrame([[extract_mfcc_feature(row[0]), row[1]] for row in metadata], columns=["feature", "class"])

# print("\ntest train split..")

# x_train, x_test, y_train, y_test = train_test_split(dataset["feature"], dataset["class"], test_size=0.3,
#                                                     random_state=random.randint(0, 100), shuffle=True)

# x_train = np.array([x for x in x_train])
# x_test = np.array([x for x in x_test])


# label_encoder = LabelEncoder()
# y_train_encoded = label_encoder.fit_transform(y_train)
# y_test_encoded = label_encoder.transform(y_test)

# num_classes = len(label_encoder.classes_)
# y_train_onehot = tf.keras.utils.to_categorical(y_train_encoded, num_classes=num_classes)
# y_test_onehot = tf.keras.utils.to_categorical(y_test_encoded, num_classes=num_classes)

# model = Sequential()

# model.add(Dense(100, input_shape=(40, )))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(200))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(100))
# model.add(Dropout(0.5))
# model.add(Dense(2))
# model.add(Activation('sigmoid'))

# model.compile(loss="binary_crossentropy", metrics=['accuracy'], optimizer='adam')

# model.fit(x_train, y_train_onehot, epochs=20, validation_data=(x_test, y_test_onehot))

# test_accuracy = model.evaluate(x_test, y_test_onehot, verbose=0)
# print("test accuracy:", test_accuracy)

model_path = "pmodel.keras"

# model.save(model_path)

# audio_file_path = "./test_pd/VA2lloeroun56F2605161927.wav"

# print("\n\n#################################################")

model = load_model(model_path)

# print("test for parkinsons:")
# for files in os.listdir("./test_pd"):
#     result = predict_audio(os.path.join("./test_pd",files), model)
#     print("Predicted class probabilities:", classify(result))

# print("\n\n")

# print("test for non parkinsons:")
# for files in os.listdir("./test_npd"):
#     result = predict_audio(os.path.join("./test_npd",files), model)
#     classification = classify(result)
#     print("Predicted class probabilities:", classification)

def main():
    st.title('Voice as an assisting tool for detection of Parkinson’s disease ')
    st.sidebar.title('Upload Audio File')
    uploaded_file = st.sidebar.file_uploader("Choose an audio file", type=['wav'])
    
    if uploaded_file is not None:
        # Display the uploaded audio file
        st.audio(uploaded_file)
        
        # Extract features from the uploaded audio file
        # features = extract_features(uploaded_file, mfcc=True, chroma=False, mel=False)
        # features = features.reshape(1, features.shape[0], 1)

        result = predict_audio(uploaded_file, model)
        #classification = classify(result)
        
        # Make prediction
        # prediction = model.predict(features)
        # predicted_class = np.argmax(prediction)
        
        # Map prediction index to class label
        # class_labels = ['Normal', 'Parkinsons']
        # predicted_label = class_labels[predicted_class]
        

        time = datetime.datetime.now()

        st.write(f"Predicted Class({time}): {result}")

        for file in os.listdir("temp"):
            os.unlink(os.path.join("temp", file))

if __name__ == "__main__":
    main()