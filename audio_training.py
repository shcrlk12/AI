import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from keras import layers, models
from sklearn.preprocessing import LabelEncoder
import itertools
from keras._tf_keras.keras.layers import LSTM, Dense, TimeDistributed

# Common Voice 데이터 경로
config_path = './audio/ko/'  # 데이터셋이 있는 디렉토리 경로
audio_path = './audio/ko/clips/'
# 데이터 로드 (TSV 파일을 로드)
df = pd.read_csv(os.path.join(config_path, 'validated.tsv'), sep='\t')

X = []
y = []
label_idx = 0

for index, row in df.iterrows():
    audio, sr = librosa.load(os.path.join(audio_path, row['path']), sr=None)

    # STFT(단기 푸리에 변환) 수행
    stft = librosa.stft(audio)

    # Mel 필터 뱅크 적용
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)

    # Log Mel Spectrogram으로 변환
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)   
    X.append(log_mel_spectrogram)
    encoder = LabelEncoder()
    
    # plt.figure(figsize=(10, 4))
    # librosa.display.specshow(log_mel_spectrogram, x_axis='time', y_axis='mel', sr=sr)
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Log Mel Spectrogram')
    # plt.show()

sentences = df['sentence'].values
vocab = []

max_len = 0
for s in sentences:
    vocab.append(list((set(s))))
    if len(s) > max_len:  # 문자열의 길이를 구할 때는 len()을 사용합니다.
        max_len = len(s)  # max_len을 업데이트

vocab = list(itertools.chain(*vocab))

# nested_array = np.array(vocab, dtype=object)
# vocab = nested_array.flatten()
vocab = sorted(set(vocab))

char_to_index = {char: idx + 1 for idx, char in enumerate(vocab)}  # 0은 padding 용도
index_to_char = {idx: char for char, idx in char_to_index.items()}

encoded_all = [[char_to_index.get(c, 0) for c in s] for s in sentences]
padded_all = tf.keras.preprocessing.sequence.pad_sequences(encoded_all, maxlen=max_len, padding='post')

y = padded_all
# 데이터 크기 조정 (시간축 길이가 동일하도록 패딩)
max_length = max([x.shape[1] for x in X])
X_pad = np.array([np.pad(x, ((0, 0), (0, max_length - x.shape[1])), mode='constant') for x in X])

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2, random_state=42)
X_train = np.array(X_train, dtype=np.float32)

# 모델 설계
model = models.Sequential([
    layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
    layers.Conv1D(64, kernel_size=3, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    layers.Conv1D(128, kernel_size=3, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    layers.Conv1D(256, kernel_size=3, activation='relu'),
    layers.GlobalAveragePooling1D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(56)  # 마지막 출력은 실수
])

# model = models.Sequential([
#     LSTM(128, return_sequences=False, input_shape=(X_train.shape[2], X_train.shape[1])),
#     TimeDistributed(Dense(len(vocab) + 1, activation='softmax'))
# ])

# 모델 컴파일
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 모델 훈련
model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_data=(X_test, y_test))
model.save('iris.h5')

# 모델 평가
y_pred = model.predict(X_test)
y_pred_int = list(map(int, y_pred[0])) 

pred = [index_to_char.get(c, "")  for c in y_pred_int]
pred_test = [index_to_char.get(c, "")  for c in y_test[0]]

print (y_pred_int)
print(y_test[0])
print (pred)
print(pred_test)
# loss, accuracy = model.evaluate(X_test, y_test)
# print(f"Test Loss: {loss}")
# print(f"Test Accuracy: {accuracy}")