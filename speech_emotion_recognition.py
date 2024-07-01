import librosa
import numpy as np
import os
from keras.models import load_model
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk

imgFolder = os.path.join('static', 'assets')

def speech_emotion_recognition(wave_file):
    data, sr = librosa.load(wave_file)
    
    def cqtspec(audio, sr, min_freq=30, octave_resolution=14):
        max_frequency = sr / 2
        num_freq = round(octave_resolution * np.log2(max_frequency / min_freq))
        cqt_spectrogram = np.abs(librosa.cqt(data, sr=sr, fmin=min_freq, bins_per_octave=octave_resolution, n_bins=num_freq))
        return cqt_spectrogram

    def cqhc(audio, sr, min_freq=30, octave_resolution=14, num_coeff=20):
        cqt_spectrogram = np.power(cqtspec(audio, sr, min_freq, octave_resolution), 2)
        num_freq = np.shape(cqt_spectrogram)[0]
        ftcqt_spectrogram = np.fft.fft(cqt_spectrogram, 2 * num_freq - 1, axis=0)
        absftcqt_spectrogram = abs(ftcqt_spectrogram)
        pitch_component = np.real(np.fft.ifft(ftcqt_spectrogram / (absftcqt_spectrogram + 1e-14), axis=0)[0:num_freq, :])
        coeff_indices = np.round(octave_resolution * np.log2(np.arange(1, num_coeff + 1))).astype(int)
        audio_cqhc = pitch_component[coeff_indices, :]
        return audio_cqhc
    
    feat = cqhc(data, sr, min_freq=30, octave_resolution=14, num_coeff=20)
    shape = 290 - feat.shape[1]
    feat_pad = np.pad(feat, ((0, 0), (0, shape)), 'constant')
    feat_pad = feat_pad.reshape(1, 20, 290, 1)
    
    model = load_model('emotion_h5_file.h5')
    ans = model(feat_pad)
    output = np.argmax(ans)
    if output == 0:
        labels = os.path.join(imgFolder, 'angry.gif')
        labels1 = 'Angry'
    elif output == 1:
        labels = os.path.join(imgFolder, 'happy.gif')
        labels1 = 'Happy'
    elif output == 2:
        labels = os.path.join(imgFolder, 'neutral.gif')
        labels1 = 'Neutral'
    elif output == 3:
        labels = os.path.join(imgFolder, 'sad.gif')
        labels1 = 'Sad'
    elif output == 4:
        labels = os.path.join(imgFolder, 'surprise.gif')
        labels1 = 'Surprise'
    
    return labels, labels1

def display_gif(gif_path):
    root = tk.Tk()
    root.title("Speech Emotion Recognition")

    img = Image.open(gif_path)
    photo = ImageTk.PhotoImage(img)

    label = Label(root, image=photo)
    label.image = photo
    label.pack()

    root.mainloop()

labels, labels1 = speech_emotion_recognition(r'test audio\neutral_1.wav')
display_gif(labels)
