
# from matplotlib import pyplot
from scipy.io import wavfile
# from oct2py import octave

import sklearn
import numpy
from modulee import detect_peaks
import librosa


def peaks_indexes(file_path):
    sample_rate, data = wavfile.read(file_path)
    cb = numpy.array(data, dtype=numpy.int16)
    test = [d[0] for d in cb]
    test = numpy.array(test)

    indexes = detect_peaks(test, mph=3000, mpd=500, show=False)
    # indexes = indexes.mean(axis=1)

    return indexes


def peaks_count(file_path):
    indexes = peaks_indexes(file_path)
    return [len(indexes)]


# def peaks_old():
#    sample_rate, data = wavfile.read("datasets/initial/train/banana/0.wav")
#    # data = [ 0, 6, 25, 20, 15, 8, 15, 6, 0, 6, 0, -5, -15, -3, 4, 10, 8, 13, 8, 10, 3, 1, 20, 7, 3, 0 ]
#
#    cb = np.array(data, dtype=np.int16)
#    test = [d[0] for d in cb]
#    octave.eval("pkg load signal")
#    (the_peaks, indexes) = octave.findpeaks(np.array(test), 'DoubleSided', nout=2)
#    print the_peaks
#    pyplot.plot(indexes)
#    pyplot.show()


def spectral_bandwidth(file_path):
    y, sr = librosa.load(file_path)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    result = sklearn.preprocessing.scale(spec_bw, axis=1)
    return result


def spectral_flatness(file_path):
    y, sr = librosa.load(file_path)
    flatness = librosa.feature.spectral_flatness(y=y)
    return flatness


def melspectogram(file_path):
    y, sr = librosa.load(file_path)
    melspectogram = librosa.feature.melspectrogram(y=y, sr=sr)
    result = sklearn.preprocessing.scale(melspectogram, axis=1)
    return result


#def poly_features(file_path):
#    y, sr = librosa.load(file_path)
#    s = numpy.abs(librosa.stft(y))
#    return librosa.feature.poly_features(S=s, order=2)


def wav2mfcc(file_path, max_pad_len=11):
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    wave = wave[::3]
    mfcc = librosa.feature.mfcc(wave, sr=16000)
    pad_width = max_pad_len - mfcc.shape[1]
    mfcc = numpy.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

    mfccs = sklearn.preprocessing.scale(mfcc, axis=1)
    mfccs = mfccs.mean(axis=1)
    mfccs = mfccs.var(axis=1)
    return mfccs
