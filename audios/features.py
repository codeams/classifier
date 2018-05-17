
from scipy.io import wavfile
import sklearn
import numpy
from modulee import detect_peaks
import librosa


def peaks_indexes(file_path):
    _, data = wavfile.read(file_path)
    cb = numpy.array(data, dtype=numpy.int16)
    test = [d[0] for d in cb]
    test = numpy.array(test)
    indexes = detect_peaks(test, mph=3000, mpd=500, show=False)
    return indexes


def peaks_count(file_path):
    indexes = peaks_indexes(file_path)
    return [len(indexes)]


def spectral_bandwidth(file_path):
    y, sr = librosa.load(file_path)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    result = sklearn.preprocessing.scale(spec_bw, axis=1)
    return result
