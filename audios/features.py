
# from matplotlib import pyplot
# from scipy.io import wavfile
# from oct2py import octave

import numpy
from modulee import detect_peaks
import librosa


def peaks_indexes(data):
    # sample_rate, data = wavfile.read(file_path)
    cb = numpy.array(data, dtype=numpy.int16)
    test = [d[0] for d in cb]
    test = numpy.array(test)

    indexes = detect_peaks(test, mph=8000, mpd=2000, show=True)
    return indexes


def peaks_count(data):
    indexes = peaks_indexes(data)
    return len(indexes)


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
    return spec_bw
