
import pyaudio
import wave
from array import array

import config

FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100  # Try some lower value like 2000
CHUNK = 1024  # ??
RECORD_SECONDS = 5


def record_wav(path, name):
    filer = path + "/" + name + ".wav"

    # Instantiate the pyaudio
    audio = pyaudio.PyAudio()

    # Recording prerequisites
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

    # starting recording
    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        data_chunk = array('h', data)
        vol = max(data_chunk)

        frames.append(data)

        # if vol >= 2000:
            # print("something said\n")
            # frames.append(data)
        # else:
            # print("nothing\n")

    # End of recording
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Writing to file
    wav_file = wave.open(filer, 'wb')
    wav_file.setnchannels(CHANNELS)
    wav_file.setsampwidth(audio.get_sample_size(FORMAT))
    wav_file.setframerate(RATE)

    # Append frames recorded to file
    wav_file.writeframes(b''.join(frames))
    wav_file.close()

    print("{} has been saved".format(filer))


if __name__ == '__main__':
    for label in config.LABELS:
        for i in range(0, config.WAV_FILES_PER_LABEL):
            record_wav(config.TRAIN_PATH, "{}/{}".format(label, i))
