import numpy as np
from scipy.signal import hilbert, chirp
import matplotlib.pyplot as plt


fs = 50176.0  # sampling frequency
duration = 1.0  # duration of the signal
t = np.arange(int(fs * duration)) / fs  # time base


from scipy.io import wavfile
sample_rate, data = wavfile.read("datasets/initial/train/banana/0.wav")
cb = np.array(data, dtype=np.int16)
test = [d[0] for d in cb]
test = np.array(test)
a_t = test

# a_t = 1.0 + 0.7 * np.sin(2.0 * np.pi * 3.0 * t)  # information signal
c_t = chirp(t, 20.0, t[-1], 80)  # chirp carrier
x = a_t * c_t  # modulated signal

plt.subplot(2, 1, 1)
plt.plot(x)  # plot the modulated signal

z = hilbert(x)  # form the analytical signal
inst_amplitude = np.abs(z)  # envelope extraction
inst_phase = np.unwrap(np.angle(z))  # inst phase
inst_freq = np.diff(inst_phase) / (2 * np.pi) * fs  # inst frequency

# Regenerate the carrier from the instantaneous phase
regenerated_carrier = np.cos(inst_phase)

plt.plot(inst_amplitude, 'r')  # overlay the extracted envelope
plt.title('Modulated signal and extracted envelope')
plt.xlabel('n')
plt.ylabel('x(t) and |z(t)|')
plt.subplot(2, 1, 2)
plt.plot(regenerated_carrier)
plt.title('Extracted carrier or TFS')
plt.xlabel('n')
plt.ylabel('cos[\omega(t)]')
plt.show()
