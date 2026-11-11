import logging
import numpy as np
import copy

try:
    import pyroomacoustics as pra
    from pyroomacoustics.parameters import constants
    from scipy.signal import resample_poly
except ImportError:
    logging.warning("pyroomacoustics is not installed. "
                    "Please install it with `pip install pyroomacoustics`.")

class WhamRoom(pra.room.ShoeBox):

    def __init__(self, p, mics, s1, T60, fs=16000,
                 t0=0., sigma2_awgn=None):

        self.T60 = T60
        self.max_rir_len = np.ceil(T60*fs).astype(int)
        self.cut_point = int(fs * 0.05)

        volume = p[0]*p[1]*p[2]
        surface_area = 2*(p[0]*p[1] + p[0]*p[2] + p[1]*p[2])
        absorption = 24 * volume * np.log(10.0) / (constants.get('c') * surface_area * T60)

        # minimum max order to guarantee complete filter of length T60
        max_order = np.ceil(T60 * constants.get('c') / min(p)).astype(int)

        super().__init__(p, fs=fs, t0=t0, absorption=absorption,
                         max_order=max_order, sigma2_awgn=sigma2_awgn,
                         sources=None, mics=None)

        self.add_source(s1)
        self.add_microphone_array(pra.MicrophoneArray(np.array(mics).T, fs))

    def add_audio(self, audios):
        for i in range(len(audios)):
            self.sources[i].add_signal(audios[i])

    def generate_rirs(self):

        original_max_order = self.max_order

        self.compute_rir()
        self.rir_reverberant = copy.deepcopy(self.rir) 
        self.rir_anechoic = copy.deepcopy(self.rir)
        for m in range(len(self.rir)):
            for s in range(len(self.rir[m])):
                self.rir_reverberant[m][s] = self.rir_reverberant[m][s][:self.max_rir_len]
                self.rir_anechoic[m][s] = self.rir_anechoic[m][s][:self.max_rir_len]
        
        for i in range(len(self.rir_anechoic)):
            for j in range(len(self.rir_anechoic[i])):
                peak_idx = np.argmax(self.rir_anechoic[i][j])
                if peak_idx > 20:
                    if peak_idx + 20 < len(self.rir_anechoic[i][j]):
                        self.rir_anechoic[i][j][:peak_idx-20] = 0
                        self.rir_anechoic[i][j][peak_idx+20:] = 0
                    else:
                        self.rir_anechoic[i][j][:peak_idx-20] = 0
                else:
                   if peak_idx + 20 < len(self.rir_anechoic[i][j]):
                        self.rir_anechoic[i][j][peak_idx+20:] = 0
                
    def generate_audio(self, anechoic=False, fs=16000):

        if not self.rir:
            self.generate_rirs()
        if anechoic:
            self.rir = self.rir_anechoic
        else:
            self.rir = self.rir_reverberant
        audio_array = self.simulate(return_premix=True, recompute_rir=False)

        if type(fs) is not list:
            fs_array = [fs]
        else:
            fs_array = fs
        audio_out = []
        for elem in fs_array:
            if type(elem) is str:
                elem = int(elem.replace('k','000'))
            if elem != self.fs:
                assert(self.fs % elem == 0)
                audio_out.append(resample_poly(audio_array, elem, self.fs, axis=2))
            else:
                audio_out.append(audio_array)
        if type(fs) is not list:
            return audio_out[0] # array of shape (n_sources, n_mics, n_samples)
        else:
            return audio_out