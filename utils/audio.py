import librosa
import librosa.filters
import numpy as np
from scipy import signal
from scipy.io import wavfile
from omegaconf import OmegaConf 

#hp = OmegaConf.load('wav2lipHD/base_config.yaml').audio

class AudioTools:
    def __init__(self, config):
        self.cfg = config # subconfig from the global config: cfg.audio

    @staticmethod
    def load_wav(path, sr):
        return librosa.core.load(path, sr=sr)[0]

    @staticmethod
    def save_wav(wav, path, sr):
        wav *= 32767 / max(0.01, np.max(np.abs(wav)))
        #proposed by @dsmiller
        wavfile.write(path, sr, wav.astype(np.int16))

    @staticmethod
    def save_wavenet_wav(wav, path, sr):
        librosa.output.write_wav(path, wav, sr=sr)

    @staticmethod
    def preemphasis(wav, k, preemphasize=True):
        if preemphasize:
            return signal.lfilter([1, -k], [1], wav)
        return wav

    @staticmethod
    def inv_preemphasis(wav, k, inv_preemphasize=True):
        if inv_preemphasize:
            return signal.lfilter([1], [1, -k], wav)
        return wav

    def get_hop_size(self):
        hop_size = self.cfg.hop_size
        if hop_size is None:
            assert self.cfg.frame_shift_ms is not None
            hop_size = int(self.cfg.frame_shift_ms / 1000 * self.cfg.sample_rate)
        return hop_size

    def linearspectrogram(self, wav):
        D = self._stft(self.preemphasis(wav, self.cfg.preemphasis, self.cfg.preemphasize))
        S = self._amp_to_db(np.abs(D)) - self.cfg.ref_level_db
    
        if self.cfg.signal_normalization:
            return self._normalize(S)
        return S

    def melspectrogram(self, wav):
        D = self._stft(self.preemphasis(wav, self.cfg.preemphasis, self.cfg.preemphasize))
        S = self._amp_to_db(self._linear_to_mel(np.abs(D))) - self.cfg.ref_level_db
    
        if self.cfg.signal_normalization:
            return self._normalize(S)
        return S

    def _lws_processor(self):
        import lws
        return lws.lws(self.cfg.n_fft, self.get_hop_size(), fftsize=self.cfg.win_size, mode="speech")

    
    def _stft(self, y):
        if self.cfg.use_lws:
            return self._lws_processor().stft(y).T
        else:
            return librosa.stft(y=y, n_fft=self.cfg.n_fft, hop_length=self.get_hop_size(), win_length=self.cfg.win_size)

        
    ##########################################################
    #Those are only correct when using lws!!! (This was messing with Wavenet quality for a long time!)

    @staticmethod
    def num_frames(length, fsize, fshift):
        """Compute number of time frames of spectrogram
        """
        pad = (fsize - fshift)
        if length % fshift == 0:
            M = (length + pad * 2 - fsize) // fshift + 1
        else:
            M = (length + pad * 2 - fsize) // fshift + 2
        return M

    @staticmethod
    def pad_lr(x, fsize, fshift):
        """Compute left and right padding
        """
        M = AudioTools.num_frames(len(x), fsize, fshift)
        pad = (fsize - fshift)
        T = len(x) + 2 * pad
        r = (M - 1) * fshift + fsize - T
        return pad, pad + r
    
    ##########################################################
    #Librosa correct padding
    @staticmethod
    def librosa_pad_lr(x, fsize, fshift):
        return 0, (x.shape[0] // fshift + 1) * fshift - x.shape[0]

    def _linear_to_mel(self, spectogram):
        if not hasattr(self, 'mel_basis'):
            self.mel_basis = self._build_mel_basis()
        return np.dot(self.mel_basis, spectogram)

    def _build_mel_basis(self):
        assert self.cfg.fmax <= self.cfg.sample_rate // 2
        return librosa.filters.mel(self.cfg.sample_rate, self.cfg.n_fft, n_mels=self.cfg.num_mels,
                                   fmin=self.cfg.fmin, fmax=self.cfg.fmax)

    def _amp_to_db(self, x):
        min_level = np.exp(self.cfg.min_level_db / 20 * np.log(10))
        return 20 * np.log10(np.maximum(min_level, x))

    @staticmethod
    def _db_to_amp(x):
        return np.power(10.0, (x) * 0.05)

    
    def _normalize(self, S):
        if self.cfg.allow_clipping_in_normalization:
            if self.cfg.symmetric_mels:
                return np.clip((2 * self.cfg.max_abs_value) * ((S - self.cfg.min_level_db) / (-self.cfg.min_level_db)) - self.cfg.max_abs_value,
                               -self.cfg.max_abs_value, self.cfg.max_abs_value)
            else:
                return np.clip(self.cfg.max_abs_value * ((S - self.cfg.min_level_db) / (-self.cfg.min_level_db)), 0, self.cfg.max_abs_value)
    
        assert S.max() <= 0 and S.min() - self.cfg.min_level_db >= 0
        if self.cfg.symmetric_mels:
            return (2 * self.cfg.max_abs_value) * ((S - self.cfg.min_level_db) / (-self.cfg.min_level_db)) - self.cfg.max_abs_value
        else:
            return self.cfg.max_abs_value * ((S - self.cfg.min_level_db) / (-self.cfg.min_level_db))

    def _denormalize(self, D):
        if self.cfg.allow_clipping_in_normalization:
            if self.cfg.symmetric_mels:
                return (((np.clip(D, -self.cfg.max_abs_value,
                                  self.cfg.max_abs_value) + self.cfg.max_abs_value) * -self.cfg.min_level_db / (2 * self.cfg.max_abs_value))
                + self.cfg.min_level_db)
            else:
                return ((np.clip(D, 0, self.cfg.max_abs_value) * -self.cfg.min_level_db / self.cfg.max_abs_value) + self.cfg.min_level_db)
    
        if self.cfg.symmetric_mels:
            return (((D + self.cfg.max_abs_value) * -self.cfg.min_level_db / (2 * self.cfg.max_abs_value)) + self.cfg.min_level_db)
        else:
            return ((D * -self.cfg.min_level_db / self.cfg.max_abs_value) + self.cfg.min_level_db)


    # julian add
    @staticmethod
    def wav_add_noise(wave_data):
        noise_factor = np.random.random() * max(wave_data) * 0.1
        noise = np.random.randn(len(wave_data))
        res =  wave_data  + noise * noise_factor
        return res.astype(wave_data.dtype)

    @staticmethod
    def wav_shift_pitch(data, sample_rate = 16000):
        pitch_factor = np.random.randint(-6, 7) 
        return librosa.effects.pitch_shift(data, sample_rate, pitch_factor)
