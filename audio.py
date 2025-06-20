
import librosa
import numpy as np

def _build_mel_basis():
    import hparams as hp
    return librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft, n_mels=hp.num_mels)

def melspectrogram(wav):
    import hparams as hp
    D = _stft(wav)
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - hp.ref_level_db
    return _normalize(S)

def _stft(y):
    import hparams as hp
    return librosa.stft(y=y, n_fft=hp.n_fft, hop_length=hp.hop_length, win_length=hp.win_length)

def _linear_to_mel(spectrogram):
    mel_basis = _build_mel_basis()
    return np.dot(mel_basis, spectrogram)

def _amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))

def _normalize(S):
    import hparams as hp
    return np.clip((S - hp.min_level_db) / -hp.min_level_db, 0, 1)
