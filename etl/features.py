import librosa
import numpy as np

def extract_features_from_file(path, sr=22050):
    """
    Returns simple feature dict: rms, zcr, mfcc_mean (list)
    """
    y, sr = librosa.load(path, sr=sr)
    rms = float(np.mean(librosa.feature.rms(y=y)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = [float(np.mean(m)) for m in mfcc]
    return {
        "rms": rms,
        "zcr": zcr,
        "mfcc_mean": mfcc_mean
    }

# Example usage:
# f = extract_features_from_file("sample.wav")
# print(f)
