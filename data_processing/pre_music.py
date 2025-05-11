import librosa
import numpy as np
import os
import sys
import wave
from tqdm import tqdm
import librosa as lr

# FPS = 30 #* 5
FPS = 60
HOP_LENGTH = 512
SR = FPS * HOP_LENGTH
EPS = 1e-6

# HOP_LENGTH = 160
# SR = 16000

# audio_dir = 'data/finedance/music_wav'
# audio_dir = '/home/human/datasets/aist_plusplus_final/music'
# audio_dir = "/home/human/datasets/data/Clip/music_clip_rhythm"

target_dir_ori = 'E:\GestureGeneration\Speech_driven_gesture_generation_with_autoencoder-GENEA_2022\dataset\music'


# target_dir_ori = "E:\DeepMotion\dataset\processed\pre_music"


# AIST++
def _get_tempo(audio_name):
    """Get tempo (BPM) for a music by parsing music name."""
    # a lot of stuff, only take the 5th element
    audio_name = audio_name.split("_")[4]
    assert len(audio_name) == 4
    if audio_name[0:3] in [
        "mBR",
        "mPO",
        "mLO",
        "mMH",
        "mLH",
        "mWA",
        "mKR",
        "mJS",
        "mJB",
    ]:
        return int(audio_name[3]) * 10 + 80
    elif audio_name[0:3] == "mHO":
        return int(audio_name[3]) * 5 + 110
    else:
        assert False, audio_name


def extract_music_feature(audio_filename):
    audio_name = os.path.basename(audio_filename)
    audio_name = audio_name.split('.')[0]
    # audio_name = audio_filename[:-4]

    path = os.path.join(os.path.dirname(audio_filename), 'pre_music')
    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(path, f"{audio_name}_{FPS}fps.npy")
    music_file = audio_filename
    # music_file = os.path.join(audio_dir, file)

    signal, _ = librosa.load(music_file, sr=SR)

    audio_harmonic, audio_percussvie = librosa.effects.hpss(signal)
    envelope = librosa.onset.onset_strength(audio_percussvie, sr=SR)  # (seq_len,)
    # melspe = librosa.feature.melspectrogram(y=data, sr=SR)
    # melspe_db = librosa.power_to_db(melspe, ref=np.max)
    # mfcc = librosa.feature.mfcc(S=melspe_db, n_mfcc=20).T  # (seq_len, 20)
    mfcc = librosa.feature.mfcc(signal, sr=SR, n_mfcc=20).T  # (seq_len, 20)


    chroma = librosa.feature.chroma_cens(signal, sr=SR, hop_length=HOP_LENGTH, n_chroma=12).T  # (seq_len, 12)

    spectral_centroid = librosa.feature.spectral_centroid(signal, sr=SR)[0]

    peak_idxs = librosa.onset.onset_detect(
        onset_envelope=envelope.flatten(), sr=SR, hop_length=HOP_LENGTH
    )
    peak_onehot = np.zeros_like(envelope, dtype=np.float32)
    peak_onehot[peak_idxs] = 1.0  # (seq_len,)

    try:
        start_bpm = _get_tempo(audio_name)
    except:
        # determine manually
        start_bpm = lr.beat.tempo(y=lr.load(music_file)[0])[0]

    tempo, beat_idxs = librosa.beat.beat_track(
        onset_envelope=envelope,
        sr=SR,
        hop_length=HOP_LENGTH,
        start_bpm=start_bpm,
        tightness=100,
    )
    beat_onehot = np.zeros_like(envelope, dtype=np.float32)
    beat_onehot[beat_idxs] = 1.0  # (seq_len,)

    tempogram = librosa.feature.tempogram(onset_envelope=envelope, sr=SR)

    # tempogram = tempogram.transpose()

    print(f'envelope.shape: {envelope.shape}\n'
          f'mfcc.shape: {mfcc.shape}\n'
          f'chroma.shape: {chroma.shape}\n'
          f'peak_onehot.shape: {peak_onehot.shape}\n'
          f'beat_onehot.shape: {beat_onehot.shape}\n'
          f'spectral_centroid.shape: {spectral_centroid.shape}'
    )

    audio_feature = np.concatenate(
        [envelope[:, None], mfcc, chroma, peak_onehot[:, None], beat_onehot[:, None], spectral_centroid[:, None]],
        axis=-1
    )
    print('audio_feature.shape: ', audio_feature.shape)
    np.save(save_path, audio_feature)
    print(os.path.join(save_path, f'{audio_name}.npy'))

    return audio_feature


# if __name__ == '__main__':
#     #     data_path = sys.argv[1]
#     file = 'E:/DeepMotion/dataset/test/Mozart_17.wav'
#     extract_music_feature(file)



#
# import librosa
# import numpy as np
# import os
# import sys
# import wave
# from tqdm import  tqdm
# import librosa as lr
#
# # FPS = 30 #* 5
# FPS = 60
# HOP_LENGTH = 512
# SR = FPS * HOP_LENGTH
# EPS = 1e-6
#
# # HOP_LENGTH = 160
# # SR = 16000
#
# # audio_dir = 'data/finedance/music_wav'
# # audio_dir = '/home/human/datasets/aist_plusplus_final/music'
# # audio_dir = "/home/human/datasets/data/Clip/music_clip_rhythm"
#
# target_dir_ori = 'E:\GestureGeneration\Speech_driven_gesture_generation_with_autoencoder-GENEA_2022\dataset\music'
# # target_dir_ori = "E:\DeepMotion\dataset\processed\pre_music"
#
#
#
# # AIST++
# def _get_tempo(audio_name):
#     """Get tempo (BPM) for a music by parsing music name."""
#     # a lot of stuff, only take the 5th element
#     audio_name = audio_name.split("_")[4]
#     assert len(audio_name) == 4
#     if audio_name[0:3] in [
#         "mBR",
#         "mPO",
#         "mLO",
#         "mMH",
#         "mLH",
#         "mWA",
#         "mKR",
#         "mJS",
#         "mJB",
#     ]:
#         return int(audio_name[3]) * 10 + 80
#     elif audio_name[0:3] == "mHO":
#         return int(audio_name[3]) * 5 + 110
#     else:
#         assert False, audio_name
#
#
# def extract_music_feature(audio_filename):
#
#     audio_name = os.path.basename(audio_filename)
#     audio_name = audio_name.split('.')[0]
#     # audio_name = audio_filename[:-4]
#
#     path = os.path.join(os.path.dirname(audio_filename), 'pre_music')
#     os.makedirs(path, exist_ok=True)
#     save_path = os.path.join(path, f"{audio_name}_60fps.npy")
#     music_file = audio_filename
#     # music_file = os.path.join(audio_dir, file)
#
#
#     data, _ = librosa.load(music_file, sr=SR)
#
#     envelope = librosa.onset.onset_strength(y=data, sr=SR)  # (seq_len,)
#     # melspe = librosa.feature.melspectrogram(y=data, sr=SR)
#     # melspe_db = librosa.power_to_db(melspe, ref=np.max)
#     # mfcc = librosa.feature.mfcc(S=melspe_db, n_mfcc=20).T  # (seq_len, 20)
#     mfcc = librosa.feature.mfcc(y=data, sr=SR, n_mfcc=20).T  # (seq_len, 20)
#     chroma = librosa.feature.chroma_cens(
#         y=data, sr=SR, hop_length=HOP_LENGTH, n_chroma=12
#     ).T  # (seq_len, 12)
#
#     peak_idxs = librosa.onset.onset_detect(
#         onset_envelope=envelope.flatten(), sr=SR, hop_length=HOP_LENGTH
#     )
#     peak_onehot = np.zeros_like(envelope, dtype=np.float32)
#     peak_onehot[peak_idxs] = 1.0  # (seq_len,)
#
#     try:
#         start_bpm = _get_tempo(audio_name)
#     except:
#         # determine manually
#         start_bpm = lr.beat.tempo(y=lr.load(music_file)[0])[0]
#
#     tempo, beat_idxs = librosa.beat.beat_track(
#         onset_envelope=envelope,
#         sr=SR,
#         hop_length=HOP_LENGTH,
#         start_bpm=start_bpm,
#         tightness=100,
#     )
#     beat_onehot = np.zeros_like(envelope, dtype=np.float32)
#     beat_onehot[beat_idxs] = 1.0  # (seq_len,)
#
#     audio_feature = np.concatenate(
#         [envelope[:, None], mfcc, chroma, peak_onehot[:, None], beat_onehot[:, None]],
#         axis=-1,
#     )
#     print('audio_feature.shape: ', audio_feature.shape)
#     np.save(save_path, audio_feature)
#     print(os.path.join(save_path, f'{audio_name}.npy'))
#
#     return audio_feature
#
# #
# # if __name__ == '__main__':
# # #     data_path = sys.argv[1]
# #     file = 'E:/DeepMotion/dataset/test/Mozart_17.wav'
# #     extract_music_feature(file)