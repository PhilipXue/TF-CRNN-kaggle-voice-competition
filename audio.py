import functools

import librosa
import numpy as np

try:
    import cv2
except ImportError:
    pass

@functools.lru_cache()
def _load_audio(audio_file_name):

    y, sr = librosa.load(audio_file_name, sr=None)

    return y, sr


def audio_length(audio_filename):
    """Audio lenght (mill second)"""
    y, sr = _load_audio(audio_filename)
    duration = librosa.core.get_duration(y=y, sr=sr)
    result = round(duration * 1000)

    return result

def melspectrogram(audio_file_name, hop_length, offset=0.0, duration=None, sr=22050):
    """Create melspectrogram from audio file.
    1. Load the audio with raw sample rate.
    2. Cut the audio samples with target offset and duration.
    3. Do resample the cutted samples.
    4. Mel
    """
    all_y, native_sample_rate = _load_audio(audio_file_name)

    # In edge case, librosa.core.time_to_samples() return difference value.
    # e.g. native_sample_rate=44100, sr=22050, offset = 1.15
    # Then, I decide to use `int(np.round(native_sample_rate * offset))`
    # reffer to https://github.com/librosa/librosa/blob/0.5.1/librosa/core/audio.py#L111
    #
    # sample_start = int(librosa.core.time_to_samples(offset, sr=raw_sample_rate))
    sample_start = int(np.round(native_sample_rate * offset))

    if duration:
        sample_duration = int(librosa.core.time_to_samples(duration, sr=native_sample_rate))
        sample_end = sample_start + sample_duration
    else:
        sample_end = all_y.size

    cut_y = all_y[sample_start:sample_end]
    y = librosa.core.resample(cut_y, orig_sr=native_sample_rate, target_sr=sr)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, hop_length=hop_length)

    return mel


def mel_to_db(mel):
    '''
    Covert mel-spetrogram into db style and map db into pixel strength
    '''
    db = librosa.power_to_db(mel, ref=np.min)

    # values are to [0.0, 255.0]
    db -= db.min()
    db *= 255 / (db.max()**(db.max() != 0))

    # TODO(wakisaka): flip vertical. why does phlip do it?
    return np.flip(db, 0)


# TODO: sr(sample rate)
def melspectrogram_width(audio_length, hop_length, sr=22050):
    """Compute mel spectrogram image width from audio_length and hop_length
    Args:
       audio_lenght(int): milli second audio length. (slice duration)
       hop_lenght(int): mel hop length
       sr(int): sample rate
    """
    return int((audio_length/1000 * sr) / hop_length) + 1


def compute_hop_lenght(audio_length, width, sr=22050):
    """Compute hop length from mel spectrogram image width and audio_length
    It is inverse function of melspectrogram_width().
    Args:
       audio_lenght(int): milli second audio length. (slice duration)
       width(int): mel spectrogram image width
       sr(int): sample rate
    """
    return int((audio_length/1000 * sr) / int(width - 1))


def signal_extract_mask(audio_file, mel_hop_length, threshold=1.2):
    """Create a mask to extract the signal part of the audio
    Please refer the the github issue #3
    """

    audiolen = audio_length(audio_file)

    # TODO(wakisaka): sample rate with config.
    mel = melspectrogram(audio_file, mel_hop_length)
    db = mel_to_db(mel)

    col_mean = np.mean(db, axis=0)
    row_mean = np.mean(db, axis=1)
    col = col_mean.shape[0]

    row_mean_tile = threshold * np.transpose(np.tile(row_mean, (col, 1)))
    col_mean_tile = threshold * np.tile(col_mean, (128, 1))

    se = db * (db > row_mean_tile) * (db > col_mean_tile)
    kernel = np.ones((4, 4), np.uint8)
    se = cv2.erode(se, kernel)
    se = cv2.dilate(se, kernel)
    se = cv2.erode(se, kernel)
    se = cv2.dilate(se, kernel)

    # mask.shape -> [db.shape[1],]
    mask = np.sum(se, axis=0)

    assert(mask.shape[0] == melspectrogram_width(audiolen, mel_hop_length))

    return mask