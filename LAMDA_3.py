import warnings
warnings.filterwarnings(action='ignore')

import os, sys, json
import numpy as np
import librosa
import time
from scipy import signal
import scipy.signal
from datetime import timedelta as td


def _stft(y, n_fft, hop_length, win_length):
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def _istft(y, hop_length, win_length):
    return librosa.istft(y, hop_length, win_length)


def _amp_to_db(x):
    return librosa.core.amplitude_to_db(x, ref=1.0, amin=1e-20, top_db=80.0)


def _db_to_amp(x,):
    return librosa.core.db_to_amplitude(x, ref=1.0)


def removeNoise(
    audio_clip,
    noise_clip,
    n_grad_freq=2,
    n_grad_time=4,
    n_fft=2048,
    win_length=2048,
    hop_length=512,
    n_std_thresh=1.5,
    prop_decrease=1.0,
    verbose=False,
    visual=False,
):
    """Remove noise from audio based upon a clip containing only noise

    Args:
        audio_clip (array): The first parameter.
        noise_clip (array): The second parameter.
        n_grad_freq (int): how many frequency channels to smooth over with the mask.
        n_grad_time (int): how many time channels to smooth over with the mask.
        n_fft (int): number audio of frames between STFT columns.
        win_length (int): Each frame of audio is windowed by `window()`. The window will be of length `win_length` and then padded with zeros to match `n_fft`..
        hop_length (int):number audio of frames between STFT columns.
        n_std_thresh (int): how many standard deviations louder than the mean dB of the noise (at each frequency level) to be considered signal
        prop_decrease (float): To what extent should you decrease noise (1 = all, 0 = none)
        visual (bool): Whether to plot the steps of the algorithm

    Returns:
        array: The recovered signal with noise subtracted

    """
    if verbose:
        start = time.time()
    # STFT over noise
    noise_stft = _stft(noise_clip, n_fft, hop_length, win_length)
    noise_stft_db = _amp_to_db(np.abs(noise_stft))  # convert to dB
    # Calculate statistics over noise
    mean_freq_noise = np.mean(noise_stft_db, axis=1)
    std_freq_noise = np.std(noise_stft_db, axis=1)
    noise_thresh = mean_freq_noise + std_freq_noise * n_std_thresh
    if verbose:
        print("STFT on noise:", td(seconds=time.time() - start))
        start = time.time()
    # STFT over signal
    if verbose:
        start = time.time()
    sig_stft = _stft(audio_clip, n_fft, hop_length, win_length)
    sig_stft_db = _amp_to_db(np.abs(sig_stft))
    if verbose:
        print("STFT on signal:", td(seconds=time.time() - start))
        start = time.time()
    # Calculate value to mask dB to
    mask_gain_dB = np.min(_amp_to_db(np.abs(sig_stft)))
    #print(noise_thresh, mask_gain_dB)
    # Create a smoothing filter for the mask in time and frequency
    smoothing_filter = np.outer(
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_freq + 1, endpoint=False),
                np.linspace(1, 0, n_grad_freq + 2),
            ]
        )[1:-1],
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_time + 1, endpoint=False),
                np.linspace(1, 0, n_grad_time + 2),
            ]
        )[1:-1],
    )
    smoothing_filter = smoothing_filter / np.sum(smoothing_filter)
    # calculate the threshold for each frequency/time bin
    db_thresh = np.repeat(
        np.reshape(noise_thresh, [1, len(mean_freq_noise)]),
        np.shape(sig_stft_db)[1],
        axis=0,
    ).T
    # mask if the signal is above the threshold
    sig_mask = sig_stft_db < db_thresh
    if verbose:
        print("Masking:", td(seconds=time.time() - start))
        start = time.time()
    # convolve the mask with a smoothing filter
    sig_mask = scipy.signal.fftconvolve(sig_mask, smoothing_filter, mode="same")
    sig_mask = sig_mask * prop_decrease
    if verbose:
        print("Mask convolution:", td(seconds=time.time() - start))
        start = time.time()
    # mask the signal
    sig_stft_db_masked = (
        sig_stft_db * (1 - sig_mask)
        + np.ones(np.shape(mask_gain_dB)) * mask_gain_dB * sig_mask
    )  # mask real
    sig_imag_masked = np.imag(sig_stft) * (1 - sig_mask)
    sig_stft_amp = (_db_to_amp(sig_stft_db_masked) * np.sign(sig_stft)) + (
        1j * sig_imag_masked
    )
    if verbose:
        print("Mask application:", td(seconds=time.time() - start))
        start = time.time()
    # recover the signal
    recovered_signal = _istft(sig_stft_amp, hop_length, win_length)
    recovered_spec = _amp_to_db(
        np.abs(_stft(recovered_signal, n_fft, hop_length, win_length))
    )
    if verbose:
        print("Signal recovery:", td(seconds=time.time() - start))

    return recovered_signal


if __name__ == '__main__':

    # 압축 문제로 한글 깨질 수도 있기 때문에
    # 다시 path 타이핑 요청드림.^^
    wav_list = os.listdir("./문제3")

    answer = []
    for wav_file in wav_list:

        a = {}

        # 압축 문제로 한글 깨질 수도 있기 때문에
        # 다시 path 타이핑 요청드림.^^
        wav, sr = librosa.load("./문제3/"+wav_file)
        
        # 노이즈 제거
        noise1 = wav[0:1*sr]
        wav1 = removeNoise(audio_clip=wav, noise_clip=noise1,
            n_grad_freq=2,
            n_grad_time=4,
            n_fft=2048,
            win_length=2048,
            hop_length=512,
            n_std_thresh=1.5,
            prop_decrease=1.0,
            verbose=False,
            visual=False)

        # 앞, 뒤 0.1초씩 제거 후, zero-padding
        # 음성 녹음 시 발생하는 딜레이로 인한 노이즈 배제
        removal_sec = round(0.1*sr) # 0.1초
        wav2 = wav1[removal_sec:-removal_sec]
        wav2 = np.pad(wav2, (removal_sec,removal_sec), 'constant', constant_values=0)

        # top_db = threshold(20) 보다 큰값만 추출
        # -> 유효 음성 구간 추출
        clip = librosa.effects.trim(wav2, top_db= 20)

        begin = float("{:.3f}".format(clip[1][0]/22050))
        end = float("{:.3f}".format(clip[1][1]/22050))

        a['filename'] = wav_file
        a['begin'] = begin
        a['end'] = end

        answer.append(a)
    
    json_path = 'LAMDA.json'

    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    data["Q3"] = answer

    # print(data["Q3"])

    # dump json file
    with open(json_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent='\t', ensure_ascii=False)