#!/venv/Scripts/python
# -*- coding:utf-8 -*-
# @version   : V1.0
# @author    : zhongjiafeng
# @time      : 2022/10/21 13:34
# @function  : the script is used to do something

import librosa
from spafe.fbanks import gammatone_fbanks
import numpy as np
from scipy.fftpack import dct


def pre_emphasis(sig, pre_emph_coeff=0.97):
    return np.append(sig[0], sig[1:] - pre_emph_coeff * sig[:-1])

def stride_trick(a, stride_length, stride_step):
    a = np.array(a)
    nrows = ((a.size - stride_length) // stride_step) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(
        a, shape=(nrows, stride_length), strides=(stride_step * n, n)
    )

def zero_handling(x):
    return np.where(x == 0, np.finfo(float).eps, x)

def framing(sig, win_len=2048, win_hop=512):
    # compute frame length and frame step (convert from seconds to samples)
    frame_length = win_len
    frame_step = win_hop

    # make sure to use integers as indices
    frames = stride_trick(sig, frame_length, frame_step)

    if len(frames[-1]) < frame_length:
        frames[-1] = np.append(
            frames[-1], np.array([0] * (frame_length - len(frames[0])))
        )

    return frames, frame_length

def windowing(frames, frame_len, win_type="hamming"):
    return {
        "hanning": np.hanning(frame_len) * frames,
        "bartlet": np.bartlett(frame_len) * frames,
        "kaiser": np.kaiser(frame_len, beta=14) * frames,
        "blackman": np.blackman(frame_len) * frames,
    }.get(win_type, np.hamming(frame_len) * frames)

def scale_fbank(scale, nfilts):
    return {
        "ascendant": np.array([i / nfilts for i in range(1, nfilts + 1)]).reshape(
            nfilts, 1
        ),
        "descendant": np.array([i / nfilts for i in range(nfilts, 0, -1)]).reshape(
            nfilts, 1
        ),
        "constant": np.ones(shape=(nfilts, 1)),
    }[scale]

def gtcc(
        sig,
        fs=16000,
        num_ceps=13,
        pre_emph=0,
        pre_emph_coeff=0.97,
        win_len=2048,
        win_hop=512,
        win_type="hamming",
        nfft=2048,
        nfilts=24,
        low_freq=0,
        high_freq=None,
        scale="constant",
        order=4,
        dct_type=2
):
    gamma_fbanks, gamma_freq = gammatone_fbanks.gammatone_filter_banks(
        nfilts=nfilts,
        nfft=nfft,
        fs=fs,
        low_freq=low_freq,
        high_freq=high_freq,
        scale=scale,
        order=order
    )

    # pre-emphasis
    if pre_emph:
        sig = pre_emphasis(sig=sig, pre_emph_coeff=pre_emph_coeff)

    # framing
    frames, frame_length = framing(sig=sig, win_len=win_len, win_hop=win_hop)

    # windowing
    windows = windowing(frames=frames, frame_len=frame_length, win_type=win_type)

    # FFT
    fourrier_transform = np.absolute(np.fft.rfft(windows, nfft))

    # Power Spectrum
    abs_fft_values = (1.0 / nfft) * np.square(fourrier_transform)

    gt_spectrogram = np.dot(abs_fft_values, gamma_fbanks.T)
    gt_spectrogram = zero_handling(gt_spectrogram)

    # log
    log_gt_spectrogram = np.log(gt_spectrogram)

    # DCT
    gtcc = dct(x=log_gt_spectrogram, type=dct_type, axis=1, norm="ortho")[:, :num_ceps]
    return gtcc

def linear_fitlerbank(
    nfilts=24,
    nfft=512,
    fs=16000,
    low_freq=0,
    high_freq=None,
    scale="constant",
):
    # init freqs
    high_freq = high_freq or fs / 2

    # compute points evenly spaced in frequency (points are in Hz)
    delta_hz = abs(high_freq - low_freq) / (nfilts + 1)
    scale_freqs = low_freq + delta_hz * np.arange(0, nfilts + 2)

    # assign freqs
    lower_edges_hz = scale_freqs[:-2]
    upper_edges_hz = scale_freqs[2:]
    center_freqs_hz = scale_freqs[1:-1]
    center_freqs = center_freqs_hz

    freqs = np.linspace(low_freq, high_freq, nfft // 2 + 1)
    fbank = np.zeros((nfilts, nfft // 2 + 1))

    for j, (center, lower, upper) in enumerate(
            zip(center_freqs_hz, lower_edges_hz, upper_edges_hz)
    ):
        left_slope = (freqs >= lower) == (freqs <= center)
        fbank[j, left_slope] = (freqs[left_slope] - lower) / (center - lower)

        right_slope = (freqs >= center) == (freqs <= upper)
        fbank[j, right_slope] = (upper - freqs[right_slope]) / (upper - center)

    # compute scaling
    scaling = scale_fbank(scale=scale, nfilts=nfilts)
    fbank = fbank * scaling

    return fbank, center_freqs

def LFB(
        sig,
        fs=16000,
        pre_emph=0,
        pre_emph_coeff=0.97,
        win_len=2048,
        win_hop=512,
        win_type="hamming",
        nfft=2048,
        nfilts=24,
        low_freq=0,
        high_freq=None,
        scale="constant",
        mode = 'LFBs',
):
    filterbank, center_freqs = linear_fitlerbank(
        nfilts=nfilts, nfft=nfft, fs=fs, low_freq=low_freq, high_freq=high_freq, scale=scale,
    )

    # pre-emphasis
    if pre_emph:
        sig = pre_emphasis(sig=sig, pre_emph_coeff=pre_emph_coeff)

    # framing
    frames, frame_length = framing(sig=sig, win_len=win_len, win_hop=win_hop)

    # windowing
    windows = windowing(frames=frames, frame_len=frame_length, win_type=win_type)

    # FFT
    fourrier_transform = np.absolute(np.fft.rfft(windows, nfft))

    # Power Spectrum
    abs_fft_values = (1.0 / nfft) * np.square(fourrier_transform)
    feature = np.dot(abs_fft_values, filterbank.T)

    # log
    if mode == 'LLFB':
        feature = zero_handling(feature)
        feature = np.log(feature)

    return feature

