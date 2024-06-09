import functools
import multiprocessing as mp
import numba
import numpy as np
import torch
import torchutil

import penn


###############################################################################
# PYIN (from librosa)
###############################################################################


def from_audio(
    audio,
    sample_rate=penn.SAMPLE_RATE,
    hopsize=penn.HOPSIZE_SECONDS,
    fmin=penn.FMIN,
    fmax=penn.FMAX):
    """Estimate pitch with yin"""
    # Pad
    pad = int(
        penn.WINDOW_SIZE - penn.convert.seconds_to_samples(hopsize)) // 2
    audio = torch.nn.functional.pad(audio, (pad, pad))

    # Infer pitch bin probabilities
    with torchutil.time.context('infer'):
        logits = infer(audio, sample_rate, hopsize, fmin, fmax)

    # Decode pitch and periodicity
    with torchutil.time.context('postprocess'):
        return penn.postprocess(logits)[1:]


def from_file(
    file,
    hopsize=penn.HOPSIZE_SECONDS,
    fmin=penn.FMIN,
    fmax=penn.FMAX):
    """Estimate pitch and periodicity with pyin from audio on disk"""
    # Load
    with torchutil.time.context('load'):
        audio = penn.load.audio(file)

    # Infer
    return from_audio(audio, penn.SAMPLE_RATE, hopsize, fmin, fmax)


def from_file_to_file(
    file,
    output_prefix=None,
    hopsize=penn.HOPSIZE_SECONDS,
    fmin=penn.FMIN,
    fmax=penn.FMAX):
    """Estimate pitch and periodicity with pyin and save to disk"""
    # Infer
    pitch, periodicity = from_file(file, hopsize, fmin, fmax)

    # Save to disk
    with torchutil.time.context('save'):

        # Maybe use same filename with new extension
        if output_prefix is None:
            output_prefix = file.parent / file.stem

        # Save
        torch.save(pitch, f'{output_prefix}-pitch.pt')
        torch.save(periodicity, f'{output_prefix}-periodicity.pt')


def from_files_to_files(
    files,
    output_prefixes=None,
    hopsize=penn.HOPSIZE_SECONDS,
    fmin=penn.FMIN,
    fmax=penn.FMAX):
    """Estimate pitch and periodicity with pyin and save to disk"""
    pitch_fn = functools.partial(
        from_file_to_file,
        hopsize=hopsize,
        fmin=fmin,
        fmax=fmax)
    iterator = zip(files, output_prefixes)

    # Turn off multiprocessing for benchmarking
    if penn.BENCHMARK:
        for item in torchutil.iterator(
            iterator,
            f'{penn.CONFIG}',
            total=len(files)
        ):
            pitch_fn(*item)
    else:
        with mp.get_context('spawn').Pool() as pool:
            pool.starmap(pitch_fn, iterator)


###############################################################################
# Utilities
###############################################################################


def cumulative_mean_normalized_difference(frames, min_period, max_period):
    import librosa

    a = np.fft.rfft(frames, 2 * penn.WINDOW_SIZE, axis=-2)
    b = np.fft.rfft(
        frames[..., penn.WINDOW_SIZE:0:-1, :],
        2 * penn.WINDOW_SIZE,
        axis=-2)
    acf_frames = np.fft.irfft(
        a * b, 2 * penn.WINDOW_SIZE, axis=-2)[..., penn.WINDOW_SIZE:, :]
    acf_frames[np.abs(acf_frames) < 1e-6] = 0

    # Energy terms
    energy_frames = np.cumsum(frames ** 2, axis=-2)
    energy_frames = (
        energy_frames[..., penn.WINDOW_SIZE:, :] -
        energy_frames[..., :-penn.WINDOW_SIZE, :])
    energy_frames[np.abs(energy_frames) < 1e-6] = 0

    # Difference function
    yin_frames = energy_frames[..., :1, :] + energy_frames - 2 * acf_frames

    # Cumulative mean normalized difference function
    yin_numerator = yin_frames[..., min_period: max_period + 1, :]

    # Broadcast to have leading ones
    tau_range = librosa.util.expand_to(
        np.arange(1, max_period + 1), ndim=yin_frames.ndim, axes=-2)

    cumulative_mean = (
        np.cumsum(yin_frames[..., 1: max_period + 1, :], axis=-2) / tau_range)

    yin_denominator = cumulative_mean[..., min_period - 1: max_period, :]
    yin_frames = yin_numerator / \
        (yin_denominator + librosa.util.tiny(yin_denominator))

    return yin_frames


def infer(
    audio,
    sample_rate=penn.SAMPLE_RATE,
    hopsize=penn.HOPSIZE_SECONDS,
    trough_threshold: float = 0.1,
    fmin=penn.FMIN,
    fmax=penn.FMAX):
    hopsize = int(penn.convert.seconds_to_samples(hopsize))
    import scipy

    # Debug: Print audio length and sample rate
    # print(f"Audio length: {audio.shape[-1]}")
    # print(f"Sample rate: {sample_rate}")
    # print(f"Hopsize in samples: {hopsize}")

    # Pad audio to center-align frames
    pad = penn.WINDOW_SIZE // 2
    padded = torch.nn.functional.pad(audio, (0, 2 * pad))
    # Debug: Print padded audio length
    # print(f"Padded audio length: {padded.shape[-1]}")
    # Slice and chunk audio
    frames = torch.nn.functional.unfold(
        padded[:, None, None],
        kernel_size=(1, 2 * penn.WINDOW_SIZE),
        # stride=(1, penn.HOPSIZE))[0]
        stride=(1, penn.HOPSIZE))[0]
    # print(f"Frames shape: {frames.shape}")
    # exit()

    # Calculate minimum and maximum periods
    min_period = max(int(np.floor(sample_rate / fmax)), 1)
    max_period = min(
        int(np.ceil(sample_rate / fmin)),
        penn.WINDOW_SIZE - 1)

    # Calculate cumulative mean normalized difference function
    yin_frames = cumulative_mean_normalized_difference(
        frames.numpy(),
        min_period,
        max_period)
    # print("yin_frames shape:", yin_frames.shape)

    # Parabolic interpolation
    parabolic_shifts = parabolic_interpolation(yin_frames)

    # Find local minima.
    is_trough = localmin(yin_frames, axis=-2)
    is_trough[..., 0, :] = yin_frames[..., 0, :] < yin_frames[..., 1, :]

    # Find minima below peak threshold.
    is_threshold_trough = np.logical_and(is_trough, yin_frames < trough_threshold)

    # Absolute threshold.
    # "The solution we propose is to set an absolute threshold and choose the
    # smallest value of tau that gives a minimum of d' deeper than
    # this threshold. If none is found, the global minimum is chosen instead."
    target_shape = list(yin_frames.shape)
    target_shape[-2] = 1

    global_min = np.argmin(yin_frames, axis=-2)
    yin_period = np.argmax(is_threshold_trough, axis=-2)

    global_min = global_min.reshape(target_shape)
    yin_period = yin_period.reshape(target_shape)

    no_trough_below_threshold = np.all(~is_threshold_trough, axis=-2, keepdims=True)
    yin_period[no_trough_below_threshold] = global_min[no_trough_below_threshold]

    # Refine peak by parabolic interpolation.

    yin_period = (
        min_period
        + yin_period
        + np.take_along_axis(parabolic_shifts, yin_period, axis=-2)
    )[..., 0, :]

    # Convert period to fundamental frequency.
    f0: np.ndarray = sample_rate / yin_period

    f0 = torch.from_numpy(f0)
    # print(probs.shape)
    return f0[None]


def parabolic_interpolation(frames):
    """Piecewise parabolic interpolation for yin and pyin"""
    import librosa

    parabolic_shifts = np.zeros_like(frames)
    parabola_a = (
        frames[..., :-2, :] +
        frames[..., 2:, :] -
        2 * frames[..., 1:-1, :]
    ) / 2
    parabola_b = (frames[..., 2:, :] - frames[..., :-2, :]) / 2
    parabolic_shifts[..., 1:-1, :] = \
        -parabola_b / (2 * parabola_a + librosa.util.tiny(parabola_a))
    parabolic_shifts[np.abs(parabolic_shifts) > 1] = 0
    return parabolic_shifts


@numba.stencil
def _localmin_sten(x):  # pragma: no cover
    """Numba stencil for local minima computation"""
    return (x[0] < x[-1]) & (x[0] <= x[1])

def localmin(x: np.ndarray, *, axis: int = 0) -> np.ndarray:
    """Find local minima in an array

    An element ``x[i]`` is considered a local minimum if the following
    conditions are met:

    - ``x[i] < x[i-1]``
    - ``x[i] <= x[i+1]``

    Note that the first condition is strict, and that the first element
    ``x[0]`` will never be considered as a local minimum.

    Examples
    --------
    >>> x = np.array([1, 0, 1, 2, -1, 0, -2, 1])
    >>> librosa.util.localmin(x)
    array([False,  True, False, False,  True, False,  True, False])

    >>> # Two-dimensional example
    >>> x = np.array([[1,0,1], [2, -1, 0], [2, 1, 3]])
    >>> librosa.util.localmin(x, axis=0)
    array([[False, False, False],
           [False,  True,  True],
           [False, False, False]])

    >>> librosa.util.localmin(x, axis=1)
    array([[False,  True, False],
           [False,  True, False],
           [False,  True, False]])

    Parameters
    ----------
    x : np.ndarray [shape=(d1,d2,...)]
        input vector or array
    axis : int
        axis along which to compute local minimality

    Returns
    -------
    m : np.ndarray [shape=x.shape, dtype=bool]
        indicator array of local minimality along ``axis``

    See Also
    --------
    localmax
    """
    # Rotate the target axis to the end
    xi = x.swapaxes(-1, axis)

    # Allocate the output array and rotate target axis
    lmin = np.empty_like(x, dtype=bool)
    lmini = lmin.swapaxes(-1, axis)

    # Call the vectorized stencil
    _localmin(xi, lmini)

    # Handle the edge condition not covered by the stencil
    lmini[..., -1] = xi[..., -1] < xi[..., -2]

    return lmin

@numba.guvectorize(
    [
        "void(int16[:], bool_[:])",
        "void(int32[:], bool_[:])",
        "void(int64[:], bool_[:])",
        "void(float32[:], bool_[:])",
        "void(float64[:], bool_[:])",
    ],
    "(n)->(n)",
    cache=True,
    nopython=True,
)
def _localmin(x, y):  # pragma: no cover
    """Vectorized wrapper for the localmin stencil"""
    y[:] = _localmin_sten(x)


