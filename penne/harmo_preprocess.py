# monophonic pitch estimator using harmonic_net.

# torch
from random import shuffle
import torch
import torch.cuda
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader

# system
from tqdm import tqdm
from datetime import datetime
import os
from glob import glob

import numpy as np
import matplotlib.pyplot as plt

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class PitchTracker():
    def __init__(self, 
        checkpoint_path = None,
        fmin = 27.5,
        sample_rate = 16000,
        hop_length = 160,
        frame_len = 1024,
        frames_per_step = 1,
        post_processing = True,
        high_threshold=0.8, 
        low_threshold=0.1, 
        n_beam = 5, 
        min_pitch_dur = 0.1,
        freq_bins_in = 88*4,
        freq_bins_out = 88*4,
        bins_per_octave_in = 48,
        bins_per_octave_out = 48,
        device = "cuda" if torch.cuda.is_available() else "cpu",
        dataset = "MDB"
    ) -> None:

        self.hop_length = hop_length
        self.frame_len = frame_len
        self.frames_per_step = frames_per_step
        # post processing
        self.min_pitch_len = min_pitch_dur * sample_rate / hop_length
        self.post_processing = post_processing
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.n_beam = n_beam

        self.device = device

        self.freq_bins_in = freq_bins_in
        self.freq_bins_out = freq_bins_out
        self.bins_per_octave_in = bins_per_octave_in
        self.bins_per_octave_out = bins_per_octave_out
        self.fmin = fmin
        self.sample_rate = sample_rate
        self.dataset = dataset

    def pred(self, waveform, sr):
        # inputs: 
        #     waveform:
        #     sr: 16000
        # returns:
        #     time, freq, activation, activation_map
        #     [T], [T], [T], [T x 352]

        if isinstance(waveform,np.ndarray):
            waveform = torch.tensor(waveform)
        if(len(waveform.size()) == 1):
            waveform = waveform[None, :]
        
        if(sr != self.sample_rate):
            print("convert sr from %d to %d"%(sr, self.sample_rate))
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # start from the 0
        if self.dataset == "MDB":
            waveform = F.pad(waveform, [self.frame_len//2, self.frame_len//2], mode='reflect')
        b, wav_len = waveform.shape
        assert b == 1
        num_frames = int((wav_len - self.frame_len)//self.hop_length) + 1
        batch = torch.zeros([1, num_frames, self.frame_len])
        for i in range(num_frames):
            begin = i * self.hop_length
            end = begin + self.frame_len
            batch[:, i, :] = waveform[:, begin:end]

        times = np.arange(num_frames) * (self.hop_length/self.sample_rate)

        result_dict = {
            # 'pred_freqs':[],
            # 'pred_activations':[],
            'pred_activations_map':[],
        }


        steps = int(np.ceil(num_frames / self.frames_per_step))
        result = np.zeros((1, self.frame_len, steps))
        for i in tqdm(range(steps)):
            begin = i * self.frames_per_step
            end = begin + self.frames_per_step
            waveforms = batch[:, begin:end]
            result[0,:,i] = waveforms
        return result

    def pred_file(self, audio_path, output_dir=None, save_activation=True):
        wav_path_list = []
        if os.path.isdir(audio_path):
            all_files = glob(os.path.join(audio_path, "*"))
            for path in all_files:
                _, ext = os.path.splitext(path)
                if ext.lower() in ['.wav', '.mp3', '.flac']:
                    wav_path_list.append(path)
        else:
            wav_path_list.append(audio_path)

        for i, wav_path in enumerate(wav_path_list):
            
            result_dir, basename = os.path.split(wav_path)
            if(output_dir != None):
                result_dir = str(output_dir)
                os.makedirs(result_dir, exist_ok=True)
            wav_name, ext = os.path.splitext(basename)
            pred_path = os.path.join(result_dir, wav_name + ".f0.txt")

            waveform, sr = torchaudio.load(wav_path)
            waveform = torch.sum(waveform, dim=0, keepdim=True)

            return self.pred(waveform, sr)

    def hz_to_onehot(self, hz, freq_bins, bins_per_octave):
        # input: [b x T]
        # output: [b x T x freq_bins]

        fmin = self.fmin

        indexs = ( torch.log((hz+0.0000001)/fmin) / np.log(2.0**(1.0/bins_per_octave)) + 0.5 ).long()
        assert(torch.max(indexs) < freq_bins)
        mask = (indexs >= 0).long()
        # => [b x T x 1]
        mask = torch.unsqueeze(mask, dim=2)
        # => [b x T x freq_bins]
        onehot = F.one_hot(torch.clip(indexs, 0), freq_bins)
        onehot = onehot * mask # mask the freq below fmin
        return onehot

    def onehot_to_hz(self, onehot, bins_per_octave, threshold = 0.6):
        # input: [b x T x freq_bins]
        # output: [b x T]
        fmin = self.fmin
        max_onehot = torch.max(onehot, dim=2)
        indexs = max_onehot[1]
        mask = (max_onehot[0] > threshold).float()

        hz = fmin * (2**(indexs/bins_per_octave))
        hz = hz * mask # set freq to 0 if activate val below threshold
        
        return hz, max_onehot[0]

