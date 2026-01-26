import os
import random

import numpy as np
import soundfile as sf
import soxr
import torch
from omegaconf import ListConfig
from torch.utils.data import Dataset

from linear_cae.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def find_files_with_extensions(path, extensions=[".wav", ".flac"]):
    """
    Recursively finds all files with specific extensions in a given directory or list of directories.

    Args:
        path (str or list): Root directory or list of directories.
        extensions (list): List of file extensions to include.
    """
    found_files = []
    # if path is a list, let's unwrap it
    if isinstance(path, (list, ListConfig)):
        paths = path
    else:
        paths = [path]

    # Recursively traverse the directory
    for path in paths:
        for foldername, subfolders, filenames in os.walk(path, followlinks=True):
            for filename in filenames:
                # Check if the file has an extension from the specified list
                if any(filename.lower().endswith(ext.lower()) for ext in extensions):
                    if filename.startswith("._"):
                        continue
                    # Build the full path to the file
                    file_path = os.path.join(foldername, filename)
                    found_files.append(file_path)

    return found_files


class AudioDataset(Dataset):
    def __init__(
        self,
        wav_paths,
        hop,
        fac,
        data_length,
        data_fractions,
        rms_min=0.001,
        sample_rate=44100,
        data_extensions=[".wav", ".flac"],
        tot_samples=None,
        random_sampling=True,
    ):
        self.random_sampling = random_sampling
        if data_fractions is None:
            data_fractions = [1 / len(wav_paths) for _ in wav_paths]
        tot_fractions = sum(data_fractions)
        data_fractions = [el / tot_fractions for el in data_fractions]
        self.tot_samples = tot_samples
        self.rms_min = rms_min
        self.paths = []
        self.num_samples = []
        self.num_tot_samples = []
        self.num_repetitions = []
        for path, fraction in zip(wav_paths, data_fractions):
            paths = find_files_with_extensions(path, extensions=data_extensions)
            seed_value = 42
            shuffling_random = random.Random(seed_value)
            shuffling_random.shuffle(paths)
            num_samples = len(paths)
            # print(f'Found {num_samples} samples.')
            self.paths.append(paths)
            self.num_samples.append(num_samples)
            if tot_samples is None:
                self.num_tot_samples.append(int(num_samples))
            else:
                self.num_tot_samples.append(int(tot_samples))
            self.num_repetitions.append(self.num_tot_samples[-1] // num_samples)

        log.info(f"Found {sum(self.num_samples)} train samples across {len(self.paths)} folders.")

        self.hop = hop
        self.data_length = data_length
        self.wv_length = hop * data_length + (fac - 1) * hop if data_length > 0 else -1
        self.data_fractions = torch.tensor(data_fractions)
        self.sample_rate = sample_rate
        self.cached_metadata = {}

    def __len__(self):
        return int(self.tot_samples) if self.random_sampling else self.num_samples

    def _get_audio_info(self, path):
        if path not in self.cached_metadata:
            info = sf.info(path)
            self.cached_metadata[path] = {
                "sample_rate": info.samplerate,
                "num_frames": int(info.duration * info.samplerate),
            }
        return self.cached_metadata[path]

    def __getitem__(self, idx):
        data_id = torch.multinomial(self.data_fractions, 1).item()
        if self.random_sampling:
            if idx > (self.num_samples[data_id] * self.num_repetitions[data_id]):
                idx = torch.randint(self.num_samples[data_id], size=(1,)).item()
            else:
                idx = idx % self.num_samples[data_id]
        else:
            idx = idx % self.num_samples[data_id]
        path = self.paths[data_id][idx]
        try:
            audio_info = self._get_audio_info(path)
            orig_sr = audio_info["sample_rate"]
            num_frames = audio_info["num_frames"]

            n_frames_to_load = self.wv_length
            if n_frames_to_load == -1:
                n_frames_to_load = num_frames
                rand_start = 0
            else:
                if self.sample_rate != orig_sr:
                    # We need to adjust the number of frames to load based on the original sample rate
                    n_frames_to_load = int(np.ceil((n_frames_to_load) * orig_sr / self.sample_rate))
                if num_frames <= n_frames_to_load:
                    idx = torch.randint(self.tot_samples, size=(1,)).item()
                    return self.__getitem__(idx)

                rand_start = torch.randint(num_frames - n_frames_to_load, size=(1,)).item()

            waveform, loaded_sr = sf.read(
                path,
                frames=n_frames_to_load,
                start=rand_start,
                dtype="float32",
                always_2d=True,
            )
            if loaded_sr != self.sample_rate:
                waveform = soxr.resample(
                    waveform,
                    orig_sr,
                    self.sample_rate,
                    # quality='high',
                )
            waveform = torch.from_numpy(waveform).permute(1, 0)

            if waveform.shape[0] > 1:
                mono = waveform.mean(dim=0, keepdim=True)
            else:
                mono = waveform
            stereo = torch.cat([mono, mono], dim=0)
            out = stereo[torch.randint(stereo.shape[0], size=(1,)).item(), :]

            rms = torch.sqrt(torch.mean(out**2))
            if rms < self.rms_min:
                idx = torch.randint(self.tot_samples, size=(1,)).item()
                return self.__getitem__(idx)

        except Exception as e:
            # print(e)
            log.warning(f"Error loading audio {path}: {e}")
            idx = torch.randint(self.tot_samples, size=(1,)).item()
            # return torch.zeros((2, self.wv_length))[0]
            return self.__getitem__(idx)
        return out.clone()


class TestAudioDataset(Dataset):
    """
    Dataset for evaluation/testing that loads audio files, optionally crops them,
    and ensures specific consistency in length and sample rate.
    """

    def __init__(
        self,
        wav_path,
        hop,
        fac,
        data_length,
        crop_to_length=False,
        tot_samples=None,
        random_sampling=True,
        sample_rate=44100,
        max_samples=None,
    ):
        self.random_sampling = random_sampling
        self.paths = find_files_with_extensions(wav_path, extensions=[".wav", ".flac", ".mp3"])
        # sort paths
        self.paths = sorted(self.paths)
        if max_samples is not None:
            # Lets shuffle to get some variety
            seed_value = 42
            shuffling_random = random.Random(seed_value)
            shuffling_random.shuffle(self.paths)

            self.paths = self.paths[:max_samples]
            log.info(f"Limiting test dataset to {max_samples} samples")

        seed_value = 42
        shuffling_random = random.Random(seed_value)
        shuffling_random.shuffle(self.paths)
        self.data_samples = len(self.paths)
        # print(f'Found {self.data_samples} samples.')
        log.info(f"Found {self.data_samples} test samples")
        self.hop = hop
        if tot_samples is None:
            self.tot_samples = self.data_samples
        else:
            self.tot_samples = tot_samples
        self.num_repetitions = self.tot_samples // self.data_samples
        self.wv_length = hop * data_length + (fac - 1) * hop
        self.crop_to_length = crop_to_length
        self.sample_rate = sample_rate

    def __len__(self):
        return int(self.tot_samples)

    def __getitem__(self, idx):
        if idx > (self.data_samples * self.num_repetitions):
            idx = torch.randint(self.data_samples, size=(1,)).item()
        else:
            idx = idx % self.data_samples
        path = self.paths[idx]
        frames = self.wv_length
        # We need to change the value of frames if are are resampling
        info = sf.info(path)
        if info.samplerate != self.sample_rate:
            # COrrect the frames so we load frames in samplerate rather than self.sample_rate

            # Eg. if 1 second in self.sample_rate is 16000, and samplerate is 44100, we need to load 16000 * 44100 / 16000 = 44100 frames
            frames = int(np.ceil(frames * info.samplerate / self.sample_rate))
        try:
            if self.crop_to_length:
                wv, sr = sf.read(
                    path,
                    dtype="float32",
                    always_2d=True,
                    frames=frames,
                    start=0,
                    stop=None,
                )
                # Let's check if it's mostly silence, if so, we shift the start  5s and try again
                if np.sum(np.abs(wv)) < 0.05:
                    wv, sr = sf.read(
                        path,
                        dtype="float32",
                        always_2d=True,
                        frames=frames,
                        start=5 * self.sample_rate,
                        stop=None,
                    )
            else:
                wv, sr = sf.read(path, dtype="float32", always_2d=True)
            if wv.shape[0] < self.wv_length:
                idx = torch.randint(self.tot_samples, size=(1,)).item()
                return self.__getitem__(idx)
            if sr != self.sample_rate:
                wv = soxr.resample(
                    wv,
                    sr,
                    self.sample_rate,
                    # quality='high',
                )
            wv = torch.from_numpy(wv)
            wv = wv.permute(1, 0)  # soundfiles returns (time, channels), we need (channels, time)
            # wv = F.resample(wv, orig_freq=sr, new_freq=self.sample_rate)
            if self.crop_to_length:
                wv = wv[:, : self.wv_length]

                # TODO maybe pad to self.wv_length if needed
            # convert to mono
            if wv.shape[0] > 1:
                wv = wv.mean(dim=0, keepdim=True)
            if wv.shape[0] == 1:
                wv = torch.cat([wv, wv], dim=0)
            wv = wv[:2]  # Ensure we have only 2 channels
            # if not stereo:
            wv = wv[torch.randint(wv.shape[0], size=(1,)).item(), :]
        except Exception as e:
            print(e)
            idx = torch.randint(self.tot_samples, size=(1,)).item()
            return self.__getitem__(idx)
        return wv
