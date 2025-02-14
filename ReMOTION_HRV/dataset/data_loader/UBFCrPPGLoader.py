import os
import re
import cv2
import glob
import numpy as np
from tqdm import tqdm
from scipy.integrate import simps
from scipy.signal import find_peaks
from dataset.data_loader.BaseLoader import BaseLoader
from multiprocessing import Process, Manager

# PRV metrics calculation functions
def calculate_sdnn(rr_intervals):
    """Calculate the standard deviation of RR intervals (SDNN)."""
    return np.std(rr_intervals)

def calculate_rmssd(rr_intervals):
    """Calculate the root mean square of successive differences (RMSSD)."""
    diff = np.diff(rr_intervals)
    return np.sqrt(np.mean(diff**2))
'''
def calculate_lf_hf_(rr_intervals, fs=30):
    """
    Calculate LF and HF powers using the power spectral density of the RR intervals.
    """
    rr_signal = np.interp(np.arange(0, len(rr_intervals), 1/fs),
                          np.arange(0, len(rr_intervals)), rr_intervals)
    freqs, psd = np.fft.fftfreq(len(rr_signal), 1/fs), np.abs(np.fft.fft(rr_signal))**2
    psd = psd[:len(psd)//2]
    lf_range = (0.04, 0.15)
    hf_range = (0.15, 0.4)
    lf = simps(psd[(freqs >= lf_range[0]) & (freqs <= lf_range[1])],
               freqs[(freqs >= lf_range[0]) & (freqs <= lf_range[1])])
    hf = simps(psd[(freqs >= hf_range[0]) & (freqs <= hf_range[1])],
               freqs[(freqs >= hf_range[0]) & (freqs <= hf_range[1])])
    return lf, hf
'''
def calculate_lf_hf(rr_intervals, fs=30):
    """
    Calculate LF and HF powers using the power spectral density of the RR intervals.
    """
    rr_signal = np.interp(np.arange(0, len(rr_intervals), 1/fs),
                          np.arange(0, len(rr_intervals)), rr_intervals)
    freqs = np.fft.rfftfreq(len(rr_signal), 1/fs)
    psd = np.abs(np.fft.rfft(rr_signal))**2
    lf_range = (0.04, 0.15)
    hf_range = (0.15, 0.4)
    lf = simps(psd[(freqs >= lf_range[0]) & (freqs <= lf_range[1])],
               freqs[(freqs >= lf_range[0]) & (freqs <= lf_range[1])])
    hf = simps(psd[(freqs >= hf_range[0]) & (freqs <= hf_range[1])],
               freqs[(freqs >= hf_range[0]) & (freqs <= hf_range[1])])
    return lf, hf


def calculate_prv_metrics(bvp_signal, fs=30):
    """
    Calculate PRV metrics from a BVP signal.
    Returns SDNN, RMSSD, LF, and HF.
    """
    peaks, _ = find_peaks(bvp_signal, distance=fs//2)
    rr_intervals = np.diff(peaks) / fs  # in seconds
    sdnn = calculate_sdnn(rr_intervals)
    rmssd = calculate_rmssd(rr_intervals)
    lf, hf = calculate_lf_hf(rr_intervals, fs)
    return sdnn, rmssd, lf, hf

class UBFCrPPGLoader(BaseLoader):
    """
    UBFC-rPPG data loader.
    This loader reads video and BVP signals and computes PRV metrics (SDNN, RMSSD, LF, HF)
    from the BVP. The computed PRV metrics are saved to a separate file and recorded in the file list.
    """
    def __init__(self, name, data_path, config_data):
        super().__init__(name, data_path, config_data)
        self.pr_mode = getattr(config_data.TRAIN, "PR_MODE", False)

    def get_raw_data(self, data_path):
        data_dirs = glob.glob(os.path.join(data_path, "subject*"))
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")
        dirs = [{"index": re.search(r'subject(\d+)', data_dir).group(0), "path": data_dir} for data_dir in data_dirs]
        return dirs

    def split_raw_data(self, data_dirs, begin, end):
        if begin == 0 and end == 1:
            return data_dirs
        file_num = len(data_dirs)
        choose_range = range(int(begin * file_num), int(end * file_num))
        return [data_dirs[i] for i in choose_range]

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        filename = os.path.split(data_dirs[i]['path'])[-1]
        saved_filename = data_dirs[i]['index']

        # Read video frames
        if 'None' in config_preprocess.DATA_AUG:
            frames = self.read_video(os.path.join(data_dirs[i]['path'], "vid.avi"))
        elif 'Motion' in config_preprocess.DATA_AUG:
            frames = self.read_npy_video(glob.glob(os.path.join(data_dirs[i]['path'], '*.npy')))
        else:
            raise ValueError(f'Unsupported DATA_AUG for {self.dataset_name}!')

        # Read BVP signal
        if config_preprocess.USE_PSUEDO_PPG_LABEL:
            bvps = self.generate_pos_psuedo_labels(frames, fs=self.config_data.FS)
        else:
            bvps = self.read_wave(os.path.join(data_dirs[i]['path'], "ground_truth.txt"))

        # preprocess
        frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
        input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, saved_filename)

        # config.TRAIN.PR_MODE=True -> calculate and save PRV Mectric
        if self.pr_mode:
            sdnn, rmssd, lf, hf = calculate_prv_metrics(bvps, fs=self.config_data.FS)
            prv_metrics = np.array([sdnn, rmssd, lf, hf])
            print(f"PRV Metrics for subject {saved_filename}: SDNN={sdnn}, RMSSD={rmssd}, LF={lf}, HF={hf}")
            prv_file_name = os.path.join(self.cached_path, f"{saved_filename}_prv.npy")
            prv_dir = os.path.dirname(prv_file_name)
            if not os.path.exists(prv_dir):
                os.makedirs(prv_dir, exist_ok=True)
            np.save(prv_file_name, prv_metrics)
            file_list_dict[i] = {"input": input_name_list, "prv": prv_file_name}
        else:
            file_list_dict[i] = {"input": input_name_list}
    @staticmethod
    def read_video(video_file):
        if not os.path.exists(video_file):
            raise FileNotFoundError(f"Video file not found: {video_file}")
        VidObj = cv2.VideoCapture(video_file)
        if not VidObj.isOpened():
            raise ValueError(f"Could not open video file: {video_file}")
        frames = []
        while True:
            success, frame = VidObj.read()
            if not success:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        VidObj.release()
        if len(frames) == 0:
            raise ValueError(f"No frames extracted from video: {video_file}")
        return np.asarray(frames)

    @staticmethod
    def read_wave(bvp_file):
        with open(bvp_file, "r") as f:
            data = f.read().strip().split("\n")
            bvp = [float(x) for x in data[0].split()]
        return np.asarray(bvp)


'''
"""The dataloader for UBFC-rPPG dataset.

Details for the UBFC-rPPG Dataset see https://sites.google.com/view/ybenezeth/ubfcrppg.
If you use this dataset, please cite this paper:
S. Bobbia, R. Macwan, Y. Benezeth, A. Mansouri, J. Dubois, "Unsupervised skin tissue segmentation for remote photoplethysmography", Pattern Recognition Letters, 2017.
"""
import os
import re
import cv2
import glob
import numpy as np

from tqdm import tqdm
from scipy.integrate import simps
from scipy.signal import find_peaks
from dataset.data_loader.BaseLoader import BaseLoader
from multiprocessing import Pool, Process, Value, Array, Manager

class UBFCrPPGLoader(BaseLoader):
    """The data loader for the UBFC-rPPG dataset."""

    def __init__(self, name, data_path, config_data):
        """Initializes an UBFC-rPPG dataloader.
            Args:
                data_path(str): path of a folder which stores raw video and bvp data.
                e.g. data_path should be "RawData" for below dataset structure:
                -----------------
                     RawData/
                     |   |-- subject1/
                     |       |-- vid.avi
                     |       |-- ground_truth.txt
                     |   |-- subject2/
                     |       |-- vid.avi
                     |       |-- ground_truth.txt
                     |...
                     |   |-- subjectn/
                     |       |-- vid.avi
                     |       |-- ground_truth.txt
                -----------------
                name(string): name of the dataloader.
                config_data(CfgNode): data settings(ref:config.py).
        """
        super().__init__(name, data_path, config_data)

    def get_raw_data(self, data_path):
        """Returns data directories under the path(For UBFC-rPPG dataset)."""
        data_dirs = glob.glob(data_path + os.sep + "subject*")
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")
        dirs = [{"index": re.search(
            'subject(\d+)', data_dir).group(0), "path": data_dir} for data_dir in data_dirs]
        return dirs

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values."""
        if begin == 0 and end == 1:  # return the full directory if begin == 0 and end == 1
            return data_dirs

        file_num = len(data_dirs)
        choose_range = range(int(begin * file_num), int(end * file_num))
        data_dirs_new = []

        for i in choose_range:
            data_dirs_new.append(data_dirs[i])

        return data_dirs_new

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """ invoked by preprocess_dataset for multi_process."""
        filename = os.path.split(data_dirs[i]['path'])[-1]
        saved_filename = data_dirs[i]['index']

        # Read Frames
        if 'None' in config_preprocess.DATA_AUG:
            # Utilize dataset-specific function to read video
            frames = self.read_video(
                os.path.join(data_dirs[i]['path'],"vid.avi"))
        elif 'Motion' in config_preprocess.DATA_AUG:
            # Utilize general function to read video in .npy format
            frames = self.read_npy_video(
                glob.glob(os.path.join(data_dirs[i]['path'],'*.npy')))
        else:
            raise ValueError(f'Unsupported DATA_AUG specified for {self.dataset_name} dataset! Received {config_preprocess.DATA_AUG}.')

        # Read Labels
        if config_preprocess.USE_PSUEDO_PPG_LABEL:
            bvps = self.generate_pos_psuedo_labels(frames, fs=self.config_data.FS)
        else:
            bvps = self.read_wave(
                os.path.join(data_dirs[i]['path'],"ground_truth.txt"))
            
        frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
        input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, saved_filename)
        file_list_dict[i] = input_name_list
    @staticmethod
    def read_video(video_file):
        """Reads a video file, returns frames(T, H, W, 3)"""
        
        if not os.path.exists(video_file):
            raise FileNotFoundError(f"Video file not found: {video_file}")

        VidObj = cv2.VideoCapture(video_file)
        
        if not VidObj.isOpened():
            raise ValueError(f"Could not open video file: {video_file}")

        frames = []
        while True:
            success, frame = VidObj.read()
            if not success:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        VidObj.release()

        if len(frames) == 0:
            raise ValueError(f"No frames extracted from video: {video_file}")

        return np.asarray(frames)
    
    @staticmethod
    def read_wave(bvp_file):
        """Reads a bvp signal file."""
        with open(bvp_file, "r") as f:
            str1 = f.read()
            str1 = str1.split("\n")
            bvp = [float(x) for x in str1[0].split()]
        return np.asarray(bvp)
'''
