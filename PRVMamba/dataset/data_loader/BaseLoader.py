"""The Base Class for data-loading.

Provides a pytorch-style data-loader for end-to-end training pipelines.
Extend the class to support specific datasets.
Dataset already supported: UBFC-rPPG, PURE, SCAMPS, BP4D+, and UBFC-PHYS.
"""
import csv
import glob
import os
import re
from math import ceil
from scipy import signal
from scipy import sparse
from unsupervised_methods.methods import POS_WANG
from unsupervised_methods import utils
import math
from multiprocessing import Pool, Process, Value, Array, Manager

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm
from retinaface import RetinaFace   # Source code: https://github.com/serengil/retinaface


class BaseLoader(Dataset):
    """The base class for data loading based on pytorch Dataset.

    The dataloader supports both providing data for pytorch training and common data-preprocessing methods,
    including reading files, resizing each frame, chunking, and video-signal synchronization.
    """

    @staticmethod
    def add_data_loader_args(parser):
        """Adds arguments to parser for training process"""
        parser.add_argument("--cached_path", default=None, type=str)
        parser.add_argument("--preprocess", default=None, action='store_true')
        return parser

    def __init__(self, dataset_name, raw_data_path, config_data):
        """Inits dataloader with lists of files.

        Args:
            dataset_name (str): name of the dataloader.
            raw_data_path (str): path to the folder containing all data.
            config_data (CfgNode): data settings (ref: config.py).
        """
        self.inputs = list()
        self.labels = list()
        self.prv_files = []  # PRV 파일 경로를 저장할 리스트 (PR_MODE용)
        self.dataset_name = dataset_name
        self.raw_data_path = raw_data_path
        self.cached_path = config_data.CACHED_PATH
        self.file_list_path = config_data.FILE_LIST_PATH
        self.preprocessed_data_len = 0
        self.data_format = config_data.DATA_FORMAT
        self.do_preprocess = config_data.DO_PREPROCESS
        self.config_data = config_data

        assert (config_data.BEGIN < config_data.END)
        assert (config_data.BEGIN >= 0)
        assert (config_data.END <= 1)
        if config_data.DO_PREPROCESS:
            self.raw_data_dirs = self.get_raw_data(self.raw_data_path)
            self.preprocess_dataset(self.raw_data_dirs, config_data.PREPROCESS, config_data.BEGIN, config_data.END)
        else:
            if not os.path.exists(self.cached_path):
                print('CACHED_PATH:', self.cached_path)
                raise ValueError(self.dataset_name,
                                 'Please set DO_PREPROCESS to True. Preprocessed directory does not exist!')
            if not os.path.exists(self.file_list_path):
                print('File list does not exist... generating now...')
                self.raw_data_dirs = self.get_raw_data(self.raw_data_path)
                self.build_file_list_retroactive(self.raw_data_dirs, config_data.BEGIN, config_data.END)
                print('File list generated.', end='\n\n')
            self.load_preprocessed_data()
        print('Cached Data Path', self.cached_path, end='\n\n')
        print('File List Path', self.file_list_path)
        print(f"{self.dataset_name} Preprocessed Dataset Length: {self.preprocessed_data_len}", end='\n\n')

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.inputs)

    def __getitem__(self, index):
        """Returns a clip of video (3,T,W,H) and its corresponding signals (T)."""
        data = np.load(self.inputs[index])
        label = np.load(self.labels[index])
        if self.data_format == 'NDCHW':
            data = np.transpose(data, (0, 3, 1, 2))
        elif self.data_format == 'NCDHW':
            data = np.transpose(data, (3, 0, 1, 2))
        elif self.data_format == 'NDHWC':
            pass
        else:
            raise ValueError('Unsupported Data Format!')
        data = np.float32(data)
        label = np.float32(label)
        # item_path: e.g., /.../501_input0.npy
        item_path = self.inputs[index]
        item_path_filename = os.path.split(item_path)[-1]
        split_idx = item_path_filename.rindex('_')
        filename = item_path_filename[:split_idx]
        chunk_id = item_path_filename[split_idx + 6:].split('.')[0]
        return data, label, filename, chunk_id

    def get_raw_data(self, raw_data_path):
        """Returns raw data directories under the path.
        
        Args:
            raw_data_path (str): a list of video_files.
        """
        raise Exception("'get_raw_data' Not Implemented")

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values.
        
        Args:
            data_dirs (List[str]): a list of video_files.
            begin (float): starting index fraction.
            end (float): ending index fraction.
        """
        raise Exception("'split_raw_data' Not Implemented")

    def read_npy_video(self, video_file):
        """Reads a video file in numpy format (.npy), returns frames (T,H,W,3)."""
        frames = np.load(video_file[0])
        if np.issubdtype(frames.dtype, np.integer) and np.min(frames) >= 0 and np.max(frames) <= 255:
            processed_frames = [frame.astype(np.uint8)[..., :3] for frame in frames]
        elif np.issubdtype(frames.dtype, np.floating) and np.min(frames) >= 0.0 and np.max(frames) <= 1.0:
            processed_frames = [(np.round(frame * 255)).astype(np.uint8)[..., :3] for frame in frames]
        else:
            raise Exception(f'Loaded frames are of an incorrect type or range! '
                            f'Received frames of type {frames.dtype} and range {np.min(frames)} to {np.max(frames)}.')
        return np.asarray(processed_frames)

    def generate_pos_psuedo_labels(self, frames, fs=30):
        """Generates POS-based PPG pseudo labels for training.
        
        Returns:
            env_norm_bvp: Hilbert envelope-normalized POS PPG signal.
        """
        WinSec = 1.6
        RGB = POS_WANG._process_video(frames)
        N = RGB.shape[0]
        H = np.zeros((1, N))
        l = math.ceil(WinSec * fs)
        for n in range(N):
            m = n - l
            if m >= 0:
                Cn = np.true_divide(RGB[m:n, :], np.mean(RGB[m:n, :], axis=0))
                Cn = np.mat(Cn).H
                S = np.matmul(np.array([[0, 1, -1], [-2, 1, 1]]), Cn)
                h = S[0, :] + (np.std(S[0, :]) / np.std(S[1, :])) * S[1, :]
                mean_h = np.mean(h)
                for temp in range(h.shape[1]):
                    h[0, temp] = h[0, temp] - mean_h
                H[0, m:n] = H[0, m:n] + (h[0])
        bvp = H
        bvp = utils.detrend(np.mat(bvp).H, 100)
        bvp = np.asarray(np.transpose(bvp))[0]
        min_freq = 0.70
        max_freq = 3
        b, a = signal.butter(2, [(min_freq) / fs * 2, (max_freq) / fs * 2], btype='bandpass')
        pos_bvp = signal.filtfilt(b, a, bvp.astype(np.double))
        analytic_signal = signal.hilbert(pos_bvp)
        amplitude_envelope = np.abs(analytic_signal)
        env_norm_bvp = pos_bvp / amplitude_envelope
        return np.array(env_norm_bvp)

    def preprocess_dataset(self, data_dirs, config_preprocess, begin, end):
        """Parses and preprocesses all raw data based on the split.
        """
        data_dirs_split = self.split_raw_data(data_dirs, begin, end)
        file_list_dict = self.multi_process_manager(data_dirs_split, config_preprocess)
        self.build_file_list(file_list_dict)
        self.load_preprocessed_data()
        print("Total Number of raw files preprocessed:", len(data_dirs_split), end='\n\n')

    def preprocess(self, frames, bvps, config_preprocess):
        """Preprocesses video frames and BVP signal.
        """
        frames = self.crop_face_resize(
            frames,
            config_preprocess.CROP_FACE.DO_CROP_FACE,
            config_preprocess.CROP_FACE.BACKEND,
            config_preprocess.CROP_FACE.USE_LARGE_FACE_BOX,
            config_preprocess.CROP_FACE.LARGE_BOX_COEF,
            config_preprocess.CROP_FACE.DETECTION.DO_DYNAMIC_DETECTION,
            config_preprocess.CROP_FACE.DETECTION.DYNAMIC_DETECTION_FREQUENCY,
            config_preprocess.CROP_FACE.DETECTION.USE_MEDIAN_FACE_BOX,
            config_preprocess.RESIZE.W,
            config_preprocess.RESIZE.H)
        data = []
        for data_type in config_preprocess.DATA_TYPE:
            f_c = frames.copy()
            if data_type == "Raw":
                data.append(f_c)
            elif data_type == "DiffNormalized":
                data.append(BaseLoader.diff_normalize_data(f_c))
            elif data_type == "Standardized":
                data.append(BaseLoader.standardized_data(f_c))
            else:
                raise ValueError("Unsupported data type!")
        data = np.concatenate(data, axis=-1)
        if config_preprocess.LABEL_TYPE == "DiffNormalized":
            bvps = BaseLoader.diff_normalize_label(bvps)
        elif config_preprocess.LABEL_TYPE == "Standardized":
            bvps = BaseLoader.standardized_label(bvps)
        elif config_preprocess.LABEL_TYPE != "Raw":
            raise ValueError("Unsupported label type!")
        if config_preprocess.DO_CHUNK:
            frames_clips, bvps_clips = self.chunk(data, bvps, config_preprocess.CHUNK_LENGTH)
        else:
            frames_clips = np.array([data])
            bvps_clips = np.array([bvps])
        return frames_clips, bvps_clips

    def face_detection(self, frame, backend, use_larger_box=False, larger_box_coef=1.0):
        """Detects face in a frame."""
        if backend == "HC":
            detector = cv2.CascadeClassifier('./dataset/haarcascade_frontalface_default.xml')
            face_zone = detector.detectMultiScale(frame)
            if len(face_zone) < 1:
                print("ERROR: No Face Detected")
                face_box_coor = [0, 0, frame.shape[0], frame.shape[1]]
            elif len(face_zone) >= 2:
                max_width_index = np.argmax(face_zone[:, 2])
                face_box_coor = face_zone[max_width_index]
                print("Warning: More than one face detected. Cropping the biggest one.")
            else:
                face_box_coor = face_zone[0]
        elif backend == "RF":
            res = RetinaFace.detect_faces(frame)
            if len(res) > 0:
                highest_score_face = max(res.values(), key=lambda x: x['score'])
                face_zone = highest_score_face['facial_area']
                x_min, y_min, x_max, y_max = face_zone
                x = x_min
                y = y_min
                width = x_max - x_min
                height = y_max - y_min
                center_x = x + width // 2
                center_y = y + height // 2
                square_size = max(width, height)
                new_x = center_x - (square_size // 2)
                new_y = center_y - (square_size // 2)
                face_box_coor = [new_x, new_y, square_size, square_size]
            else:
                print("ERROR: No Face Detected")
                face_box_coor = [0, 0, frame.shape[0], frame.shape[1]]
        else:
            raise ValueError("Unsupported face detection backend!")
        if use_larger_box:
            face_box_coor[0] = max(0, face_box_coor[0] - (larger_box_coef - 1.0) / 2 * face_box_coor[2])
            face_box_coor[1] = max(0, face_box_coor[1] - (larger_box_coef - 1.0) / 2 * face_box_coor[3])
            face_box_coor[2] = larger_box_coef * face_box_coor[2]
            face_box_coor[3] = larger_box_coef * face_box_coor[3]
        return face_box_coor

    def crop_face_resize(self, frames, use_face_detection, backend, use_larger_box, larger_box_coef,
                         use_dynamic_detection, detection_freq, use_median_box, width, height):
        """Crops faces in frames and resizes them."""
        if frames.shape[0] == 0:
            print("⚠️ Warning: No frames available for face cropping. Skipping...")
            return np.array([])
        if use_dynamic_detection:
            num_dynamic_det = ceil(frames.shape[0] / detection_freq)
        else:
            num_dynamic_det = 1
        face_region_all = []
        for idx in range(num_dynamic_det):
            if use_face_detection:
                face_region_all.append(self.face_detection(frames[detection_freq * idx], backend, use_larger_box, larger_box_coef))
            else:
                face_region_all.append([0, 0, frames.shape[1], frames.shape[2]])
        face_region_all = np.asarray(face_region_all, dtype='int')
        if use_median_box:
            face_region_median = np.median(face_region_all, axis=0).astype('int')
        resized_frames = np.zeros((frames.shape[0], height, width, 3))
        for i in range(frames.shape[0]):
            frame = frames[i]
            reference_index = i // detection_freq if use_dynamic_detection else 0
            if use_face_detection:
                face_region = face_region_median if use_median_box else face_region_all[reference_index]
                frame = frame[max(face_region[1], 0):min(face_region[1] + face_region[3], frame.shape[0]),
                              max(face_region[0], 0):min(face_region[0] + face_region[2], frame.shape[1])]
            resized_frames[i] = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        return resized_frames

    def chunk(self, frames, bvps, chunk_length):
        """Chunks the data into snippets."""
        clip_num = frames.shape[0] // chunk_length
        frames_clips = [frames[i * chunk_length:(i + 1) * chunk_length] for i in range(clip_num)]
        bvps_clips = [bvps[i * chunk_length:(i + 1) * chunk_length] for i in range(clip_num)]
        return np.array(frames_clips), np.array(bvps_clips)

    def save(self, frames_clips, bvps_clips, filename):
        """Saves the chunked data."""
        if not os.path.exists(self.cached_path):
            os.makedirs(self.cached_path, exist_ok=True, mode=0o777)
        count = 0
        for i in range(len(bvps_clips)):
            assert (len(self.inputs) == len(self.labels))
            input_path_name = os.path.join(self.cached_path, f"{filename}_input{count}.npy")
            label_path_name = os.path.join(self.cached_path, f"{filename}_label{count}.npy")
            self.inputs.append(input_path_name)
            self.labels.append(label_path_name)
            np.save(input_path_name, frames_clips[i])
            np.save(label_path_name, bvps_clips[i])
            count += 1
        return count

    def save_multi_process(self, frames_clips, bvps_clips, filename):
        """Saves the chunked data using multi-processing."""
        if not os.path.exists(self.cached_path):
            os.makedirs(self.cached_path, exist_ok=True, mode=0o777)
        count = 0
        input_path_name_list = []
        label_path_name_list = []
        for i in range(len(bvps_clips)):
            assert (len(self.inputs) == len(self.labels))
            input_path_name = os.path.join(self.cached_path, f"{filename}_input{count}.npy")
            label_path_name = os.path.join(self.cached_path, f"{filename}_label{count}.npy")
            input_path_name_list.append(input_path_name)
            label_path_name_list.append(label_path_name)
            np.save(input_path_name, frames_clips[i])
            np.save(label_path_name, bvps_clips[i])
            count += 1
        return input_path_name_list, label_path_name_list

    def multi_process_manager(self, data_dirs, config_preprocess, multi_process_quota=8):
        """Distributes dataset preprocessing across multiple processes."""
        print('Preprocessing dataset...')
        file_num = len(data_dirs)
        choose_range = range(0, file_num)
        pbar = tqdm(list(choose_range))
        manager = Manager()
        file_list_dict = manager.dict()
        p_list = []
        running_num = 0
        for i in choose_range:
            process_flag = True
            while process_flag:
                if running_num < multi_process_quota:
                    p = Process(target=self.preprocess_dataset_subprocess, 
                                args=(data_dirs, config_preprocess, i, file_list_dict))
                    p.start()
                    p_list.append(p)
                    running_num += 1
                    process_flag = False
                for p_ in p_list:
                    if not p_.is_alive():
                        p_list.remove(p_)
                        p_.join()
                        running_num -= 1
                        pbar.update(1)
        for p_ in p_list:
            p_.join()
            pbar.update(1)
        pbar.close()
        return file_list_dict

    def build_file_list(self, file_list_dict):
        """
        Builds a list of files used by the dataloader (e.g., for train/val/test) and saves it as a CSV.
        When PR_MODE is enabled, associates each input file with its corresponding PRV file.
        """
        input_files = []
        prv_files = []
        if self.config_data.PR_MODE:
            # 각 프로세스 결과가 dict인 경우
            for process_num, entry in file_list_dict.items():
                if isinstance(entry, dict) and "input" in entry and "prv" in entry:
                    for in_file in entry["input"]:
                        input_files.append(in_file)
                        prv_files.append(entry["prv"])
                elif isinstance(entry, list):
                    input_files.extend(entry)
                else:
                    raise ValueError(f"Unexpected format in file_list_dict for process {process_num}: {entry}")
            if not input_files:
                raise ValueError(self.dataset_name, 'No files in file list')
            file_list_df = pd.DataFrame({'input_files': input_files, 'prv_files': prv_files})
        else:
            for process_num, file_paths in file_list_dict.items():
                input_files.extend(file_paths)
            if not input_files:
                raise ValueError(self.dataset_name, 'No files in file list')
            file_list_df = pd.DataFrame({'input_files': input_files})
        os.makedirs(os.path.dirname(self.file_list_path), exist_ok=True, mode=0o777)
        file_list_df.to_csv(self.file_list_path, index=False)

    def build_file_list_retroactive(self, data_dirs, begin, end):
        """Retroactively builds a file list if not already generated and saves it as a CSV."""
        data_dirs_subset = self.split_raw_data(data_dirs, begin, end)
        filename_list = list({data_dirs_subset[i]['index'] for i in range(len(data_dirs_subset))})
        file_list = []
        for fname in filename_list:
            processed_file_data = list(glob.glob(os.path.join(self.cached_path, f"{fname}_input*.npy")))
            file_list += processed_file_data
        if not file_list:
            raise ValueError(self.dataset_name,
                             'File list empty. Check preprocessed data folder exists and is not empty.')
        file_list_df = pd.DataFrame(file_list, columns=['input_files'])
        os.makedirs(os.path.dirname(self.file_list_path), exist_ok=True, mode=0o777)
        file_list_df.to_csv(self.file_list_path, index=False)

    def load_preprocessed_data(self):
        """Loads preprocessed data from the CSV file.
        Also loads PRV file paths if the CSV contains a 'prv_files' column.
        If PR_MODE is enabled but no 'prv_files' column exists,
        derive the PRV file path for each input by extracting the subject ID.
        """
        file_list_path = self.file_list_path
        file_list_df = pd.read_csv(file_list_path)
        inputs = file_list_df['input_files'].tolist()
        if not inputs:
            raise ValueError(self.dataset_name + ' dataset loading data error!')
        # 사용된 CSV의 순서를 그대로 사용합니다.
        self.inputs = inputs
        self.labels = [input_file.replace("input", "label") for input_file in self.inputs]
        self.preprocessed_data_len = len(self.inputs)
        if 'prv_files' in file_list_df.columns:
            self.prv_files = file_list_df['prv_files'].tolist()
        else:
            # 입력 파일명에서 subject ID를 추출하여 PRV 파일 경로를 생성합니다.
            # 예를 들어, "subject48_input8.npy" -> subject = "subject48"
            # 그리고 PRV 파일은 "subject48_prv.npy" 로 저장되었으므로:
            self.prv_files = []
            for inp in self.inputs:
                # 파일명에서 subject ID 추출
                basename = os.path.basename(inp)  # 예: "subject48_input8.npy"
                subject = basename.split('_')[0]    # "subject48"
                # 디렉토리 경로는 inp와 동일하게 사용
                dir_path = os.path.dirname(inp)
                prv_file = os.path.join(dir_path, f"{subject}_prv.npy")
                self.prv_files.append(prv_file)




    
    @staticmethod
    def diff_normalize_data(data):
        """Calculates discrete difference in video data along the time-axis and normalizes by its standard deviation."""
        n, h, w, c = data.shape
        diffnormalized_len = n - 1
        diffnormalized_data = np.zeros((diffnormalized_len, h, w, c), dtype=np.float32)
        diffnormalized_data_padding = np.zeros((1, h, w, c), dtype=np.float32)
        for j in range(diffnormalized_len):
            diffnormalized_data[j, :, :, :] = (data[j + 1, :, :, :] - data[j, :, :, :]) / (data[j + 1, :, :, :] + data[j, :, :, :] + 1e-7)
        diffnormalized_data = diffnormalized_data / np.std(diffnormalized_data)
        diffnormalized_data = np.append(diffnormalized_data, diffnormalized_data_padding, axis=0)
        diffnormalized_data[np.isnan(diffnormalized_data)] = 0
        return diffnormalized_data

    @staticmethod
    def diff_normalize_label(label):
        """Calculates discrete difference in labels along the time-axis and normalizes by its standard deviation."""
        diff_label = np.diff(label, axis=0)
        diffnormalized_label = diff_label / np.std(diff_label)
        diffnormalized_label = np.append(diffnormalized_label, np.zeros(1), axis=0)
        diffnormalized_label[np.isnan(diffnormalized_label)] = 0
        return diffnormalized_label

    @staticmethod
    def standardized_data(data):
        """Z-score standardization for video data."""
        data = data - np.mean(data)
        data = data / np.std(data)
        data[np.isnan(data)] = 0
        return data

    @staticmethod
    def standardized_label(label):
        """Z-score standardization for label signal."""
        label = label - np.mean(label)
        label = label / np.std(label)
        label[np.isnan(label)] = 0
        return label

    @staticmethod
    def resample_ppg(input_signal, target_length):
        """Resamples a PPG sequence to a specific length."""
        return np.interp(np.linspace(1, input_signal.shape[0], target_length),
                         np.linspace(1, input_signal.shape[0], input_signal.shape[0]), input_signal)
