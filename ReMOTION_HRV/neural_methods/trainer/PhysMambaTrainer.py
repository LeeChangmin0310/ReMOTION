import os
import math
import logging
import numpy as np
import torch
import torch.optim as optim
from evaluation.metrics import calculate_metrics
from neural_methods.loss.PhysNetNegPearsonLoss import Neg_Pearson
from neural_methods.model.PhysMamba import PhysMamba
from neural_methods.trainer.BaseTrainer import BaseTrainer
from tqdm import tqdm
from scipy.signal import welch

# Import the PRV metrics calculation function from the loader module.
from dataset.data_loader.UBFCrPPGLoader import calculate_prv_metrics


class PhysMambaTrainer(BaseTrainer):
    def __init__(self, config, data_loader) -> None:
        """
        Initialize the trainer.
        - Sets up the model, loss function, optimizer, and learning rate scheduler.
        - If PR_MODE is enabled, additional settings such as the PRV loss weight are configured.
        """
        super().__init__()
        self.config = config
        self.device = torch.device(config.DEVICE)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.frame_rate = config.TRAIN.DATA.FS

        # Store the PR_MODE flag as an instance variable.
        self.prv_mode = getattr(config.TRAIN, "PR_MODE", False)

        # Setup logging.
        os.makedirs(config.LOG.EXPERIMENT_DIR, exist_ok=True)
        log_file = os.path.join(config.LOG.EXPERIMENT_DIR, f"{self.model_file_name}.log")
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initialized PhysMambaTrainer.")

        # Initialize the model and set up multi-GPU processing if available.
        self.model = PhysMamba(prv_mode=self.prv_mode).to(self.device)
        if self.num_of_gpu > 0:
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))

        if config.TOOLBOX_MODE == "train_and_test":
            self.num_train_batches = len(data_loader["train"])
            self.criterion_Pearson = Neg_Pearson()
            self.optimizer = optim.Adam(self.model.parameters(), lr=config.TRAIN.LR, weight_decay=0.0005)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=config.TRAIN.LR, epochs=config.TRAIN.EPOCHS, steps_per_epoch=self.num_train_batches)
            if self.prv_mode:
                self.prv_loss_weight = getattr(config.TRAIN, "PR_LOSS_WEIGHT", 0.5)
        elif config.TOOLBOX_MODE == "only_test":
            self.criterion_Pearson_test = Neg_Pearson()
        else:
            raise ValueError("PhysNet trainer initialized in incorrect toolbox mode!")
        
        self.min_valid_loss = None
        self.best_epoch = 0

    def normalize_signal(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Normalize the given signal using its mean and standard deviation.
        """
        mean = torch.mean(signal, dim=-1, keepdim=True)
        std = torch.std(signal, dim=-1, keepdim=True)
        # Avoid division by zero by adding a small constant.
        std = std + 1e-8
        return (signal - mean) / std

    def evaluate_prv_metrics(self, rPPG_pred: torch.Tensor, gt_prv: torch.Tensor) -> float:
        """
        Calculate PRV metrics from the predicted rPPG signal and return the MSE compared to the ground truth.
        """
        rPPG_np = rPPG_pred.cpu().detach().numpy()
        metrics_pred = [calculate_prv_metrics(rPPG_np[i]) for i in range(rPPG_np.shape[0])]
        metrics_pred = np.array(metrics_pred)
        gt_np = gt_prv.cpu().detach().numpy()
        mse = np.mean((metrics_pred - gt_np) ** 2)
        return mse

    def aggregate_prv_metrics(self, rPPG_pred: torch.Tensor, gt_prv: torch.Tensor):
        """
        For all samples in the batch, calculate the PRV metrics and return the average predicted metrics and
        average ground truth metrics.
        """
        rPPG_np = rPPG_pred.cpu().detach().numpy()
        pred_metrics = [calculate_prv_metrics(rPPG_np[i]) for i in range(rPPG_np.shape[0])]
        pred_metrics = np.array(pred_metrics)
        gt_metrics = gt_prv.cpu().detach().numpy()
        avg_pred = np.mean(pred_metrics, axis=0)
        avg_gt = np.mean(gt_metrics, axis=0)
        return avg_pred, avg_gt

    def train(self, data_loader: dict) -> None:
        """
        Training loop:
         - For each batch, compute the rPPG reconstruction loss and, if PR_MODE is enabled, also compute the PRV metric loss.
         - Log the losses and PRV metrics at regular intervals and save the model after each epoch.
        """
        if "train" not in data_loader or data_loader["train"] is None:
            raise ValueError("No training data provided.")
        self.logger.info('Starting training...')
        for epoch in range(self.max_epoch_num):
            self.logger.info(f"==== Training Epoch: {epoch} ====")
            self.model.train()
            running_loss = running_ppg_loss = running_prv_error = 0.0
            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description(f"Train epoch {epoch}")
                # Unpack the batch: batch[0]: input video, batch[1]: BVP ground truth, batch[4]: Ground truth PRV metrics (if PR_MODE)
                data = batch[0].float().to(self.device)
                labels = batch[1].float().to(self.device)
                if self.prv_mode:
                    gt_prv = batch[4].float().to(self.device)
                self.optimizer.zero_grad()
                # Forward pass: if PR_MODE is enabled, take the first element as the rPPG signal.
                pred_ppg = self.model(data)[0] if self.prv_mode else self.model(data)
                # Normalize signals for Pearson loss.
                pred_ppg_norm = self.normalize_signal(pred_ppg)
                labels_norm = self.normalize_signal(labels)
                loss_ppg = self.criterion_Pearson(pred_ppg_norm, labels_norm)
                loss = loss_ppg
                running_ppg_loss += loss_ppg.item()
                if self.prv_mode:
                    prv_error = self.evaluate_prv_metrics(pred_ppg, gt_prv)
                    loss += self.prv_loss_weight * torch.tensor(prv_error, device=self.device)
                    running_prv_error += prv_error
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                running_loss += loss.item()
                tbar.set_postfix(loss=loss.item())
                if (idx + 1) % 100 == 0:
                    if self.prv_mode:
                        avg_ppg_loss = running_ppg_loss / 100
                        avg_prv_error = running_prv_error / 100
                        self.logger.info(f"[Epoch {epoch}, Batch {idx+1}] Avg Pearson Loss = {avg_ppg_loss:.4f}, Avg PRV MSE = {avg_prv_error:.4f}")
                        running_ppg_loss = running_prv_error = 0.0
                    self.logger.info(f"[Epoch {epoch}, Batch {idx+1}] Combined Loss = {running_loss/100:.4f}")
                    running_loss = 0.0
                    torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            self.save_model(epoch)
            # Validation phase
            if not self.config.TEST.USE_LAST_EPOCH:
                valid_results = self.valid(data_loader)
                valid_loss = valid_results[0]
                self.logger.info(f"Validation Combined Loss = {valid_loss:.4f}")
                if self.prv_mode:
                    valid_ppg_loss, valid_prv_error, avg_pred_metrics, avg_gt_metrics = valid_results[1:]
                    diff_metrics = np.abs(avg_pred_metrics - avg_gt_metrics)
                    self.logger.info(f"Validation Pearson Loss = {valid_ppg_loss:.4f}, Validation PRV MSE = {valid_prv_error:.4f}")
                    self.logger.info(f"Validation PRV Metrics: Predicted = {avg_pred_metrics}, Ground Truth = {avg_gt_metrics}, Abs Diff = {diff_metrics}")
                if self.min_valid_loss is None or valid_loss < self.min_valid_loss:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    self.logger.info(f"Updated Best Model - Epoch: {self.best_epoch}")
                    self.save_best_model()
            torch.cuda.empty_cache()
        self.logger.info(f"Best Trained Epoch: {self.best_epoch}, Min Validation Loss: {self.min_valid_loss}")

    def valid(self, data_loader: dict):
        """
        Validation loop:
         - For each batch, compute the loss and, if PR_MODE is enabled, calculate the PRV metrics.
         - Return the average combined loss, average Pearson loss, average PRV MSE, average predicted PRV metrics,
           and average ground truth PRV metrics.
        """
        self.logger.info("Starting validation...")
        valid_losses = []
        valid_ppg_losses = []
        valid_prv_errors = []
        all_predicted_metrics = []
        all_gt_metrics = []
        self.model.eval()
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for valid_batch in vbar:
                data = valid_batch[0].float().to(self.device)
                labels = valid_batch[1].float().to(self.device)
                if self.prv_mode:
                    gt_prv = valid_batch[4].float().to(self.device)
                    pred_ppg = self.model(data)[0]
                    pred_ppg_norm = self.normalize_signal(pred_ppg)
                    labels_norm = self.normalize_signal(labels)
                    loss_ppg = self.criterion_Pearson(pred_ppg_norm, labels_norm)
                    prv_error = self.evaluate_prv_metrics(pred_ppg, gt_prv)
                    loss = loss_ppg + self.prv_loss_weight * prv_error
                    valid_ppg_losses.append(loss_ppg.item())
                    valid_prv_errors.append(prv_error)
                    pred_metrics, gt_metrics = self.aggregate_prv_metrics(pred_ppg, gt_prv)
                    all_predicted_metrics.append(pred_metrics)
                    all_gt_metrics.append(gt_metrics)
                else:
                    pred_ppg = self.model(data)
                    pred_ppg_norm = self.normalize_signal(pred_ppg)
                    labels_norm = self.normalize_signal(labels)
                    loss = self.criterion_Pearson(pred_ppg_norm, labels_norm)
                valid_losses.append(loss.item())
                vbar.set_postfix(loss=loss.item())
        avg_loss = np.mean(valid_losses)
        if self.prv_mode:
            avg_ppg_loss = np.mean(valid_ppg_losses)
            avg_prv_error = np.mean(valid_prv_errors)
            avg_pred_metrics = np.mean(np.array(all_predicted_metrics), axis=0)
            avg_gt_metrics = np.mean(np.array(all_gt_metrics), axis=0)
            return avg_loss, avg_ppg_loss, avg_prv_error, avg_pred_metrics, avg_gt_metrics
        else:
            return avg_loss, None, None, None, None

    def test(self, data_loader: dict) -> None:
        """
        Testing loop:
         - Load the best (or last) model and evaluate on the test dataset.
         - Save the predictions and ground truth, and compute final evaluation metrics.
         - If PR_MODE is enabled, log the average predicted PRV metrics, ground truth PRV metrics,
           their absolute difference, as well as the Pearson and PRV losses.
        """
        self.logger.info("Starting testing...")
        predictions = {}
        labels_dict = {}
        test_losses = []
        test_ppg_losses = []
        test_prv_errors = []
        all_predicted_metrics = []
        all_gt_metrics = []
        if self.config.TEST.USE_LAST_EPOCH:
            model_path = os.path.join(self.model_dir, f"{self.model_file_name}_Epoch{self.max_epoch_num - 1}.pth")
        else:
            model_path = os.path.join(self.model_dir, f"{self.model_file_name}_Epoch{self.best_epoch}.pth")
        if not os.path.exists(model_path):
            self.logger.error(f"Model file {model_path} not found!")
            raise FileNotFoundError(f"Model file {model_path} does not exist.")
        self.logger.info(f"Loading model from: {model_path}")
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        with torch.no_grad():
            for test_batch in tqdm(data_loader["test"], ncols=80):
                data = test_batch[0].float().to(self.device)
                labels_batch = test_batch[1].float().to(self.device)
                if self.prv_mode:
                    gt_prv = test_batch[4].float().to(self.device)
                    pred_ppg = self.model(data)[0]
                    pred_ppg_norm = self.normalize_signal(pred_ppg)
                    labels_norm = self.normalize_signal(labels_batch)
                    loss_ppg = self.criterion_Pearson(pred_ppg_norm, labels_norm)
                    prv_error = self.evaluate_prv_metrics(pred_ppg, gt_prv)
                    loss = loss_ppg + self.prv_loss_weight * prv_error
                    test_ppg_losses.append(loss_ppg.item())
                    test_prv_errors.append(prv_error)
                    pred_metrics, gt_metrics = self.aggregate_prv_metrics(pred_ppg, gt_prv)
                    all_predicted_metrics.append(pred_metrics)
                    all_gt_metrics.append(gt_metrics)
                else:
                    pred_ppg = self.model(data)
                    pred_ppg_norm = self.normalize_signal(pred_ppg)
                    labels_norm = self.normalize_signal(labels_batch)
                    loss = self.criterion_Pearson(pred_ppg_norm, labels_norm)
                test_losses.append(loss.item())
                # Save predictions and labels per sample (using subject index and sort index)
                for idx in range(data.shape[0]):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    predictions.setdefault(subj_index, {})[sort_index] = pred_ppg[idx].cpu()
                    labels_dict.setdefault(subj_index, {})[sort_index] = labels_batch[idx].cpu()
        avg_test_loss = np.mean(test_losses)
        self.logger.info(f"Test Completed - Average Combined Loss: {avg_test_loss:.4f}")
        if self.prv_mode:
            avg_test_ppg = np.mean(test_ppg_losses)
            avg_test_prv = np.mean(test_prv_errors)
            avg_test_pred_metrics = np.mean(np.array(all_predicted_metrics), axis=0)
            avg_test_gt_metrics = np.mean(np.array(all_gt_metrics), axis=0)
            diff_metrics = np.abs(avg_test_pred_metrics - avg_test_gt_metrics)
            self.logger.info(f"Test Pearson Loss: {avg_test_ppg:.4f}, Test PRV Error (MSE): {avg_test_prv:.4f}")
            self.logger.info(f"Test PRV Metrics: Predicted = {avg_test_pred_metrics}, Ground Truth = {avg_test_gt_metrics}, Abs Diff = {diff_metrics}")
        self.logger.info("Calculating test metrics...")
        calculate_metrics(predictions, labels_dict, self.config)

    def save_model(self, epoch: int) -> None:
        """
        Save the model state for the current epoch.
        """
        os.makedirs(self.model_dir, exist_ok=True)
        model_path = os.path.join(self.model_dir, f"{self.model_file_name}_Epoch{epoch}.pth")
        torch.save(self.model.state_dict(), model_path)
        self.logger.info(f"Saved Model Path: {model_path}")

    def save_best_model(self) -> None:
        """
        Save the best model (based on validation loss).
        """
        best_model_path = os.path.join(self.model_dir, f"{self.model_file_name}_Best.pth")
        torch.save(self.model.state_dict(), best_model_path)
        self.logger.info(f"Best Model Saved: {best_model_path}")

    def get_hr(self, y: np.ndarray, sr: int = 30, min_hr: int = 30, max_hr: int = 180) -> float:
        """
        Calculate heart rate (HR) from a ground truth BVP signal using Welch's method.
        The frequency range is limited to 30-180 bpm; if no valid component is found, returns 0.0.
        """
        nfft = int(1e5 / sr)
        nperseg = int(min(len(y) - 1, 256))
        freqs, psd = welch(y, sr, nfft=nfft, nperseg=nperseg)
        # Apply frequency mask for the HR range (in Hz)
        mask = (freqs > min_hr / 60) & (freqs < max_hr / 60)
        if not np.any(mask):
            self.logger.warning("No frequency component found in the specified HR range.")
            return 0.0
        valid_freqs = freqs[mask]
        valid_psd = psd[mask]
        hr = valid_freqs[np.argmax(valid_psd)] * 60
        return hr

'''
"""PhysMamba Trainer."""
import os
from collections import OrderedDict

import math
import logging
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import random
from evaluation.metrics import calculate_metrics
from neural_methods.loss.PhysNetNegPearsonLoss import Neg_Pearson
from neural_methods.model.PhysMamba import PhysMamba
from neural_methods.trainer.BaseTrainer import BaseTrainer
from torch.autograd import Variable
from tqdm import tqdm
from scipy.signal import welch


class PhysMambaTrainer(BaseTrainer):

    def __init__(self, config, data_loader):
        """Inits parameters from args and the writer for TensorboardX."""
        super().__init__()
        # self.criterion_PRV = nn.MSELoss() #### PRV MODE ####
        self.device = torch.device(config.DEVICE)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.base_len = self.num_of_gpu
        self.config = config
        self.min_valid_loss = None
        self.best_epoch = 0
        self.diff_flag = 0
        if config.TRAIN.DATA.PREPROCESS.LABEL_TYPE == "DiffNormalized":
            self.diff_flag = 1
        self.frame_rate = config.TRAIN.DATA.FS
        
        self.log_dir = config.LOG.EXPERIMENT_DIR
        os.makedirs(self.log_dir, exist_ok=True, mode=0o777)
        ## LOGGING ##
        log_file = os.path.join(self.log_dir, f"{self.model_file_name}.log")
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initialized PhysMambaTrainer.")

        self.model = PhysMamba().to(self.device)  # [3, T, 128,128]
        if self.num_of_gpu > 0:
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))

        if config.TOOLBOX_MODE == "train_and_test":
            self.num_train_batches = len(data_loader["train"])
            self.criterion_Pearson = Neg_Pearson()
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=config.TRAIN.LR, weight_decay = 0.0005)
            # See more details on the OneCycleLR scheduler here: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=config.TRAIN.LR, epochs=config.TRAIN.EPOCHS, steps_per_epoch=self.num_train_batches)
        elif config.TOOLBOX_MODE == "only_test":
            self.criterion_Pearson_test = Neg_Pearson()
            pass
        else:
            raise ValueError("PhysNet trainer initialized in incorrect toolbox mode!")

    def train(self, data_loader):
        """Training routine for model"""
        if data_loader["train"] is None:
            raise ValueError("No data for train")

        self.logger.info('Starting training...')
        
        for epoch in range(self.max_epoch_num):
            print('')
            # print(f"====Training Epoch: {epoch}====")
            self.logger.info(f"==== Training Epoch: {epoch} ====")
            self.model.train()
            # loss_PRV_avg = [] #### PRV MODE ####
            loss_rPPG_avg = []
            running_loss = 0.0
            
            # Model Training
            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)
                data, labels = batch[0].float(), batch[1].float()
                N, D, C, H, W = data.shape

                data = data.to(self.device)
                # labels = labels.to(self.device) #### PRV MODE ####

                self.optimizer.zero_grad()
                pred_ppg = self.model(data)
                # pred_ppg, pred_prv = self.model(data) #### PRV MODE ####

                pred_ppg = (pred_ppg-torch.mean(pred_ppg, axis=-1).view(-1, 1))/torch.std(pred_ppg, axis=-1).view(-1, 1)    # normalize
                
                labels = (labels - torch.mean(labels)) / torch.std(labels)
                loss = self.criterion_Pearson(pred_ppg, labels)
                
                #### PRV MODE ####
                # loss_PRV = self.criterion_PRV(pred_prv, torch.zeros_like(pred_prv))  # PRV의 목표값은 0으로 가정

                loss.backward()
                running_loss += loss.item()
                self.optimizer.step()
                self.scheduler.step()
                
                running_loss += loss.item()
                tbar.set_postfix(loss=loss.item())

                if idx % 100 == 99:
                    self.logger.info(f"[{epoch}, {idx + 1}] loss: {running_loss / 100:.3f}")
                    running_loss = 0.0
            ############## OOM ##############
                    torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            #################################
            self.save_model(epoch)
            
            if not self.config.TEST.USE_LAST_EPOCH:
                valid_loss = self.valid(data_loader)
                self.logger.info(f"Validation Loss: {valid_loss}")

                if self.min_valid_loss is None or valid_loss < self.min_valid_loss:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    self.logger.info(f"Updated Best Model - Epoch: {self.best_epoch}")
                    self.save_best_model()
            torch.cuda.empty_cache()
            
        self.logger.info(f"Best Trained Epoch: {self.best_epoch}, Min Validation Loss: {self.min_valid_loss}")
        
        
    def valid(self, data_loader):
        """ Runs the model on valid sets."""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")

        print('')
        # print(" ====Validing===")
        self.logger.info("Starting validation...")
        valid_loss = []
        self.model.eval()
        valid_step = 0
        
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for valid_idx, valid_batch in enumerate(vbar):
                valid_step += 1
                vbar.set_description("Validation")
                BVP_label = valid_batch[1].to(torch.float32).to(self.device)
                rPPG = self.model(valid_batch[0].to(torch.float32).to(self.device))
                
                rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # normalize
                BVP_label = (BVP_label - torch.mean(BVP_label)) / torch.std(BVP_label)  # normalize
                
                loss_ecg = self.criterion_Pearson(rPPG, BVP_label)
                valid_loss.append(loss_ecg.item())
                
                self.logger.info(f"Validation Batch {valid_step} - Loss: {loss_ecg.item():.6f}")
                vbar.set_postfix(loss=loss_ecg.item())
            valid_loss = np.asarray(valid_loss)
        
        avg_valid_loss = np.mean(valid_loss)
        self.logger.info(f"Validation Completed - Average Loss: {avg_valid_loss:.6f}")
        
        return np.mean(valid_loss)

    def test(self, data_loader):
        """Runs the model on test sets."""
        if data_loader["test"] is None:
            raise ValueError("No data for testing")

        self.logger.info("Starting testing...")
        predictions = dict()
        labels = dict()
        test_losses = []

        if self.config.TEST.USE_LAST_EPOCH:
            model_path = os.path.join(self.model_dir, self.model_file_name + '_Epoch' + str(self.max_epoch_num - 1) + '.pth')
        else:
            model_path = os.path.join(self.model_dir, self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')

        if not os.path.exists(model_path):
            self.logger.error(f"Model file {model_path} not found!")
            raise FileNotFoundError(f"Model file {model_path} does not exist.")

        self.logger.info(f"Loading model from: {model_path}")
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        with torch.no_grad():
            tbar = tqdm(data_loader["test"], ncols=80)
            for batch_idx, test_batch in enumerate(tbar):
                data, label = test_batch[0].to(self.device), test_batch[1].to(self.device)
                pred_ppg_test = self.model(data)

                loss = self.criterion_Pearson(pred_ppg_test, label)
                test_losses.append(loss.item())

                # 배치별 loss 로깅
                self.logger.info(f"Test Batch {batch_idx + 1} - Loss: {loss.item():.6f}")
                tbar.set_postfix(loss=loss.item())

                for idx in range(data.shape[0]):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    predictions.setdefault(subj_index, {})[sort_index] = pred_ppg_test[idx]
                    labels.setdefault(subj_index, {})[sort_index] = label[idx]

        avg_test_loss = np.mean(test_losses)
        self.logger.info(f"Test Completed - Average Loss: {avg_test_loss:.6f}")
        self.logger.info("Calculating test metrics...")
        calculate_metrics(predictions, labels, self.config)

    def save_model(self, epoch):
        """Saves the model state."""
        os.makedirs(self.model_dir, exist_ok=True, mode=0o777)
        model_path = os.path.join(self.model_dir, f"{self.model_file_name}_Epoch{epoch}.pth")
        torch.save(self.model.state_dict(), model_path)
        self.logger.info(f"Saved Model: {model_path}")

    def save_best_model(self):
        """Saves the best model based on validation loss."""
        best_model_path = os.path.join(self.model_dir, f"{self.model_file_name}_Best.pth")
        torch.save(self.model.state_dict(), best_model_path)
        self.logger.info(f"Best Model Saved: {best_model_path}")

    # HR calculation based on ground truth label
    def get_hr(self, y, sr=30, min=30, max=180):
        p, q = welch(y, sr, nfft=1e5/sr, nperseg=np.min((len(y)-1, 256)))
        return p[(p>min/60)&(p<max/60)][np.argmax(q[(p>min/60)&(p<max/60)])]*60
'''