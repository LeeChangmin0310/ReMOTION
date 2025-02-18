"""PhysMamba Trainer."""
import os
import logging
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from scipy.signal import welch

from evaluation.metrics import calculate_metrics
from neural_methods.loss.PhysNetNegPearsonLoss import Neg_Pearson
from neural_methods.model.PhysMamba import PhysMamba
from neural_methods.trainer.BaseTrainer import BaseTrainer
from dataset.data_loader.UBFCrPPGLoader import calculate_prv_metrics


class PhysMambaTrainer(BaseTrainer):
    def __init__(self, config, data_loader) -> None:
        """
        Initialize the trainer by setting up the model, loss function, optimizer, and scheduler.
        Including PRV related settings according to config.py
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
        self.prv_mode = config.TRAIN.DATA.PR_MODE

        # Setup logging
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

        # Initialize model and multi-GPU setting
        self.model = PhysMamba().to(self.device)
        if self.num_of_gpu > 0:
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(self.num_of_gpu)))

        # Mode-specific settings
        if config.TOOLBOX_MODE == "train_and_test":
            self.num_train_batches = len(data_loader["train"])
            self.criterion_Pearson = Neg_Pearson()
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=config.TRAIN.LR, weight_decay=0.0005
            )
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=config.TRAIN.LR,
                epochs=config.TRAIN.EPOCHS, steps_per_epoch=self.num_train_batches
            )
            if self.prv_mode:
                self.prv_loss_weight = getattr(config.TRAIN, "PR_LOSS_WEIGHT", 0.5)
        elif config.TOOLBOX_MODE == "only_test":
            self.criterion_Pearson_test = Neg_Pearson()
        else:
            raise ValueError("PhysMamba trainer initialized in incorrect toolbox mode!")

        self.min_valid_loss = None
        self.best_epoch = 0

    def normalize_signal(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Normalize the signal using mean and standard deviation.
        """
        mean = torch.mean(signal, dim=-1, keepdim=True)
        std = torch.std(signal, dim=-1, keepdim=True) + 1e-8  # prevent division by zero
        return (signal - mean) / std

    def _process_batch(self, batch, training: bool = True):
        """
        Process a single batch to compute predictions and loss.
        Args:
            batch: Tuple or list containing batch data.
            training: Boolean flag indicating training mode (affects forward pass).
        Returns:
            A tuple (loss, pred_ppg, labels, additional_metrics) where additional_metrics is a dict
            containing PRV metrics if enabled.
        """
        # Unpack common items
        data = batch[0].float().to(self.device)
        labels = batch[1].float().to(self.device)
        gt_prv = batch[4].float().to(self.device) if self.prv_mode else None

        # Forward pass (if PRV mode, assume model returns tuple and we need first element)
        pred_ppg = self.model(data)
        if self.prv_mode:
            pred_ppg = pred_ppg[0]

        # Ensure proper tensor dimensions
        if pred_ppg.ndim == 1:
            pred_ppg = pred_ppg.unsqueeze(0)

        # Normalize for loss calculation
        pred_norm = self.normalize_signal(pred_ppg)
        labels_norm = self.normalize_signal(labels)
        loss_ppg = self.criterion_Pearson(pred_norm, labels_norm)
        loss = loss_ppg

        additional = {}
        if self.prv_mode:
            prv_error = self.evaluate_prv_metrics(pred_ppg, gt_prv)
            loss += self.prv_loss_weight * torch.tensor(prv_error, device=self.device)
            additional = {
                "prv_error": prv_error,
                "loss_ppg": loss_ppg.item()
            }
        return loss, pred_ppg, labels, additional

    def evaluate_prv_metrics(self, rPPG_pred: torch.Tensor, gt_prv: torch.Tensor) -> float:
        """
        Compute PRV metrics MSE between predicted and ground truth values.
        """
        rPPG_np = rPPG_pred.cpu().detach().numpy()
        metrics_pred = np.array([calculate_prv_metrics(rPPG_np[i]) for i in range(rPPG_np.shape[0])])
        gt_np = gt_prv.cpu().detach().numpy()
        mse = np.mean((metrics_pred - gt_np) ** 2)
        return mse

    def aggregate_prv_metrics(self, rPPG_pred: torch.Tensor, gt_prv: torch.Tensor):
        """
        Compute and aggregate average predicted and ground truth PRV metrics for a batch.
        """
        rPPG_np = rPPG_pred.cpu().detach().numpy()
        pred_metrics = np.array([calculate_prv_metrics(rPPG_np[i]) for i in range(rPPG_np.shape[0])])
        gt_metrics = gt_prv.cpu().detach().numpy()
        return np.mean(pred_metrics, axis=0), np.mean(gt_metrics, axis=0)

    def train(self, data_loader: dict) -> None:
        """
        Main training loop.
        """
        if "train" not in data_loader or data_loader["train"] is None:
            raise ValueError("No training data provided.")

        self.logger.info("Starting training...")
        for epoch in range(self.max_epoch_num):
            self.logger.info(f"==== Training Epoch: {epoch} ====")
            self.model.train()
            running_loss = 0.0
            running_ppg_loss = 0.0
            running_prv_error = 0.0

            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                self.optimizer.zero_grad()
                loss, _, _, additional = self._process_batch(batch, training=True)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                running_loss += loss.item()
                if self.prv_mode:
                    running_ppg_loss += additional.get("loss_ppg", 0.0)
                    running_prv_error += additional.get("prv_error", 0.0)
                tbar.set_postfix(loss=loss.item())

                if (idx + 1) % 100 == 0:
                    if self.prv_mode:
                        avg_ppg_loss = running_ppg_loss / 100
                        avg_prv_error = running_prv_error / 100
                        self.logger.info(
                            f"[Epoch {epoch}, Batch {idx+1}] Avg Pearson Loss = {avg_ppg_loss:.4f}, Avg PRV MSE = {avg_prv_error:.4f}"
                        )
                        running_ppg_loss = running_prv_error = 0.0
                    self.logger.info(f"[Epoch {epoch}, Batch {idx+1}] Combined Loss = {running_loss/100:.4f}")
                    running_loss = 0.0
                    torch.cuda.empty_cache()

            torch.cuda.empty_cache()
            self.save_model(epoch)
            self._validate_and_update(data_loader, epoch)
        self.logger.info(f"Best Trained Epoch: {self.best_epoch}, Min Validation Loss: {self.min_valid_loss}")

    def _validate_and_update(self, data_loader: dict, epoch: int) -> None:
        """
        Run the validation loop and update the best model if necessary.
        """
        if self.config.TEST.USE_LAST_EPOCH:
            return  # Skip validation if using the last epoch for testing

        valid_results = self.valid(data_loader)
        valid_loss = valid_results[0]
        self.logger.info(f"Validation Combined Loss = {valid_loss:.4f}")
        if self.prv_mode:
            valid_ppg_loss, valid_prv_error, avg_pred_metrics, avg_gt_metrics = valid_results[1:]
            diff_metrics = np.abs(avg_pred_metrics - avg_gt_metrics)
            self.logger.info(
                f"Validation Pearson Loss = {valid_ppg_loss:.4f}, Validation PRV MSE = {valid_prv_error:.4f}"
            )
            self.logger.info(
                f"Validation PRV Metrics: Predicted = {avg_pred_metrics}, Ground Truth = {avg_gt_metrics}, Abs Diff = {diff_metrics}"
            )
        if self.min_valid_loss is None or valid_loss < self.min_valid_loss:
            self.min_valid_loss = valid_loss
            self.best_epoch = epoch
            self.logger.info(f"Updated Best Model - Epoch: {self.best_epoch}")
            self.save_best_model()


    def valid(self, data_loader: dict):
        """
        Validation loop.
        Returns:
            Tuple: (avg_loss, avg_ppg_loss, avg_prv_error, avg_pred_metrics, avg_gt_metrics)
            or
            PR_MODE=False (avg_loss, None, None, None, None)
        """
        self.logger.info("Starting validation...")
        losses = []
        ppg_losses = []
        prv_errors = []
        pred_metrics_list = []
        gt_metrics_list = []

        self.model.eval()
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for batch in vbar:
                loss, pred_ppg, labels, additional = self._process_batch(batch, training=False)
                losses.append(loss.item())
                if self.prv_mode:
                    ppg_losses.append(additional.get("loss_ppg", 0.0))
                    prv_errors.append(additional.get("prv_error", 0.0))
                    gt_prv = batch[4].float().to(self.device)
                    pred_m, gt_m = self.aggregate_prv_metrics(pred_ppg, gt_prv)
                    pred_metrics_list.append(pred_m)
                    gt_metrics_list.append(gt_m)
                vbar.set_postfix(loss=loss.item())

        avg_loss = np.mean(losses)
        if self.prv_mode:
            avg_ppg_loss = np.mean(ppg_losses)
            avg_prv_error = np.mean(prv_errors)
            avg_pred_metrics = np.mean(np.array(pred_metrics_list), axis=0)
            avg_gt_metrics = np.mean(np.array(gt_metrics_list), axis=0)
            return avg_loss, avg_ppg_loss, avg_prv_error, avg_pred_metrics, avg_gt_metrics
        else:
            return avg_loss, None, None, None, None

    def test(self, data_loader: dict) -> None:
        """
        Testing loop:
         - Load the best (or last) model.
         - Evaluate on test data and save predictions, ground truth, and final metrics.
        """
        self.logger.info("Starting testing...")
        predictions = {}
        labels_dict = {}
        test_losses = []
        ppg_losses = []
        prv_errors = []
        pred_metrics_list = []
        gt_metrics_list = []

        # Select model file based on configuration
        model_path = os.path.join(
            self.model_dir,
            f"{self.model_file_name}_Epoch{self.max_epoch_num - 1}.pth" if self.config.TEST.USE_LAST_EPOCH
            else f"{self.model_file_name}_Epoch{self.best_epoch}.pth"
        )
        if not os.path.exists(model_path):
            self.logger.error(f"Model file {model_path} not found!")
            raise FileNotFoundError(f"Model file {model_path} does not exist.")
        self.logger.info(f"Loading model from: {model_path}")
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        with torch.no_grad():
            for batch in tqdm(data_loader["test"], ncols=80):
                loss, pred_ppg, labels, additional = self._process_batch(batch, training=False)
                test_losses.append(loss.item())
                if self.prv_mode:
                    ppg_losses.append(additional.get("loss_ppg", 0.0))
                    prv_errors.append(additional.get("prv_error", 0.0))
                    gt_prv = batch[4].float().to(self.device)
                    pred_m, gt_m = self.aggregate_prv_metrics(pred_ppg, gt_prv)
                    pred_metrics_list.append(pred_m)
                    gt_metrics_list.append(gt_m)
                # Save predictions and labels per sample using provided indices
                for idx in range(batch[0].shape[0]):
                    subj_index = batch[2][idx]
                    sort_index = int(batch[3][idx])
                    predictions.setdefault(subj_index, {})[sort_index] = pred_ppg[idx].cpu()
                    labels_dict.setdefault(subj_index, {})[sort_index] = labels[idx].cpu()

        avg_test_loss = np.mean(test_losses)
        self.logger.info(f"Test Completed - Average Combined Loss: {avg_test_loss:.4f}")
        if self.prv_mode:
            avg_ppg_loss = np.mean(ppg_losses)
            avg_prv_error = np.mean(prv_errors)
            avg_pred_metrics = np.mean(np.array(pred_metrics_list), axis=0)
            avg_gt_metrics = np.mean(np.array(gt_metrics_list), axis=0)
            diff_metrics = np.abs(avg_pred_metrics - avg_gt_metrics)
            self.logger.info(f"Test Pearson Loss: {avg_ppg_loss:.4f}, Test PRV Error (MSE): {avg_prv_error:.4f}")
            self.logger.info(
                f"Test PRV Metrics: Predicted = {avg_pred_metrics}, Ground Truth = {avg_gt_metrics}, Abs Diff = {diff_metrics}"
            )
        self.logger.info("Calculating test metrics...")
        calculate_metrics(predictions, labels_dict, self.config)

    def save_model(self, epoch: int) -> None:
        """
        Save the model state for the given epoch.
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
        Calculate heart rate (HR) from a BVP signal using Welch's method.
        """
        nfft = int(1e5 / sr)
        nperseg = int(min(len(y) - 1, 256))
        freqs, psd = welch(y, sr, nfft=nfft, nperseg=nperseg)
        # Limit frequency range to valid HR (Hz)
        mask = (freqs > min_hr / 60) & (freqs < max_hr / 60)
        if not np.any(mask):
            self.logger.warning("No frequency component found in the specified HR range.")
            return 0.0
        valid_freqs = freqs[mask]
        valid_psd = psd[mask]
        hr = valid_freqs[np.argmax(valid_psd)] * 60
        return hr
