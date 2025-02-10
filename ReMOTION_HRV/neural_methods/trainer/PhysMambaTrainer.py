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
    '''
    def test(self, data_loader):
        """ Runs the model on test sets."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")
        
        print('')
        self.logger.info("Starting testing...")        predictions = dict()
        predictions = dict()
        labels = dict()
        test_losses = []
    '''
    '''
        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
            print("Testing uses pretrained model!")
            print(self.config.INFERENCE.MODEL_PATH)
        else:
            if self.config.TEST.USE_LAST_EPOCH:
                last_epoch_model_path = os.path.join(
                self.model_dir, self.model_file_name + '_Epoch' + str(self.max_epoch_num - 1) + '.pth')
                print("Testing uses last epoch as non-pretrained model!")
                print(last_epoch_model_path)
                self.model.load_state_dict(torch.load(last_epoch_model_path))
            else:
                best_model_path = os.path.join(
                    self.model_dir, self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')
                print("Testing uses best epoch selected using model selection as non-pretrained model!")
                print(best_model_path)
                self.model.load_state_dict(torch.load(best_model_path))
    '''
    '''   
        if self.config.TEST.USE_LAST_EPOCH:
            model_path = os.path.join(self.model_dir, self.model_file_name + '_Epoch' + str(self.max_epoch_num - 1) + '.pth')
        else:
            model_path = os.path.join(self.model_dir, self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')

        if not os.path.exists(model_path):
            self.logger.error(f"Model file {model_path} not found!")
            raise FileNotFoundError(f"Model file {model_path} does not exist.")

        self.logger.info(f"Loading model from: {model_path}")
        # self.model.load_state_dict(torch.load(model_path))
        # self.model.eval()
        self.model = self.model.to(self.config.DEVICE)
        self.model.eval()
        
        print("Running model evaluation on the testing dataset!")
        with torch.no_grad():
            tbar = tqdm(data_loader["test"], ncols=80)
            for _, test_batch in enumerate(tbar):
                batch_size = test_batch[0].shape[0]
                data, label = test_batch[0].to(
                    self.config.DEVICE), test_batch[1].to(self.config.DEVICE)
                pred_ppg_test = self.model(data)

                if self.config.TEST.OUTPUT_SAVE_DIR:
                    label = label.cpu()
                    pred_ppg_test = pred_ppg_test.cpu()

                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    predictions[subj_index][sort_index] = pred_ppg_test[idx]
                    labels[subj_index][sort_index] = label[idx]

        print('')
        calculate_metrics(predictions, labels, self.config)
        if self.config.TEST.OUTPUT_SAVE_DIR: # saving test outputs 
            self.save_test_outputs(predictions, labels, self.config)
    '''
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
    ''' 
    def save_best_model(self):
        """Saves the best model based on validation loss."""
        best_model_path = os.path.join(self.model_dir, self.model_file_name + '_Best.pth')
        torch.save(self.model.state_dict(), best_model_path)
        print(f'Best Model Saved: {best_model_path}')

    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
        torch.save(self.model.state_dict(), model_path)
        print('Saved Model Path: ', model_path)
    '''

    # HR calculation based on ground truth label
    def get_hr(self, y, sr=30, min=30, max=180):
        p, q = welch(y, sr, nfft=1e5/sr, nperseg=np.min((len(y)-1, 256)))
        return p[(p>min/60)&(p<max/60)][np.argmax(q[(p>min/60)&(p<max/60)])]*60
