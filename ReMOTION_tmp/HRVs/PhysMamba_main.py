""" The main function of rPPG deep learning pipeline with wavelet-based PRV calculation."""

import argparse
import random
import time

import numpy as np
import torch
from torch import nn
from config import get_config
from dataset import data_loader
from neural_methods import trainer
from unsupervised_methods.unsupervised_predictor import unsupervised_predict
from torch.utils.data import DataLoader
import os

RANDOM_SEED = 100
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Create a general generator for use with the validation dataloader,
# the test dataloader, and the unsupervised dataloader
general_generator = torch.Generator()
general_generator.manual_seed(RANDOM_SEED)
# Create a training generator to isolate the train dataloader from
# other dataloaders and better control non-deterministic behavior
train_generator = torch.Generator()
train_generator.manual_seed(RANDOM_SEED)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def add_args(parser):
    """Adds arguments for parser."""
    parser.add_argument('--config_file', required=False,
                        default="configs/train_configs/PURE_PURE_UBFC-rPPG_TSCAN_BASIC.yaml", type=str, help="The name of the model.")
    return parser


class WaveletPRVModule(nn.Module):
    """Wavelet-based PRV calculation module."""
    def __init__(self, wavelet_name='db4', levels=4, learnable_params=True):
        super(WaveletPRVModule, self).__init__()
        self.wavelet_name = wavelet_name
        self.levels = levels
        self.learnable_params = learnable_params

        if self.learnable_params:
            # 학습 가능한 파라미터 추가
            self.wavelet_weights = nn.Parameter(torch.randn(levels, 1))
            self.wavelet_bias = nn.Parameter(torch.randn(levels, 1))

    def forward(self, rPPG):
        # rPPG: [batch_size, length]
        batch_size, length = rPPG.shape
        coeffs = []

        for i in range(batch_size):
            signal = rPPG[i].cpu().numpy()
            coeff = pywt.wavedec(signal, self.wavelet_name, level=self.levels)
            coeff = [torch.tensor(c, dtype=torch.float32).to(rPPG.device) for c in coeff]
            coeffs.append(coeff)

        if self.learnable_params:
            # 학습 가능한 파라미터 적용
            for i in range(len(coeffs)):
                for j in range(len(coeffs[i])):
                    coeffs[i][j] = coeffs[i][j] * self.wavelet_weights[j] + self.wavelet_bias[j]

        # PRV 계산 (예: 최대값과 최소값의 차이)
        prv = [torch.max(coeff[-1]) - torch.min(coeff[-1]) for coeff in coeffs]
        prv = torch.stack(prv).to(rPPG.device)

        return prv


class PhysMambaWithPRV(nn.Module):
    """PhysMamba model with wavelet-based PRV calculation."""
    def __init__(self, physmamba_model, wavelet_module):
        super(PhysMambaWithPRV, self).__init__()
        self.physmamba = physmamba_model  # 기존 PhysMamba 모델
        self.wavelet_module = wavelet_module  # 새로운 웨이블릿 모듈

    def forward(self, x):
        # 기존 PhysMamba 모델을 통해 PPG 신호 복원
        rPPG = self.physmamba(x)
        
        # 웨이블릿 모듈을 통해 PRV 계산
        prv = self.wavelet_module(rPPG)
        
        return rPPG, prv


def train_and_test_with_prv(config, data_loader_dict):
    """Trains and tests the model with PRV calculation."""
    # 기존 PhysMamba 모델 로드
    physmamba_model = trainer.PhysMambaTrainer.PhysMambaTrainer(config, data_loader_dict).model
    physmamba_model.load_state_dict(torch.load(config.TEST.MODEL_PATH))  # Pretrained weights 로드

    # 새로운 웨이블릿 모듈 생성
    wavelet_module = WaveletPRVModule()

    # 전체 모델 조합
    model = PhysMambaWithPRV(physmamba_model, wavelet_module).to(config.DEVICE)

    # 기존 PhysMamba 모델의 가중치 고정
    for param in physmamba_model.parameters():
        param.requires_grad = False

    # Optimizer 및 학습 설정
    optimizer = torch.optim.Adam(wavelet_module.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(config.TRAIN.EPOCHS):
        model.train()
        for batch in data_loader_dict['train']:
            data, labels = batch[0].to(config.DEVICE), batch[1].to(config.DEVICE)
            optimizer.zero_grad()
            rPPG, prv = model(data)
            loss = criterion(prv, torch.zeros_like(prv))  # PRV 손실 계산
            loss.backward()
            optimizer.step()

        # Validation 및 테스트
        if not config.TEST.USE_LAST_EPOCH:
            valid_loss = validate_with_prv(model, data_loader_dict['valid'], criterion)
            print(f'Epoch {epoch}, Validation Loss: {valid_loss}')

    # 테스트
    test_with_prv(model, data_loader_dict['test'])


def validate_with_prv(model, valid_loader, criterion):
    """Validates the model with PRV calculation."""
    model.eval()
    valid_loss = []
    with torch.no_grad():
        for batch in valid_loader:
            data, labels = batch[0].to(config.DEVICE), batch[1].to(config.DEVICE)
            rPPG, prv = model(data)
            loss = criterion(prv, torch.zeros_like(prv))
            valid_loss.append(loss.item())
    return np.mean(valid_loss)


def test_with_prv(model, test_loader):
    """Tests the model with PRV calculation."""
    model.eval()
    predictions = dict()
    labels = dict()
    with torch.no_grad():
        for batch in test_loader:
            data, label = batch[0].to(config.DEVICE), batch[1].to(config.DEVICE)
            rPPG, prv = model(data)
            # 결과 저장 및 평가
            # ...


if __name__ == "__main__":
    # Parse arguments.
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    parser = trainer.BaseTrainer.BaseTrainer.add_trainer_args(parser)
    parser = data_loader.BaseLoader.BaseLoader.add_data_loader_args(parser)
    args = parser.parse_args()

    # Configurations.
    config = get_config(args)
    print('Configuration:')
    print(config, end='\n\n')

    # Data loaders.
    data_loader_dict = data_loader.load_data_loaders(config)

    # Train and test with PRV calculation.
    if config.TOOLBOX_MODE == "train_and_test":
        train_and_test_with_prv(config, data_loader_dict)
    else:
        raise ValueError("TOOLBOX_MODE must be 'train_and_test' for PRV calculation.")