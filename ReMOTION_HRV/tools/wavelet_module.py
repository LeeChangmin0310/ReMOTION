import pywt
import torch
import torch.nn as nn

class LearnableWaveletTransform(nn.Module):
    def __init__(self, levels=4):
        super().__init__()
        self.levels = levels
        self.wavelet_weights = nn.Parameter(torch.randn(levels, 1))  # Trainable Parameters
        self.wavelet_bias = nn.Parameter(torch.randn(levels, 1))

    def forward(self, x):
        # x: [batch_size, signal_length]
        coeffs = []
        for i in range(x.size(0)):
            signal = x[i].cpu().numpy()
            coeff = pywt.wavedec(signal, 'db4', level=self.levels)
            coeff = [torch.tensor(c, dtype=torch.float32).to(x.device) for c in coeff]
            coeffs.append(coeff)
        
        # Trainable weights
        weighted_coeffs = []
        for i in range(len(coeffs)):
            level_coeffs = []
            for j in range(len(coeffs[i])):
                modified = coeffs[i][j] * self.wavelet_weights[j] + self.wavelet_bias[j]
                level_coeffs.append(modified)
            weighted_coeffs.append(level_coeffs)
        
        return weighted_coeffs

'''class LearnableWaveletTransform(nn.Module):
    def __init__(self, wavelet_name='db4', levels=4, learnable_params=True):
        super(LearnableWaveletTransform, self).__init__()
        self.wavelet_name = wavelet_name
        self.levels = levels
        self.learnable_params = learnable_params

        if self.learnable_params:
            # 학습 가능한 파라미터 추가
            self.wavelet_weights = nn.Parameter(torch.randn(levels, 1))
            self.wavelet_bias = nn.Parameter(torch.randn(levels, 1))

    def forward(self, x):
        # x: [batch_size, length]
        batch_size, length = x.shape
        coeffs = []

        for i in range(batch_size):
            signal = x[i].cpu().numpy()
            coeff = pywt.wavedec(signal, self.wavelet_name, level=self.levels)
            coeff = [torch.tensor(c, dtype=torch.float32).to(x.device) for c in coeff]
            coeffs.append(coeff)

        if self.learnable_params:
            # 학습 가능한 파라미터 적용
            for i in range(len(coeffs)):
                for j in range(len(coeffs[i])):
                    coeffs[i][j] = coeffs[i][j] * self.wavelet_weights[j] + self.wavelet_bias[j]

        return coeffs'''