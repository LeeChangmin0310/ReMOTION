import pywt
import torch
import torch.nn as nn

class LearnableWaveletTransform(nn.Module):
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

        return coeffs