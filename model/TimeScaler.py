import torch
import torch.nn as nn
from pytorch_wavelets import DWT1DForward, DWT1DInverse


class RevIN(nn.Module):
    """
    Reversible Instance Normalization to mitigate distribution shift.
    """
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        return x

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


class ODBlock(nn.Module):
    """
    [OD Block: Orthogonal Decoupling]
    Projects original sequences into mathematically disjoint sub-sequences.
    """
    def __init__(self, seq_len, pred_len, wave_level=3, wave_basis='db4', device='cuda'):
        super(ODBlock, self).__init__()
        self.device = device
        self.wave_level = wave_level
        self.wave_basis = wave_basis
        
        # Wavelet Decomposition & Reconstruction objects
        self.dwt = DWT1DForward(wave=wave_basis, J=wave_level).to(device)
        self.idwt = DWT1DInverse(wave=wave_basis).to(device)
        
        # Pre-calculate shapes to initialize the specific branches dynamically
        self.in_shapes = self._get_coeff_shapes(seq_len)
        self.out_shapes = self._get_coeff_shapes(pred_len)

    def _get_coeff_shapes(self, length):
        # Create a dummy tensor to inference output shapes from DWT
        dummy = torch.randn(1, 1, length).to(self.device)
        yl, yh = self.dwt(dummy)
        # Returns list: [Approx_len, Detail_1_len, Detail_2_len, ...]
        return [yl.shape[-1]] + [d.shape[-1] for d in yh]

    def forward(self, x):
        """Forward Transform (Decomposition)"""
        yl, yh = self.dwt(x)
        return yl, yh

    def inverse(self, yl, yh):
        """Inverse Transform (Reconstruction)"""
        return self.idwt((yl, yh))

class SPBlock(nn.Module):
    """
    [SP Block: Structured Prediction]
    The atomic unit of the Structured Prediction module. It performs prediction within a specific structural component (scale) by evolving the spectral representation of that component.
    """
    def __init__(self, in_len, out_len, hidden_dim=256, dropout=0.1):
        super(SPBlock, self).__init__()
        self.freq_in = in_len // 2 + 1
        self.freq_out = out_len // 2 + 1
        self.out_len = out_len
        
        # Input/Output dimensions for the MLP (Real + Imag parts)
        self.mlp_in_dim = self.freq_in * 2
        self.mlp_out_dim = self.freq_out * 2
        
        self.revin = RevIN(num_features=1, affine=True)
        
        self.mlp = nn.Sequential(
            nn.Linear(self.mlp_in_dim, hidden_dim),
            nn.GELU(), # Maybe try nn.ReLU() as well
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.mlp_out_dim)
        )

    def forward(self, x):
        
        B, C, L = x.shape
        
        # 1. Local RevIN (Handling scale differences) 
        x = x.transpose(1, 2) 
        x = self.revin(x, 'norm') 
        x = x.transpose(1, 2) 
        
        # 2. FFT: Time Domain -> Spectral Domain
        x_fft = torch.fft.rfft(x, dim=-1, norm='ortho')
        x_vals = torch.view_as_real(x_fft)
        x_flat = x_vals.reshape(B, C, -1)
        
        # 3. Prediction: Evolving dynamics in the spectral domain
        y_flat = self.mlp(x_flat)
        
        # 4. IFFT: Spectral Domain -> Time Domain
        y_vals = y_flat.reshape(B, C, self.freq_out, 2)
        y_fft = torch.view_as_complex(y_vals.contiguous())
        y = torch.fft.irfft(y_fft, n=self.out_len, dim=-1, norm='ortho')
        
        # 5. Branch Denormalization
        y = y.transpose(1, 2)
        y = self.revin(y, 'denorm') 
        y = y.transpose(1, 2)
        
        return y

class TimeScaler(nn.Module):
    def __init__(self, seq_len, pred_len, c_in, d_model=256, dropout=0.1, 
                 wave_level=3, wave_basis='db4', device=torch.device('cuda')):
        super(TimeScaler, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # 1. Global RevIN (Handling non-stationarity)
        self.global_revin = RevIN(c_in)
        
        # 2. OD Block (Orthogonal Decoupling)
        self.odb = ODBlock(seq_len, pred_len, wave_level, wave_basis, device)
        
        # 3. SP Blocks (Structured Prediction)
        self.sp_blocks = nn.ModuleList()
        
        # Approximation Predictor (Trend)
        self.sp_blocks.append(
            SPBlock(self.odb.in_shapes[0], self.odb.out_shapes[0], hidden_dim=d_model, dropout=dropout)
        )
        
        # Detail Predictors (Seasonality/Noise)
        for i in range(len(self.odb.in_shapes) - 1):
            self.sp_blocks.append(
                SPBlock(self.odb.in_shapes[i+1], self.odb.out_shapes[i+1], hidden_dim=d_model, dropout=dropout)
            )
            
    def forward(self, x, return_decomposition=False):

        B, L, C = x.shape
        
        # Global Normalization
        x = self.global_revin(x, 'norm')
        
        # Reshape for Wavelet Processing: [Batch * Channels, 1, Length]
        x = x.permute(0, 2, 1).reshape(B * C, 1, L)
        
        # 1. Decoupling (OD Block)
        yl_in, yh_in = self.odb(x)
        
        # 2. Structured Prediction (SP Block)
        # Predict Approximation
        yl_out = self.sp_blocks[0](yl_in)
        
        # Predict Details
        yh_out = []
        for i, detail_coeff in enumerate(yh_in):
            pred_detail = self.sp_blocks[i+1](detail_coeff)
            yh_out.append(pred_detail)
            
        # 3. Reconstruction
        y_pred = self.odb.inverse(yl_out, yh_out)
        
        # handling DWT padding
        y_pred = y_pred[..., -self.pred_len:] 
        
        # Reshape back to [Batch, Length, Channels]
        y_pred = y_pred.reshape(B, C, self.pred_len).permute(0, 2, 1)
        
        # Global Denormalization
        y_pred = self.global_revin(y_pred, 'denorm')
        
        if return_decomposition:
            # Return prediction plus components for Structured Loss supervision
            return y_pred, yl_out, yh_out
        
        return y_pred

class StructuredLoss(nn.Module):
    """
    [Structured Loss]
    Acts as a structural regularizer by supervising both the reconstructed signal and its decoupled components.
    """
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0):
        super(StructuredLoss, self).__init__()
        self.criterion = nn.MSELoss() # Maybe try L1Loss as well?
        self.alpha = alpha # Weight for Reconstruction loss
        self.beta = beta   # Weight for Trend (Approx) loss
        self.gamma = gamma # Weight for Detail loss

    def forward(self, pred, true, pred_yl, pred_yh_list, odb_module, revin_module):
        """
        Calculates the weighted sum of reconstruction loss and component losses.
        Args:
            pred: Reconstructed prediction [B, L, C]
            true: Ground Truth [B, L, C]
            pred_yl: Predicted approximation coeffs
            pred_yh_list: Predicted detail coeffs
            odb_module: The ODBlock used in the model
            revin_module: The Global RevIN used in the model
        """
        B, L, C = true.shape
        
        # 1. Normalize Ground Truth to align with model's latent space
        true_norm = revin_module(true, 'norm')  
        true_norm = true_norm.permute(0, 2, 1).reshape(B * C, 1, L)
        
        # 2. Decompose Ground Truth using OD Block (No Grad)
        with torch.no_grad():
            true_yl, true_yh_list = odb_module(true_norm)
        
        # 3. Component Supervision
        # Trend (Approximation) Loss
        loss_trend = self.criterion(pred_yl, true_yl)
        
        # Detail Loss (Sum over all detail levels)
        loss_detail = 0
        for p_h, t_h in zip(pred_yh_list, true_yh_list):
            loss_detail += self.criterion(p_h, t_h)
            
        # 4. Reconstruction Loss
        loss_recon = self.criterion(pred, true)
        
        # 5. Weighted Sum
        total_loss = (self.alpha * loss_recon) + (self.beta * loss_trend) + (self.gamma * loss_detail)
        return total_loss
