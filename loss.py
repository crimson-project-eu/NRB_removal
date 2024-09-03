import torch
import numpy as np
from scipy import signal

import torch.nn.functional as F

from typing import Union


class WeightedHuberLoss(torch.nn.Module):

    def __init__(self,
                 wtp: float = 20,
                 wfp: float = 13,
                 wfn: float = 18,
                 wtn: float = 5,
                 ts: Union[str, float] = 0.005,
                 **kwargs):

        super(WeightedHuberLoss, self).__init__()
        self.Wtp = wtp
        self.Wfp = wfp
        self.Wtn = wtn
        self.Wfn = wfn
        if isinstance(ts, str):
            assert ts == 'scipy', "Ts must be either scipy or float"
            self.kwargs = kwargs
        self.Ts = ts

    def forward(self, x, y):

        y_pred = x.reshape(-1)
        y_true = y.reshape(-1)
        # print(self.Ts)

        if isinstance(self.Ts, float):
            below_ts_true_mask = y_true <= self.Ts
            below_ts_pred_mask = y_pred <= self.Ts
        else:

            y_pred_np = y_pred.detach().cpu().numpy()
            y_true_np = y_true.detach().cpu().numpy()

            pred_peaks = signal.find_peaks(y_pred_np, **self.kwargs)
            below_ts_pred_mask_np = np.zeros_like(y_pred_np, dtype=bool)
            below_ts_pred_mask_np[pred_peaks[0]] = False
            below_ts_pred_mask = torch.tensor(below_ts_pred_mask_np, dtype=torch.float64, device='cuda')

            true_peaks = signal.find_peaks(y_true_np, **self.kwargs)
            below_ts_true_mask_np = np.zeros_like(y_true_np, dtype=bool)
            below_ts_true_mask_np[true_peaks[0]] = False
            below_ts_true_mask = torch.tensor(below_ts_true_mask_np, dtype=torch.float64, device='cuda')

        t1 = torch.where(below_ts_true_mask & below_ts_pred_mask, self.Wtn, torch.tensor(0.0))
        t2 = torch.where(below_ts_true_mask & ~below_ts_pred_mask, self.Wfp, torch.tensor(0.0))
        t3 = torch.where(~below_ts_true_mask & ~below_ts_pred_mask, self.Wtp, torch.tensor(0.0))
        t4 = torch.where(~below_ts_true_mask & below_ts_pred_mask, self.Wfn, torch.tensor(0.0))

        loss = torch.nn.HuberLoss(reduction='none', delta=1)(y_pred, y_true)

        weighted_losses = t1 * loss + t2 * loss + t3 * loss + t4 * loss

        # weighted_losses = loss
        sample_losses = torch.mean(weighted_losses)

        return torch.sqrt(sample_losses)
