import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

from s2m2.core.utils.model_utils import load_model, run_stereo_matching


class TorchBackend:
    """PyTorch-eager s2m2 inference backend."""

    def __init__(
        self,
        weights_dir: str,
        model_type: str = "L",
        refine_iter: int = 3,
        use_positivity: bool = True,
        device: str = "cuda:0",
    ):
        self.device = torch.device(device)
        self.model = load_model(
            pretrain_path=weights_dir,
            model_type=model_type,
            use_positivity=use_positivity,
            refine_iter=refine_iter,
            device=self.device,
        )
        if self.model is None:
            raise RuntimeError(
                f"s2m2 load_model returned None (weights_dir={weights_dir}, "
                f"model_type={model_type}). Check that CH*NTR*.pth exists."
            )

    def warmup(self, height: int, width: int):
        h = (height // 32) * 32
        w = (width // 32) * 32
        dummy = torch.zeros(1, 3, h, w, device=self.device, dtype=torch.float32)
        run_stereo_matching(self.model, dummy, dummy, self.device, N_repeat=1)

    def infer(self, left_t: torch.Tensor, right_t: torch.Tensor):
        pred_disp, _pred_occ, pred_conf, _avg_conf, _ms = run_stereo_matching(
            self.model, left_t, right_t, self.device, N_repeat=1
        )
        return pred_disp, pred_conf
