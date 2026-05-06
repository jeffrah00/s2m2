import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

from s2m2.core.utils.image_utils import image_pad, image_crop


class TensorRTBackend:
    """TensorRT engine backend for s2m2.

    The engine must be produced ahead of time via demo/export_tensorrt.py and
    is fixed-shape: incoming images must already match the engine's HxW (after
    the same pad-to-32 that run_stereo_matching applies internally).
    """

    def __init__(self, engine_path: str, device: str = "cuda:0"):
        if not os.path.isfile(engine_path):
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")

        import tensorrt as trt
        import pycuda.autoinit  # noqa: F401  (initializes a CUDA context)
        import pycuda.driver as cuda

        self._trt = trt
        self._cuda = cuda
        self.device = torch.device(device)

        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.input_names = []
        self.output_names = []
        self.tensor_shapes = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            self.tensor_shapes[name] = shape
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        if len(self.input_names) < 2:
            raise RuntimeError(
                f"Expected >=2 inputs (left, right), got {self.input_names}"
            )
        if len(self.output_names) < 1:
            raise RuntimeError(
                f"Expected >=1 output (disparity), got {self.output_names}"
            )

        self.engine_h, self.engine_w = self.tensor_shapes[self.input_names[0]][-2:]

        self._buffers = {}
        for name, shape in self.tensor_shapes.items():
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            host = np.empty(shape, dtype=dtype)
            dev = cuda.mem_alloc(host.nbytes)
            self._buffers[name] = (host, dev)
            self.context.set_tensor_address(name, int(dev))

        self.stream = cuda.Stream()

    def warmup(self, height: int, width: int):
        pass

    def _check_shape(self, left_t: torch.Tensor):
        h, w = left_t.shape[-2:]
        # mirror what image_pad inside run_stereo_matching would produce
        import math

        h_padded = math.ceil(h / 32) * 32
        w_padded = math.ceil(w / 32) * 32
        if (h_padded, w_padded) != (self.engine_h, self.engine_w):
            raise RuntimeError(
                "Incoming image shape (after pad-to-32) "
                f"{h_padded}x{w_padded} does not match TRT engine shape "
                f"{self.engine_h}x{self.engine_w}. Re-export the engine."
            )

    def infer(self, left_t: torch.Tensor, right_t: torch.Tensor):
        self._check_shape(left_t)
        h, w = left_t.shape[-2:]

        left_pad = image_pad(left_t, 32).contiguous()
        right_pad = image_pad(right_t, 32).contiguous()

        host_l, dev_l = self._buffers[self.input_names[0]]
        host_r, dev_r = self._buffers[self.input_names[1]]

        np.copyto(host_l, left_pad.detach().cpu().numpy().astype(host_l.dtype))
        np.copyto(host_r, right_pad.detach().cpu().numpy().astype(host_r.dtype))
        self._cuda.memcpy_htod_async(dev_l, host_l, self.stream)
        self._cuda.memcpy_htod_async(dev_r, host_r, self.stream)

        self.context.execute_async_v3(stream_handle=self.stream.handle)

        outputs = {}
        for name in self.output_names:
            host, dev = self._buffers[name]
            self._cuda.memcpy_dtoh_async(host, dev, self.stream)
            outputs[name] = host
        self.stream.synchronize()

        # Convention: first output is disparity. If a confidence output is
        # present (heuristic: a different name containing "conf"), use it;
        # otherwise return ones.
        disp_np = outputs[self.output_names[0]].copy()
        conf_np = None
        for name in self.output_names[1:]:
            if "conf" in name.lower():
                conf_np = outputs[name].copy()
                break

        disp_t = torch.from_numpy(disp_np).to(self.device).float()
        if conf_np is not None:
            conf_t = torch.from_numpy(conf_np).to(self.device).float()
        else:
            conf_t = torch.ones_like(disp_t)

        # Match run_stereo_matching post-processing: crop and squeeze
        disp_t = image_crop(disp_t.unsqueeze(0) if disp_t.ndim == 3 else disp_t, (h, w)).squeeze().float()
        conf_t = image_crop(conf_t.unsqueeze(0) if conf_t.ndim == 3 else conf_t, (h, w)).squeeze().float()
        return disp_t, conf_t
