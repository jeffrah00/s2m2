import torch
import os

def export_onnx(model, onnx_path, left_torch, right_torch):

    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    model = model.cpu().eval()
    left_torch, right_torch = left_torch.cpu(), right_torch.cpu()
    try:
        # do_constant_folding=True is required so that shape-derived slice
        # bounds (starts/ends/axes) are folded into constants; otherwise
        # TensorRT's importSlice fails with "axes.allValuesKnown && this
        # version of tensorrt does not support dynamic axes" even though
        # dynamic_axes=None.
        torch.onnx.export(model,
                          (left_torch, right_torch),
                          onnx_path,
                          export_params=True,
                          dynamo=True,
                          opset_version=16,
                          verbose=True,
                          do_constant_folding=True,
                          input_names=['input_left', 'input_right'],
                          output_names=['output_disp', 'output_occ', 'output_conf'],
                          dynamic_axes=None)
        print('success onnx conversion')

    except Exception as e:
        print(f"error type:{type(e).__name__}")
        print(f"error :{e}")
        import traceback
        traceback.print_exc()


def export_torchscript(model, torchscript_path, left_torch, right_torch):

    os.makedirs(os.path.dirname(torchscript_path), exist_ok=True)
    try:
        with torch.inference_mode():
                exported_mod = torch.export.export(model, (left_torch, right_torch))

        torch.export.save(exported_mod, torchscript_path)
        print('success torchscript conversion')

    except Exception as e:
        print(f"error type:{type(e).__name__}")
        print(f"error :{e}")
        import traceback
        traceback.print_exc()


