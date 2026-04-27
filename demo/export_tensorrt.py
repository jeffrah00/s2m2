import os
import subprocess
import torch
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='S')
    parser.add_argument('--img_width', type=int, default=800)
    parser.add_argument('--img_height', type=int, default=1088)
    parser.add_argument('--precision', type=str, choices=['fp16', 'tf32', 'fp32'], default='fp16')

    return parser

def main(args):
    print(project_root)
    onnx_path = os.path.join(project_root, "weights/onnx_save")
    trt_path = os.path.join(project_root, "weights/trt_save")
    os.makedirs(trt_path, exist_ok=True)

    torch_version = torch.__version__
    onnx_file_path = os.path.join(onnx_path, f'S2M2_{args.model_type}_{args.img_width}_{args.img_height}_v2_torch{torch_version[0]}{torch_version[2]}.onnx')
    trt_file_path = os.path.join(trt_path, f'S2M2_{args.model_type}_{args.img_width}_{args.img_height}_{args.precision}.engine')

    command = f'trtexec --onnx={onnx_file_path} --saveEngine={trt_file_path}'

    if args.precision == 'fp16':
        command += ' --fp16'
    elif args.precision == 'fp32':
        command += ' --noTF32'

    # On Jetson AGX Orin (JetPack 5.1.1) only stub versions of libnvrtc.so
    # and libnvidia-ml.so are present. trtexec detects stubs at startup and
    # prints "you are running using the stub version of nvrtc/nvml" on every
    # line of output. Filter those lines so the real build log stays readable.
    proc = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in proc.stdout:
        if 'stub version' not in line:
            print(line, end='', flush=True)
    proc.wait()

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
