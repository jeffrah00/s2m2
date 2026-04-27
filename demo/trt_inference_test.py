"""TensorRT inference benchmark via trtexec (no tensorrt Python package required)."""
import argparse
import os
import re
import subprocess
import sys
import tempfile

import cv2
import numpy as np

from s2m2.core.utils.image_utils import read_images

TRTEXEC = '/usr/src/tensorrt/bin/trtexec'
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='S')
    parser.add_argument('--img_width', type=int, default=800)
    parser.add_argument('--img_height', type=int, default=1088)
    parser.add_argument('--precision', type=str, choices=['fp16', 'tf32', 'fp32'], default='fp16')
    parser.add_argument('--warmup', type=int, default=500, help='warmup duration in ms')
    parser.add_argument('--iterations', type=int, default=100)
    return parser


def load_inputs(left_path, right_path, img_height, img_width):
    left, right = read_images(left_path, right_path)
    if left.shape[1] >= img_width and left.shape[0] >= img_height:
        left = left[:img_height, :img_width]
        right = right[:img_height, :img_width]
    else:
        left = cv2.resize(left, dsize=(img_width, img_height))
        right = cv2.resize(right, dsize=(img_width, img_height))
    # NCHW float32 — trtexec --loadInputs expects raw binary
    left_arr = np.ascontiguousarray(left.transpose(2, 0, 1)[np.newaxis].astype(np.float32))
    right_arr = np.ascontiguousarray(right.transpose(2, 0, 1)[np.newaxis].astype(np.float32))
    return left_arr, right_arr


def run_benchmark(engine_path, left_bin, right_bin, warmup_ms, iterations):
    cmd = (
        f'{TRTEXEC}'
        f' --loadEngine={engine_path}'
        f' --loadInputs=input_left:{left_bin},input_right:{right_bin}'
        f' --warmUp={warmup_ms}'
        f' --iterations={iterations}'
    )
    print(f'cmd: {cmd}\n')

    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines = []
    for line in proc.stdout:
        if 'stub version' not in line:
            print(line, end='', flush=True)
        lines.append(line)
    proc.wait()

    if proc.returncode != 0:
        print(f'\ntrtexec exited with code {proc.returncode}', file=sys.stderr)
        return

    # Parse "GPU Compute Time: min = X ms, max = X ms, mean = X ms, ..."
    for line in lines:
        m = re.search(r'GPU Compute Time:.*mean\s*=\s*([0-9.]+)\s*ms', line)
        if m:
            avg_ms = float(m.group(1))
            print(f'\nAvg GPU compute: {avg_ms:.3f} ms  |  FPS: {1000 / avg_ms:.1f}')
            return

    print('\n(Could not parse GPU Compute Time from trtexec output)', file=sys.stderr)


def main(args):
    engine_path = os.path.join(
        project_root,
        f'weights/trt_save/S2M2_{args.model_type}_{args.img_width}_{args.img_height}_{args.precision}.engine',
    )
    print(f'Engine : {engine_path}')
    if not os.path.exists(engine_path):
        print(f'ERROR: engine file not found: {engine_path}', file=sys.stderr)
        sys.exit(1)

    left_path = os.path.join(project_root, 'data', 'samples/Web/0025_L.png')
    right_path = os.path.join(project_root, 'data', 'samples/Web/0025_R.png')
    left_arr, right_arr = load_inputs(left_path, right_path, args.img_height, args.img_width)
    print(f'Input : left={left_arr.shape} right={right_arr.shape}  dtype={left_arr.dtype}')

    with tempfile.TemporaryDirectory() as tmpdir:
        left_bin = os.path.join(tmpdir, 'input_left.bin')
        right_bin = os.path.join(tmpdir, 'input_right.bin')
        left_arr.tofile(left_bin)
        right_arr.tofile(right_bin)
        run_benchmark(engine_path, left_bin, right_bin, args.warmup, args.iterations)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    print(args)
    main(args)
