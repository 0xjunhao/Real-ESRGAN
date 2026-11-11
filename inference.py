#!/usr/bin/env python3
import argparse
from collections import defaultdict
from fractions import Fraction
import tempfile
import time
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

import ffmpeg

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from torch import nn
import torch

ROOT = Path(__file__).resolve().parent
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}
VIDEO_EXTS = {'.mp4', '.mov', '.mkv', '.m4v', '.avi', '.webm', '.flv', '.ts'}
MODEL_URLS = {
    'RealESRGAN_x4plus': [
        'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'],
    'RealESRNet_x4plus': [
        'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth'],
    'RealESRGAN_x4plus_anime_6B': [
        'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth'],
    'RealESRGAN_x2plus': [
        'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'],
    'realesr-animevideov3': [
        'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth'],
    'realesr-general-x4v3': [
        'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
        'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'],
}
MODEL_SCALES = {
    'RealESRGAN_x4plus': 4,
    'RealESRNet_x4plus': 4,
    'RealESRGAN_x4plus_anime_6B': 4,
    'RealESRGAN_x2plus': 2,
    'realesr-animevideov3': 4,
    'realesr-general-x4v3': 4,
}


def probe_video(path: Path) -> dict:
    meta = ffmpeg.probe(str(path))
    streams = [s for s in meta['streams'] if s['codec_type'] == 'video']
    stream = streams[0]
    rate = stream.get('avg_frame_rate') or stream.get('r_frame_rate') or '0/0'
    fps = float(Fraction(rate)) if rate != '0/0' else 0.0
    count = stream.get('nb_frames')
    duration = stream.get('duration') or meta['format'].get('duration')
    frame_count = int(count) if count and count != 'N/A' else (
        round(float(duration) * fps) if duration else None)
    has_audio = any(s['codec_type'] == 'audio' for s in meta['streams'])
    return dict(width=int(stream['width']),
                height=int(stream['height']),
                fps=fps, has_audio=has_audio,
                count=frame_count)


class VideoReader:
    def __init__(self, source: Path):
        self.source = source
        self.idx = 0
        self.process = None
        meta = probe_video(source)
        self.width, self.height = meta['width'], meta['height']
        self.fps, self.count = meta['fps'], meta['count']
        self.audio_source = source if meta['has_audio'] else None
        self.process = (
            ffmpeg.input(str(source))
            .output('pipe:', format='rawvideo', pix_fmt='bgr24', loglevel='error')
            .run_async(pipe_stdout=True)
        )

    def read(self) -> np.ndarray:
        n = self.width * self.height * 3
        data = self.process.stdout.read(n)
        if not data:
            return None
        assert len(data) == n, f'Failed to read complete frame from video for "{
            self.source}".'
        frame = np.frombuffer(data, np.uint8).reshape((self.height, self.width, 3))
        return frame

    def close(self) -> None:
        if self.process is not None:
            self.process.stdout.close()
            assert self.process.wait() == 0, f'FFmpeg decoder failed for {self.source}'


class VideoWriter:
    def __init__(self, args: argparse.Namespace, output_path: Path,
                 width: int, height: int, fps: float, audio_source=None):
        width = int(width * args.outscale)
        height = int(height * args.outscale)
        source = ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24',
                              s=f'{width}x{height}', framerate=fps)
        self.output_path = output_path
        kwargs = dict(pix_fmt='yuv420p', vcodec=args.video_encoder, loglevel='error')
        if audio_source is not None:
            tracks = [
                s for s in ffmpeg.probe(
                    str(audio_source))['streams'] if s.get('codec_type') == 'audio']
            track = next((s for s in tracks if s['codec_name'] == 'aac'), tracks[0])
            acodec = 'copy' if track and track['codec_name'] == 'aac' else 'aac'
            audio = ffmpeg.input(str(audio_source))[str(track['index'])]
            output = ffmpeg.output(source, audio, str(
                output_path), acodec=acodec, **kwargs)
        else:
            output = ffmpeg.output(source, str(output_path), **kwargs)
        self.process = output.overwrite_output().run_async(pipe_stdin=True)

    def write(self, frame: np.ndarray) -> None:
        self.process.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())

    def close(self):
        self.process.stdin.close()
        assert self.process.wait() == 0, f'FFmpeg encoder failed for {self.output_path}'


def resolve_device(args: argparse.Namespace) -> torch.device:
    if args.device != 'auto':
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda:0')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    return device


def prepare_weights(args: argparse.Namespace) -> None:
    model_name = args.model_name
    assert model_name in MODEL_URLS, f'Model "{model_name}" is not supported.'
    model_dir = ROOT / 'weights'
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / (model_name + '.pth')
    if not model_path.is_file():
        for url in MODEL_URLS[model_name]:
            load_file_from_url(
                url=url,
                model_dir=str(model_dir),
                progress=True,
                file_name=None)
        assert model_path.is_file(), f'Failed to load model "{model_name}".'


def create_model(name: str) -> nn.Module:
    if name in ('RealESRGAN_x4plus', 'RealESRNet_x4plus'):
        return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                       num_block=23, num_grow_ch=32, scale=4)
    if name == 'RealESRGAN_x4plus_anime_6B':
        return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                       num_block=6, num_grow_ch=32, scale=4)
    if name == 'RealESRGAN_x2plus':
        return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                       num_block=23, num_grow_ch=32, scale=2)
    assert False, f'Model "{name}" is not supported.'


def make_upsampler(args: argparse.Namespace, device: torch.device) -> RealESRGANer:
    model = create_model(args.model_name)
    model_path = ROOT / 'weights' / (args.model_name + '.pth')
    upsampler = RealESRGANer(
        scale=MODEL_SCALES[args.model_name], model_path=str(model_path), dni_weight=None,
        model=model, tile=args.tile, tile_pad=args.tile_pad, pre_pad=args.pre_pad,
        half=not args.fp32, device=device)
    if args.torch_compile:
        print(f'Compiling model (backend={args.compile_backend}, '
              f'mode={args.compile_mode}, device={device}); first run may be slow', flush=True)
        upsampler.model.compile(backend=args.compile_backend, mode=args.compile_mode)
    return upsampler


def process_images(args: argparse.Namespace, paths: list[Path]) -> None:
    device = resolve_device(args)
    print(f'Images: model={args.model_name}, device={device}, count={len(paths)}')
    upsampler = make_upsampler(args, device)
    for path in tqdm(paths, desc='images', unit='image'):
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        assert image is not None, f'Failed to read image "{path}".'
        output, _ = upsampler.enhance(
            image, outscale=args.outscale, alpha_upsampler=args.alpha_upsampler)
        has_alpha = image.ndim == 3 and image.shape[2] == 4
        ext = path.suffix.lower().lstrip('.') if args.ext == 'auto' else args.ext
        if has_alpha:
            ext = 'png'
        elif ext == 'jpeg':
            ext = 'jpg'
        suffix = f'_{args.suffix}' if args.suffix else ''
        out_path = Path(args.output) / f'{path.stem}{suffix}.{ext}'
        assert cv2.imwrite(
            str(out_path), output), f'Failed to write image "{out_path}".'


@torch.no_grad()
def upscale_video_batch(upsampler: RealESRGANer, frames: list, outscale: float, timings: dict = None):
    def checkpoint(label, start):
        if timings is not None:
            if upsampler.device.type == 'mps':
                torch.mps.synchronize()
            elif upsampler.device.type == 'cuda':
                torch.cuda.synchronize(upsampler.device)
            timings[label] = time.perf_counter() - start


    if len(frames) == 1:
        begin = time.perf_counter()
        result = [upsampler.enhance(frames[0], outscale=outscale)[0]]
        checkpoint('upscale_total', begin)
        return result

    begin = time.perf_counter()
    tensors = []
    for frame in frames:
        rgb = cv2.cvtColor(frame.astype(np.float32) / 255.0,
                           cv2.COLOR_BGR2RGB)
        upsampler.pre_process(rgb)
        tensors.append(upsampler.img)
    upsampler.img = torch.cat(tensors, dim=0)
    checkpoint('prepare', begin)
    begin = time.perf_counter()
    if upsampler.tile_size > 0:
        upsampler.tile_process()
    else:
        upsampler.process()
    checkpoint('model', begin)
    begin = time.perf_counter()
    result = upsampler.post_process().float().cpu().clamp_(0, 1).numpy()
    result = result[:, [2, 1, 0], :, :].transpose(0, 2, 3, 1)
    height, width = frames[0].shape[:2]
    output = []
    for image in result:
        image = (image * 255.0).round().astype(np.uint8)
        if outscale != float(upsampler.scale):
            image = cv2.resize(image,
                               (int(width * outscale), int(height * outscale)),
                               interpolation=cv2.INTER_LANCZOS4)
        output.append(image)
    checkpoint('postprocess', begin)
    return output


def process_video_worker(args: argparse.Namespace, source_path: Path,
                         output_path: Path, device: torch.device) -> None:
    print(
        f'Video: model={
            args.model_name}, device={device}, output={output_path}',
        flush=True)
    upsampler = make_upsampler(args, device)
    reader = VideoReader(source_path)
    writer = VideoWriter(
        args,
        output_path,
        reader.width,
        reader.height,
        reader.fps,
        reader.audio_source)
    try:
        profile_total = defaultdict(float)
        profiled_frames = 0
        with tqdm(total=reader.count, unit='frame', desc=str(device)) as progress:
            while True:
                start = time.perf_counter()
                frames = []
                for _ in range(args.batch_size):
                    frame = reader.read()
                    if frame is None:
                        break
                    frames.append(frame)
                if not frames:
                    break
                decode_seconds = time.perf_counter() - start

                inputs = frames
                if args.torch_compile and len(frames) < args.batch_size:
                    inputs = frames + [frames[-1]] * (args.batch_size - len(frames))
                stage_times = {} if args.profile else None
                start = time.perf_counter()
                enhanced = upscale_video_batch(upsampler, inputs, args.outscale, stage_times)
                upscale_seconds = time.perf_counter() - start

                start = time.perf_counter()
                for frame in enhanced[:len(frames)]:
                    writer.write(frame)
                encode_seconds = time.perf_counter() - start
                progress.update(len(frames))
                if args.profile:
                    profiled_frames += len(frames)
                    profile_total['decode'] += decode_seconds
                    profile_total['upscale'] += upscale_seconds
                    profile_total['encode'] += encode_seconds
                    for name, duration in stage_times.items():
                        profile_total[name] += duration
                    per_frame = lambda key: profile_total[key] / profiled_frames
                    seconds = sum(profile_total[k] for k in ('decode', 'upscale', 'encode'))
                    msg = (f'[profile] {profiled_frames} frames '
                               f'{profiled_frames/seconds:.3f} fps; '
                               f'decode={per_frame("decode"):.3f}s/frame, '
                               f'upscale={per_frame("upscale"):.3f}s/frame, '
                               f'write={per_frame("encode"):.3f}s/frame')
                    if args.batch_size > 1:
                        msg += (f' [prepare={per_frame("prepare"):.3f}, '
                                f'model={per_frame("model"):.3f}, '
                                f'post={per_frame("postprocess"):.3f}]')
                    tqdm.write(msg)

    finally:
        try:
            reader.close()
        finally:
            writer.close()


def process_video(args: argparse.Namespace, video_path: Path) -> None:
    device = resolve_device(args)
    stem = video_path.stem
    suffix = f'_{args.suffix}' if args.suffix else ''
    output_path = Path(args.output) / f'{stem}{suffix}.mp4'
    process_video_worker(args, video_path, output_path, device)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--alpha_upsampler',
        default='realesrgan',
        help='Alpha upsampler to use')
    parser.add_argument('--batch_size', '--batch-size', type=int, default=1,
                        help='Batch size for video processing')
    parser.add_argument('--compile_backend', choices=['inductor', 'eager', 'aot_eager'],
                        default='inductor')
    parser.add_argument('--compile_mode',
                        choices=['default', 'reduce-overhead', 'max-autotune',
                                 'max-autotune-no-cudagraphs'],
                        default='default')
    parser.add_argument('--device', default='auto',
                        choices=['auto', 'cuda:0', 'mps', 'cpu'],
                        help='device to use for execution')
    parser.add_argument('--ext', choices=['auto', 'jpg', 'png'], default='auto')
    parser.add_argument('-i', '--input', default='inputs',
                        help='Image, video, or directory')
    parser.add_argument('-n', '--model_name', default='RealESRGAN_x2plus')
    parser.add_argument('-o', '--output', default='results', help='Output directory')
    parser.add_argument('--pre_pad', type=int, default=0)
    parser.add_argument('--profile', action='store_true', help='Enable profiling')
    parser.add_argument('-s', '--outscale', type=float, default=2.0)
    parser.add_argument('--suffix', default='out', help='Filename suffix')
    parser.add_argument('-t', '--tile', type=int, default=0)
    parser.add_argument('--tile_pad', type=int, default=10)
    parser.add_argument(
        '--fp32',
        action='store_true',
        help='Use FP32 instead of default FP16')
    parser.add_argument(
        '--video_encoder',
        choices=[
            'libx264',
            'h264_nvenc',
            'hevc_nvenc',
            'h264_videotoolbox',
            'hevc_videotoolbox'],
        default='libx264',
    )
    parser.add_argument('--torch_compile', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    prepare_weights(args)
    source = Path(args.input).expanduser().resolve()
    assert source.exists(), f'Input path "{source}" does not exist.'
    paths = sorted(path for path in source.iterdir() if path.is_file()
                   ) if source.is_dir() else [source]
    images = [p for p in paths if p.suffix.lower() in IMAGE_EXTS]
    videos = [p for p in paths if p.suffix.lower() in VIDEO_EXTS]
    if images:
        process_images(args, images)
    for path in videos:
        process_video(args, path)


if __name__ == '__main__':
    main()
