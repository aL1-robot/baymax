#!/usr/bin/env python3
"""
Standalone "Hey Baymax" wake word tester (microWakeWord streaming model).
Bypasses pymicro_wakeword by driving tflite_runtime + pymicro_features
directly. Live audio comes from `arecord` (ALSA plug layer does the
resampling properly), so no sounddevice and no custom resampler.

Install (inside your venv, Orange Pi / Debian Bullseye aarch64):
    TFVER=2.15.0.post1
    PYVER=39
    ARCH=aarch64
    pip install numpy pymicro-features
    pip install https://github.com/PINTO0309/TensorflowLite-bin/releases/download/v${TFVER}/tflite_runtime-${TFVER/-/}-cp${PYVER}-none-linux_${ARCH}.whl

Usage:
    python wakeword_standalone.py hey_baymax.json --wav test.wav
    python wakeword_standalone.py hey_baymax.json --wav clips/*.wav
    python wakeword_standalone.py hey_baymax.json --alsa plughw:3,0 --benchmark
"""

import argparse
import json
import subprocess
import sys
import time
import wave
from collections import deque
from pathlib import Path

import numpy as np

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    print("tflite_runtime not found - see the install block at the top "
          "of this file.", file=sys.stderr)
    raise

from pymicro_features import MicroFrontend

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 160              # 10 ms
CHUNK_BYTES = CHUNK_SAMPLES * 2  # 16-bit mono
RAW_FEATURE_SCALE = 0.0390625    # micro frontend integer output -> float

_printed_debug_shape = False


def extract_feature_vectors(output):
    """Pull zero or more feature vectors out of a MicroFrontendOutput."""
    global _printed_debug_shape
    if not _printed_debug_shape:
        print(f"(debug) raw process_samples() output: {output!r}", file=sys.stderr)
        _printed_debug_shape = True

    if output is None:
        return []
    feats = getattr(output, "features", output)
    if feats is None:
        return []
    feats = list(feats)
    if not feats:
        return []

    first = feats[0]
    if hasattr(first, "__len__"):
        return feats
    return [feats]


class MinimalMicroWakeWord:
    """quantize -> invoke -> dequantize -> sliding-window average -> cutoff.

    The model is a streaming model with internal state. Each invoke() must
    get `frames_per_step` NEW, NON-OVERLAPPING feature frames. The buffer is
    therefore cleared after every inference (stride == frames_per_step).
    """

    def __init__(self, tflite_path: Path, probability_cutoff: float,
                 sliding_window_size: int, feature_scale="auto"):
        self.interpreter = tflite.Interpreter(model_path=str(tflite_path))
        self.interpreter.allocate_tensors()

        in_detail = self.interpreter.get_input_details()[0]
        out_detail = self.interpreter.get_output_details()[0]

        self.input_index = in_detail["index"]
        self.input_shape = in_detail["shape"]
        self.input_dtype = in_detail["dtype"]
        in_q = in_detail["quantization_parameters"]
        self.input_scale = float(in_q["scales"][0]) if len(in_q["scales"]) else 0.0
        self.input_zero_point = int(in_q["zero_points"][0]) if len(in_q["zero_points"]) else 0

        self.output_index = out_detail["index"]
        out_q = out_detail["quantization_parameters"]
        self.output_scale = float(out_q["scales"][0]) if len(out_q["scales"]) else 0.0
        self.output_zero_point = int(out_q["zero_points"][0]) if len(out_q["zero_points"]) else 0

        print(f"  input:  shape={self.input_shape} dtype={self.input_dtype} "
              f"scale={self.input_scale} zero_point={self.input_zero_point}")
        print(f"  output: scale={self.output_scale} zero_point={self.output_zero_point}")

        self.frames_per_step = int(self.input_shape[1])
        self.feature_size = int(self.input_shape[2])
        self.frame_buffer: deque = deque(maxlen=self.frames_per_step)
        self.step_ms = self.frames_per_step * 10
        print(f"  (model consumes {self.frames_per_step} new {self.feature_size}-bin "
              f"frames per inference = one inference every {self.step_ms} ms)")

        # Feature scaling: "auto" decides on the first frame with signal.
        # Raw micro-frontend output is integers up to ~670; scaled is ~0-26.
        self.feature_scale_mode = feature_scale
        if feature_scale == "auto":
            self.feature_scale = None
        else:
            self.feature_scale = float(feature_scale)

        self.probability_cutoff = probability_cutoff
        self.sliding_window_size = sliding_window_size
        self.window = deque(maxlen=sliding_window_size)
        self._debug_count = 0

    def reset(self):
        self.window.clear()
        self.frame_buffer.clear()
        try:
            self.interpreter.reset_all_variables()  # clear streaming state
        except Exception:
            pass

    def _scale_features(self, vec):
        vec = np.asarray(vec, dtype=np.float32)
        if self.feature_scale is None:
            if vec.max() > 0:
                self.feature_scale = RAW_FEATURE_SCALE if vec.max() > 40 else 1.0
                print(f"(feature scale auto-detected: {self.feature_scale}, "
                      f"first frame max={vec.max():.2f})", file=sys.stderr)
            else:
                return vec  # all zeros, nothing to scale yet
        return vec * self.feature_scale

    def process(self, feature_vector):
        """Feed one 40-bin frame. Returns (avg_prob, hit) after an inference,
        or (None, False) while still collecting the next stride."""
        self.frame_buffer.append(self._scale_features(feature_vector))

        if len(self.frame_buffer) < self.frames_per_step:
            return None, False

        arr = np.concatenate(list(self.frame_buffer))
        self.frame_buffer.clear()  # consume the stride - do NOT slide

        if self.input_scale:
            q = np.round(arr / self.input_scale + self.input_zero_point)
        else:
            q = arr
        info = np.iinfo(self.input_dtype)
        q = np.clip(q, info.min, info.max).astype(self.input_dtype)
        q = q.reshape(self.input_shape)

        self.interpreter.set_tensor(self.input_index, q)
        self.interpreter.invoke()
        raw_out = self.interpreter.get_tensor(self.output_index).flatten()[0]

        if self.output_scale:
            prob = (float(raw_out) - self.output_zero_point) * self.output_scale
        else:
            prob = float(raw_out)

        if self._debug_count < 5:
            print(f"(debug) feature: min={arr.min():.2f} max={arr.max():.2f} "
                  f"mean={arr.mean():.2f} | quantized: min={q.min()} max={q.max()} "
                  f"| raw_out={raw_out} prob={prob:.4f}", file=sys.stderr)
            self._debug_count += 1

        self.window.append(prob)
        avg = sum(self.window) / len(self.window)
        return avg, avg >= self.probability_cutoff


def load_model(config_path: Path, window_override=None, cutoff_override=None,
               feature_scale="auto"):
    with open(config_path) as f:
        config = json.load(f)

    micro = config["micro"]
    cutoff = cutoff_override if cutoff_override is not None else micro["probability_cutoff"]
    window = window_override if window_override is not None else micro["sliding_window_size"]
    tflite_path = config_path.parent / config["model"]

    print(f"loaded: {config_path.name} (wake word: {config.get('wake_word', '?')})")
    print(f"  model file: {tflite_path}")
    print(f"  cutoff: {cutoff}  window: {window} inferences")

    return MinimalMicroWakeWord(tflite_path, cutoff, window, feature_scale)


def run_wav(mww, frontend, path: Path) -> bool:
    with wave.open(str(path), "rb") as wf:
        if wf.getframerate() != SAMPLE_RATE or wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            print(f"  SKIP {path.name}: need 16 kHz mono 16-bit "
                  f"(got {wf.getframerate()} Hz, {wf.getnchannels()} ch, "
                  f"{wf.getsampwidth() * 8}-bit)")
            return False
        raw = wf.readframes(wf.getnframes())

    mww.reset()
    detected = False
    peak = 0.0
    audio_peak = int(np.abs(np.frombuffer(raw, dtype=np.int16)).max()) if raw else 0

    for offset in range(0, len(raw) - CHUNK_BYTES + 1, CHUNK_BYTES):
        output = frontend.process_samples(raw[offset:offset + CHUNK_BYTES])
        for feat in extract_feature_vectors(output):
            avg, hit = mww.process(feat)
            if avg is None:
                continue
            peak = max(peak, avg)
            if hit:
                detected = True

    print(f"  [{'DETECT' if detected else '  --  '}] {path.name}  "
          f"peak_prob={peak:.3f}  audio_peak={audio_peak}")
    return detected


def run_mic(mww, frontend, alsa_device: str, benchmark: bool, refractory: float):
    cmd = ["arecord", "-q", "-D", alsa_device, "-f", "S16_LE",
           "-r", str(SAMPLE_RATE), "-c", "1", "-t", "raw"]
    print("capturing with:", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    print("listening... say the wake word. ctrl-c to stop.\n")

    last_hit = 0.0
    proc_times = []
    last_stat_print = time.monotonic()
    last_meter_print = time.monotonic()
    level_accum = []
    max_prob_since_print = 0.0
    inferences = 0

    try:
        while True:
            chunk = proc.stdout.read(CHUNK_BYTES)
            if len(chunk) < CHUNK_BYTES:
                err = proc.stderr.read().decode(errors="replace").strip()
                print(f"arecord stopped. {err}", file=sys.stderr)
                break
            now = time.monotonic()

            level_accum.append(int(np.abs(np.frombuffer(chunk, dtype=np.int16)).max()))

            t0 = time.perf_counter()
            output = frontend.process_samples(chunk)
            for feat in extract_feature_vectors(output):
                avg, hit = mww.process(feat)
                if avg is None:
                    continue
                inferences += 1
                max_prob_since_print = max(max_prob_since_print, avg)
                if hit and (now - last_hit) > refractory:
                    last_hit = now
                    print(f">>> DETECTED at {time.strftime('%H:%M:%S')} (prob={avg:.3f})")
            proc_times.append(time.perf_counter() - t0)

            if benchmark and (now - last_stat_print) > 2.0 and proc_times:
                avg_ms = 1000 * sum(proc_times) / len(proc_times)
                print(f"(avg {avg_ms:.2f} ms of work per 10ms chunk)")
                proc_times.clear()
                last_stat_print = now

            if (now - last_meter_print) > 0.5:
                peak_level = max(level_accum) if level_accum else 0
                print(f"[meter] mic peak={peak_level:5d}/32767  "
                      f"inferences={inferences}  max_prob={max_prob_since_print:.3f}  "
                      f"cutoff={mww.probability_cutoff}")
                level_accum.clear()
                max_prob_since_print = 0.0
                last_meter_print = now
    finally:
        proc.terminate()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path, help="model json manifest")
    p.add_argument("--wav", nargs="+", type=Path, help="test 16kHz mono wav file(s)")
    p.add_argument("--alsa", default="plughw:3,0",
                   help="ALSA device for arecord (default: plughw:3,0)")
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--window", type=int, default=None,
                   help="sliding window size in inferences (each = one stride)")
    p.add_argument("--cutoff", type=float, default=None)
    p.add_argument("--refractory", type=float, default=1.0)
    p.add_argument("--feature-scale", default="auto",
                   help="'auto', or a number (e.g. 0.0390625 for raw frontend ints, 1.0 for none)")
    args = p.parse_args()

    mww = load_model(args.config, args.window, args.cutoff, args.feature_scale)
    frontend = MicroFrontend()

    if args.wav:
        paths = []
        for entry in args.wav:
            paths.extend(sorted(entry.parent.glob(entry.name))
                         if any(c in entry.name for c in "*?[")
                         else [entry])
        hits = sum(run_wav(mww, frontend, path) for path in paths)
        print(f"\n{hits}/{len(paths)} detected")
    else:
        try:
            run_mic(mww, frontend, args.alsa, args.benchmark, args.refractory)
        except KeyboardInterrupt:
            print("\nstopped.")


if __name__ == "__main__":
    main()
