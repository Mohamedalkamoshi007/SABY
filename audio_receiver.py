#!/usr/bin/env python3
"""
audio_receiver.py  -  Receive ESP32-C3 UDP audio stream, clean it, save WAV, optional live playback.

Packet format (little-endian):
    uint16 magic (0xA51D) | uint16 sample_rate | uint32 seq | uint16 samples[N]  (raw 12-bit ADC)
"""
import argparse
import os
import subprocess
import queue
import socket
import struct
import sys
import time
import wave
from datetime import datetime

import numpy as np
from scipy import signal

MAGIC = 0xA51D
HEADER = struct.Struct("<HHI")
MAX_GAP_PACKETS = 50          # larger jumps are treated as a stream restart


class Processor:
    """Stateful, block-by-block cleaning chain."""

    def __init__(self, rate, gain, hp_hz, lp_hz, spike):
        self.rate = rate
        self.gain = gain
        self.spike = spike
        self.hp = signal.butter(2, hp_hz, "highpass", fs=rate, output="sos")
        self.lp = signal.butter(4, min(lp_hz, 0.45 * rate), "lowpass", fs=rate, output="sos")
        self.fade_short = max(8, int(0.004 * rate))   # 4 ms after a gap
        self.fade_start = int(0.020 * rate)           # 20 ms at stream start
        self.reset()

    def reset(self):
        self.hp_zi = None
        self.lp_zi = np.zeros((self.lp.shape[0], 2))
        self.carry = None
        self.last = 0.0
        self._start_fade(self.fade_start)

    def _start_fade(self, n):
        self.fade_total = n
        self.fade_pos = 0

    def process(self, raw):
        x = raw.astype(np.float64)

        # 1) De-spike: replace isolated outliers by the median of their neighbours
        if self.spike > 0:
            if self.carry is None:
                self.carry = np.array([x[0], x[0]])
            a = np.concatenate([self.carry, x])
            self.carry = a[-2:].copy()
            mid = a[1:-1]
            med = np.median(np.stack([a[:-2], mid, a[2:]]), axis=0)
            x = np.where(np.abs(mid - med) > self.spike, med, mid)

        # 2) DC offset removal (high-pass), state initialised at first sample -> no start click
        if self.hp_zi is None:
            self.hp_zi = signal.sosfilt_zi(self.hp) * x[0]
        y, self.hp_zi = signal.sosfilt(self.hp, x, zi=self.hp_zi)

        # 3) Gentle low-pass
        y, self.lp_zi = signal.sosfilt(self.lp, y, zi=self.lp_zi)

        # 4) Normalise (12-bit centred) and apply software gain
        y = y * (self.gain / 2048.0)

        # 5) Fade-in after start / packet loss (raised cosine)
        if self.fade_pos < self.fade_total:
            k = min(len(y), self.fade_total - self.fade_pos)
            idx = (self.fade_pos + np.arange(k)) / self.fade_total
            y[:k] *= 0.5 - 0.5 * np.cos(np.pi * idx)
            self.fade_pos += k

        # 6) Soft limiter
        y = np.tanh(y)
        self.last = float(y[-1])
        return y

    def conceal(self, n):
        """Fill a gap of n lost samples with a smooth exponential decay (no click)."""
        tau = 0.002 * self.rate
        out = self.last * np.exp(-np.arange(1, n + 1) / tau)
        self.last = float(out[-1])
        self.carry = None
        self._start_fade(self.fade_short)
        return out


def to_pcm(y):
    return np.clip(np.round(y * 32000.0), -32768, 32767).astype(np.int16)


class LivePlayer:
    """Small jitter buffer + sounddevice output."""

    def __init__(self, rate, prebuffer_blocks=5, max_blocks=40):
        import sounddevice as sd
        self.q = queue.Queue(maxsize=max_blocks)
        self.cur = np.zeros(0, dtype=np.int16)
        self.prebuffer = prebuffer_blocks
        self.started = False
        self.stream = sd.OutputStream(samplerate=rate, channels=1, dtype="int16",
                                      callback=self._callback)

    def put(self, block):
        try:
            self.q.put_nowait(block)
        except queue.Full:                      # PC clock slower than ESP32: drop oldest
            try:
                self.q.get_nowait()
            except queue.Empty:
                pass
            self.q.put_nowait(block)
        if not self.started and self.q.qsize() >= self.prebuffer:
            self.stream.start()
            self.started = True

    def _callback(self, outdata, frames, time_info, status):
        out = np.zeros(frames, dtype=np.int16)
        filled = 0
        while filled < frames:
            if self.cur.size == 0:
                try:
                    self.cur = self.q.get_nowait()
                except queue.Empty:
                    break
            k = min(frames - filled, self.cur.size)
            out[filled:filled + k] = self.cur[:k]
            self.cur = self.cur[k:]
            filled += k
        outdata[:, 0] = out

    def close(self):
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser(description="ESP32-C3 UDP audio receiver")
    ap.add_argument("--port", type=int, default=5005)
    ap.add_argument("--out", default=None, help="output WAV path (default: timestamped)")
    ap.add_argument("--gain", type=float, default=3.0, help="software gain (default 3.0)")
    ap.add_argument("--hp", type=float, default=20.0, help="DC-block high-pass cutoff Hz")
    ap.add_argument("--lp", type=float, default=3500.0, help="low-pass cutoff Hz")
    ap.add_argument("--spike", type=float, default=350.0, help="de-spike threshold in ADC counts (0=off)")
    ap.add_argument("--seconds", type=float, default=0, help="stop after N seconds (0 = until Ctrl+C)")
    ap.add_argument("--live", action="store_true", help="play audio live (use headphones!)")
    args = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
    sock.bind(("0.0.0.0", args.port))
    sock.settimeout(0.5)

    out_path = args.out or datetime.now().strftime("audio_%Y%m%d_%H%M%S.wav")
    proc = wav = player = None
    expected = None
    pkts = lost = late = 0
    t_start = None
    t_stat = time.monotonic()
    st = {"ss": 0.0, "n": 0, "rmin": 4095, "rmax": 0}

    def emit(y):
        pcm = to_pcm(y)
        wav.writeframes(pcm.tobytes())
        if player:
            player.put(pcm)
        f = pcm.astype(np.float64) / 32768.0
        st["ss"] += float(np.sum(f * f))
        st["n"] += len(f)

    print(f"Listening on UDP port {args.port} ... (Ctrl+C to stop)")
    try:
        while True:
            if args.seconds and t_start and time.monotonic() - t_start >= args.seconds:
                break
            try:
                data, addr = sock.recvfrom(4096)
            except socket.timeout:
                continue

            if len(data) < HEADER.size + 2 or (len(data) - HEADER.size) % 2:
                continue
            magic, rate, seq = HEADER.unpack_from(data)
            if magic != MAGIC:
                continue
            raw = np.frombuffer(data, dtype="<u2", offset=HEADER.size)

            if proc is None:
                proc = Processor(rate, args.gain, args.hp, args.lp, args.spike)
                wav = wave.open(out_path, "wb")
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(rate)
                if args.live:
                    player = LivePlayer(rate)
                t_start = time.monotonic()
                print(f"Stream from {addr[0]}  rate={rate} Hz  ->  {out_path}")
            elif rate != proc.rate:
                continue

            # ----- sequence handling -----
            if expected is not None:
                d = ((seq - expected + 2**31) % 2**32) - 2**31
                if d < 0:
                    if d >= -MAX_GAP_PACKETS:
                        late += 1          # duplicate / late packet: ignore
                        continue
                    proc.reset()           # ESP32 restarted
                elif d > MAX_GAP_PACKETS:
                    proc.reset()
                elif d > 0:
                    lost += d
                    emit(proc.conceal(d * len(raw)))
            expected = (seq + 1) & 0xFFFFFFFF

            pkts += 1
            st["rmin"] = min(st["rmin"], int(raw.min()))
            st["rmax"] = max(st["rmax"], int(raw.max()))
            emit(proc.process(raw))

            # ----- status line once per second -----
            now = time.monotonic()
            if now - t_stat >= 1.0:
                t_stat = now
                rms = (st["ss"] / max(st["n"], 1)) ** 0.5
                db = 20 * np.log10(max(rms, 1e-6))
                total = pkts + lost
                clip = "  CLIPPING! lower mic gain" if (st["rmin"] <= 5 or st["rmax"] >= 4090) else ""
                line = (f"[{now - t_start:6.1f}s] pkts={pkts} lost={lost} "
                        f"({100.0 * lost / max(total, 1):.2f}%) late={late} "
                        f"level={db:6.1f} dBFS raw=[{st['rmin']}..{st['rmax']}]{clip}")
                sys.stdout.write("\r" + line.ljust(110))
                sys.stdout.flush()
                st.update(ss=0.0, n=0, rmin=4095, rmax=0)

    except KeyboardInterrupt:
        pass
    finally:
        print()
        if wav:
            wav.close()
            here = os.path.dirname(os.path.abspath(__file__))
            subprocess.run([sys.executable, os.path.join(here, "denoise.py"), out_path])
            clean_path = out_path[:-4] + "_clean.wav"
            if os.path.exists(clean_path):
                os.replace(clean_path, out_path)
                print(f"Saved (clean): {out_path}   (packets={pkts}, lost={lost})")
            else:
                print(f"Denoise failed, raw file kept: {out_path}")
        else:
            print("No audio received.")
        if player:
            player.close()
        sock.close()


if __name__ == "__main__":
    main()