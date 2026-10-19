#!/usr/bin/env python3
"""
mic_receiver.py - receive INMP441 UDP stream (int16 @ 16 kHz), save WAV, optional live playback.
Packet (little-endian): uint16 magic(0xA51E) | uint16 rate | uint32 seq | int16 samples[N]
"""
import argparse
import os
import queue
import socket
import struct
import subprocess
import sys
import time
import wave
from datetime import datetime

import numpy as np
from scipy import signal

MAGIC = 0xA51E
HEADER = struct.Struct("<HHI")
MAX_GAP_PACKETS = 50


class Processor:
    def __init__(self, rate, gain, hp_hz):
        self.rate = rate
        self.gain = gain
        self.hp = signal.butter(2, hp_hz, "highpass", fs=rate, output="sos") if hp_hz > 0 else None
        self.fade_short = max(8, int(0.004 * rate))
        self.fade_start = int(0.020 * rate)
        self.reset()

    def reset(self):
        self.hp_zi = None
        self.last = 0.0
        self._start_fade(self.fade_start)

    def _start_fade(self, n):
        self.fade_total = n
        self.fade_pos = 0

    def process(self, raw):
        x = raw.astype(np.float64) / 32768.0
        if self.hp is not None:
            if self.hp_zi is None:
                self.hp_zi = signal.sosfilt_zi(self.hp) * x[0]
            x, self.hp_zi = signal.sosfilt(self.hp, x, zi=self.hp_zi)
        y = x * self.gain
        if self.fade_pos < self.fade_total:
            k = min(len(y), self.fade_total - self.fade_pos)
            idx = (self.fade_pos + np.arange(k)) / self.fade_total
            y[:k] *= 0.5 - 0.5 * np.cos(np.pi * idx)
            self.fade_pos += k
        y = np.tanh(y)
        self.last = float(y[-1])
        return y

    def conceal(self, n):
        tau = 0.002 * self.rate
        out = self.last * np.exp(-np.arange(1, n + 1) / tau)
        self.last = float(out[-1])
        self._start_fade(self.fade_short)
        return out


def to_pcm(y):
    return np.clip(np.round(y * 32000.0), -32768, 32767).astype(np.int16)


class LivePlayer:
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
        except queue.Full:
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
    ap = argparse.ArgumentParser(description="INMP441 UDP audio receiver")
    ap.add_argument("--port", type=int, default=5005)
    ap.add_argument("--out", default=None, help="output WAV path (default: timestamped)")
    ap.add_argument("--gain", type=float, default=8.0, help="software gain (default 8)")
    ap.add_argument("--hp", type=float, default=80.0, help="high-pass cutoff Hz (0 = off)")
    ap.add_argument("--seconds", type=float, default=0, help="stop after N seconds (0 = until Ctrl+C)")
    ap.add_argument("--live", action="store_true", help="play live (use headphones!)")
    ap.add_argument("--denoise", action="store_true", help="run denoise.py after recording")
    ap.add_argument("--no-report", action="store_true", help="skip the report image")
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
    st = {"ss": 0.0, "n": 0, "peak": 0.0}

    def emit(y):
        pcm = to_pcm(y)
        wav.writeframes(pcm.tobytes())
        if player:
            player.put(pcm)

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
            raw = np.frombuffer(data, dtype="<i2", offset=HEADER.size)

            if proc is None:
                proc = Processor(rate, args.gain, args.hp)
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

            if expected is not None:
                d = ((seq - expected + 2**31) % 2**32) - 2**31
                if d < 0:
                    if d >= -MAX_GAP_PACKETS:
                        late += 1
                        continue
                    proc.reset()
                elif d > MAX_GAP_PACKETS:
                    proc.reset()
                elif d > 0:
                    lost += d
                    emit(proc.conceal(d * len(raw)))
            expected = (seq + 1) & 0xFFFFFFFF

            pkts += 1
            f = raw.astype(np.float64) / 32768.0
            st["ss"] += float(np.sum(f * f))
            st["n"] += len(f)
            st["peak"] = max(st["peak"], float(np.max(np.abs(f))))
            emit(proc.process(raw))

            now = time.monotonic()
            if now - t_stat >= 1.0:
                t_stat = now
                rms = (st["ss"] / max(st["n"], 1)) ** 0.5
                db = 20 * np.log10(max(rms, 1e-6))
                pk = 20 * np.log10(max(st["peak"], 1e-6))
                total = pkts + lost
                clip = "  CLIPPING! speak farther or lower --gain" if st["peak"] >= 0.98 else ""
                line = (f"[{now - t_start:6.1f}s] pkts={pkts} lost={lost} "
                        f"({100.0 * lost / max(total, 1):.2f}%) "
                        f"rms={db:6.1f} dBFS peak={pk:6.1f} dBFS{clip}")
                sys.stdout.write("\r" + line.ljust(110))
                sys.stdout.flush()
                st.update(ss=0.0, n=0, peak=0.0)

    except KeyboardInterrupt:
        pass
    finally:
        print()
        if wav:
            wav.close()
            if args.denoise:
                here = os.path.dirname(os.path.abspath(__file__))
                subprocess.run([sys.executable, os.path.join(here, "denoise.py"), out_path])
                clean = out_path[:-4] + "_clean.wav"
                if os.path.exists(clean):
                    os.replace(clean, out_path)
            print(f"Saved: {out_path}   (packets={pkts}, lost={lost})")
            if not args.no_report:
                here = os.path.dirname(os.path.abspath(__file__))
                subprocess.run([sys.executable, os.path.join(here, "report.py"), out_path, str(pkts), str(lost)])
        else:
            print("No audio received.")
        if player:
            player.close()
        sock.close()


if __name__ == "__main__":
    main()