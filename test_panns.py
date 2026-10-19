# test_panns.py - checks that the pretrained PANNs model can be downloaded and used on your CPU
import sys
import time

import librosa
from panns_inference import AudioTagging

path = sys.argv[1] if len(sys.argv) > 1 else "final4/back.wav"
audio, _ = librosa.load(path, sr=32000, mono=True, duration=3.0)

t0 = time.time()
at = AudioTagging(checkpoint_path=None, device="cpu")     # downloads the model (about 300 MB) on the first run
print("model loaded in", round(time.time() - t0, 1), "s")

t0 = time.time()
clipwise, embedding = at.inference(audio[None, :])
print("embedding shape:", embedding.shape, " time:", round(time.time() - t0, 2), "s")