#!/usr/bin/env python3
"""Warm white at 20% brightness. Stays on until Ctrl+C, then turns the strip off."""
import signal
import time

import spidev

BUS, DEV = 4, 1
NUM_LEDS = 30
WARM_WHITE = (255, 147, 41)
BRIGHTNESS = 0.1

spi = spidev.SpiDev()
spi.open(BUS, DEV)
spi.max_speed_hz = 2_400_000
spi.mode = 0

LUT = []
for v in range(256):
    bits = 0
    for i in range(7, -1, -1):
        bits = (bits << 3) | (0b110 if (v >> i) & 1 else 0b100)
    LUT.append(bits.to_bytes(3, "big"))

RESET = bytes(300)  # long low gap so every LED latches before and after the data


def show(r, g, b, repeat=3):
    frame = RESET + (LUT[g] + LUT[r] + LUT[b]) * NUM_LEDS + RESET
    for _ in range(repeat):
        spi.xfer2(list(frame))
        time.sleep(0.02)


def stop(*_):
    show(0, 0, 0, repeat=5)
    spi.close()
    raise SystemExit


signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGTERM, stop)

r, g, b = (int(c * BRIGHTNESS + 0.5) for c in WARM_WHITE)
show(r, g, b)
print("Glowing. Press Ctrl+C to turn off.")
while True:
    signal.pause()
