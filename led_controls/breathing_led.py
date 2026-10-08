#!/usr/bin/env python3
"""Warm white breathing animation for a WS2812B strip on Orange Pi 5 (SPI4 M0)."""
import math
import signal
import sys
import time

import spidev

# ---- settings ----
BUS, DEV = 4, 1
NUM_LEDS = 10
WARM_WHITE = (255, 147, 41)   # roughly 2700K; lower the G/B values for warmer
PERIOD = 4.0                  # seconds per full breath
MAX_BRIGHTNESS = 0.2          # 0.0 to 1.0, keep low to be easy on your supply
MIN_BRIGHTNESS = 0.02         # never fully off between breaths
FPS = 60
GAMMA = 2.2                   # makes the fade look even to the eye

# ---- SPI setup ----
spi = spidev.SpiDev()
spi.open(BUS, DEV)
spi.max_speed_hz = 2_400_000
spi.mode = 0

# each color byte -> 3 SPI bytes (0 => 100, 1 => 110)
LUT = []
for v in range(256):
    bits = 0
    for i in range(7, -1, -1):
        bits = (bits << 3) | (0b110 if (v >> i) & 1 else 0b100)
    LUT.append(bits.to_bytes(3, "big"))

RESET = bytes(100)


def show_color(r, g, b):
    """Send the same color to every LED (WS2812B wants GRB order)."""
    pixel = LUT[g] + LUT[r] + LUT[b]
    spi.xfer2(list(pixel * NUM_LEDS + RESET))


def breath(t):
    """Smooth breathing curve, 0..1. Slow at the dim end, like inhaling."""
    x = math.sin(2 * math.pi * t / PERIOD - math.pi / 2)  # starts at the dim point
    return (math.exp(x) - 1 / math.e) / (math.e - 1 / math.e)


def clear_and_exit(*_):
    show_color(0, 0, 0)
    spi.close()
    sys.exit(0)


signal.signal(signal.SIGINT, clear_and_exit)
signal.signal(signal.SIGTERM, clear_and_exit)

start = time.monotonic()
frame_time = 1.0 / FPS

while True:
    t = time.monotonic() - start
    level = MIN_BRIGHTNESS + (MAX_BRIGHTNESS - MIN_BRIGHTNESS) * breath(t)
    level = level ** GAMMA
    r, g, b = (int(c * level + 0.5) for c in WARM_WHITE)
    show_color(r, g, b)
    time.sleep(frame_time)
