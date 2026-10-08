import spidev

BUS, DEV = 4, 1          # match your /dev/spidevX.Y
NUM_LEDS = 30

spi = spidev.SpiDev()
spi.open(BUS, DEV)
spi.max_speed_hz = 2_400_000
spi.mode = 0

# Precompute: each color byte -> 3 SPI bytes (0 => 100, 1 => 110)
LUT = []
for v in range(256):
    bits = 0
    for i in range(7, -1, -1):
        bits = (bits << 3) | (0b110 if (v >> i) & 1 else 0b100)
    LUT.append(bits.to_bytes(3, "big"))

def show(pixels):  # pixels = [(r, g, b), ...]
    buf = bytearray()
    for r, g, b in pixels:
        buf += LUT[g] + LUT[r] + LUT[b]   # WS2812B wants GRB order
    buf += bytes(20)                       # >50 µs low = latch/reset
    spi.xfer2(list(buf))

show([(255, 0, 0)] * NUM_LEDS)   # all red
