import spidev

BUS, DEV = 4, 1
NUM_LEDS = 30

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

def show(pixels):
    buf = bytearray()
    for r, g, b in pixels:
        buf += LUT[g] + LUT[r] + LUT[b]
    buf += bytes(100)   # longer reset, safer for newer WS2812B batches
    spi.xfer2(list(buf))

show([(20, 0, 0)] * NUM_LEDS)   # dim red first, easier on your supply
