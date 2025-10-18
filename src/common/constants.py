from enum import Enum, IntEnum

"""Common physical prefixes."""
KILO = 1e3
MEGA = 1e6
GIGA = 1e9
MILLI = 1e-3
MICRO = 1e-6
NANO = 1e-9


"""Time Prefixes in seconds."""
MILLISECOND = 1e-3
MICROSECOND = 1e-6
NANOSECOND = 1e-9


class SamplingRate(IntEnum):
    """Common sampling rates in Hz."""

    FS_44K1_HZ = 44100
    FS_48K_HZ = 48000
    FS_96K_HZ = 96000
    FS_192K_HZ = 192000


class Unit(Enum):
    """Common physical units."""

    HERTZ = "Hz"
    DECIBEL = "dB"
    PASCAL = "Pa"
    NEWTON = "N"
    METER = "m"
    SECOND = "s"
    WATT = "W"
    VOLT = "V"
    AMPERE = "A"
    OHM = "Ω"
