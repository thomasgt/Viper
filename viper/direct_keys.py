import ctypes
import time
from enum import Enum

# Internal implementation
_SendInput = ctypes.windll.user32.SendInput
_PUL = ctypes.POINTER(ctypes.c_ulong)


class _KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", _PUL)]


class _HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]


class _MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time",ctypes.c_ulong),
                ("dwExtraInfo", _PUL)]


class _InputI(ctypes.Union):
    _fields_ = [("ki", _KeyBdInput),
                ("mi", _MouseInput),
                ("hi", _HardwareInput)]


class _Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", _InputI)]


# External interface
def press_key(key):
    extra = ctypes.c_ulong(0)
    ii_ = _InputI()
    ii_.ki = _KeyBdInput(0, key.value, 0x0008, 0, ctypes.pointer(extra))
    x = _Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


def release_key(key):
    extra = ctypes.c_ulong(0)
    ii_ = _InputI()
    ii_.ki = _KeyBdInput(0, key.value, 0x0008 | 0x0002, 0, ctypes.pointer(extra))
    x = _Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


class Key(Enum):
    W = 0x11
    A = 0x1E
    S = 0x1F
    D = 0x20

if __name__ == '__main__':
    for t in range(5, 0, -1):
        print("Starting in {}...".format(t))
        time.sleep(1)

    while (True):
        press_key(Key.W)
        time.sleep(5)
        release_key(Key.W)
        press_key(Key.S)
        time.sleep(5)
        release_key(Key.S)