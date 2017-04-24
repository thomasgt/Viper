import ctypes
import time
from enum import Enum
import threading
import queue
import numpy as np

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


class KeyThread(threading.Thread):
    # Internal states
    class State(Enum):
        IDLE = 0
        ENABLED = 1
        DISABLED = 2

    def __init__(self, key, period=0.1):
        super(KeyThread, self).__init__()
        self.key = key
        self.task_queue = queue.Queue()
        self.duty_cycle = 0
        self.period = period
        self.daemon = True
        self.state = self.State.IDLE

    def set_duty_cycle(self, duty_cycle):
        self.duty_cycle = np.clip(duty_cycle, 0, 1)

    def start(self):
        self.state = self.State.ENABLED
        super(KeyThread, self).start()

    def run(self):
        while True:
            if self.state == self.State.ENABLED:
                # Press/release key
                start_time = time.time()
                # if self.duty_cycle * self.period > 0.01:
                press_key(self.key)
                time.sleep(self.duty_cycle * self.period)
                on_time = time.time() - start_time
                release_key(self.key)
                time.sleep((1 - self.duty_cycle) * self.period)
                total_time = time.time()  - start_time
                # print("   Actual period:     %5.1f" % total_time)
                # print("   Actual duty cycle: %5.1f%%" % (on_time / total_time * 100))
            elif self.state == self.State.DISABLED:
                release_key(self.key)
                self.state = self.State.IDLE
            elif self.state == self.State.IDLE:
                time.sleep(0.1)

    def disable(self):
        self.state = self.State.DISABLED

    def enable(self):
        self.state = self.State.ENABLED



if __name__ == '__main__':
    for t in range(5, 0, -1):
        print("Starting in {}...".format(t))
        time.sleep(1)

    forward_thread = KeyThread(Key.W, 2)
    forward_thread.start()

    for duty_cycle in np.arange(0.3, 1, 0.1):
        print("Setting duty cycle to %0.0f%%" % (duty_cycle * 100))
        forward_thread.set_duty_cycle(duty_cycle)
        time.sleep(5)

    forward_thread.set_duty_cycle(0)

    time.sleep(10)
