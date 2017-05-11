import win32gui
import time
import numpy as np
from PIL import ImageGrab
import cv2


class Window:
    def __init__(self, title=None):
        self.title = title
        self.hwnd = None
        self.bbox = (0, 0, 0, 0)

    def _enum_handler(self, hwnd, param):
        if self.title in win32gui.GetWindowText(hwnd):
            self.hwnd = hwnd

    def link_handler(self, title):
        self.title = title
        self.hwnd = None
        win32gui.EnumWindows(self._enum_handler, None)
        return self.hwnd is not None

    def focus_window(self):
        if self.hwnd is not None:
            win32gui.SetForegroundWindow(self.hwnd)

    def move_window(self, x, y):
        bbox = win32gui.GetWindowRect(self.hwnd)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        win32gui.MoveWindow(self.hwnd, x, y, w, h, True)

    def capture_screen(self):
        bbox = win32gui.GetWindowRect(self.hwnd)
        return cv2.cvtColor(np.array(ImageGrab.grab(bbox)), cv2.COLOR_RGB2BGR)


if __name__ == "__main__":
    gta_window = Window()
    gta_window.link_handler("Grand Theft Auto V")
    gta_window.focus_window()
    gta_window.move_window(0, 0)

    time.sleep(2)

    im = gta_window.capture_screen()
    print(np.shape(im))
    cv2.imshow("Test", im)
    cv2.waitKey()
    cv2.destroyAllWindows()

    print("Starting fps calculation...")

    startTime = lastTime = currentTime = time.time()
    frame_count = 0
    while currentTime - startTime < 20:
        im = gta_window.capture_screen()
        frame_count += 1
        currentTime = time.time()
    print("FPS: ", frame_count / 20.)
