import numpy

import ctypes
from ctypes import windll
from ctypes import wintypes


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [('biSize', wintypes.DWORD),
                ('biWidth', wintypes.LONG),
                ('biHeight', wintypes.LONG),
                ('biPlanes', wintypes.WORD),
                ('biBitCount', wintypes.WORD),
                ('biCompression', wintypes.DWORD),
                ('biSizeImage', wintypes.DWORD),
                ('biXPelsPerMeter', wintypes.LONG),
                ('biYPelsPerMeter', wintypes.LONG),
                ('biClrUsed', wintypes.DWORD),
                ('biClrImportant', wintypes.DWORD)]


class RGBQUAD(ctypes.Structure):
    _fields_ = [('rgbBlue', wintypes.BYTE),
                ('rgbGreen', wintypes.BYTE),
                ('rgbRed', wintypes.BYTE),
                ('rgbReserved', wintypes.BYTE)]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [('bmiHeader', BITMAPINFOHEADER),
                ('bmiColors', RGBQUAD)]


class BI:
    RGB = 0


class DIB:
    RGB_COLORS = 0


class PW:
    CLIENTONLY = 1
    RENDERFULLCONTENT = 2


class WindowDC:
    def __init__(self, window):
        self._window = window
        self._hdc = None

    def __enter__(self):
        self._hdc = windll.user32.GetWindowDC(self._window.handle())
        return self

    def __exit__(self, *args):
        windll.user32.ReleaseDC(self._window.handle(), self._hdc)
        self._hdc = None

    def handle(self):
        return self._hdc


class Window:
    def __init__(self, caption):
        self._hwnd = windll.user32.FindWindowW(None, caption)

    def handle(self):
        return self._hwnd

    def dc(self):
        return WindowDC(self)

    def print(self, dc):
        windll.user32.PrintWindow(self.handle(), dc.handle(), PW.CLIENTONLY | PW.RENDERFULLCONTENT)

    def size(self):
        rect = wintypes.RECT()
        windll.user32.GetClientRect(self._hwnd, ctypes.byref(rect))

        height = rect.bottom - rect.top
        width = rect.right - rect.left

        return width, height


class CompatibleDC:
    def __init__(self, dc):
        self._dc = dc
        self._hdc = None

    def __enter__(self):
        self._hdc = windll.gdi32.CreateCompatibleDC(self._dc.handle())
        return self

    def __exit__(self, *args):
        windll.gdi32.DeleteDC(self._hdc)
        self._hdc = None

    def handle(self):
        return self._hdc

    def select(self, obj):
        windll.gdi32.SelectObject(self.handle(), obj.handle())

    def bitmap_data(self, bitmap):
        return bitmap.data(self)


class CompatibleBitmap:
    def __init__(self, dc, width, height):
        self._dc = dc
        self._height = height
        self._width = width
        self._hbitmap = None

    def __enter__(self):
        self._hbitmap = windll.gdi32.CreateCompatibleBitmap(self._dc.handle(), self._width, self._height)
        return self

    def __exit__(self, *args):
        windll.gdi32.DeleteObject(self._hbitmap)
        self._hbitmap = None

    def handle(self):
        return self._hbitmap

    def data(self, dc):
        bitmap_info = BITMAPINFO()
        bitmap_info.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bitmap_info.bmiHeader.biWidth = self._width
        bitmap_info.bmiHeader.biHeight = -self._height
        bitmap_info.bmiHeader.biPlanes = 1
        bitmap_info.bmiHeader.biBitCount = 32
        bitmap_info.bmiHeader.biCompression = BI.RGB

        image_buffer_len = self._width * self._height * 4
        image_buffer = ctypes.create_string_buffer(image_buffer_len)
        windll.gdi32.GetDIBits(dc.handle(), self.handle(), 0, self._height, image_buffer,
                               ctypes.byref(bitmap_info), DIB.RGB_COLORS)

        image_buffer = numpy.frombuffer(image_buffer.raw, dtype=numpy.uint8)
        image_buffer.shape = (self._height, self._width, 4)

        return numpy.copy(image_buffer[:, :, :3])


class GameWindowManager:
    def __init__(self, window_caption):
        self._window = Window(window_caption)

    def get_picture(self):
        width, height = self._window.size()

        with self._window.dc() as window_dc:
            with CompatibleDC(window_dc) as target_dc:
                with CompatibleBitmap(window_dc, width, height) as bitmap:
                    target_dc.select(bitmap)
                    self._window.print(target_dc)

                    return target_dc.bitmap_data(bitmap)


class MsMinesweeperWindowManager(GameWindowManager):
    def __init__(self):
        super(MsMinesweeperWindowManager, self).__init__('Microsoft Minesweeper')
