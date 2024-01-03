import numpy

import ctypes
from ctypes import windll
from ctypes.wintypes import WORD, DWORD, LONG, HWND, LPARAM, RECT


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [('biSize', DWORD),
                ('biWidth', LONG),
                ('biHeight', LONG),
                ('biPlanes', WORD),
                ('biBitCount', WORD),
                ('biCompression', DWORD),
                ('biSizeImage', DWORD),
                ('biXPelsPerMeter', LONG),
                ('biYPelsPerMeter', LONG),
                ('biClrUsed', DWORD),
                ('biClrImportant', DWORD)]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [('bmiHeader', BITMAPINFOHEADER)]


EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_int, HWND, LPARAM)

BI_RGB = 0
DIB_RGB_COLORS = 0
PW_CLIENTONLY = 1
PW_RENDERFULLCONTENT = 2


class GameWindowManager:
    def __init__(self):
        self._hwnd = self._get_window()

    @staticmethod
    def _get_window():
        minesweeper_window_caption = 'Microsoft Minesweeper'
        window_caption_buffer = ctypes.create_unicode_buffer(len(minesweeper_window_caption) + 1)

        minesweeper_window_hwnd = None

        def _window_enumeration_callback(hwnd, param):
            nonlocal minesweeper_window_hwnd
            windll.user32.GetWindowTextW(hwnd, window_caption_buffer, len(window_caption_buffer))
            if window_caption_buffer.value == minesweeper_window_caption:
                minesweeper_window_hwnd = hwnd
                return 0
            return 1

        windll.user32.EnumWindows(EnumWindowsProc(_window_enumeration_callback), 0)

        return minesweeper_window_hwnd

    def get_picture(self):
        rect = RECT()
        windll.user32.GetClientRect(self._hwnd, ctypes.byref(rect))

        height = rect.bottom - rect.top
        width = rect.right - rect.left

        window_dc = windll.user32.GetWindowDC(self._hwnd)
        target_dc = windll.gdi32.CreateCompatibleDC(window_dc)

        bitmap = windll.gdi32.CreateCompatibleBitmap(window_dc, width, height)
        windll.gdi32.SelectObject(target_dc, bitmap)

        windll.user32.PrintWindow(self._hwnd, target_dc, PW_CLIENTONLY | PW_RENDERFULLCONTENT)

        bitmap_info = BITMAPINFO()
        bitmap_info.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bitmap_info.bmiHeader.biWidth = width
        bitmap_info.bmiHeader.biHeight = -height
        bitmap_info.bmiHeader.biPlanes = 1
        bitmap_info.bmiHeader.biBitCount = 32
        bitmap_info.bmiHeader.biCompression = BI_RGB

        image_buffer_len = width * height * 4
        image_buffer = ctypes.create_string_buffer(image_buffer_len)
        windll.gdi32.GetDIBits(target_dc, bitmap, 0, height, image_buffer, ctypes.byref(bitmap_info), DIB_RGB_COLORS)

        windll.gdi32.DeleteObject(bitmap)
        windll.gdi32.DeleteDC(target_dc)
        windll.user32.ReleaseDC(self._hwnd, window_dc)

        image_buffer = numpy.frombuffer(image_buffer.raw, dtype=numpy.uint8)
        image_buffer.shape = (height, width, 4)

        return numpy.copy(image_buffer[:, :, :3])

