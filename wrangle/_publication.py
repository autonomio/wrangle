"""Atomic directory publication that never replaces an existing result."""
from __future__ import annotations

import ctypes
import errno
import os
import sys

from ._core import WrangleError


def publish_directory(source, destination):
    """Rename on the same filesystem with the OS's exclusive-destination control."""
    try:
        if sys.platform == "win32":
            # Windows rename rejects an existing destination atomically.
            os.rename(source, destination)
            return
        library = ctypes.CDLL(None, use_errno=True)
        if sys.platform == "darwin":
            function = library.renamex_np
            function.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
            arguments = (os.fsencode(source), os.fsencode(destination), 0x00000004)  # RENAME_EXCL, sys/stdio.h
        elif sys.platform.startswith("linux"):
            function = library.renameat2
            function.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
            arguments = (-100, os.fsencode(source), -100, os.fsencode(destination), 1)  # AT_FDCWD, RENAME_NOREPLACE
        else:
            raise AttributeError("No supported exclusive rename API")
        function.restype = ctypes.c_int
        if function(*arguments) == 0:
            return
        code = ctypes.get_errno()
        if code in {errno.ENOSYS, errno.EINVAL, errno.ENOTSUP}:
            raise AttributeError("Exclusive rename is unavailable on this filesystem")
        raise OSError(code, os.strerror(code), str(destination))
    except FileExistsError as error:
        raise WrangleError("OUTPUT_EXISTS", "Choose a new output directory; existing results are never overwritten.", {"path": str(destination)}) from error
    except AttributeError as error:
        raise WrangleError("ATOMIC_PUBLICATION_UNSUPPORTED", "Use a filesystem/platform with exclusive atomic directory publication; results are never published through a weaker fallback.", {"path": str(destination), "platform": sys.platform}) from error
