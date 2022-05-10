# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Union


class TapqirException(Exception):
    """
    Base class for all tapqir exception.
    """

    def __init__(self, msg, *args):
        assert msg
        self.msg = msg
        super().__init__(msg, *args)


class TapqirFileNotFoundError(TapqirException):
    """
    Thrown if a file/directory is not found as an output in any pipeline.

    :param name: file type name.
    :param path: file path.
    """

    def __init__(self, name: str, path: Union[str, Path]):
        self.name = name
        self.path = path
        super().__init__(f"Unable to find {name} file '{path}'")


class CudaOutOfMemoryError(TapqirException):
    """
    Thrown if CUDA is out of memory.
    """

    def __init__(self):
        super().__init__("CUDA out of memory. Try to use smaller AOI/frame batch size")
