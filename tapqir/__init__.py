# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: GPL-3.0-or-later

from ._version import get_versions

name = "tapqir"

__version__ = get_versions()["version"]
del get_versions
