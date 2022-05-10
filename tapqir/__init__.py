# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: Apache-2.0

from ._version import get_versions

name = "tapqir"

__version__ = get_versions()["version"]
del get_versions
