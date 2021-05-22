# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: GPL-3.0-or-later

import sys

from cliff.app import App
from cliff.commandmanager import CommandManager

from tapqir import __version__ as tapqir_version


class MyApp(App):
    def __init__(self):
        super().__init__(
            description="Bayesian analysis of single molecule image data",
            version=tapqir_version,
            command_manager=CommandManager("tapqir.app"),
            deferred_help=True,
        )


def main(argv=sys.argv[1:]):
    myapp = MyApp()
    return myapp.run(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
