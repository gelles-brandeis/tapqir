import logging
import subprocess
import os
from cosmos import __path__

from cliff.command import Command

class View(Command):
    "A command to view fit results"

    def take_action(self, args):
        subprocess.run("voila {}".format(os.path.join(__path__[0], "view.ipynb")),
            shell=True)
