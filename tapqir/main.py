# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: Apache-2.0

import logging
from collections import defaultdict
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Optional

import colorama
import typer
import yaml

from tapqir import __version__

app = typer.Typer()

# option DEFAULTS
DEFAULTS = defaultdict(lambda: None)


def version_callback(value: bool):
    if value:
        typer.echo(__version__)
        raise typer.Exit()


def format_link(link):
    return "<{blue}{link}{nc}>".format(
        blue=colorama.Fore.CYAN, link=link, nc=colorama.Fore.RESET
    )


# available models
class Model(str, Enum):
    cosmos = "cosmos"


def _get_default(key):
    return DEFAULTS[key]


def deactivate_prompts(ctx, param, value):
    if value:
        for p in ctx.command.params:
            if isinstance(p, typer.core.TyperOption) and p.prompt is not None:
                p.prompt = None
    return value


@app.command()
def init():
    """
    Initialize Tapqir in the working directory.

    Initializing a Tapqir workspace creates a ``.tapqir`` sub-directory for storing ``config.yml``
    file, ``loginfo`` file, and files that are created by commands such as ``tapqir fit``.
    """
    global DEFAULTS
    cd = DEFAULTS.pop("cd")

    TAPQIR_PATH = cd / ".tapqir"

    if TAPQIR_PATH.is_dir():
        typer.echo(".tapqir already exists.")
        raise typer.Exit()

    # initialize directory
    TAPQIR_PATH.mkdir()
    CONFIG_FILE = TAPQIR_PATH / "config.yml"
    # CONFIG_FILE.touch(exist_ok=True)
    with open(CONFIG_FILE, "w") as cfg_file:
        DEFAULTS["P"] = 14
        DEFAULTS["nbatch-size"] = 10
        DEFAULTS["fbatch-size"] = 512
        DEFAULTS["learning-rate"] = 0.005
        DEFAULTS["num-channels"] = 1
        yaml.dump(dict(DEFAULTS), cfg_file, sort_keys=False)

    typer.echo(
        (
            "{yellow}Tapqir is a Bayesian program for single-molecule data analysis.{nc}\n"
            "{yellow}---------------------------------------------------------------{nc}\n"
            f"- Checkout the documentation: {format_link('https://tapqir.readthedocs.io/')}\n"
            f"- Get help on our forum: {format_link('https://github.com/gelles-brandeis/tapqir/discussions')}\n"
            f"- Star us on GitHub: {format_link('https://github.com/gelles-brandeis/tapqir')}\n"
            "\nInitialized Tapqir in the working directory."
        ).format(yellow=colorama.Fore.YELLOW, nc=colorama.Fore.RESET)
    )


@app.command()
def glimpse(
    dataset: str = typer.Option(
        partial(_get_default, "dataset"), help="Dataset name", prompt="Dataset name"
    ),
    P: int = typer.Option(
        partial(_get_default, "P"),
        "--aoi-size",
        "-P",
        help="AOI image size - number of pixels along the axis",
        prompt="AOI image size - number of pixels along the axis",
    ),
    num_channels: int = typer.Option(
        partial(_get_default, "num-channels"),
        "--num-channels",
        "-C",
        help="Number of color channels",
        prompt="Number of color channels",
    ),
    overwrite: bool = typer.Option(
        True,
        "--overwrite",
        "-w",
        help="Overwrite defaults values.",
        prompt="Overwrite defaults values?",
    ),
    no_input: bool = typer.Option(
        False,
        "--no-input",
        help="Disable interactive prompt.",
        is_eager=True,
        callback=deactivate_prompts,
    ),
):
    """
    Extract AOIs from raw glimpse images.

    Analyzing data acquired with Glimpse and pre-processed with  the imscroll program
    will require the following files:

    * image data in glimpse format and header file\n
    * aoiinfo file designating the locations of target molecules (on-target AOIs) in the binder channel\n
    * (optional) aoiinfo file designating the off-target control locations (off-target AOIs) in the binder channel\n
    * driftlist file recording the stage movement that took place during the experiment
    """

    from tapqir.imscroll import read_glimpse

    global DEFAULTS
    cd = DEFAULTS.pop("cd")

    # inputs descriptions
    desc = {}
    desc["frame-start"] = "First frame to include in the analysis"
    desc["frame-end"] = "Last frame to include in the analysis"
    desc["name"] = "Channel name"
    desc["glimpse-folder"] = "Path to the header/glimpse folder"
    desc["driftlist"] = "Path to the driftlist file"
    desc["ontarget-aoiinfo"] = "Path to the on-target AOI locations file"
    desc["offtarget-aoiinfo"] = "Path to the off-target control AOI locations file"
    desc["ontarget-labels"] = "On-target AOI binding labels"
    desc["offtarget-labels"] = "Off-target AOI binding labels"

    DEFAULTS["dataset"] = dataset
    DEFAULTS["P"] = P
    DEFAULTS["num-channels"] = num_channels

    if not no_input:
        frame_range = typer.confirm(
            "Specify frame range?", default=("frame-start" in DEFAULTS)
        )
        if frame_range:
            DEFAULTS["frame-start"] = typer.prompt(
                desc["frame-start"], default=DEFAULTS["frame-start"], type=int
            )
            DEFAULTS["frame-end"] = typer.prompt(
                desc["frame-end"], default=DEFAULTS["frame-end"], type=int
            )
        else:
            if "frame-start" in DEFAULTS:
                del DEFAULTS["frame-start"]
            if "frame-end" in DEFAULTS:
                del DEFAULTS["frame-end"]

        if "channels" not in DEFAULTS:
            DEFAULTS["channels"] = []
        for c in range(num_channels):
            if len(DEFAULTS["channels"]) < c + 1:
                DEFAULTS["channels"].append(defaultdict(lambda: None))
            else:
                DEFAULTS["channels"][c] = defaultdict(
                    lambda: None, DEFAULTS["channels"][c]
                )
            keys = [
                "name",
                "glimpse-folder",
                "ontarget-aoiinfo",
                "offtarget-aoiinfo",
                "driftlist",
                "ontarget-labels",
            ]
            typer.echo(f"\nINPUTS FOR CHANNEL #{c}\n")
            for key in keys:
                if key == "offtarget-aoiinfo":
                    offtarget = typer.confirm(
                        "Add off-target AOI locations?",
                        default=("offtarget-aoiinfo" in DEFAULTS["channels"][c]),
                    )
                    if not offtarget:
                        if "offtarget-aoiinfo" in DEFAULTS["channels"][c]:
                            del DEFAULTS["channels"][c]["offtarget-aoiinfo"]
                        continue
                elif key == "ontarget-labels":
                    labels = typer.confirm(
                        "Add on-target labels?",
                        default=("ontarget-labels" in DEFAULTS["channels"][c]),
                    )
                    if not labels:
                        if "ontarget-labels" in DEFAULTS["channels"][c]:
                            del DEFAULTS["channels"][c]["ontarget-labels"]
                        continue
                DEFAULTS["channels"][c][key] = typer.prompt(
                    desc[key], default=DEFAULTS["channels"][c][key]
                )

    DEFAULTS = dict(DEFAULTS)
    for i, channel in enumerate(DEFAULTS["channels"]):
        DEFAULTS["channels"][i] = dict(channel)

    if overwrite:
        with open(cd / ".tapqir" / "config.yml", "w") as cfg_file:
            yaml.dump(DEFAULTS, cfg_file, sort_keys=False)

    read_glimpse(
        path=cd,
        **DEFAULTS,
    )


@app.command()
def fit(
    model: Model = typer.Option("cosmos", help="Tapqir model", prompt="Tapqir model"),
    channels: str = typer.Option(
        "0",
        help="Color-channel numbers to analyze",
        prompt="Channel numbers (space separated if multiple)",
    ),
    marginal: bool = typer.Option(
        partial(_get_default, "marginal"),
        "--marginal/--full",
        help="Use the marginalized or the full model",
        prompt="Use the marginalized model?",
        show_default=False,
    ),
    cuda: bool = typer.Option(
        partial(_get_default, "cuda"),
        "--cuda/--cpu",
        help="Run computations on GPU or CPU",
        prompt="Run computations on GPU?",
        show_default=False,
    ),
    nbatch_size: int = typer.Option(
        partial(_get_default, "nbatch-size"),
        "--nbatch-size",
        "-n",
        help="AOI batch size",
        prompt="AOI batch size",
    ),
    fbatch_size: int = typer.Option(
        partial(_get_default, "fbatch-size"),
        "--fbatch-size",
        "-f",
        help="Frame batch size",
        prompt="Frame batch size",
    ),
    learning_rate: float = typer.Option(
        partial(_get_default, "learning-rate"),
        "--learning-rate",
        "-lr",
        help="Learning rate",
        prompt="Learning rate",
    ),
    num_epochs: int = typer.Option(
        0,
        "--num-epochs",
        "-e",
        help="Number of epochs",
        prompt="Number of epochs",
    ),
    k_max: int = typer.Option(
        2, "--k-max", "-k", help="Maximum number of spots per image"
    ),
    funsor: bool = typer.Option(
        False, "--funsor/--pyro", help="Use funsor or pyro backend"
    ),
    pykeops: bool = typer.Option(
        True,
        "--pykeops",
        help="Use pykeops backend for offset marginalization",
    ),
    overwrite: bool = typer.Option(
        True,
        "--overwrite",
        "-w",
        help="Overwrite defaults values.",
        prompt="Overwrite defaults values?",
    ),
    no_input: bool = typer.Option(
        False,
        "--no-input",
        help="Disable interactive prompt.",
        is_eager=True,
        callback=deactivate_prompts,
    ),
):
    """
    Fit the data to the selected model.

    Available models:

    * cosmos: single-color time-independent co-localization model.\n
    """
    global DEFAULTS
    cd = DEFAULTS.pop("cd")

    from pyroapi import pyro_backend

    from tapqir.models import models

    settings = {}
    settings["S"] = 1
    settings["K"] = k_max
    settings["channels"] = [int(c) for c in channels.split()]
    settings["device"] = "cuda" if cuda else "cpu"
    settings["dtype"] = "double"
    settings["use_pykeops"] = pykeops
    settings["marginal"] = marginal

    if overwrite:
        DEFAULTS["marginal"] = marginal
        DEFAULTS["cuda"] = cuda
        DEFAULTS["nbatch-size"] = nbatch_size
        DEFAULTS["fbatch-size"] = fbatch_size
        DEFAULTS["learning-rate"] = learning_rate
        with open(cd / ".tapqir" / "config.yml", "w") as cfg_file:
            yaml.dump(dict(DEFAULTS), cfg_file, sort_keys=False)

    backend = "funsor" if funsor else "pyro"
    if backend == "pyro":
        PYRO_BACKEND = "pyro"
    elif backend == "funsor":
        import funsor
        import pyro.contrib.funsor  # noqa: F401

        funsor.set_backend("torch")
        PYRO_BACKEND = "contrib.funsor"
    else:
        raise ValueError("Only pyro and funsor backends are supported.")

    with pyro_backend(PYRO_BACKEND):

        model = models[model](**settings)
        model.load(cd)

        model.init(learning_rate, nbatch_size, fbatch_size)
        model.run(num_epochs)


@app.command()
def stats(
    model: Model = typer.Option("cosmos", help="Tapqir model", prompt="Tapqir model"),
    channels: str = typer.Option(
        "0",
        help="Color-channel numbers to analyze",
        prompt="Channel numbers (space separated if multiple)",
    ),
    cuda: bool = typer.Option(
        partial(_get_default, "cuda"),
        "--cuda/--cpu",
        help="Run computations on GPU or CPU",
        prompt="Run computations on GPU?",
        show_default=False,
    ),
    nbatch_size: int = typer.Option(
        partial(_get_default, "nbatch-size"),
        "--nbatch-size",
        "-n",
        help="AOI batch size",
        prompt="AOI batch size",
    ),
    fbatch_size: int = typer.Option(
        partial(_get_default, "fbatch-size"),
        "--fbatch-size",
        "-f",
        help="Frame batch size",
        prompt="Frame batch size",
    ),
    matlab: bool = typer.Option(
        False,
        "--matlab",
        help="Save parameters in matlab format",
        prompt="Save parameters in matlab format?",
    ),
    funsor: bool = typer.Option(
        False, "--funsor/--pyro", help="Use funsor or pyro backend"
    ),
    no_input: bool = typer.Option(
        False,
        "--no-input",
        help="Use defaults values.",
        is_eager=True,
        callback=deactivate_prompts,
    ),
):
    from pyroapi import pyro_backend

    from tapqir.models import models
    from tapqir.utils.stats import save_stats

    global DEFAULTS
    cd = DEFAULTS.pop("cd")

    dtype = "double"
    device = "cuda" if cuda else "cpu"
    backend = "funsor" if funsor else "pyro"
    channels = [int(c) for c in channels.split()]

    # pyro backend
    if backend == "pyro":
        PYRO_BACKEND = "pyro"
    elif backend == "funsor":
        import funsor
        import pyro.contrib.funsor  # noqa: F401

        funsor.set_backend("torch")
        PYRO_BACKEND = "contrib.funsor"
    else:
        raise ValueError("Only pyro and funsor backends are supported.")

    with pyro_backend(PYRO_BACKEND):

        model = models[model](1, 2, channels, device, dtype)
        model.load(cd)
        model.load_checkpoint(param_only=True)
        model.nbatch_size = nbatch_size
        model.fbatch_size = fbatch_size

        save_stats(model, cd, save_matlab=matlab)


@app.command()
def show(
    model: Model = typer.Option("cosmos", help="Tapqir model", prompt="Tapqir model"),
    channels: str = typer.Option(
        "0",
        help="Color-channel numbers to analyze",
        prompt="Channel numbers (space separated if multiple)",
    ),
    funsor: bool = typer.Option(
        False, "--funsor/--pyro", help="Use funsor or pyro backend"
    ),
    no_input: bool = typer.Option(
        False,
        "--no-input",
        help="Use defaults values.",
        is_eager=True,
        callback=deactivate_prompts,
    ),
):
    import sys

    from pyroapi import pyro_backend
    from PySide2.QtWidgets import QApplication

    from tapqir.commands.qtgui import MainWindow
    from tapqir.models import models

    global DEFAULTS
    cd = DEFAULTS.pop("cd")

    backend = "funsor" if funsor else "pyro"
    channels = [int(c) for c in channels.split()]
    # pyro backend
    if backend == "pyro":
        PYRO_BACKEND = "pyro"
    elif backend == "funsor":
        PYRO_BACKEND = "contrib.funsor"
    else:
        raise ValueError("Only pyro and funsor backends are supported.")

    with pyro_backend(PYRO_BACKEND):
        model = models[model](1, 2, channels, "cpu", "float")
        app = QApplication(sys.argv)
        MainWindow(model, cd)
        sys.exit(app.exec_())


@app.command()
def log():
    """Show logging info."""
    import pydoc

    global DEFAULTS
    cd = DEFAULTS.pop("cd")

    log_file = cd / ".tapqir" / "loginfo"
    with open(log_file, "r") as f:
        pydoc.pager(f.read())


@app.callback()
def main(
    cd: Path = typer.Option(
        Path.cwd(),
        help="Change working directory.",
        show_default=False,
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        callback=version_callback,
        help="Show Tapqir version and exit.",
        is_eager=True,
    ),
):
    """
    Bayesian analysis of co-localization single-molecule microscopy image data.
    """

    global DEFAULTS
    # set working directory
    DEFAULTS["cd"] = cd

    # create logger
    logger = logging.getLogger("tapqir")

    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(levelname)s - %(message)s",
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if (cd / ".tapqir").is_dir():
        fh = logging.FileHandler(cd / ".tapqir" / "loginfo")
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%m/%d/%Y %I:%M %p",
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # read defaults from config file
    if (cd / ".tapqir" / "config.yml").is_file():
        with open(cd / ".tapqir" / "config.yml", "r") as cfg_file:
            cfg_defaults = yaml.safe_load(cfg_file) or {}
            DEFAULTS.update(cfg_defaults)


# click object is required to generate docs with sphinx-click
typer_click_object = typer.main.get_command(app)

if __name__ == "__main__":
    app()
