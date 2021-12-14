# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: Apache-2.0

import logging
from collections import defaultdict
from enum import Enum
from functools import partial
from pathlib import Path
from typing import List, Optional

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


def get_default(key):
    if key in [
        "name",
        "glimpse-folder",
        "ontarget-aoiinfo",
        "offtarget-aoiinfo",
        "driftlist",
    ]:
        return [c.get(key, None) for c in DEFAULTS["channels"]]
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
            "Initialized Tapqir in the working directory.\n"
            "{yellow}---------------------------------------------------------------{nc}\n"
            f"- Checkout the documentation: {format_link('https://tapqir.readthedocs.io/')}\n"
            f"- Get help on our forum: {format_link('https://github.com/gelles-brandeis/tapqir/discussions')}"
        ).format(yellow=colorama.Fore.YELLOW, nc=colorama.Fore.RESET)
    )


@app.command()
def glimpse(
    dataset: str = typer.Option(
        partial(get_default, "dataset"), help="Dataset name", prompt="Dataset name"
    ),
    P: int = typer.Option(
        partial(get_default, "P"),
        "--aoi-size",
        "-P",
        help="AOI image size - number of pixels along the axis",
        prompt="AOI image size - number of pixels along the axis",
    ),
    num_channels: int = typer.Option(
        partial(get_default, "num-channels"),
        "--num-channels",
        "-C",
        help="Number of color channels",
        prompt="Number of color channels",
    ),
    name: Optional[List[str]] = typer.Option(partial(get_default, "name")),
    glimpse_folder: Optional[List[str]] = typer.Option(
        partial(get_default, "glimpse-folder")
    ),
    ontarget_aoiinfo: Optional[List[str]] = typer.Option(
        partial(get_default, "ontarget-aoiinfo")
    ),
    offtarget_aoiinfo: Optional[List[str]] = typer.Option(
        partial(get_default, "offtarget-aoiinfo")
    ),
    driftlist: Optional[List[str]] = typer.Option(partial(get_default, "driftlist")),
    frame_start: Optional[int] = typer.Option(partial(get_default, "frame-start")),
    frame_end: Optional[int] = typer.Option(partial(get_default, "frame-end")),
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
    labels: bool = typer.Option(
        False,
        "--labels",
        "-l",
        help="Add on-target binding labels.",
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
    breakpoint()

    # inputs descriptions
    desc = {}
    desc["frame-start"] = "First frame to include in the analysis"
    desc["frame-end"] = "Last frame to include in the analysis"
    desc["name"] = "Channel name"
    desc["glimpse-folder"] = "Header/glimpse folder"
    desc["driftlist"] = "Driftlist file"
    desc["ontarget-aoiinfo"] = "Target molecule locations file"
    desc["offtarget-aoiinfo"] = "Off-target control locations file"
    desc["ontarget-labels"] = "On-target AOI binding labels"
    desc["offtarget-labels"] = "Off-target AOI binding labels"

    DEFAULTS["dataset"] = dataset
    DEFAULTS["P"] = P
    DEFAULTS["num-channels"] = num_channels
    DEFAULTS["frame-start"] = frame_start
    DEFAULTS["frame-end"] = frame_end

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
            DEFAULTS["frame-start"] = None
            DEFAULTS["frame-end"] = None

        if "channels" not in DEFAULTS:
            DEFAULTS["channels"] = []
        for c in range(num_channels):
            if len(DEFAULTS["channels"]) < c + 1:
                DEFAULTS["channels"].append({})
            DEFAULTS["channels"][c]["name"] = name[c] if c < len(name) else None
            DEFAULTS["channels"][c]["glimpse-folder"] = (
                glimpse_folder[c] if c < len(glimpse_folder) else None
            )
            DEFAULTS["channels"][c]["ontarget-aoiinfo"] = (
                ontarget_aoiinfo[c] if c < len(ontarget_aoiinfo) else None
            )
            DEFAULTS["channels"][c]["offtarget-aoiinfo"] = (
                offtarget_aoiinfo[c] if c < len(offtarget_aoiinfo) else None
            )
            DEFAULTS["channels"][c]["driftlist"] = (
                driftlist[c] if c < len(driftlist) else None
            )
            keys = [
                "name",
                "glimpse-folder",
                "ontarget-aoiinfo",
                "offtarget-aoiinfo",
                "driftlist",
            ]
            if labels:
                keys += ["ontarget-labels", "offtarget-labels"]
            else:
                if "ontarget-labels" in DEFAULTS["channels"][c]:
                    del DEFAULTS["channels"][c]["ontarget-labels"]
                elif "offtarget-labels" in DEFAULTS["channels"][c]:
                    del DEFAULTS["channels"][c]["offtarget-labels"]
            typer.echo(f"\nINPUTS FOR CHANNEL #{c}\n")
            for key in keys:
                if key == "offtarget-aoiinfo":
                    offtarget = typer.confirm(
                        "Add off-target AOI locations?",
                        default=(
                            DEFAULTS["channels"][c]["offtarget-aoiinfo"] is not None
                        ),
                    )
                    if not offtarget:
                        DEFAULTS["channels"][c]["offtarget-aoiinfo"] = None
                        continue

                DEFAULTS["channels"][c][key] = typer.prompt(
                    desc[key], default=DEFAULTS["channels"][c][key]
                )

    DEFAULTS = dict(DEFAULTS)

    if overwrite:
        with open(cd / ".tapqir" / "config.yml", "w") as cfg_file:
            yaml.dump(DEFAULTS, cfg_file, sort_keys=False)

    typer.echo("Extracting AOIs ...")
    read_glimpse(
        path=cd,
        **DEFAULTS,
    )
    typer.echo("Extracting AOIs: Done")


@app.command()
def fit(
    model: Model = typer.Option("cosmos", help="Tapqir model", prompt="Tapqir model"),
    channels: str = typer.Option(
        "0",
        help="Color-channel numbers to analyze",
        prompt="Channel numbers (space separated if multiple)",
    ),
    cuda: bool = typer.Option(
        partial(get_default, "cuda"),
        "--cuda/--cpu",
        help="Run computations on GPU or CPU",
        prompt="Run computations on GPU?",
        show_default=False,
    ),
    nbatch_size: int = typer.Option(
        partial(get_default, "nbatch-size"),
        "--nbatch-size",
        "-n",
        help="AOI batch size",
        prompt="AOI batch size",
    ),
    fbatch_size: int = typer.Option(
        partial(get_default, "fbatch-size"),
        "--fbatch-size",
        "-f",
        help="Frame batch size",
        prompt="Frame batch size",
    ),
    learning_rate: float = typer.Option(
        partial(get_default, "learning-rate"),
        "--learning-rate",
        "-lr",
        help="Learning rate",
        prompt="Learning rate",
    ),
    num_iter: int = typer.Option(
        0,
        "--num-iter",
        "-it",
        help="Number of iterations",
        prompt="Number of iterations",
    ),
    k_max: int = typer.Option(
        2, "--k-max", "-k", help="Maximum number of spots per image"
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
    pykeops: bool = typer.Option(
        True,
        "--pykeops/--no-pykeops",
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
    from tapqir.utils.stats import save_stats

    settings = {}
    settings["S"] = 1
    settings["K"] = k_max
    settings["channels"] = [int(c) for c in channels.split()]
    settings["device"] = "cuda" if cuda else "cpu"
    settings["dtype"] = "double"
    settings["use_pykeops"] = pykeops

    if overwrite:
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
        typer.echo("Fitting the data ...")
        exit_code = model.run(num_iter)
        if exit_code == 0:
            typer.echo("Fitting the data: Done")
            typer.echo("Computing stats ...")
            save_stats(model, cd, save_matlab=matlab)
            typer.echo("Computing stats: Done")
        elif exit_code == 1:
            typer.echo("The model hasn't converged!")


@app.command()
def stats(
    model: Model = typer.Option("cosmos", help="Tapqir model", prompt="Tapqir model"),
    channels: str = typer.Option(
        "0",
        help="Color-channel numbers to analyze",
        prompt="Channel numbers (space separated if multiple)",
    ),
    cuda: bool = typer.Option(
        partial(get_default, "cuda"),
        "--cuda/--cpu",
        help="Run computations on GPU or CPU",
        prompt="Run computations on GPU?",
        show_default=False,
    ),
    nbatch_size: int = typer.Option(
        partial(get_default, "nbatch-size"),
        "--nbatch-size",
        "-n",
        help="AOI batch size",
        prompt="AOI batch size",
    ),
    fbatch_size: int = typer.Option(
        partial(get_default, "fbatch-size"),
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

        typer.echo("Computing stats ...")
        save_stats(model, cd, save_matlab=matlab)
        typer.echo("Computing stats: Done")


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
    breakpoint()

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
