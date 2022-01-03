# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: Apache-2.0

import logging
import sys
from collections import defaultdict
from enum import Enum
from functools import partial
from pathlib import Path
from typing import List, Optional

import colorama
import matplotlib.pyplot as plt
import torch
import typer
import yaml
from tqdm import tqdm

from tapqir import __version__

app = typer.Typer()

# option DEFAULTS
DEFAULTS = defaultdict(lambda: None)
DEFAULTS["channels"] = []


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
    if param.name == "no_input" and value:
        for p in ctx.command.params:
            if isinstance(p, typer.core.TyperOption) and p.prompt is not None:
                p.prompt = None
    elif param.name == "frame_range" and not value:
        for p in ctx.command.params:
            if p.name in ("frame_start", "frame_end"):
                p.prompt = None
    return value


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
    frame_range: bool = typer.Option(
        partial(get_default, "frame-range"),
        help="Specify frame range.",
        prompt="Specify frame range?",
        callback=deactivate_prompts,
    ),
    frame_start: Optional[int] = typer.Option(
        partial(get_default, "frame-start"),
        help="First frame.",
        prompt="First frame",
    ),
    frame_end: Optional[int] = typer.Option(
        partial(get_default, "frame-end"),
        help="Last frame.",
        prompt="Last frame",
    ),
    num_channels: int = typer.Option(
        partial(get_default, "num-channels"),
        "--num-channels",
        "-C",
        help="Number of color channels",
        prompt="Number of color channels",
    ),
    use_offtarget: bool = typer.Option(
        partial(get_default, "use-offtarget"),
        help="Use off-target AOI locations.",
        prompt="Use off-target AOI locations?",
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
    progress_bar=None,
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

    torch.set_default_tensor_type(torch.FloatTensor)

    global DEFAULTS
    cd = DEFAULTS["cd"]

    if progress_bar is None:
        progress_bar = tqdm

    # fill in default values
    DEFAULTS["dataset"] = dataset
    DEFAULTS["P"] = P
    DEFAULTS["num-channels"] = num_channels
    DEFAULTS["frame-range"] = frame_range
    DEFAULTS["frame-start"] = frame_start
    DEFAULTS["frame-end"] = frame_end
    DEFAULTS["use-offtarget"] = use_offtarget
    DEFAULTS["labels"] = labels

    # inputs descriptions
    desc = {}
    desc["name"] = "Channel name"
    desc["glimpse-folder"] = "Header/glimpse folder"
    desc["driftlist"] = "Driftlist file"
    desc["ontarget-aoiinfo"] = "Target molecule locations file"
    desc["offtarget-aoiinfo"] = "Off-target control locations file"
    desc["ontarget-labels"] = "On-target AOI binding labels"
    desc["offtarget-labels"] = "Off-target AOI binding labels"

    keys = [
        "name",
        "glimpse-folder",
        "ontarget-aoiinfo",
        "offtarget-aoiinfo",
        "driftlist",
    ]
    flags = [name, glimpse_folder, ontarget_aoiinfo, offtarget_aoiinfo, driftlist]

    for c in range(num_channels):
        if len(DEFAULTS["channels"]) < c + 1:
            DEFAULTS["channels"].append({})
        for flag, key in zip(flags, keys):
            DEFAULTS["channels"][c][key] = flag[c] if c < len(flag) else None

    # interactive input
    if not no_input:
        for c in range(num_channels):
            if labels:
                keys += ["ontarget-labels", "offtarget-labels"]
            typer.echo(f"\nINPUTS FOR CHANNEL #{c}\n")
            for key in keys:
                if key == "offtarget-aoiinfo" and not use_offtarget:
                    continue

                DEFAULTS["channels"][c][key] = typer.prompt(
                    desc[key], default=DEFAULTS["channels"][c][key]
                )

    DEFAULTS = dict(DEFAULTS)

    if overwrite:
        with open(cd / ".tapqir" / "config.yaml", "w") as cfg_file:
            yaml.dump(
                {key: value for key, value in DEFAULTS.items() if key != "cd"},
                cfg_file,
                sort_keys=False,
            )

    typer.echo("Extracting AOIs ...")
    read_glimpse(
        path=cd,
        progress_bar=progress_bar,
        **DEFAULTS,
    )
    typer.echo("Extracting AOIs: Done")


@app.command()
def fit(
    model: Model = typer.Option("cosmos", help="Tapqir model", prompt="Tapqir model"),
    channels: List[int] = typer.Option(
        [0],
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
    progress_bar=None,
):
    """
    Fit the data to the selected model.

    Available models:

    * cosmos: single-color time-independent co-localization model.\n
    """
    global DEFAULTS
    cd = DEFAULTS["cd"]

    if progress_bar is None:
        progress_bar = tqdm

    from pyroapi import pyro_backend

    from tapqir.models import models
    from tapqir.utils.stats import save_stats

    settings = {}
    settings["S"] = 1
    settings["K"] = k_max
    settings["channels"] = channels
    settings["device"] = "cuda" if cuda else "cpu"
    settings["dtype"] = "double"
    settings["use_pykeops"] = pykeops

    if overwrite:
        DEFAULTS["cuda"] = cuda
        DEFAULTS["nbatch-size"] = nbatch_size
        DEFAULTS["fbatch-size"] = fbatch_size
        DEFAULTS["learning-rate"] = learning_rate
        with open(cd / ".tapqir" / "config.yaml", "w") as cfg_file:
            yaml.dump(
                {key: value for key, value in DEFAULTS.items() if key != "cd"},
                cfg_file,
                sort_keys=False,
            )

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
        exit_code = model.run(num_iter, progress_bar=progress_bar)
        if exit_code == 0:
            typer.echo("Fitting the data: Done")
        elif exit_code == 1:
            typer.echo("The model hasn't converged!")
        typer.echo("Computing stats ...")
        save_stats(model, cd, save_matlab=matlab)
        typer.echo("Computing stats: Done")


@app.command()
def stats(
    model: Model = typer.Option("cosmos", help="Tapqir model", prompt="Tapqir model"),
    channels: List[int] = typer.Option(
        [0],
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
    cd = DEFAULTS["cd"]

    dtype = "double"
    device = "cuda" if cuda else "cpu"
    backend = "funsor" if funsor else "pyro"

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


def config_axis(ax, label, f1, f2, ymin, ymax, xticklabels=False):
    plt.minorticks_on()
    ax.tick_params(
        direction="in",
        which="minor",
        length=1,
        bottom=True,
        top=True,
        left=True,
        right=True,
    )
    ax.tick_params(
        direction="in",
        which="major",
        length=2,
        bottom=True,
        top=True,
        left=True,
        right=True,
    )
    ax.set_ylabel(label)
    ax.set_xlim(f1 - 2, f2 + 1)
    ax.set_ylim(ymin, ymax)
    if not xticklabels:
        ax.set_xticklabels([])


@app.command()
def show(
    model: Model = typer.Option("cosmos", help="Tapqir model", prompt="Tapqir model"),
    channels: List[int] = typer.Option(
        [0],
        help="Color-channel numbers to analyze",
        prompt="Channel numbers (space separated if multiple)",
    ),
    n: int = typer.Option(0, help="n", prompt="n"),
    f1: Optional[int] = None,
    f2: Optional[int] = None,
):
    from tapqir.models import models

    global DEFAULTS
    cd = DEFAULTS["cd"]

    model = models[model](1, 2, channels, "cpu", "float")
    model.load(cd, data_only=False)
    if f1 is None:
        f1 = 0
    if f2 is None:
        f2 = model.data.F

    width, height, dpi = 10, 8, 100
    fig = plt.figure(figsize=(width, height), dpi=dpi)
    gs = fig.add_gridspec(
        nrows=6,
        ncols=1,
        top=0.99,
        bottom=0.05,
        left=0.07,
        right=0.98,
        hspace=0.1,
    )
    ax = {}
    item = {}

    ax["pspecific"] = fig.add_subplot(gs[0])
    config_axis(ax["pspecific"], r"$p(\mathsf{specific})$", f1, f2, -0.1, 1.1)
    (item["pspecific"],) = ax["pspecific"].plot(
        torch.arange(0, model.data.F),
        model.params["z_probs"][n],
        "o-",
        ms=3,
        lw=1,
        color="C2",
    )

    ax["height"] = fig.add_subplot(gs[1])
    config_axis(ax["height"], r"$h$", f1, f2, -100, 8000)

    ax["width"] = fig.add_subplot(gs[2])
    config_axis(ax["width"], r"$w$", f1, f2, 0.5, 2.5)

    ax["x"] = fig.add_subplot(gs[3])
    config_axis(ax["x"], r"$x$", f1, f2, -9, 9)

    ax["y"] = fig.add_subplot(gs[4])
    config_axis(ax["y"], r"$y$", f1, f2, -9, 9)

    ax["background"] = fig.add_subplot(gs[5])
    config_axis(ax["background"], r"$b$", f1, f2, 0, 500, True)
    ax["background"].set_xlabel("Time (frame)")

    for p in ["height", "width", "x", "y"]:
        for k in range(model.K):
            (item[f"{p}_{k}_mean"],) = ax[p].plot(
                torch.arange(0, model.data.F),
                model.params[p]["Mean"][k, n],
                "o-",
                ms=3,
                lw=1,
                color=f"C{k}",
            )
            item[f"{p}_{k}_fill"] = ax[p].fill_between(
                torch.arange(0, model.data.F),
                model.params[p]["LL"][k, n],
                model.params[p]["UL"][k, n],
                alpha=0.3,
                color=f"C{k}",
            )
    (item["background_mean"],) = ax["background"].plot(
        torch.arange(0, model.data.F),
        model.params["background"]["Mean"][n],
        "o-",
        ms=3,
        lw=1,
        color="k",
    )
    item["background_fill"] = ax["background"].fill_between(
        torch.arange(0, model.data.F),
        model.params["background"]["LL"][n],
        model.params["background"]["UL"][n],
        alpha=0.3,
        color="k",
    )
    plt.show()
    return model, fig, item, ax


@app.command()
def log():
    """Show logging info."""
    import pydoc

    global DEFAULTS
    cd = DEFAULTS["cd"]

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

    Initialize Tapqir in the working directory. Initializing a Tapqir workspace creates
    a ``.tapqir`` sub-directory for storing ``config.yaml`` file, ``loginfo`` file,
    and files that are created by commands such as ``tapqir fit``.
    """

    global DEFAULTS
    # set working directory
    DEFAULTS["cd"] = cd

    TAPQIR_PATH = cd / ".tapqir"

    if not TAPQIR_PATH.is_dir():
        # initialize directory
        TAPQIR_PATH.mkdir()
    CONFIG_PATH = TAPQIR_PATH / "config.yaml"
    if not CONFIG_PATH.is_file():
        with open(CONFIG_PATH, "w") as cfg_file:
            DEFAULTS["P"] = 14
            DEFAULTS["nbatch-size"] = 10
            DEFAULTS["fbatch-size"] = 512
            DEFAULTS["learning-rate"] = 0.005
            DEFAULTS["num-channels"] = 1
            DEFAULTS["cuda"] = "true"
            yaml.dump(
                {key: value for key, value in DEFAULTS.items() if key != "cd"},
                cfg_file,
                sort_keys=False,
            )

        typer.echo(
            (
                f"Initialized Tapqir at {TAPQIR_PATH}.\n"
                "{yellow}---------------------------------------------------------------{nc}\n"
                f"- Checkout the documentation: {format_link('https://tapqir.readthedocs.io/')}\n"
                f"- Get help on our forum: {format_link('https://github.com/gelles-brandeis/tapqir/discussions')}"
            ).format(yellow=colorama.Fore.YELLOW, nc=colorama.Fore.RESET)
        )

    # read defaults from config file
    with open(CONFIG_PATH, "r") as cfg_file:
        cfg_defaults = yaml.safe_load(cfg_file) or {}
        DEFAULTS.update(cfg_defaults)
        typer.echo(f"Configuration options are read from {CONFIG_PATH}.")

    # create logger
    logger = logging.getLogger("tapqir")

    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(levelname)s - %(message)s",
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(cd / ".tapqir" / "loginfo")
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %I:%M %p",
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)


# click object is required to generate docs with sphinx-click
typer_click_object = typer.main.get_command(app)

if __name__ == "__main__":
    app()
