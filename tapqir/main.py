# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: Apache-2.0

import logging
import sys
from collections import defaultdict
from enum import Enum
from functools import partial
from pathlib import Path
from typing import List, Optional

import click
import colorama
import matplotlib.pyplot as plt
import torch
import typer
import yaml
from tqdm import tqdm

from tapqir import __version__
from tapqir.distributions.util import gaussian_spots
from tapqir.exceptions import CudaOutOfMemoryError, TapqirFileNotFoundError

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
        min=5,
        max=50,
        help="AOI image size - number of pixels along the axis",
        prompt="AOI image size",
    ),
    frame_range: bool = typer.Option(
        partial(get_default, "frame-range"),
        help="Specify frame range.",
        prompt="Specify frame range?",
        callback=deactivate_prompts,
    ),
    frame_start: Optional[int] = typer.Option(
        partial(get_default, "frame-start"),
        min=0,
        help="Starting frame.",
        prompt="Starting frame",
    ),
    frame_end: Optional[int] = typer.Option(
        partial(get_default, "frame-end"),
        min=1,
        help="Ending frame.",
        prompt="Ending frame",
    ),
    use_offtarget: bool = typer.Option(
        partial(get_default, "use-offtarget"),
        help="Use off-target AOI locations.",
        prompt="Use off-target AOI locations?",
    ),
    num_channels: int = typer.Option(
        partial(get_default, "num-channels"),
        "--num-channels",
        "-C",
        min=1,
        help="Number of color channels",
        prompt="Number of color channels",
    ),
    name: Optional[List[str]] = typer.Option(None),
    glimpse_folder: Optional[List[Path]] = typer.Option(
        None,
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    driftlist: Optional[List[Path]] = typer.Option(
        None,
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
    ),
    ontarget_aoiinfo: Optional[List[Path]] = typer.Option(
        None,
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
    ),
    offtarget_aoiinfo: Optional[List[Path]] = typer.Option(
        None,
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
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

    logger = logging.getLogger("tapqir")

    global DEFAULTS
    cd = DEFAULTS["cd"]

    if progress_bar is None:
        progress_bar = tqdm

    # read parameter input values
    DEFAULTS["dataset"] = dataset
    DEFAULTS["P"] = P
    DEFAULTS["frame-range"] = frame_range
    DEFAULTS["frame-start"] = frame_start
    DEFAULTS["frame-end"] = frame_end
    DEFAULTS["use-offtarget"] = use_offtarget
    DEFAULTS["num-channels"] = num_channels
    DEFAULTS["labels"] = labels

    # parameter names
    param_names = [
        "name",
        "glimpse-folder",
        "driftlist",
        "ontarget-aoiinfo",
        "offtarget-aoiinfo",
    ]
    if labels:
        param_names += ["ontarget-labels", "offtarget-labels"]

    # parameter flags
    param_flag = {}
    param_flag["name"] = name
    param_flag["glimpse-folder"] = glimpse_folder
    param_flag["driftlist"] = driftlist
    param_flag["ontarget-aoiinfo"] = ontarget_aoiinfo
    param_flag["offtarget-aoiinfo"] = offtarget_aoiinfo

    # parameter prompts
    param_prompt = {}
    param_prompt["name"] = "Channel name"
    param_prompt["glimpse-folder"] = "Header/glimpse folder"
    param_prompt["driftlist"] = "Driftlist file"
    param_prompt["ontarget-aoiinfo"] = "Target molecule locations file"
    param_prompt["offtarget-aoiinfo"] = "Off-target control locations file"
    param_prompt["ontarget-labels"] = "On-target AOI binding labels"
    param_prompt["offtarget-labels"] = "Off-target AOI binding labels"

    # parameter types
    param_type = {}
    param_type["name"] = str
    param_type["glimpse-folder"] = click.Path(
        exists=True, file_okay=False, dir_okay=True, resolve_path=True
    )
    param_type["driftlist"] = click.Path(
        exists=True, file_okay=True, dir_okay=False, resolve_path=True
    )
    param_type["ontarget-aoiinfo"] = click.Path(
        exists=True, file_okay=True, dir_okay=False, resolve_path=True
    )
    param_type["offtarget-aoiinfo"] = click.Path(
        exists=True, file_okay=True, dir_okay=False, resolve_path=True
    )
    param_type["ontarget-labels"] = click.Path(
        exists=True, file_okay=True, dir_okay=False, resolve_path=True
    )
    param_type["offtarget-labels"] = click.Path(
        exists=True, file_okay=True, dir_okay=False, resolve_path=True
    )

    for c in range(num_channels):
        if len(DEFAULTS["channels"]) < c + 1:
            DEFAULTS["channels"].append({})
        if not no_input:
            typer.echo(f"\nINPUTS FOR CHANNEL #{c}\n")
        for param_name in param_names:
            if param_name == "offtarget-aoiinfo" and not use_offtarget:
                continue
            if c >= len(param_flag[param_name]):
                if not no_input:
                    DEFAULTS["channels"][c][param_name] = typer.prompt(
                        param_prompt[param_name],
                        default=DEFAULTS["channels"][c][param_name],
                        type=param_type[param_name],
                    )
            else:
                DEFAULTS["channels"][c][param_name] = param_flag[param_name][c]

    DEFAULTS = dict(DEFAULTS)
    for c in range(num_channels):
        DEFAULTS["channels"][c] = dict(DEFAULTS["channels"][c])

    if overwrite:
        with open(cd / ".tapqir" / "config.yaml", "w") as cfg_file:
            yaml.dump(
                {key: value for key, value in DEFAULTS.items() if key != "cd"},
                cfg_file,
                sort_keys=False,
            )

    logger.info("Extracting AOIs ...")
    read_glimpse(
        path=cd,
        progress_bar=progress_bar,
        **DEFAULTS,
    )
    logger.info("Extracting AOIs: Done")

    return 0


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
        partial(get_default, "matlab"),
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

    logger = logging.getLogger("tapqir")

    settings = {}
    settings["S"] = 1
    settings["K"] = k_max
    settings["channels"] = channels
    settings["device"] = "cuda" if cuda else "cpu"
    settings["dtype"] = "double"
    settings["use_pykeops"] = pykeops
    # priors settings
    settings["background_mean_std"] = DEFAULTS["background_mean_std"]
    settings["background_std_std"] = DEFAULTS["background_std_std"]
    settings["lamda_rate"] = DEFAULTS["lamda_rate"]
    settings["height_std"] = DEFAULTS["height_std"]
    settings["width_min"] = DEFAULTS["width_min"]
    settings["width_max"] = DEFAULTS["width_max"]
    settings["proximity_rate"] = DEFAULTS["proximity_rate"]
    settings["gain_std"] = DEFAULTS["gain_std"]

    if overwrite:
        DEFAULTS["cuda"] = cuda
        DEFAULTS["nbatch-size"] = nbatch_size
        DEFAULTS["fbatch-size"] = fbatch_size
        DEFAULTS["learning-rate"] = learning_rate
        DEFAULTS["matlab"] = matlab
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

        logger.info("Fitting the data ...")
        model = models[model](**settings)
        try:
            model.load(cd)
        except TapqirFileNotFoundError as err:
            logger.exception(f"Failed to load {err.name} file")
            return 1

        model.init(learning_rate, nbatch_size, fbatch_size)
        try:
            model.run(num_iter, progress_bar=progress_bar)
        except CudaOutOfMemoryError:
            logger.exception("Failed to fit the data")
            return 1
        logger.info("Fitting the data: Done")

        logger.info("Computing stats ...")
        try:
            model.compute_stats(save_matlab=matlab)
        except CudaOutOfMemoryError:
            logger.exception("Failed to compute stats")
            return 1
        logger.info("Computing stats: Done")

    return 0


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
        partial(get_default, "matlab"),
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

    logger = logging.getLogger("tapqir")

    global DEFAULTS
    cd = DEFAULTS["cd"]

    dtype = "double"
    device = "cuda" if cuda else "cpu"
    backend = "funsor" if funsor else "pyro"

    settings = {}
    settings["S"] = 1
    settings["channels"] = channels
    settings["device"] = device
    settings["dtype"] = dtype

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

        logger.info("Computing stats ...")
        model = models[model](**settings)
        try:
            model.load(cd)
        except TapqirFileNotFoundError:
            logger.exception("Failed to load data file")
            return 1
        model.load_checkpoint(param_only=True)
        model.nbatch_size = nbatch_size
        model.fbatch_size = fbatch_size
        try:
            model.compute_stats(save_matlab=matlab)
        except CudaOutOfMemoryError:
            logger.exception("Failed to compute stats")
            return 1
        logger.info("Computing stats: Done")

    return 0


def config_axis(ax, label, f1, f2, ymin=None, ymax=None, xticklabels=False):
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
    ax.set_xlim(f1 - 0.5, f2 - 0.5)
    if (ymin is not None) and (ymax is not None):
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
    show_fov: bool = True,
    gui=None,
):
    from tapqir.imscroll import GlimpseDataset
    from tapqir.models import models

    logger = logging.getLogger("tapqir")

    global DEFAULTS
    cd = DEFAULTS["cd"]

    model = models[model](1, 2, channels, "cpu", "float")
    try:
        model.load(cd, data_only=False)
    except TapqirFileNotFoundError as err:
        logger.exception(f"Failed to load {err.name} file")
        return 1
    if f1 is None:
        f1 = 0
    if f2 is None:
        f2 = f1 + 15
    c = model.cdx
    color = "C2" if model.data.mask[n] else "C7"

    width, dpi = 6.25, 100
    height = 10 if show_fov else 6.25
    fig = plt.figure(figsize=(width, height), dpi=dpi)
    gs = fig.add_gridspec(
        nrows=10,
        ncols=15,
        top=0.96,
        bottom=0.39 if show_fov else 0.02,
        left=0.1,
        right=0.98,
        hspace=0.1,
        height_ratios=[0.9, 0.9, 1, 1, 1, 1, 1, 1, 1, 1],
    )
    if show_fov:
        gs2 = fig.add_gridspec(
            nrows=1,
            ncols=1,
            top=0.32,
            bottom=0.02,
            left=0.1,
            right=0.98,
        )
    ax = {}
    item = {}

    frames = torch.arange(f1, f2)
    img_ideal = (
        model.data.offset.mean
        + model.params["background"]["Mean"][n, frames, None, None]
    )
    gaussian = gaussian_spots(
        model.params["height"]["Mean"][:, n, frames],
        model.params["width"]["Mean"][:, n, frames],
        model.params["x"]["Mean"][:, n, frames],
        model.params["y"]["Mean"][:, n, frames],
        model.data.xy[n, frames, c],
        model.data.P,
    )
    img_ideal = img_ideal + gaussian.sum(-4)
    for f in range(15):
        ax[f"image_{f}"] = fig.add_subplot(gs[0, f])
        item[f"image_{f}"] = ax[f"image_{f}"].imshow(
            model.data.images[n, f, c].numpy(),
            vmin=model.data.vmin[model.cdx] - 50,
            vmax=model.data.vmax[model.cdx] + 50,
            cmap="gray",
        )
        ax[f"image_{f}"].set_title(rf"${f}$", fontsize=9)
        ax[f"image_{f}"].axis("off")
        ax[f"ideal_{f}"] = fig.add_subplot(gs[1, f])
        item[f"ideal_{f}"] = ax[f"ideal_{f}"].imshow(
            img_ideal[f].numpy(),
            vmin=model.data.vmin[model.cdx] - 50,
            vmax=model.data.vmax[model.cdx] + 50,
            cmap="gray",
        )
        ax[f"ideal_{f}"].axis("off")

    ax["z_map"] = fig.add_subplot(gs[2, :])
    config_axis(ax["z_map"], r"$z$", f1, f2, -0.1, model.S + 0.1)
    (item["z_map"],) = ax["z_map"].plot(
        torch.arange(0, model.data.F),
        model.params["z_map"][n],
        "-",
        lw=1,
        color="k",
    )

    ax["p_specific"] = fig.add_subplot(gs[3, :])
    config_axis(ax["p_specific"], r"$p(\mathsf{specific})$", f1, f2, -0.1, 1.1)
    (item["p_specific"],) = ax["p_specific"].plot(
        torch.arange(0, model.data.F),
        model.params["p_specific"][n],
        "o-",
        ms=3,
        lw=1,
        color=color,
    )

    ax["height"] = fig.add_subplot(gs[4, :])
    config_axis(
        ax["height"],
        r"$h$",
        f1,
        f2,
        model.params["height"]["vmin"],
        model.params["height"]["vmax"],
    )

    ax["width"] = fig.add_subplot(gs[5, :])
    config_axis(
        ax["width"],
        r"$w$",
        f1,
        f2,
        model.params["width"]["vmin"],
        model.params["width"]["vmax"],
    )

    ax["x"] = fig.add_subplot(gs[6, :])
    config_axis(
        ax["x"], r"$x$", f1, f2, model.params["x"]["vmin"], model.params["x"]["vmax"]
    )

    ax["y"] = fig.add_subplot(gs[7, :])
    config_axis(
        ax["y"], r"$y$", f1, f2, model.params["y"]["vmin"], model.params["y"]["vmax"]
    )

    ax["background"] = fig.add_subplot(gs[8, :])
    config_axis(
        ax["background"],
        r"$b$",
        f1,
        f2,
        model.params["background"]["vmin"],
        model.params["background"]["vmax"],
    )

    ax["chi2"] = fig.add_subplot(gs[9, :])
    config_axis(
        ax["chi2"],
        r"$\chi^2 \mathsf{test}$",
        f1,
        f2,
        model.params["chi2"]["vmin"],
        model.params["chi2"]["vmax"],
        True,
    )
    ax["chi2"].set_xlabel("Time (frame)")

    theta_mask = model.params["theta_probs"][:, n] > 0.5
    j_mask = (model.params["m_probs"][:, n] > 0.5) & ~theta_mask
    for p in ["height", "width", "x", "y"]:
        # target-nonspecific spots
        for k in range(model.K):
            f_mask = j_mask[k]
            mean = model.params[p]["Mean"][k, n] * f_mask
            ll = model.params[p]["LL"][k, n] * f_mask
            ul = model.params[p]["UL"][k, n] * f_mask
            (item[f"{p}_nonspecific{k}_mean"],) = ax[p].plot(
                torch.arange(0, model.data.F)[f_mask],
                mean[f_mask],
                "o",
                ms=2,
                lw=1,
                color=f"C{k}",
            )
            item[f"{p}_nonspecific{k}_fill"] = ax[p].fill_between(
                torch.arange(0, model.data.F),
                ll,
                ul,
                where=f_mask,
                alpha=0.3,
                color=f"C{k}",
            )
        # target-specific spots
        f_mask = theta_mask.sum(0).bool()
        mean = (model.params[p]["Mean"][:, n] * theta_mask).sum(0)
        ll = (model.params[p]["LL"][:, n] * theta_mask).sum(0)
        ul = (model.params[p]["UL"][:, n] * theta_mask).sum(0)
        (item[f"{p}_specific_mean"],) = ax[p].plot(
            torch.arange(0, model.data.F)
            if p == "height"
            else torch.arange(0, model.data.F)[f_mask],
            mean if p == "height" else mean[f_mask],
            "-" if p == "height" else "o",
            ms=2,
            color=color,
        )
        item[f"{p}_specific_fill"] = ax[p].fill_between(
            torch.arange(0, model.data.F),
            ll,
            ul,
            where=f_mask,
            alpha=0.3,
            color=color,
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

    (item["chi2"],) = ax["chi2"].plot(
        torch.arange(0, model.data.F),
        model.params["chi2"]["values"][n],
        "-",
        lw=1,
        color="k",
    )

    if show_fov:
        ax["glimpse"] = fig.add_subplot(gs2[0])
        P = DEFAULTS.pop("P")
        channels = DEFAULTS.pop("channels")
        fov = GlimpseDataset(**DEFAULTS, **channels[c], c=c)
        fov.plot(
            fov.dtypes,
            P,
            n=0,
            f=0,
            save=False,
            ax=ax["glimpse"],
            item=item,
        )
    else:
        fov = None

    if not gui:
        plt.show()
    return model, fig, item, ax, fov


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

    from tapqir.logger import ColorFormatter

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
            # model settings
            DEFAULTS["P"] = 14
            DEFAULTS["nbatch-size"] = 10
            DEFAULTS["fbatch-size"] = 512
            DEFAULTS["learning-rate"] = 0.005
            DEFAULTS["num-channels"] = 1
            DEFAULTS["cuda"] = True
            DEFAULTS["matlab"] = False
            # priors settings
            DEFAULTS["background_mean_std"] = 1000
            DEFAULTS["background_std_std"] = 100
            DEFAULTS["lamda_rate"] = 1
            DEFAULTS["height_std"] = 10000
            DEFAULTS["width_min"] = 0.75
            DEFAULTS["width_max"] = 2.25
            DEFAULTS["proximity_rate"] = 1
            DEFAULTS["gain_std"] = 50
            # offset settings
            DEFAULTS["offset_x"] = 10
            DEFAULTS["offset_y"] = 10
            DEFAULTS["offset_P"] = 30
            DEFAULTS["bin_size"] = 1
            # save config file
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
                f"- Get help on our forum: {format_link('https://github.com/gelles-brandeis/tapqir/discussions')}\n"
            ).format(yellow=colorama.Fore.YELLOW, nc=colorama.Fore.RESET)
        )

    # create logger
    logger = logging.getLogger("tapqir")

    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(ColorFormatter())
    logger.addHandler(ch)

    fh = logging.FileHandler(cd / ".tapqir" / "loginfo")
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %I:%M %p",
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # read defaults from config file
    with open(CONFIG_PATH, "r") as cfg_file:
        cfg_defaults = yaml.safe_load(cfg_file) or {}
        DEFAULTS.update(cfg_defaults)
        logger.info(f"Configuration options are read from {CONFIG_PATH}.")


# click object is required to generate docs with sphinx-click
typer_click_object = typer.main.get_command(app)

if __name__ == "__main__":
    app()
