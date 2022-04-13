# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from collections import defaultdict
from functools import partial, singledispatch
from pathlib import Path

import ipywidgets as widgets
import matplotlib.pyplot as plt
import torch
from ipyevents import Event
from ipyfilechooser import FileChooser
from IPython.display import display
from tensorboard import notebook
from tqdm import tqdm_notebook
from traitlets.utils.bunch import Bunch

from tapqir.distributions.util import gaussian_spots
from tapqir.main import fit, glimpse, log, main, show

logger = logging.getLogger("tapqir")

# disable pyplot keyboard shortcuts
plt.rcParams["keymap.zoom"].remove("o")
plt.rcParams["keymap.home"].remove("h")
plt.rcParams["keymap.yscale"].remove("l")
plt.rcParams["keymap.xscale"].remove("k")


@singledispatch
def get_value(x):
    return x


@get_value.register
def _(x: Bunch):
    return x.new


@get_value.register
def _(x: widgets.Widget):
    return x.value


@singledispatch
def get_path(x):
    return x


@get_path.register
def _(x: str):
    return Path(x)


@get_path.register
def _(x: FileChooser):
    return Path(x.selected_path)


def initUI(DEFAULTS):
    """
    Main layout.

    - Set working directory.
    - Commands tabs.
    - Output.
    """

    layout = widgets.VBox(layout={"width": "850px", "border": "2px solid black"})

    # widgets
    cd = FileChooser(".", title="Select working directory")
    cd.show_only_dirs = True

    out = widgets.Output()

    # layout
    layout.children = [cd, out]
    # callbacks
    cd.register_callback(
        partial(
            cdCmd,
            DEFAULTS=DEFAULTS,
            out=out,
            layout=layout,
        )
    )
    return layout


def cdCmd(path, DEFAULTS, out, layout):
    """
    Set working directory and load default parameters (main).
    """
    import sys
    from platform import uname

    IN_COLAB = "google.colab" in sys.modules
    IN_WSL2 = "microsoft-standard" in uname().release
    IN_LINUX = not IN_COLAB and not IN_WSL2

    path = get_path(path)
    with out:
        main(cd=path)
        logger.info("Loading configuration data ...")

    # Tabs
    tab = widgets.Tab()
    tab.children = [glimpseUI(out, DEFAULTS), fitUI(out, DEFAULTS)]
    if IN_LINUX:
        tensorboard = widgets.Output()
        tab.children = tab.children + (showUI(out, DEFAULTS), tensorboard)
    elif IN_WSL2:
        tensorboard = None
        tensorboard_info = widgets.Label(
            value=(
                'Run "tensorboard --logdir=<working directory>" in the terminal'
                ' and then open "localhost:6006" in the browser'
            )
        )
        tab.children = tab.children + (showUI(out, DEFAULTS), tensorboard_info)
    elif IN_COLAB:
        tensorboard = None
        tab.children = tab.children + (
            widgets.Label(value="Disabled in Colab"),
            widgets.Label(value="Disabled in Colab"),
        )
    tab.children = tab.children + (logUI(out),)
    tab.set_title(0, "Extract AOIs")
    tab.set_title(1, "Fit the data")
    tab.set_title(2, "View results")
    tab.set_title(3, "Tensorboard")
    tab.set_title(4, "View logs")

    if tensorboard is not None:
        with tensorboard:
            notebook.start(f"--logdir '{path}'")
            notebook.display(height=1000)

    # insert tabs into GUI
    wd = widgets.Label(value=f"Working directory: {path}")
    layout.children = (wd, tab) + layout.children[1:]

    with out:
        logger.info("Loading configuration data: Done")

    out.clear_output(wait=True)


def glimpseUI(out, DEFAULTS):
    """
    Glimpse command layout.
    """
    # Extract AOIs
    layout = widgets.VBox()
    # Dataset name
    datasetName = widgets.Text(
        value=DEFAULTS["dataset"],
        description="Dataset name",
        style={"description_width": "initial"},
    )
    # AOI image size
    aoiSize = widgets.IntText(
        value=DEFAULTS["P"],
        min=5,
        max=50,
        description="AOI image size",
        style={"description_width": "initial"},
    )
    # Specify frame range?
    specifyFrame = widgets.Checkbox(
        value=DEFAULTS["frame-range"], description="Specify frame range?", indent=False
    )
    # First frame
    firstFrame = widgets.IntText(
        value=DEFAULTS["frame-start"],
        disabled=not DEFAULTS["frame-range"],
        description="Starting frame",
        style={"description_width": "initial"},
    )
    # Last frame
    lastFrame = widgets.IntText(
        value=DEFAULTS["frame-end"],
        disabled=not DEFAULTS["frame-range"],
        description="Ending frame",
        style={"description_width": "initial"},
    )
    # Specify frame range?
    useOfftarget = widgets.Checkbox(
        value=DEFAULTS["use-offtarget"],
        description="Use off-target AOI locations?",
        indent=False,
    )
    # Number of channels
    numChannels = widgets.BoundedIntText(
        value=DEFAULTS["num-channels"],
        min=1,
        max=4,
        step=1,
        description="Number of color channels",
        style={"description_width": "initial"},
    )
    # Channel tabs
    channelTabs = widgets.Tab()
    channelUI(numChannels.value, channelTabs, useOfftarget, DEFAULTS)
    # extract AOIs
    extractAOIs = widgets.Button(description="Extract AOIs")
    # Layout
    layout.children = [
        datasetName,
        aoiSize,
        specifyFrame,
        firstFrame,
        lastFrame,
        useOfftarget,
        numChannels,
        channelTabs,
        extractAOIs,
    ]
    # Callbacks
    numChannels.observe(
        partial(
            channelUI,
            channelTabs=channelTabs,
            useOfftarget=useOfftarget,
            DEFAULTS=DEFAULTS,
        ),
        names="value",
    )
    useOfftarget.observe(
        partial(toggleUseOffTarget, channelTabs=channelTabs, DEFAULTS=DEFAULTS),
        names="value",
    )
    extractAOIs.on_click(partial(glimpseCmd, layout=layout, out=out))
    specifyFrame.observe(
        partial(toggleWidgets, widgets=(firstFrame, lastFrame)), names="value"
    )
    return layout


def toggleUseOffTarget(checked, channelTabs, DEFAULTS):
    checked = get_value(checked)
    if checked:
        for c, tab in enumerate(channelTabs.children):
            offtargetLayout = FileChooser(
                ".", title="Off-target control locations file"
            )
            if DEFAULTS["channels"][c]["offtarget-aoiinfo"] is not None:
                offtargetLayout.reset(
                    path=Path(DEFAULTS["channels"][c]["offtarget-aoiinfo"]).parent,
                    filename=Path(DEFAULTS["channels"][c]["offtarget-aoiinfo"]).name,
                )
                offtargetLayout._apply_selection()
            tab.children = tab.children + (offtargetLayout,)
    else:
        for tab in channelTabs.children:
            tab.children = tab.children[:-1]


def glimpseCmd(b, layout, out):
    channelTabs = layout.children[7]
    with out:
        glimpse(
            dataset=layout.children[0].value,
            P=layout.children[1].value,
            frame_range=layout.children[2].value,
            frame_start=layout.children[3].value,
            frame_end=layout.children[4].value,
            use_offtarget=layout.children[5].value,
            num_channels=layout.children[6].value,
            name=[c.children[0].value for c in channelTabs.children],
            glimpse_folder=[c.children[1].value for c in channelTabs.children],
            ontarget_aoiinfo=[c.children[3].value for c in channelTabs.children],
            offtarget_aoiinfo=[c.children[4].value for c in channelTabs.children],
            driftlist=[c.children[2].value for c in channelTabs.children],
            no_input=True,
            progress_bar=tqdm_notebook,
            labels=False,
        )

    out.clear_output(wait=True)


def channelUI(C, channelTabs, useOfftarget, DEFAULTS):
    C = get_value(C)
    currentC = len(channelTabs.children)
    for c in range(max(currentC, C)):
        if c < C and c < currentC:
            continue
        elif c < C and c >= currentC:
            if len(DEFAULTS["channels"]) < c + 1:
                DEFAULTS["channels"].append(defaultdict(lambda: None))
            layout = widgets.VBox()
            nameLayout = widgets.Text(
                value=DEFAULTS["channels"][c]["name"],
                description="Channel name",
                style={"description_width": "initial"},
            )
            headerLayout = FileChooser(".", title="Header/glimpse folder")
            if DEFAULTS["channels"][c]["glimpse-folder"] is not None:
                headerLayout.reset(path=Path(DEFAULTS["channels"][c]["glimpse-folder"]))
                headerLayout._apply_selection()
            driftlistLayout = FileChooser(".", title="Driftlist file")
            if DEFAULTS["channels"][c]["driftlist"] is not None:
                driftlistLayout.reset(
                    path=Path(DEFAULTS["channels"][c]["driftlist"]).parent,
                    filename=Path(DEFAULTS["channels"][c]["driftlist"]).name,
                )
                driftlistLayout._apply_selection()
            ontargetLayout = FileChooser(".", title="Target molecule locations file")
            if DEFAULTS["channels"][c]["ontarget-aoiinfo"] is not None:
                ontargetLayout.reset(
                    path=Path(DEFAULTS["channels"][c]["ontarget-aoiinfo"]).parent,
                    filename=Path(DEFAULTS["channels"][c]["ontarget-aoiinfo"]).name,
                )
                ontargetLayout._apply_selection()
            layout.children = [
                nameLayout,
                headerLayout,
                driftlistLayout,
                ontargetLayout,
            ]
            if useOfftarget.value:
                offtargetLayout = FileChooser(
                    ".", title="Off-target control locations file"
                )
                if DEFAULTS["channels"][c]["offtarget-aoiinfo"] is not None:
                    offtargetLayout.reset(
                        path=Path(DEFAULTS["channels"][c]["offtarget-aoiinfo"]).parent,
                        filename=Path(
                            DEFAULTS["channels"][c]["offtarget-aoiinfo"]
                        ).name,
                    )
                    offtargetLayout._apply_selection()
                layout.children = layout.children + (offtargetLayout,)
            channelTabs.children = channelTabs.children + (layout,)
        channelTabs.set_title(c, f"Channel #{c}")
    channelTabs.children = channelTabs.children[:C]


def fitUI(out, DEFAULTS):
    # Fit the data
    layout = widgets.VBox()
    # Model
    modelSelect = widgets.Dropdown(
        description="Tapqir model",
        value="cosmos",
        options=["cosmos"],
        style={"description_width": "initial"},
    )
    # Channel numbers
    channelNumber = widgets.Dropdown(
        description="Channel numbers",
        value="0",
        options=[str(c) for c in range(DEFAULTS["num-channels"])],
        style={"description_width": "initial"},
    )
    # Run computations on GPU?
    useGpu = widgets.Checkbox(
        value=DEFAULTS["cuda"],
        description="Run computations on GPU?",
        style={"description_width": "initial"},
    )
    # AOI batch size
    aoiBatch = widgets.IntText(
        value=DEFAULTS["nbatch-size"],
        description="AOI batch size",
        style={"description_width": "initial"},
    )
    # Frame batch size
    frameBatch = widgets.IntText(
        value=DEFAULTS["fbatch-size"],
        description="Frame batch size",
        style={"description_width": "initial"},
    )
    # Learning rate
    learningRate = widgets.FloatText(
        value=DEFAULTS["learning-rate"],
        description="Learning rate",
        style={"description_width": "initial"},
    )
    # Number of iterations
    iterationNumber = widgets.IntText(
        description="Number of iterations",
        style={"description_width": "initial"},
    )
    # Save parameters in matlab format?
    saveMatlab = widgets.Checkbox(
        value=DEFAULTS["matlab"],
        description="Save parameters in matlab format?",
        style={"description_width": "initial"},
    )
    # Fit the data
    fitData = widgets.Button(description="Fit the data")
    # Layout
    layout.children = [
        modelSelect,
        channelNumber,
        useGpu,
        aoiBatch,
        frameBatch,
        learningRate,
        iterationNumber,
        saveMatlab,
        fitData,
    ]
    # Callbacks
    fitData.on_click(partial(fitCmd, layout=layout, out=out))
    return layout


def fitCmd(b, layout, out):
    with out:
        fit(
            model=layout.children[0].value,
            channels=[int(layout.children[1].value)],
            cuda=layout.children[2].value,
            nbatch_size=layout.children[3].value,
            fbatch_size=layout.children[4].value,
            learning_rate=layout.children[5].value,
            num_iter=layout.children[6].value,
            matlab=layout.children[7].value,
            k_max=2,
            funsor=False,
            pykeops=True,
            no_input=True,
            progress_bar=tqdm_notebook,
        )

    out.clear_output(wait=True)


def showUI(out, DEFAULTS):
    # View results
    layout = widgets.VBox()
    # Model
    modelSelect = widgets.Dropdown(
        description="Tapqir model",
        value="cosmos",
        options=["cosmos"],
        style={"description_width": "initial"},
    )
    # Channel numbers
    channelNumber = widgets.Dropdown(
        description="Channel numbers",
        value="0",
        options=[str(c) for c in range(DEFAULTS["num-channels"])],
        style={"description_width": "initial"},
    )
    # Fit the data
    showParams = widgets.Button(
        description="Load results", layout=widgets.Layout(margin="10px 0px 10px 0px")
    )
    controls = widgets.HBox()
    view = widgets.Output()
    params = widgets.VBox(children=[controls, view])
    summary = widgets.Output()
    sum_view = widgets.Accordion(children=[params, summary])
    sum_view.set_title(0, "AOI images and parameters")
    sum_view.set_title(1, "Summary statistics")
    sum_view.selected_index = 0
    # Layout
    layout.children = [modelSelect, channelNumber, showParams, sum_view]
    # Callbacks
    showParams.on_click(partial(showCmd, layout=layout, out=out))
    return layout


def showCmd(b, layout, out):
    params, summary = layout.children[-1].children
    controls, view = params.children
    with out:
        logger.info("Loading results ...")
        show_ret = show(
            model=layout.children[0].value,
            channels=[int(layout.children[1].value)],
            n=0,
            f1=None,
            f2=None,
            gui=True,
        )
        if show_ret == 1:
            out.clear_output(wait=True)
            return
        model, fig, item, ax = show_ret
    with view:
        plt.show()
    with summary:
        display(model.summary)
    n = widgets.BoundedIntText(
        value=0,
        min=0,
        max=model.data.Nt - 1,
        description=f"AOI (0-{model.data.Nt-1})",
        style={"description_width": "initial"},
        layout={"width": "150px"},
    )
    n_counter = widgets.BoundedIntText(
        min=0,
        max=model.data.Nt - 1,
    )
    f1_text = widgets.BoundedIntText(
        value=0,
        min=0,
        max=model.data.F - 15,
        step=1,
        description=f"Frame (0-{model.data.F-1})",
        style={"description_width": "initial"},
        layout={"width": "300px"},
    )
    f1_counter = widgets.BoundedIntText(
        min=0,
        max=model.data.F - 15,
    )
    f1_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=model.data.F - 15,
        step=1,
        continuous_update=True,
        readout=False,
        readout_format="d",
        layout={"width": "210px"},
    )
    f1_box = widgets.VBox(layout=widgets.Layout(width="310px"))
    f1_incr = widgets.Button(description="+15", layout=widgets.Layout(width="45px"))
    f1_decr = widgets.Button(description="-15", layout=widgets.Layout(width="45px"))
    f1_controls = widgets.HBox(
        children=[f1_decr, f1_slider, f1_incr],
        layout=widgets.Layout(width="305px"),
    )
    f1_box.children = [f1_text, f1_controls]
    widgets.jslink((f1_slider, "value"), (f1_text, "value"))
    widgets.jslink((f1_text, "value"), (f1_counter, "value"))
    widgets.jslink((n, "value"), (n_counter, "value"))
    zoom = widgets.Checkbox(
        value=False,
        description="Zoom out frames ['z']",
        indent=False,
        layout={"width": "240px"},
    )
    labels = widgets.Checkbox(
        value=False,
        description="Show labels",
        indent=False,
        layout={"width": "240px"},
    )
    targets = widgets.Checkbox(
        value=False,
        description="Show target location ['o']",
        indent=False,
        layout={"width": "240px"},
    )
    nonspecific = widgets.Checkbox(
        value=True,
        description="Show non-specific spots ['n']",
        indent=False,
        layout={"width": "240px"},
    )
    checkboxes = widgets.VBox(layout=widgets.Layout(width="250px"))
    if model.data.labels is None:
        checkboxes.children = [zoom, targets, nonspecific]
    else:
        checkboxes.children = [zoom, targets, nonspecific, labels]
    controls.children = [n, f1_box, checkboxes]
    n.observe(
        partial(
            updateParams,
            f1=f1_slider,
            model=model,
            fig=fig,
            item=item,
            ax=ax,
            targets=targets,
            nonspecific=nonspecific,
            labels=labels,
        ),
        names="value",
    )
    f1_slider.observe(
        partial(
            updateRange,
            n=n,
            model=model,
            fig=fig,
            item=item,
            ax=ax,
            zoom=zoom,
            targets=targets,
        ),
        names="value",
    )
    f1_incr.on_click(partial(incrementRange, x=15, counter=f1_counter))
    f1_decr.on_click(partial(incrementRange, x=-15, counter=f1_counter))
    zoom.observe(
        partial(zoomOut, f1=f1_slider, model=model, fig=fig, item=item, ax=ax),
        names="value",
    )
    labels.observe(
        partial(showLabels, n=n_counter, model=model, item=item, ax=ax),
        names="value",
    )
    targets.observe(
        partial(showTargets, n=n_counter, f1=f1_slider, model=model, item=item, ax=ax),
        names="value",
    )
    nonspecific.observe(
        partial(showNonspecific, n=n_counter, model=model, item=item, ax=ax),
        names="value",
    )
    # mouse click UI
    fig.canvas.mpl_connect(
        "button_release_event",
        partial(onFrameClick, counter=f1_counter),
    )
    # key press UI
    d = Event(source=layout, watched_events=["keyup"], prevent_default_action=True)
    d.on_dom_event(
        partial(
            onKeyPress,
            n=n_counter,
            f1=f1_counter,
            zoom=zoom,
            targets=targets,
            nonspecific=nonspecific,
        )
    )
    with out:
        logger.info("Loading results: Done")

    out.clear_output(wait=True)
    b.disabled = True


def onKeyPress(event, n, f1, zoom, targets, nonspecific):
    key = event["key"]
    if key == "h" or key == "ArrowLeft":
        f1.value = f1.value - 15
    elif key == "l" or key == "ArrowRight":
        f1.value = f1.value + 15
    elif key == "j" or key == "ArrowDown":
        n.value = n.value - 1
    elif key == "k" or key == "ArrowUp":
        n.value = n.value + 1
    elif key == "z":
        zoom.value = not zoom.value
    elif key == "o":
        targets.value = not targets.value
    elif key == "n":
        nonspecific.value = not nonspecific.value


def onFrameClick(event, counter):
    counter.value = int(event.xdata)


def incrementRange(name, x, counter):
    counter.value = counter.value + x


def updateParams(n, f1, model, fig, item, ax, targets, nonspecific, labels):
    n = get_value(n)
    f1 = get_value(f1)
    f2 = f1 + 15
    c = model.cdx

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
    for i, f in enumerate(range(f1, f2)):
        ax[f"image_{i}"].set_title(rf"${f}$", fontsize=9)
        item[f"image_{i}"].set_data(model.data.images[n, f, c].numpy())
        item[f"ideal_{i}"].set_data(img_ideal[i].numpy())
        if targets.value:
            item[f"target_{i}"].remove()
            item[f"target_{i}"] = ax[f"image_{i}"].scatter(
                model.data.x[n, f, c].item(),
                model.data.y[n, f, c].item(),
                c="C0",
                s=40,
                marker="+",
            )

    params = ["p_specific", "z_map", "height", "width", "x", "y", "background", "chi2"]
    if labels.value:
        params += ["labels"]
    theta_mask = model.params["theta_probs"][:, n] > 0.5
    j_mask = (model.params["m_probs"][:, n] > 0.5) & ~theta_mask
    for p in params:
        if p == "p_specific":
            item[p].set_ydata(model.params[p][n])
        elif p in {"z_map"}.intersection(model.params.keys()):
            item[p].set_ydata(model.params[p][n])
        elif p == "chi2":
            item[p].set_ydata(model.params[p]["values"][n])
        elif p == "background":
            item[f"{p}_fill"].remove()
            item[f"{p}_fill"] = ax[p].fill_between(
                torch.arange(0, model.data.F),
                model.params[p]["LL"][n],
                model.params[p]["UL"][n],
                alpha=0.3,
                color="k",
            )
            item[f"{p}_mean"].set_ydata(model.params[p]["Mean"][n])
        elif p == "labels":
            item[p].set_ydata(model.data.labels["z"][n, :, c])
        elif p in ["height", "width", "x", "y"]:
            # target-nonspecific spots
            if nonspecific.value:
                for k in range(model.K):
                    f_mask = j_mask[k]
                    mean = model.params[p]["Mean"][k, n] * f_mask
                    ll = model.params[p]["LL"][k, n] * f_mask
                    ul = model.params[p]["UL"][k, n] * f_mask
                    item[f"{p}_nonspecific{k}_mean"].remove()
                    (item[f"{p}_nonspecific{k}_mean"],) = ax[p].plot(
                        torch.arange(0, model.data.F)[f_mask],
                        mean[f_mask],
                        "o",
                        ms=2,
                        lw=1,
                        color=f"C{k}",
                    )
                    item[f"{p}_nonspecific{k}_fill"].remove()
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
            item[f"{p}_specific_fill"].remove()
            item[f"{p}_specific_fill"] = ax[p].fill_between(
                torch.arange(0, model.data.F),
                ll,
                ul,
                where=f_mask,
                alpha=0.3,
                color="C2",
            )
            item[f"{p}_specific_mean"].remove()
            (item[f"{p}_specific_mean"],) = ax[p].plot(
                torch.arange(0, model.data.F)
                if p == "height"
                else torch.arange(0, model.data.F)[f_mask],
                mean if p == "height" else mean[f_mask],
                "-" if p == "height" else "o",
                ms=2,
                color="C2",
            )
    fig.canvas.draw()


def updateRange(f1, n, model, fig, item, ax, zoom, targets):
    n = get_value(n)
    f1 = get_value(f1)
    f2 = f1 + 15
    c = model.cdx

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
    for i, f in enumerate(range(f1, f2)):
        ax[f"image_{i}"].set_title(rf"${f}$", fontsize=9)
        item[f"image_{i}"].set_data(model.data.images[n, f, c].numpy())
        item[f"ideal_{i}"].set_data(img_ideal[i].numpy())
        if targets.value:
            item[f"target_{i}"].remove()
            item[f"target_{i}"] = ax[f"image_{i}"].scatter(
                model.data.x[n, f, c].item(),
                model.data.y[n, f, c].item(),
                c="C0",
                s=40,
                marker="+",
            )

    if not zoom.value:
        for key, a in ax.items():
            if not key.startswith("image") and not key.startswith("ideal"):
                a.set_xlim(f1 - 0.5, f2 - 0.5)
    else:
        for p in [
            "z_map",
            "p_specific",
            "height",
            "width",
            "x",
            "y",
            "background",
            "chi2",
        ]:
            item[f"{p}_vspan"].remove()
            item[f"{p}_vspan"] = ax[p].axvspan(f1, f2, facecolor="C0", alpha=0.3)
    fig.canvas.draw()


def zoomOut(checked, f1, model, fig, item, ax):
    checked = get_value(checked)
    f1_value = get_value(f1)
    if checked:
        f1 = 0
        f2 = model.data.F
        for p in [
            "z_map",
            "p_specific",
            "height",
            "width",
            "x",
            "y",
            "background",
            "chi2",
        ]:
            item[f"{p}_vspan"] = ax[p].axvspan(
                f1_value, f1_value + 15, facecolor="C0", alpha=0.3
            )
    else:
        f1 = f1_value
        f2 = f1 + 15
        for p in [
            "z_map",
            "p_specific",
            "height",
            "width",
            "x",
            "y",
            "background",
            "chi2",
        ]:
            item[f"{p}_vspan"].remove()
    for key, a in ax.items():
        if not key.startswith("image") and not key.startswith("ideal"):
            a.set_xlim(f1 - 0.5, f2 - 0.5)
    fig.canvas.draw()


def showLabels(checked, n, model, item, ax):
    checked = get_value(checked)
    n = get_value(n)
    if checked:
        (item["labels"],) = ax["z_map"].plot(
            torch.arange(0, model.data.F),
            model.data.labels["z"][n, :, model.cdx],
            "-",
            lw=1,
            color="r",
        )
    else:
        item["labels"].remove()


def showTargets(checked, n, f1, model, item, ax):
    checked = get_value(checked)
    n = get_value(n)
    f1 = get_value(f1)
    f2 = f1 + 15
    c = model.cdx
    if checked:
        for i, f in enumerate(range(f1, f2)):
            item[f"target_{i}"] = ax[f"image_{i}"].scatter(
                model.data.x[n, f, c].item(),
                model.data.y[n, f, c].item(),
                c="C0",
                s=40,
                marker="+",
            )
    else:
        for i, f in enumerate(range(f1, f2)):
            item[f"target_{i}"].remove()


def showNonspecific(checked, n, model, item, ax):
    checked = get_value(checked)
    n = get_value(n)
    for p in ["height", "width", "x", "y"]:
        if checked:
            theta_mask = model.params["theta_probs"][:, n] > 0.5
            j_mask = (model.params["m_probs"][:, n] > 0.5) & ~theta_mask
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
        else:
            for k in range(model.K):
                item[f"{p}_nonspecific{k}_fill"].remove()
                item[f"{p}_nonspecific{k}_mean"].remove()


def toggleWidgets(b, widgets):
    checked = get_value(b)
    for w in widgets:
        w.disabled = not checked


def logUI(out):
    # View logs
    layout = widgets.VBox()
    # Load logs
    showLog = widgets.Button(description="(Re)-load logs")
    logView = widgets.Output(layout={"max_height": "850px", "overflow": "auto"})
    # Layout
    layout.children = [
        showLog,
        logView,
    ]
    # Callbacks
    showLog.on_click(partial(logCmd, layout=layout, logView=logView, out=out))
    return layout


def logCmd(b, layout, logView, out):
    with out:
        logger.info("Loading logs ...")
    with logView:
        log()
    with out:
        logger.info("Loading logs: Done")
    logView.clear_output(wait=True)
    out.clear_output(wait=True)


def app():
    import pkg_resources

    nbpath = pkg_resources.resource_filename("tapqir", "app.ipynb")
    os.system("voila " + nbpath)


def run():
    from tapqir.main import DEFAULTS

    display(initUI(DEFAULTS))
