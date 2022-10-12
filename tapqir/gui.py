# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from collections import OrderedDict, defaultdict
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
from tapqir.main import avail_models, dwelltime, fit, glimpse, log, main, show, ttfb
from tapqir.utils.dataset import save

logger = logging.getLogger("tapqir")

# disable pyplot keyboard shortcuts
plt.rcParams["keymap.zoom"].remove("o")
plt.rcParams["keymap.home"].remove("h")
plt.rcParams["keymap.yscale"].remove("l")
plt.rcParams["keymap.xscale"].remove("k")

avail_models = [e.value for e in avail_models]


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


class inputBox(widgets.VBox):
    def __init__(self, **kwargs):
        self._children = OrderedDict()
        super().__init__(children=[], **kwargs)

    def build_children(self) -> None:
        self.children = [
            child["widget"] for child in self._children.values() if not child["hide"]
        ]

    def add_child(self, name, widget, hide=False, beginning=False) -> None:
        self._children[name] = {
            "widget": widget,
            "hide": hide,
        }
        if beginning:
            self._children.move_to_end(name, last=False)
        self.build_children()

    def remove_child(self, name) -> None:
        self._children.pop(name)
        self.build_children()

    def toggle_hide(self, a=None, names=None) -> None:
        for name in names:
            self._children[name]["hide"] = not self._children[name]["hide"]
        self.build_children()

    def __getitem__(self, name):
        return self._children[name]["widget"]

    @property
    def kwargs(self):
        result = {}
        for name, v in self._children.items():
            if isinstance(
                v["widget"],
                (
                    widgets.Button,
                    widgets.Output,
                    widgets.Accordion,
                    widgets.VBox,
                    widgets.HBox,
                ),
            ):
                continue
            if isinstance(v["widget"], widgets.Tab):
                keys = v["widget"].children[0]._children.keys()
                for k in keys:
                    result[k] = [get_value(c[k]) for c in v["widget"].children]
                continue
            result[name] = get_value(v["widget"])
        return result


def initUI(DEFAULTS):
    """
    Main layout.

    - Set working directory.
    - Commands tabs.
    - Output.
    """

    layout = inputBox(layout={"width": "850px", "border": "2px solid black"})

    # widgets
    cd = FileChooser(".", title="Select working directory")
    cd.show_only_dirs = True

    out = widgets.Output()

    # layout
    layout.add_child("cd", cd)
    layout.add_child("out", out)
    # callbacks
    cd.register_callback(
        partial(
            cdCmd,
            DEFAULTS=DEFAULTS,
            layout=layout,
        )
    )
    return layout


def cdCmd(path, DEFAULTS, layout):
    """
    Set working directory and load default parameters (main).
    """
    import sys
    from platform import uname

    IN_COLAB = "google.colab" in sys.modules
    IN_WSL2 = "microsoft-standard" in uname().release
    IN_LINUX = not IN_COLAB and not IN_WSL2

    out = layout["out"]
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
    tab.children = tab.children + (postUI(out), logUI(out))
    tab.set_title(0, "Extract AOIs")
    tab.set_title(1, "Fit the data")
    tab.set_title(2, "View results")
    tab.set_title(3, "Tensorboard")
    tab.set_title(4, "Post analysis")
    tab.set_title(5, "View logs")

    if tensorboard is not None:
        with tensorboard:
            notebook.start(f"--logdir '{path}'")
            notebook.display(height=1000)

    # insert tabs into GUI
    wd = widgets.Label(value=f"Working directory: {path}")
    layout.remove_child("cd")
    layout.add_child("tab", tab, beginning=True)
    layout.add_child("wd", wd, beginning=True)

    with out:
        logger.info("Loading configuration data: Done")

    out.clear_output(wait=True)


def glimpseUI(out, DEFAULTS):
    """
    Glimpse command layout.
    """
    layout = inputBox()
    layout.add_child(
        "dataset",
        widgets.Text(
            value=DEFAULTS["dataset"],
            description="Dataset name",
            style={"description_width": "initial"},
        ),
    )
    layout.add_child(
        "P",
        widgets.BoundedIntText(
            value=DEFAULTS["P"],
            min=5,
            max=50,
            description="AOI image size",
            style={"description_width": "initial"},
        ),
    )
    layout.add_child(
        "offset_x",
        widgets.IntText(
            value=DEFAULTS["offset-x"],
            description="Offset region top-left corner (x-axis)",
            style={"description_width": "initial"},
        ),
    )
    layout.add_child(
        "offset_y",
        widgets.IntText(
            value=DEFAULTS["offset-y"],
            description="Offset region top-left corner (y-axis)",
            style={"description_width": "initial"},
        ),
    )
    layout.add_child(
        "offset_P",
        widgets.IntText(
            value=DEFAULTS["offset-P"],
            description="Offset region size",
            style={"description_width": "initial"},
        ),
    )
    layout.add_child(
        "bin_size",
        widgets.BoundedIntText(
            value=DEFAULTS["bin-size"],
            min=1,
            max=21,
            step=2,
            description="Offset histogram bin size (odd number)",
            style={"description_width": "initial"},
        ),
    )
    layout.add_child(
        "frame_range",
        widgets.Checkbox(
            value=DEFAULTS["frame-range"],
            description="Specify frame range?",
            indent=False,
        ),
    )
    layout.add_child(
        "frame_start",
        widgets.IntText(
            value=DEFAULTS["frame-start"],
            description="Starting frame",
            style={"description_width": "initial"},
        ),
        hide=not DEFAULTS["frame-range"],
    )
    layout.add_child(
        "frame_end",
        widgets.IntText(
            value=DEFAULTS["frame-end"],
            description="Ending frame",
            style={"description_width": "initial"},
        ),
        hide=not DEFAULTS["frame-range"],
    )
    layout.add_child(
        "use_offtarget",
        widgets.Checkbox(
            value=DEFAULTS["use-offtarget"],
            description="Use off-target AOI locations?",
            indent=False,
        ),
    )
    layout.add_child(
        "num_channels",
        widgets.BoundedIntText(
            value=DEFAULTS["num-channels"],
            min=1,
            max=4,
            step=1,
            description="Number of color channels",
            style={"description_width": "initial"},
        ),
    )
    layout.add_child("channels", widgets.Tab())
    channelUI(
        layout["num_channels"].value,
        layout["channels"],
        layout["use_offtarget"],
        DEFAULTS,
    )
    layout.add_child("extract_aois", widgets.Button(description="Extract AOIs"))
    # Callbacks
    layout["num_channels"].observe(
        partial(
            channelUI,
            channelTabs=layout["channels"],
            useOfftarget=layout["use_offtarget"],
            DEFAULTS=DEFAULTS,
        ),
        names="value",
    )
    layout["use_offtarget"].observe(
        partial(toggleUseOffTarget, channelTabs=layout["channels"]),
        names="value",
    )
    layout["extract_aois"].on_click(partial(glimpseCmd, layout=layout, out=out))
    layout["frame_range"].observe(
        partial(layout.toggle_hide, names=("frame_start", "frame_end")),
        names="value",
    )
    return layout


def toggleUseOffTarget(checked, channelTabs):
    for c, tab in enumerate(channelTabs.children):
        tab.toggle_hide(names=("offtarget_aoiinfo",))


def glimpseCmd(b, layout, out):
    with out:
        glimpse(
            **layout.kwargs,
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
            layout = inputBox()
            layout.add_child(
                "name",
                widgets.Text(
                    value=DEFAULTS["channels"][c]["name"],
                    description="Channel name",
                    style={"description_width": "initial"},
                ),
            )
            layout.add_child(
                "glimpse_folder", FileChooser(".", title="Header/glimpse folder")
            )
            if DEFAULTS["channels"][c]["glimpse-folder"] is not None:
                layout["glimpse_folder"].reset(
                    path=Path(DEFAULTS["channels"][c]["glimpse-folder"])
                )
                layout["glimpse_folder"]._apply_selection()
            layout.add_child("driftlist", FileChooser(".", title="Driftlist file"))
            if DEFAULTS["channels"][c]["driftlist"] is not None:
                layout["driftlist"].reset(
                    path=Path(DEFAULTS["channels"][c]["driftlist"]).parent,
                    filename=Path(DEFAULTS["channels"][c]["driftlist"]).name,
                )
                layout["driftlist"]._apply_selection()
            layout.add_child(
                "ontarget_aoiinfo",
                FileChooser(".", title="Target molecule locations file"),
            )
            if DEFAULTS["channels"][c]["ontarget-aoiinfo"] is not None:
                layout["ontarget_aoiinfo"].reset(
                    path=Path(DEFAULTS["channels"][c]["ontarget-aoiinfo"]).parent,
                    filename=Path(DEFAULTS["channels"][c]["ontarget-aoiinfo"]).name,
                )
                layout["ontarget_aoiinfo"]._apply_selection()
            layout.add_child(
                "offtarget_aoiinfo",
                FileChooser(".", title="Off-target control locations file"),
                hide=not useOfftarget.value,
            )
            if DEFAULTS["channels"][c]["offtarget-aoiinfo"] is not None:
                layout["offtarget_aoiinfo"].reset(
                    path=Path(DEFAULTS["channels"][c]["offtarget-aoiinfo"]).parent,
                    filename=Path(DEFAULTS["channels"][c]["offtarget-aoiinfo"]).name,
                )
                layout["offtarget_aoiinfo"]._apply_selection()
            channelTabs.children = channelTabs.children + (layout,)
        channelTabs.set_title(c, f"Channel #{c}")
    channelTabs.children = channelTabs.children[:C]


def fitUI(out, DEFAULTS):
    # Fit the data
    layout = inputBox()
    layout.add_child(
        "model",
        widgets.Dropdown(
            description="Tapqir model",
            value="cosmos",
            options=avail_models,
            style={"description_width": "initial"},
        ),
    )
    layout.add_child(
        "cuda",
        widgets.Checkbox(
            value=DEFAULTS["cuda"],
            description="Run computations on GPU?",
            style={"description_width": "initial"},
        ),
    )
    layout.add_child(
        "nbatch_size",
        widgets.IntText(
            value=DEFAULTS["nbatch-size"],
            description="AOI batch size",
            style={"description_width": "initial"},
        ),
    )
    layout.add_child(
        "fbatch_size",
        widgets.IntText(
            value=DEFAULTS["fbatch-size"],
            description="Frame batch size",
            style={"description_width": "initial"},
        ),
    )
    layout.add_child(
        "learning_rate",
        widgets.FloatText(
            value=DEFAULTS["learning-rate"],
            description="Learning rate",
            style={"description_width": "initial"},
        ),
    )
    layout.add_child(
        "num_iter",
        widgets.IntText(
            description="Number of iterations",
            style={"description_width": "initial"},
        ),
    )
    layout.add_child(
        "matlab",
        widgets.Checkbox(
            value=DEFAULTS["matlab"],
            description="Save parameters in matlab format?",
            style={"description_width": "initial"},
        ),
    )
    priors = inputBox()
    for name, value in DEFAULTS["priors"].items():
        priors.add_child(
            name,
            widgets.FloatText(
                value=value, description=name, style={"description_width": "initial"}
            ),
        )
    layout.add_child("priors", widgets.Accordion(children=[priors]))
    layout["priors"].set_title(0, "Priors")
    # Fit the data
    layout.add_child("fit", widgets.Button(description="Fit the data"))
    # Callbacks
    layout["fit"].on_click(partial(fitCmd, layout=layout, out=out, DEFAULTS=DEFAULTS))
    return layout


def fitCmd(b, layout, out, DEFAULTS):
    # read priors
    DEFAULTS["priors"].update(layout["priors"].children[0].kwargs)
    with out:
        fit(
            **layout.kwargs,
            k_max=2,
            funsor=False,
            pykeops=True,
            no_input=True,
            progress_bar=tqdm_notebook,
        )

    out.clear_output(wait=True)


def showUI(out, DEFAULTS):
    # View results
    layout = inputBox()
    layout.add_child(
        "model",
        widgets.Dropdown(
            description="Tapqir model",
            value="cosmos",
            options=avail_models,
            style={"description_width": "initial"},
        ),
    )
    layout.add_child(
        "show_fov",
        widgets.Checkbox(
            value=True,
            description="Show FOV images",
            indent=False,
        ),
    )
    # Load results
    layout.add_child(
        "show",
        widgets.Button(
            description="Load results",
            layout=widgets.Layout(margin="10px 0px 10px 0px"),
        ),
    )
    controls = widgets.HBox()
    view = widgets.Output()
    fov_controls = inputBox()
    params = widgets.VBox(children=[controls, view, fov_controls])
    summary = widgets.Output()
    sum_view = widgets.Accordion(children=[params, summary])
    sum_view.set_title(0, "AOI images and parameters")
    sum_view.set_title(1, "Summary statistics")
    sum_view.selected_index = 0
    layout.add_child("view", sum_view)
    # Callbacks
    layout["show"].on_click(partial(showCmd, layout=layout, out=out))
    return layout


def showCmd(b, layout, out):
    layout.toggle_hide(names=("show",))
    params, summary = layout["view"].children
    controls, view, fov_controls = params.children
    with out:
        logger.info("Loading results ...")
        show_ret = show(
            **layout.kwargs,
            n=0,
            f1=None,
            f2=None,
            gui=True,
        )
        if show_ret == 1:
            out.clear_output(wait=True)
            return
        model, fig, item, ax, fov = show_ret
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
    exclude_aoi = widgets.Checkbox(
        value=not model.data.mask[0],
        description="Exclude AOI from analysis ['e']",
        indent=False,
        layout={"width": "240px"},
    )
    checkboxes = widgets.VBox(layout=widgets.Layout(width="250px"))
    if model.data.labels is None:
        checkboxes.children = [zoom, targets, nonspecific, exclude_aoi]
    else:
        checkboxes.children = [zoom, targets, nonspecific, exclude_aoi, labels]
    controls.children = [n, f1_box, checkboxes]
    # fov controls
    if fov is not None:
        for dtype in fov.dtypes:
            fov_controls.add_child(
                dtype, widgets.Checkbox(value=True, description=f"Show {dtype} AOIs")
            )
            fov_controls[dtype].observe(
                partial(
                    showAOIs,
                    fov=fov,
                    n=n_counter,
                    item=item,
                    fig=fig,
                ),
                names="value",
            )
    fov_controls.add_child("save_data", widgets.Button(description="Save data"))
    # callbacks
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
            fov_controls=fov_controls,
            exclude_aoi=exclude_aoi,
            show_fov=layout["show_fov"],
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
            fov=fov,
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
        partial(
            showTargets,
            n=n_counter,
            f1=f1_slider,
            model=model,
            item=item,
            ax=ax,
        ),
        names="value",
    )
    nonspecific.observe(
        partial(showNonspecific, n=n_counter, model=model, item=item, ax=ax),
        names="value",
    )
    exclude_aoi.observe(
        partial(
            excludeAOI, n=n_counter, model=model, item=item, show_fov=layout["show_fov"]
        ),
        names="value",
    )
    fov_controls["save_data"].on_click(
        partial(saveData, data=model.data, path=model.path, out=out)
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
            exclude_aoi=exclude_aoi,
        )
    )
    with out:
        logger.info("Loading results: Done")

    out.clear_output(wait=True)
    b.disabled = True


def saveData(b, data, path, out):
    with out:
        logger.info("Saving data ...")
        save(data, path)


def showAOIs(checked, fov, n, item, fig):
    dtype = "ontarget" if "ontarget" in checked.owner.description else "offtarget"
    n = get_value(n)
    for key, value in item.items():
        if key.startswith("aoi_"):
            i = int(key.split("_")[1][1:])
            if n == i:
                continue
            if dtype == "ontarget" and i >= fov.N:
                continue
            if dtype == "offtarget" and i < fov.N:
                continue
            item[key].set(visible=get_value(checked))
    fig.canvas.draw()


def onKeyPress(event, n, f1, zoom, targets, nonspecific, exclude_aoi):
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
    elif key == "e":
        exclude_aoi.value = not exclude_aoi.value


def onFrameClick(event, counter):
    counter.value = int(event.xdata)


def incrementRange(name, x, counter):
    counter.value = counter.value + x


def updateParams(
    n,
    f1,
    model,
    fig,
    item,
    ax,
    targets,
    nonspecific,
    labels,
    fov_controls,
    exclude_aoi,
    show_fov,
):
    n_old = n.old
    n = get_value(n)
    f1 = get_value(f1)
    f2 = f1 + 15
    color = (
        [f"C{2+q}" for q in range(model.Q)] if model.data.mask[n] else ["C7"] * model.Q
    )

    exclude_aoi.value = not model.data.mask[n]

    frames = torch.arange(f1, f2)
    img_ideal = (
        model.data.offset.mean
        + model.params["background"]["Mean"][n, frames, :, None, None]
    )
    gaussian = gaussian_spots(
        model.params["height"]["Mean"][:, n, frames],
        model.params["width"]["Mean"][:, n, frames],
        model.params["x"]["Mean"][:, n, frames],
        model.params["y"]["Mean"][:, n, frames],
        model.data.xy[n, frames],
        model.data.P,
    )
    img_ideal = img_ideal + gaussian.sum(-5)
    for c in range(model.data.C):
        for i, f in enumerate(range(f1, f2)):
            ax[f"image_f{i}_c{c}"].set_title(rf"${f}$", fontsize=9)
            item[f"image_f{i}_c{c}"].set_data(model.data.images[n, f, c].numpy())
            item[f"ideal_f{i}_c{c}"].set_data(img_ideal[i, c].numpy())
            if targets.value:
                item[f"target_f{i}_c{c}"].remove()
                item[f"target_f{i}_c{c}"] = ax[f"image_f{i}_c{c}"].scatter(
                    model.data.x[n, f, c].item(),
                    model.data.y[n, f, c].item(),
                    c="C0",
                    s=40,
                    marker="+",
                )

    params = [
        "p_specific",
        "z_map",
        "height",
        "width",
        "x",
        "y",
        "background",
        "chi2",
    ]
    if labels.value:
        params += ["labels"]
    for q in range(model.Q):
        theta_mask = model.params["theta_probs"][:, n, :, q] > 0.5
        j_mask = (model.params["m_probs"][:, n, :, q] > 0.5) & ~theta_mask
        for p in params:
            if p == "p_specific":
                item[f"{p}_q{q}"].set_ydata(model.params[p][n, :, q])
                item[f"{p}_q{q}"].set_color(color[q])
            elif p in {"z_map"}.intersection(model.params.keys()):
                item[f"{p}_q{q}"].set_ydata(model.params[p][n, :, q])
            elif p == "labels":
                item[p].set_ydata(model.data.labels["z"][n, :, q])
            elif p in ["height", "width", "x", "y"]:
                # target-nonspecific spots
                if nonspecific.value:
                    for k in range(model.K):
                        f_mask = j_mask[k]
                        mean = model.params[p]["Mean"][k, n, :, q] * f_mask
                        ll = model.params[p]["LL"][k, n, :, q] * f_mask
                        ul = model.params[p]["UL"][k, n, :, q] * f_mask
                        item[f"{p}_nonspecific{k}_mean_q{q}"].remove()
                        (item[f"{p}_nonspecific{k}_mean_q{q}"],) = ax[p].plot(
                            torch.arange(0, model.data.F)[f_mask],
                            mean[f_mask],
                            "o",
                            ms=2,
                            lw=1,
                            color=f"C{k}",
                        )
                        item[f"{p}_nonspecific{k}_fill_q{q}"].remove()
                        item[f"{p}_nonspecific{k}_fill_q{q}"] = ax[p].fill_between(
                            torch.arange(0, model.data.F),
                            ll,
                            ul,
                            where=f_mask,
                            alpha=0.3,
                            color=f"C{k}",
                        )
                # target-specific spots
                f_mask = theta_mask.sum(0).bool()
                mean = (model.params[p]["Mean"][:, n, :, q] * theta_mask).sum(0)
                ll = (model.params[p]["LL"][:, n, :, q] * theta_mask).sum(0)
                ul = (model.params[p]["UL"][:, n, :, q] * theta_mask).sum(0)
                item[f"{p}_specific_fill_q{q}"].remove()
                item[f"{p}_specific_fill_q{q}"] = ax[p].fill_between(
                    torch.arange(0, model.data.F),
                    ll,
                    ul,
                    where=f_mask,
                    alpha=0.3,
                    color=color[q],
                )
                item[f"{p}_specific_mean_q{q}"].remove()
                (item[f"{p}_specific_mean_q{q}"],) = ax[p].plot(
                    torch.arange(0, model.data.F)
                    if p == "height"
                    else torch.arange(0, model.data.F)[f_mask],
                    mean if p == "height" else mean[f_mask],
                    "-" if p == "height" else "o",
                    ms=2,
                    color=color[q],
                )

    for c in range(model.data.C):
        for p in params:
            if p == "chi2":
                item[f"{p}_c{c}"].set_ydata(model.params[p]["values"][n, :, c])
            elif p == "background":
                item[f"{p}_fill_c{c}"].remove()
                item[f"{p}_fill_c{c}"] = ax[p].fill_between(
                    torch.arange(0, model.data.F),
                    model.params[p]["LL"][n, :, c],
                    model.params[p]["UL"][n, :, c],
                    alpha=0.3,
                    color=color[c],
                )
                item[f"{p}_mean_c{c}"].set_ydata(model.params[p]["Mean"][n, :, c])

    if get_value(show_fov):
        ax["glimpse_c0"].set_title(rf"AOI ${n}$, Frame ${f1}$", fontsize=9)
        n_old_dtype = "ontarget" if n_old < model.data.N else "offtarget"
        n_old_visible = fov_controls[n_old_dtype].value
        colors = {"ontarget": "#AA3377", "offtarget": "#CCBB44"}
        for c in range(model.data.C):
            item[f"aoi_n{n}_c{c}"].set_edgecolor(color[c])
            item[f"aoi_n{n}_c{c}"].set(zorder=2, visible=True)
            item[f"aoi_n{n_old}_c{c}"].set_edgecolor(colors[n_old_dtype])
            item[f"aoi_n{n_old}_c{c}"].set(zorder=1, visible=n_old_visible)
    fig.canvas.draw()


def updateRange(f1, n, model, fig, item, ax, zoom, targets, fov):
    n = get_value(n)
    f1 = get_value(f1)
    f2 = f1 + 15

    frames = torch.arange(f1, f2)
    img_ideal = (
        model.data.offset.mean
        + model.params["background"]["Mean"][n, frames, :, None, None]
    )
    gaussian = gaussian_spots(
        model.params["height"]["Mean"][:, n, frames],
        model.params["width"]["Mean"][:, n, frames],
        model.params["x"]["Mean"][:, n, frames],
        model.params["y"]["Mean"][:, n, frames],
        model.data.xy[n, frames],
        model.data.P,
    )
    img_ideal = img_ideal + gaussian.sum(-5)
    for c in range(model.data.C):
        for i, f in enumerate(range(f1, f2)):
            ax[f"image_f{i}_c{c}"].set_title(rf"${f}$", fontsize=9)
            item[f"image_f{i}_c{c}"].set_data(model.data.images[n, f, c].numpy())
            item[f"ideal_f{i}_c{c}"].set_data(img_ideal[i, c].numpy())
            if targets.value:
                item[f"target_f{i}_c{c}"].remove()
                item[f"target_f{i}_c{c}"] = ax[f"image_f{i}_c{c}"].scatter(
                    model.data.x[n, f, c].item(),
                    model.data.y[n, f, c].item(),
                    c="C0",
                    s=40,
                    marker="+",
                )

    if not zoom.value:
        for key, a in ax.items():
            if (
                key.startswith("image")
                or key.startswith("ideal")
                or key.startswith("glimpse")
            ):
                continue
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
    if fov is not None:
        for c in range(model.data.C):
            fov.plot(
                fov.dtypes,
                model.data.P,
                n=n,
                f=f1,
                save=False,
                ax=ax[f"glimpse_c{c}"],
                item=item,
            )
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
        if (
            key.startswith("image")
            or key.startswith("ideal")
            or key.startswith("glimpse")
        ):
            continue
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
    if checked:
        for c in range(model.data.C):
            for i, f in enumerate(range(f1, f2)):
                item[f"target_f{i}_c{c}"] = ax[f"image_f{i}_c{c}"].scatter(
                    model.data.x[n, f, c].item(),
                    model.data.y[n, f, c].item(),
                    c="C0",
                    s=40,
                    marker="+",
                )
    else:
        for c in range(model.data.C):
            for i, f in enumerate(range(f1, f2)):
                item[f"target_f{i}_c{c}"].remove()


def showNonspecific(checked, n, model, item, ax):
    checked = get_value(checked)
    n = get_value(n)
    for p in ["height", "width", "x", "y"]:
        if checked:
            # target-nonspecific spots
            for q in range(model.Q):
                theta_mask = model.params["theta_probs"][:, n, :, q] > 0.5
                j_mask = (model.params["m_probs"][:, n, :, q] > 0.5) & ~theta_mask
                for k in range(model.K):
                    f_mask = j_mask[k]
                    mean = model.params[p]["Mean"][k, n, :, q] * f_mask
                    ll = model.params[p]["LL"][k, n, :, q] * f_mask
                    ul = model.params[p]["UL"][k, n, :, q] * f_mask
                    (item[f"{p}_nonspecific{k}_mean_q{q}"],) = ax[p].plot(
                        torch.arange(0, model.data.F)[f_mask],
                        mean[f_mask],
                        "o",
                        ms=2,
                        lw=1,
                        color=f"C{k}",
                    )
                    item[f"{p}_nonspecific{k}_fill_q{q}"] = ax[p].fill_between(
                        torch.arange(0, model.data.F),
                        ll,
                        ul,
                        where=f_mask,
                        alpha=0.3,
                        color=f"C{k}",
                    )
        else:
            for q in range(model.Q):
                for k in range(model.K):
                    item[f"{p}_nonspecific{k}_fill_q{q}"].remove()
                    item[f"{p}_nonspecific{k}_mean_q{q}"].remove()


def excludeAOI(checked, n, model, item, show_fov):
    checked = get_value(checked)
    n = get_value(n)
    model.data.mask[n] = not checked
    color = [f"C{2+q}" for q in range(model.Q)] if not checked else ["C7"] * model.Q
    for q in range(model.Q):
        item[f"z_map_q{q}"].set_color(color[q])
        item[f"p_specific_q{q}"].set_color(color[q])
        item[f"height_specific_mean_q{q}"].set_color(color[q])
        item[f"height_specific_fill_q{q}"].set_color(color[q])
        item[f"width_specific_mean_q{q}"].set_color(color[q])
        item[f"width_specific_fill_q{q}"].set_color(color[q])
        item[f"x_specific_mean_q{q}"].set_color(color[q])
        item[f"x_specific_fill_q{q}"].set_color(color[q])
        item[f"y_specific_mean_q{q}"].set_color(color[q])
        item[f"y_specific_fill_q{q}"].set_color(color[q])
    if get_value(show_fov):
        for c in range(model.data.C):
            item[f"aoi_n{n}_c{c}"].set_edgecolor(color[c])


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


def postUI(out):
    layout = inputBox()
    layout.add_child("post", widgets.Tab())
    # Time-to-first binding analysis
    ttfb_layout = inputBox()
    ttfb_layout.add_child(
        "model",
        widgets.Dropdown(
            description="Tapqir model",
            value="cosmos",
            options=avail_models,
            style={"description_width": "initial"},
        ),
    )
    ttfb_layout.add_child(
        "binary",
        widgets.Checkbox(
            value=False,
            description="Plot a binary rastergram?",
            style={"description_width": "initial"},
        ),
    )
    ttfb_layout.add_child(
        "num_samples",
        widgets.IntText(
            value=2000,
            description="Number of posterior samples",
            style={"description_width": "initial"},
        ),
    )
    ttfb_layout.add_child(
        "num_iter",
        widgets.IntText(
            value=15000,
            description="Number of iterations",
            style={"description_width": "initial"},
        ),
    )
    ttfb_layout.add_child("ttfb", widgets.Button(description="Analyze"))
    ttfb_layout["ttfb"].on_click(partial(ttfbCmd, layout=ttfb_layout, out=out))
    # Dwell time analysis
    dt_layout = inputBox()
    dt_layout.add_child(
        "model",
        widgets.Dropdown(
            description="Tapqir model",
            value="cosmos",
            options=avail_models,
            style={"description_width": "initial"},
        ),
    )
    dt_layout.add_child(
        "K",
        widgets.BoundedIntText(
            value=1,
            min=1,
            max=6,
            description="Number of exponentials",
            style={"description_width": "initial"},
        ),
    )
    dt_layout.add_child("dwelltime", widgets.Button(description="Analyze"))
    dt_layout["dwelltime"].on_click(partial(dtCmd, layout=dt_layout, out=out))
    # Layout
    layout["post"].children = (ttfb_layout, dt_layout)
    layout["post"].set_title(0, "Time-to-first binding")
    layout["post"].set_title(1, "Dwell-time")
    return layout


def ttfbCmd(b, layout, out):
    layout.toggle_hide(names=("ttfb",))
    with out:
        logger.info("Time-to-first binding analysis ...")
        ttfb(**layout.kwargs)
        logger.info("Time-to-first binding analysis: Done")
    out.clear_output(wait=True)


def dtCmd(b, layout, out):
    layout.toggle_hide(names=("dwelltime",))
    with out:
        logger.info("Dwell-time analysis ...")
        dwelltime(**layout.kwargs)
        logger.info("Dwell-time analysis: Done")
    out.clear_output(wait=True)


def app():
    import pkg_resources

    nbpath = pkg_resources.resource_filename("tapqir", "app.ipynb")
    os.system("voila " + nbpath)


def run():
    from tapqir.main import DEFAULTS

    display(initUI(DEFAULTS))
