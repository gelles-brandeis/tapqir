# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
from collections import defaultdict
from collections.abc import Iterable
from functools import partial, singledispatch
from pathlib import Path

import ipywidgets as widgets
import nbformat as nbf
import torch
from ipyfilechooser import FileChooser
from traitlets.utils.bunch import Bunch

from tapqir.main import fit, glimpse, main, show


@singledispatch
def get_value(x):
    return x


@get_value.register
def _(x: Bunch):
    return x.new


@get_value.register
def _(x: widgets.Widget):
    return x.value


def widget_progress(iterable: Iterable, progress_bar: widgets.IntProgress):
    """
    Iterate over iterable and update progress bar.
    """
    progress_bar.min = 1
    progress_bar.max = len(iterable)
    for i in iterable:
        progress_bar.value = i + 1
        yield i


def initUI(DEFAULTS):
    """
    Main layout.

    - Set working directory.
    - Commands tabs.
    - Output.
    """
    layout = widgets.VBox()

    # widgets
    cd = FileChooser("/tmp/test", title="Working directory")
    cd.show_only_dirs = True

    out = widgets.Output(layout={"border": "1px solid black"})

    tab = widgets.Tab()
    tab.children = [glimpseUI(out), fitUI(out), showUI()]
    tab.set_title(0, "Extract AOIs")
    tab.set_title(1, "Fit the data")
    tab.set_title(2, "View results")
    # layout
    layout.children = [cd, tab, out]
    # callbacks
    cd.register_callback(partial(cdCmd, DEFAULTS=DEFAULTS, tab=tab))
    return layout


def cdCmd(chooser, DEFAULTS, tab):
    """
    Set working directory and load default parameters (main).
    """
    main(cd=Path((chooser.selected_path)))
    glimpseTab = tab.children[0]
    for i, flag in enumerate(
        [
            "dataset",
            "P",
            "frame-range",
            "frame-start",
            "frame-end",
            "num-channels",
            "use-offtarget",
        ]
    ):
        if DEFAULTS[flag] is not None:
            glimpseTab.children[i].value = DEFAULTS[flag]
    channelTabs = glimpseTab.children[7]
    for c, channel in enumerate(channelTabs.children):
        if len(DEFAULTS["channels"]) < c + 1:
            DEFAULTS["channels"].append(defaultdict(lambda: None))
        for widget, flag in zip(
            channel.children,
            [
                "name",
                "glimpse-folder",
                "ontarget-aoiinfo",
                "offtarget-aoiinfo",
                "driftlist",
            ],
        ):
            if DEFAULTS["channels"][c][flag] is not None:
                if isinstance(widget, FileChooser):
                    if flag == "glimpse-folder":
                        widget.reset(path=Path(DEFAULTS["channels"][c][flag]))
                    else:
                        widget.reset(
                            path=Path(DEFAULTS["channels"][c][flag]).parent,
                            filename=Path(DEFAULTS["channels"][c][flag]).name,
                        )
                    widget._apply_selection()
                elif flag:
                    widget.value = DEFAULTS["channels"][c][flag]
    fitTab = tab.children[1]
    numChannels = glimpseTab.children[5].value
    fitTab.children[1].options = [str(c) for c in range(numChannels)]
    for i, flag in enumerate(
        [False, False, "cuda", "nbatch-size", "fbatch-size", "learning-rate"]
    ):
        if flag and DEFAULTS[flag] is not None:
            fitTab.children[i].value = DEFAULTS[flag]


def glimpseUI(out):
    """
    Glimpse command layout.
    """
    # Extract AOIs
    layout = widgets.VBox()
    # Dataset name
    datasetName = widgets.Text(
        description="Dataset name",
        style={"description_width": "initial"},
    )
    # AOI image size
    aoiSize = widgets.IntText(
        description="AOI image size",
        style={"description_width": "initial"},
        disabled=False,
    )
    # Specify frame range?
    specifyFrame = widgets.Checkbox(
        value=False, description="Specify frame range?", disabled=False, indent=False
    )
    # First frame
    firstFrame = widgets.IntText(
        description="First frame",
        style={"description_width": "initial"},
        disabled=True,
    )
    # Last frame
    lastFrame = widgets.IntText(
        description="Last frame",
        style={"description_width": "initial"},
        disabled=True,
    )
    # Number of channels
    numChannels = widgets.BoundedIntText(
        min=1,
        max=4,
        step=1,
        description="Number of color channels",
        style={"description_width": "initial"},
        disabled=False,
    )
    # Specify frame range?
    useOfftarget = widgets.Checkbox(
        description="Use off-target AOI locations?",
        disabled=False,
        indent=False,
    )
    # Channel tabs
    channelTabs = widgets.Tab()
    channelUI(numChannels.value, channelTabs)
    # progress bar
    glimpseBar = widgets.IntProgress()
    # extract AOIs
    extractAOIs = widgets.Button(description="Extract AOIs")
    # Layout
    layout.children = [
        datasetName,
        aoiSize,
        specifyFrame,
        firstFrame,
        lastFrame,
        numChannels,
        useOfftarget,
        channelTabs,
        glimpseBar,
        extractAOIs,
    ]
    # Callbacks
    numChannels.observe(partial(channelUI, channelTabs=channelTabs), names="value")
    extractAOIs.on_click(partial(glimpseCmd, layout=layout, out=out))
    specifyFrame.observe(
        partial(toggleWidgets, widgets=(firstFrame, lastFrame)), names="value"
    )
    return layout


def glimpseCmd(b, layout, out):
    channelTabs = layout.children[7]
    glimpseBar = layout.children[8]
    with out:
        glimpse(
            dataset=layout.children[0].value,
            P=layout.children[1].value,
            frame_range=layout.children[2].value,
            frame_start=layout.children[3].value,
            frame_end=layout.children[4].value,
            num_channels=layout.children[5].value,
            use_offtarget=layout.children[6].value,
            name=[c.children[0].value for c in channelTabs.children],
            glimpse_folder=[c.children[1].value for c in channelTabs.children],
            ontarget_aoiinfo=[c.children[2].value for c in channelTabs.children],
            offtarget_aoiinfo=[c.children[3].value for c in channelTabs.children],
            driftlist=[c.children[4].value for c in channelTabs.children],
            no_input=True,
            progress_bar=partial(widget_progress, progress_bar=glimpseBar),
            labels=False,
        )


def channelUI(C, channelTabs):
    C = get_value(C)
    currentC = len(channelTabs.children)
    for i in range(max(currentC, C)):
        if i < C and i < currentC:
            continue
        elif i < C and i >= currentC:
            layout = widgets.VBox()
            nameLayout = widgets.Text(
                description="Channel name", style={"description_width": "initial"}
            )
            headerLayout = FileChooser(".", title="Header/glimpse folder")
            ontargetLayout = FileChooser(".", title="Target molecule locations file")
            offtargetLayout = FileChooser(
                ".", title="Off-target control locations file"
            )
            driftlistLayout = FileChooser(".", title="Driftlist file")
            layout.children = [
                nameLayout,
                headerLayout,
                ontargetLayout,
                offtargetLayout,
                driftlistLayout,
            ]
            channelTabs.children = channelTabs.children + (layout,)
        channelTabs.set_title(i, f"Channel #{i}")
    channelTabs.children = channelTabs.children[:C]


def fitUI(out):
    # Fit the data
    layout = widgets.VBox()
    # Model
    modelSelect = widgets.Combobox(
        description="Tapqir model",
        value="cosmos",
        options=["cosmos"],
        style={"description_width": "initial"},
    )
    # Channel numbers
    channelNumber = widgets.Combobox(
        description="Channel numbers",
        value="0",
        style={"description_width": "initial"},
        disabled=False,
    )
    # Run computations on GPU?
    useGpu = widgets.Checkbox(
        value=True,
        description="Run computations of GPU?",
        style={"description_width": "initial"},
        disabled=False,
    )
    # AOI batch size
    aoiBatch = widgets.IntText(
        description="AOI batch size",
        style={"description_width": "initial"},
        disabled=False,
    )
    # Frame batch size
    frameBatch = widgets.IntText(
        description="Frame batch size",
        style={"description_width": "initial"},
        disabled=False,
    )
    # Learning rate
    learningRate = widgets.FloatText(
        description="Learning rate",
        style={"description_width": "initial"},
        disabled=False,
    )
    # Number of iterations
    iterationNumber = widgets.IntText(
        description="Number of iterations",
        style={"description_width": "initial"},
        disabled=False,
    )
    # Save parameters in matlab format?
    saveMatlab = widgets.Checkbox(
        value=False,
        description="Save parameters in matlab format?",
        style={"description_width": "initial"},
        disabled=False,
    )
    # progress bar
    fitBar = widgets.IntProgress()
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
        fitBar,
        fitData,
    ]
    # Callbacks
    fitData.on_click(partial(fitCmd, layout=layout, out=out))
    return layout


def fitCmd(b, layout, out):
    fitBar = layout.children[8]
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
            progress_bar=partial(widget_progress, progress_bar=fitBar),
        )


def showUI():
    # Fit the data
    layout = widgets.VBox()
    # Model
    modelSelect = widgets.Combobox(
        description="Tapqir model",
        value="cosmos",
        options=["cosmos"],
        style={"description_width": "initial"},
    )
    # Channel numbers
    channelNumber = widgets.Combobox(
        description="Channel numbers",
        value="0",
        style={"description_width": "initial"},
        disabled=False,
    )
    # Fit the data
    showParams = widgets.Button(description="Load results")
    controls = widgets.HBox()
    view = widgets.Output(layout={"border": "1px solid red"})
    # Layout
    layout.children = [
        modelSelect,
        channelNumber,
        showParams,
        controls,
        view,
    ]
    # Callbacks
    showParams.on_click(partial(showCmd, layout=layout, view=view))
    return layout


def showCmd(b, layout, view):
    with view:
        model, fig, item, ax = show(
            model=layout.children[0].value,
            channels=[int(layout.children[1].value)],
            n=0,
            f1=None,
            f2=None,
        )
    controls = layout.children[3]
    n = widgets.IntText(
        value=0,
        description="AOI",
        style={"description_width": "initial"},
        disabled=False,
    )
    f1 = widgets.IntText(
        value=0,
        description="f1",
        style={"description_width": "initial"},
        disabled=False,
    )
    f2 = widgets.IntText(
        value=model.data.F - 1,
        description="f2",
        style={"description_width": "initial"},
        disabled=False,
    )
    controls.children = [n, f1, f2]
    n.observe(
        partial(updateParams, model=model, fig=fig, item=item, ax=ax), names="value"
    )
    f1.observe(partial(updateRange, f2=f2, fig=fig, ax=ax), names="value")
    f2.observe(partial(updateRange, f1, fig=fig, ax=ax), names="value")


def updateParams(n, model, fig, item, ax):
    n = get_value(n)
    params = ["z", "height", "width", "x", "y", "background"]
    for p in params:
        if p == "z":
            if "z_probs" in model.params:
                item["pspecific"].set_ydata(model.params["z_probs"][n])
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
        else:
            for k in range(model.K):
                item[f"{p}_{k}_fill"].remove()
                item[f"{p}_{k}_fill"] = ax[p].fill_between(
                    torch.arange(0, model.data.F),
                    model.params[p]["LL"][k, n],
                    model.params[p]["UL"][k, n],
                    alpha=0.3,
                    color=f"C{k}",
                )
                item[f"{p}_{k}_mean"].set_ydata(model.params[p]["Mean"][k, n])
    fig.canvas.draw()


def updateRange(f1, f2, fig, ax):
    f1 = get_value(f1)
    f2 = get_value(f2)
    for a in ax.values():
        a.set_xlim(f1 - 1, f2 + 1)
    fig.canvas.draw()


def toggleWidgets(b, widgets):
    checked = get_value(b)
    for w in widgets:
        w.disabled = not checked


if __name__ == "__main__":
    notebook = nbf.v4.new_notebook()
    notebook["cells"] = []
    new_cell = nbf.v4.new_code_cell(
        "%matplotlib widget\n"
        "from tapqir.v2 import initUI\n"
        "from tapqir.main import DEFAULTS\n"
        "initUI(DEFAULTS)"
    )
    notebook["cells"].append(new_cell)
    temp_dir = tempfile.mkdtemp()
    nbpath = os.path.join(temp_dir, "temp_notebook.ipynb")
    nbf.write(notebook, nbpath)
    os.system("voila " + nbpath)
