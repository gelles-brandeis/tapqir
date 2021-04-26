import configparser
from pathlib import Path

from cliff.command import Command
from pyroapi import pyro_backend

from tapqir.imscroll.utils import save_matlab
from tapqir.models import models
from tapqir.utils.stats import save_stats


class Save(Command):
    r"""
    Fit the data to the selected model.

    Command options are read from the ``options.cfg`` file and will be
    overwritten if provided explicitly.

    Output files of the analysis are saved in ``runs/model/version/S/control/lr/bs/``:

    - ``params`` and ``optimizer`` are model parameters and optimizer state saved in PyTorch format
    - ``global_params.csv`` contains values of global parameters
    - ``parameters.mat`` contains MAP estimate of parameters in MATLAB format
    - ``scalar`` folder containing global parameter values over iterations
    - ``run.log`` log file
    """

    def get_parser(self, prog_name):
        parser = super(Save, self).get_parser(prog_name)

        parser.add_argument(
            "model",
            default="cosmos",
            type=str,
            help="Available models: {}".format(", ".join(models.keys())),
        )
        parser.add_argument(
            "dataset_path", default=".", type=str, help="Path to the dataset folder"
        )
        parser.add_argument(
            "parameters_path", type=str, help="Path to the parameters folder"
        )
        parser.add_argument(
            "-backend", metavar="BACKEND", type=str, help="Pyro backend (default: pyro)"
        )

        parser.add_argument(
            "--matlab",
            action="store_true",
            help="Save parameters in matlab format (default: False)",
        )

        return parser

    def take_action(self, args):
        # read options.cfg file
        config = configparser.ConfigParser(allow_no_value=True)
        cfg_file = Path(args.dataset_path) / "options.cfg"
        config.read(cfg_file)

        backend = args.backend or config["fit"].get("backend")

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

            model = models[args.model](1, 2)
            model.load(args.dataset_path, False, "cpu")
            model.load_parameters(args.parameters_path)

            save_stats(model, args.parameters_path)
            if args.matlab:
                save_matlab(model, args.parameters_path)
