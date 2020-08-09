from cliff.command import Command
import configparser
import os


class Config(Command):
    """
    Create cosmos.cfg configuration file with default settings.
    """

    def get_parser(self, prog_name):
        parser = super(Config, self).get_parser(prog_name)

        parser.add_argument("dataset", type=str,
                            help="Path to the dataset folder")

        return parser

    def take_action(self, args):
        config = configparser.ConfigParser(allow_no_value=True)
        config["glimpse"] = {}
        config["glimpse"]["dir"] = "/path/to/glimpse/folder"
        config["glimpse"]["test_aoiinfo"] = "/path/to/test_aoiinfo_file"
        config["glimpse"]["control_aoiinfo"] = "/path/to/control_aoiinfo_file"
        config["glimpse"]["driftlist"] = "/path/to/driftlist_file"
        config["glimpse"]["labels"] = None
        config["glimpse"]["labeltype"] = None

        config["fit"] = {}
        config["fit"]["num_iter"] = "20000"
        config["fit"]["batch_size"] = "8"
        config["fit"]["learning_rate"] = "0.005"
        config["fit"]["control"] = "False"
        config["fit"]["device"] = "cuda"

        cfg_file = os.path.join(args.dataset, "options.cfg")
        with open(cfg_file, "w") as configfile:
            config.write(configfile)
