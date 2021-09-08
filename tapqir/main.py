from tapqir.cli import parse_args

def main(argv=None):
    """Run tapqir CLI command.

    argv: optional list of commands to parse. sys.argv is used by default.

    """
    args = None

    args = parse_args(argv)
