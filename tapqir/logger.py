# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: Apache-2.0

import logging

import colorama


class ColorFormatter(logging.Formatter):
    """
    The code adapted from https://github.com/iterative/dvc/blob/main/dvc/logger.py

    Spit out colored text in supported terminals.
    colorama__ makes ANSI escape character sequences work under Windows.
    See the colorama documentation for details.
    __ https://pypi.python.org/pypi/colorama
    If record has an extra `tb_only` attribute, it will not show the
    exception cause, just the message and the traceback.
    """

    color_code = {
        "TRACE": colorama.Fore.GREEN,
        "DEBUG": colorama.Fore.BLUE,
        "WARNING": colorama.Fore.YELLOW,
        "ERROR": colorama.Fore.RED,
        "CRITICAL": colorama.Fore.RED,
    }

    def format(self, record):
        record.message = record.getMessage()
        msg = self.formatMessage(record)

        if record.levelname == "INFO":
            return msg

        if record.exc_info:
            if getattr(record, "tb_only", False):
                cause = ""
            else:
                cause = ": ".join(_iter_causes(record.exc_info[1]))

            msg = "{message}{separator}{cause}".format(
                message=msg or "",
                separator=" - " if msg and cause else "",
                cause=cause,
            )

        return "{asctime}{color}{levelname}{nc}: {msg}".format(
            asctime=self.formatTime(record, self.datefmt),
            color=self.color_code[record.levelname],
            nc=colorama.Fore.RESET,
            levelname=record.levelname,
            msg=msg,
        )

    def formatTime(self, record, datefmt=None):
        return ""


def _iter_causes(exc):
    while exc:
        yield str(exc)
        exc = exc.__cause__


def _stack_trace(exc_info):
    import traceback

    return (
        "\n"
        "{red}{line}{nc}\n"
        "{trace}"
        "{red}{line}{nc}".format(
            red=colorama.Fore.RED,
            line="-" * 60,
            trace="".join(traceback.format_exception(*exc_info)),
            nc=colorama.Fore.RESET,
        )
    )
