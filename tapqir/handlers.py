# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: Apache-2.0

import funsor
from pyro.contrib.funsor.handlers.named_messenger import NamedMessenger
from pyro.contrib.funsor.handlers.plate_messenger import IndepMessenger
from pyro.contrib.funsor.handlers.primitives import to_funsor
from pyro.contrib.funsor.handlers.runtime import _DIM_STACK
from pyro.contrib.funsor.handlers.trace_messenger import _mask_fn
from pyro.poutine.broadcast_messenger import BroadcastMessenger
from pyro.poutine.runtime import effectful
from pyro.poutine.subsample_messenger import _Subsample
from pyro.poutine.trace_messenger import TraceMessenger as OrigTraceMessenger

funsor.set_backend("torch")


class vectorized_markov(NamedMessenger):
    """
    Construct for Markov chain of variables designed for efficient elimination of Markov
    dimensions using the parallel-scan algorithm. Whenever permissible,
    :class:`~pyro.contrib.funsor.vectorized_markov` is interchangeable with
    :class:`~pyro.contrib.funsor.markov`.

    The for loop generates both :class:`int` and 1-dimensional :class:`torch.Tensor` indices:
    :code:`(0, ..., history-1, torch.arange(0, size-history), ..., torch.arange(history, size))`.
    :class:`int` indices are used to initiate the Markov chain and :class:`torch.Tensor` indices
    are used to construct vectorized transition probabilities for efficient elimination by
    the parallel-scan algorithm.

    When ``history==0`` :class:`~pyro.contrib.funsor.vectorized_markov` behaves
    similar to :class:`~pyro.contrib.funsor.plate`.

    After the for loop is run, Markov variables are identified and then the ``step``
    information is constructed and added to the trace. ``step`` informs inference algorithms
    which variables belong to a Markov chain.

    .. code-block:: py

        data = torch.ones(3, dtype=torch.float)

        def model(data, vectorized=True):

            init = pyro.param("init", lambda: torch.rand(3), constraint=constraints.simplex)
            trans = pyro.param("trans", lambda: torch.rand((3, 3)), constraint=constraints.simplex)
            locs = pyro.param("locs", lambda: torch.rand(3,))

            markov_chain = \\
                pyro.vectorized_markov(name="time", size=len(data), dim=-1) if vectorized \\
                else pyro.markov(range(len(data)))
            for i in markov_chain:
                x_curr = pyro.sample("x_{}".format(i), dist.Categorical(
                    init if isinstance(i, int) and i < 1 else trans[x_prev]),

                pyro.sample("y_{}".format(i),
                            dist.Normal(Vindex(locs)[..., x_curr], 1.),
                            obs=data[i])
                x_prev = x_curr

        #  trace.nodes["time"]["value"]
        #  frozenset({('x_0', 'x_slice(0, 2, None)', 'x_slice(1, 3, None)')})
        #
        #  pyro.vectorized_markov trace
        #  ...
        #  Sample Sites:
        #      locs dist               | 3
        #          value               | 3
        #       log_prob               |
        #       x_0 dist               |
        #          value     3 1 1 1 1 |
        #       log_prob     3 1 1 1 1 |
        #       y_0 dist     3 1 1 1 1 |
        #          value               |
        #       log_prob     3 1 1 1 1 |
        #  x_slice(1, 3, None) dist   3 1 1 1 1 2 |
        #          value 3 1 1 1 1 1 1 |
        #       log_prob 3 3 1 1 1 1 2 |
        #  y_slice(1, 3, None) dist 3 1 1 1 1 1 2 |
        #          value             2 |
        #       log_prob 3 1 1 1 1 1 2 |
        #
        #  pyro.markov trace
        #  ...
        #  Sample Sites:
        #      locs dist             | 3
        #          value             | 3
        #       log_prob             |
        #       x_0 dist             |
        #          value   3 1 1 1 1 |
        #       log_prob   3 1 1 1 1 |
        #       y_0 dist   3 1 1 1 1 |
        #          value             |
        #       log_prob   3 1 1 1 1 |
        #       x_1 dist   3 1 1 1 1 |
        #          value 3 1 1 1 1 1 |
        #       log_prob 3 3 1 1 1 1 |
        #       y_1 dist 3 1 1 1 1 1 |
        #          value             |
        #       log_prob 3 1 1 1 1 1 |
        #       x_2 dist 3 1 1 1 1 1 |
        #          value   3 1 1 1 1 |
        #       log_prob 3 3 1 1 1 1 |
        #       y_2 dist   3 1 1 1 1 |
        #          value             |
        #       log_prob   3 1 1 1 1 |

    .. warning::  This is only valid if there is only one Markov
        dimension per branch.

    :param str name: A unique name of a Markov dimension to help inference algorithm
        eliminate variables in the Markov chain.
    :param int size: Length (size) of the Markov chain.
    :param int dim: An optional dimension to use for this Markov dimension.
        If specified, ``dim`` should be negative, i.e. should index from the
        right. If not specified, ``dim`` is set to the rightmost dim that is
        left of all enclosing :class:`~pyro.contrib.funsor.plate` contexts.
    :param int history: Memory (order) of the Markov chain. Also the number
        of previous contexts visible from the current context. Defaults to 1.
        If zero, this is similar to :class:`~pyro.contrib.funsor.plate`.
    :return: Returns both :class:`int` and 1-dimensional :class:`torch.Tensor` indices:
        ``(0, ..., history-1, torch.arange(size-history), ..., torch.arange(history, size))``.
    """

    def __init__(self, name=None, size=None, dim=None, history=1):
        self.name = name
        self.size = size
        self.dim = dim
        self.history = history
        super().__init__()

    @staticmethod
    @effectful(type="markov_chain")
    def _markov_chain(name=None, markov_vars=set(), suffixes=list()):
        """
        Constructs names of markov variables in the `chain`
        from markov_vars prefixes and suffixes.

        :param str name: The name of the markov dimension.
        :param set markov_vars: Markov variable name markov_vars.
        :param list suffixes: Markov variable name suffixes.
            (`0, ..., history-1, torch.arange(0, size-history), ..., torch.arange(history, size)`)
        :return: step information
        :rtype: frozenset
        """
        chain = frozenset(
            {
                tuple("{}{}".format(var, suffix) for suffix in suffixes)
                for var in markov_vars
            }
        )
        return chain

    def __iter__(self):
        self._auxiliary_to_markov = {}
        self._markov_vars = set()
        self._suffixes = []
        for self._suffix in range(self.history):
            self._suffixes.append(self._suffix)
            yield (self._suffix, self._suffix)
        with self:
            with IndepMessenger(
                name=self.name, size=self.size - self.history, dim=self.dim
            ) as time:
                time_indices = [time.indices + i for i in range(self.history + 1)]
                time_slices = [
                    slice(i, self.size - self.history + i)
                    for i in range(self.history + 1)
                ]
                self._suffixes.extend(time_slices)
                for self._suffix, self._indices in zip(time_slices, time_indices):
                    yield (self._suffix, self._indices)
        self._markov_chain(
            name=self.name, markov_vars=self._markov_vars, suffixes=self._suffixes
        )

    def _pyro_sample(self, msg):
        if type(msg["fn"]).__name__ == "_Subsample":
            return
        BroadcastMessenger._pyro_sample(msg)
        if str(self._suffix) != str(self._suffixes[-1]):
            # _do_not_score: record these sites when tracing for use with replay,
            # but do not include them in ELBO computation.
            msg["infer"]["_do_not_score"] = True
            # map auxiliary var to markov var name prefix
            # assuming that site name has a format: "markov_var{}".format(_suffix)
            # is there a better way?
            markov_var = msg["name"][: -len(str(self._suffix))]
            self._auxiliary_to_markov[msg["name"]] = markov_var

    def _pyro_post_sample(self, msg):
        """
        At the last step of the for loop identify markov variables.
        """
        if type(msg["fn"]).__name__ == "_Subsample":
            return
        # if last step in the for loop
        if str(self._suffix) == str(self._suffixes[-1]):
            funsor_log_prob = (
                msg["funsor"]["log_prob"]
                if "log_prob" in msg.get("funsor", {})
                else to_funsor(msg["fn"].log_prob(msg["value"]), output=funsor.Real)
            )
            # for auxiliary sites in the log_prob
            for name in set(funsor_log_prob.inputs) & set(self._auxiliary_to_markov):
                # add markov var name prefix to self._markov_vars
                markov_var = self._auxiliary_to_markov[name]
                self._markov_vars.add(markov_var)


class trace(OrigTraceMessenger):
    """
    Setting ``pack_online=True`` packs online instead of after the fact,
    converting all distributions and values to Funsors as soon as they are available.

    Setting ``pack_online=False`` computes information necessary to do packing after execution.
    Each sample site is annotated with a ``dim_to_name`` dictionary,
    which can be passed directly to :func:`~pyro.contrib.funsor.to_funsor`.
    """

    def __init__(self, graph_type=None, param_only=None, pack_online=True):
        super().__init__(graph_type=graph_type, param_only=param_only)
        self.pack_online = True if pack_online is None else pack_online

    def _pyro_post_sample(self, msg):
        if msg["name"] in self.trace:
            return
        if "funsor" not in msg:
            msg["funsor"] = {}
        if isinstance(msg["fn"], _Subsample):
            return super()._pyro_post_sample(msg)
        if self.pack_online:
            if (
                "log_prob" not in msg["funsor"]
                and not msg["infer"].get("_do_not_trace")
                and not msg["infer"].get("_do_not_score", False)
            ):
                # optimization: don't perform this tensor op unless we have to
                fn_masked = _mask_fn(msg["fn"], msg["mask"])
                msg["funsor"]["log_prob"] = to_funsor(
                    fn_masked.log_prob(msg["value"]), output=funsor.Real
                )
                # TODO support this pattern which uses funsor directly - blocked by casting issues
                # msg["funsor"]["log_prob"] = msg["funsor"]["fn"](**{msg["name"]: msg["funsor"]["value"]})
            if msg["scale"] is not None and "scale" not in msg["funsor"]:
                msg["funsor"]["scale"] = to_funsor(msg["scale"], output=funsor.Real)
        else:
            # this logic has the same side effect on the _DIM_STACK as the above,
            # but does not perform any tensor or funsor operations.
            msg["funsor"]["dim_to_name"] = _DIM_STACK.names_from_batch_shape(
                msg["fn"].batch_shape
            )
            msg["funsor"]["dim_to_name"].update(
                _DIM_STACK.names_from_batch_shape(
                    msg["value"].shape[
                        : len(msg["value"]).shape - len(msg["fn"].event_shape)
                    ]
                )
            )
        return super()._pyro_post_sample(msg)

    def _pyro_post_markov_chain(self, msg):
        # add step information to the trace
        self.trace.add_node(msg["name"], **msg.copy())
