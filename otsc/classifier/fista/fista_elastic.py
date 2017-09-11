
from otsc.classifier.fista.fista_base import FISTABase


class FISTAElastic(FISTABase):

    def __init__(self, _run):
        self._run = _run

        # load configuration.
