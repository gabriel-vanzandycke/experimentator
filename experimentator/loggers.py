import io
import logging
from functools import cached_property
from .base_experiment import BaseExperiment
from .utils import insert_suffix

class TqdmToLogger(io.StringIO):
    buffer = ''
    def __init__(self, logger, level=logging.DEBUG):
        super().__init__()
        self.logger = logger
        self.level = level
    def write(self, buffer):
        self.buffer = buffer.strip('\r\n\t ')
    def flush(self):
        self.logger.log(self.level, self.buffer)

class LoggingExperiment(BaseExperiment):
    @cached_property
    def logger(self):
        identity = self.cfg.get("worker_id", None)
        suffix = ".{}".format(identity) if identity is not None else ""
        logger = logging.getLogger("experimentator{}".format(suffix))
        if identity is not None and logger.parent.handlers and not logger.handlers:
            parent_handlers = [h for h in logger.parent.handlers if isinstance(h, logging.FileHandler)]
            name = insert_suffix(parent_handlers[0].baseFilename, f".{identity}")
            handler = logging.FileHandler(name, mode="w")
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)
            handler.setFormatter(logging.Formatter("[%(levelname)s]%(filename)s:%(lineno)d: %(message)s"))
            logging.info("Logging in %s", name)
        return logger

    def progress(self, generator, **kwargs):
        stream = TqdmToLogger(self.logger, level=logging.DEBUG)
        return super().progress(generator, file=stream, **kwargs)
