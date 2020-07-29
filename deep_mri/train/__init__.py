"""
Module for automatic start of the experiments defined by configuration file.
This model also includes our callbacks for the tensorflow keras learning cycle and config file parser.
"""
from .training import run_train

__all__ = ["run_train"]
