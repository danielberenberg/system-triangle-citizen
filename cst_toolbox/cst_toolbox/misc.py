#!/usr/bin/env python
# -*- coding: utf-8 -*-

import shutil
import secrets
from pathlib import Path

class text_color:
    BLACK   = '\033[30m'
    RED     = '\033[31m'
    GREEN   = '\033[32m'
    YELLOW  = '\033[33m'
    BLUE    = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN    = '\033[36m'
    WHITE   = '\033[37m'
    LT_RED  = '\033[91m'

    # special
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def exists(f):
    """Existing path 'type'"""
    f = Path(f)
    if not f.exists():
        raise FileNotFoundError(f"{f} doesn't exist")
    return f 

class MockClient(object):
    def __init__(self, cluster, **client_params):
        self.cluster = cluster

    def submit(self, func, *args, **kwargs):
        return func(*args)

    def gather(self, futures):
        return futures

class MockCluster(object):
    def __init__(self, **cluster_params):
        for k, v in cluster_params.items():
            self.__setattr__(k, v)
    def scale(self, val):
        self.absolutely_critical_value = val

class WorkingDirectory(object):
    """Context manager for setting up intermediate disk usage
    with the option to clean up afterwards."""

    def __init__(self, name=None, parent=None, path=None, cleanup=True, **setup_params):
        parent = Path("/tmp" or parent)
        name   = secrets.token_hex(16) or str(name) 
        if path is not None:
            self.__dirname = Path(path)
        else:
            self.__dirname = Path(parent / name)

        self.cleanup = cleanup
        self.__params = setup_params
        self.set_defaults(exist_ok=True, parents=True)

    @property
    def setup_params(self):
        return self.__params

    def set_defaults(self, **defaults):
        for k, v in defaults.items():
            self.__params.setdefault(k, v)
            
    @property
    def dirname(self):
        return self.__dirname

    def setup(self, exist_ok=True, parents=True):
        self.dirname.mkdir(exist_ok=exist_ok, parents=parents)
        return self

    def run_cleanup(self):
        # TODO: add exclude option?
        clean = False
        if self.cleanup:
            shutil.rmtree(self.dirname)
            clean = True
        return clean

    def __enter__(self):
        self.setup(**self.setup_params)
        return self

    def __exit__(self, type, value, traceback):
        self.run_cleanup()
        return self

    def __str__(self):
        return self.dirname.__str__()

    def __repr__(self):
        return f"WorkingDirectory({self.dirname}, cleanup={self.cleanup})"




if __name__ == '__main__':
    pass

