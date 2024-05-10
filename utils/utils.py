# import ctypes
# import fnmatch
# import importlib
# import inspect
# import numpy as np
# import os
# import shutil
# import sys
# import types
# import io
# import pickle
# import re
# import requests
# import html
# import hashlib
# import glob
# import tempfile
# import urllib
# import urllib.request
# import uuid

import json
import numpy as np
import os
from typing import Any, List, Tuple, Union, Optional


class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]

# for serializing numpy arrays to json
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

# save dictionary to filepath as json
def dict_to_json(dictionary, filepath):
    json_object = json.dumps(dictionary, indent=4, cls=NpEncoder)
    with open(filepath, "w") as outfile:
        outfile.write(json_object)

# read json into dictionary
def json_to_dict(filepath):
    with open(filepath, "r") as infile:
        return json.load(infile)
    
# save to numpy file
def save_array(array, filepath, compression=True):
    filepath = os.path.splitext(filepath)[0]
    if compression:
        filepath = filepath + ".npz"
        np.savez_compressed(filepath, array=array)
    else:
        filepath = filepath + ".npy"
        np.save(filepath, array)
    
# load numpy file
def load_array(filepath):
    ext = os.path.splitext(filepath)[1]
    if ext == ".npz":
        return np.load(filepath)["array"]
    elif ext == ".npy":
        return np.load(filepath)
    elif len(ext) == 0:
        if os.path.isfile(filepath+".npz"):
            return np.load(filepath+".npz")["array"]
        elif os.path.isfile(filepath+".npy"):
            return np.load(filepath+".npy")
    else:
        print("Error: file extension not recognized, must be .npy or .npz")
        return None