import sys
import os
from os import path as osp

# print(sys.path)


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
        # print(sys.path)
        # print("PYTHONPATH:", os.environ["PYTHONPATH"])
        try:
            os.environ["PYTHONPATH"] = path + ":" + os.environ["PYTHONPATH"]
        except KeyError:
            os.environ["PYTHONPATH"] = path


this_dir = osp.dirname(__file__)

# Add lib to PYTHONPATH
lib_path = os.path.abspath(osp.join(this_dir, '../', 'lib'))
# print(lib_path)
add_path(lib_path)
