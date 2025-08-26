import os
import sys

def get_data_path(filename):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(project_root, "src", "data", filename)
