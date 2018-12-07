import os

with open(
        os.path.join(os.path.dirname(__file__), os.path.pardir, 'VERSION')
) as version_file:
    __version__ = version_file.read().strip()
