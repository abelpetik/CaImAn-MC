#!/usr/bin/env python

import importlib.metadata

# Keep import-time dependencies small for motion-correction-only usage.
from caiman.base.movies import load, load_movie_chain, movie
from caiman.base.timeseries import concatenate
from caiman.cluster import start_server, stop_server

__version__ = importlib.metadata.version('caiman')
