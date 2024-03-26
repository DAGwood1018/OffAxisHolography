# OffAxisHolography
Code for performing off-axis holography.

Only extract_phase.py uses the C++ implementation of hologram filtering and phase unwrapping. Alternatively, hologram_filtering.py and phase_unwrapping.py give a pure Python implementation. To compile the C++ code, use CMake. C++ dependencies can be installed using the setup bash file. It can be installed as a Python package by navigation to the downloaded repo and executing "pip install .", but this is only currently setup/tested in a Linux environment. 
