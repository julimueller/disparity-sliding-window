# Disparity Sliding Window: Object Proposals from Disparity Images
This respository provides code for our disparity sliding window method, a real-time proposal generator. Paper is submitted for IROS 2018. Code is written in C++ and also provides a wrapper for usage in Python.

## Installing and Usage

```
cd build
cmake .. && make -j12 install
disparity_sliding_window_test
```

Note: Install path has to be in LD_LIBRARY_PATH. For Python:

```
PYTHONPATH=PYTHONPATH:"dir_to_dsw_python.so"
import dsw_python
```
