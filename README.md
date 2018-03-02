# Disparity Sliding Window: Object Proposals from Disparity Images
This respository provides code for our disparity sliding window method, a real-time proposal generator. Paper is submitted for IROS 2018. Code is written in C++ and also provides a wrapper for usage in Python.

### Contents
1. [Requirements:](#requirements)
2. [Installation:](#installation)
3. [Demo:](#demo)

### Requirements:
```
1. OpenCV (we used 2.4)
2. Boost
```

### Installation:

1. Clone the DSW respository
```Shell
git clone https://github.com/julimueller/disparity-sliding-window
```
2. Build everything
```Shell
2. mkdir build && cd build
3. cmake .. -DCMAKE_INSTALL_PREFIX="YOUR_PATH" && make -j12 install
```
Note: "YOUR_PATH" has to be in LD_LIBRARY_PATH.

### Demo:
Demo in C++:
```Shell
disparity_sliding_window_test
```

Demo in Python:

```
PYTHONPATH=PYTHONPATH:"dir_to_dsw_python.so"
import dsw_python
```
