# Disparity Sliding Window: Object Proposals from Disparity Images
This respository provides code for our disparity sliding window method, a real-time proposal generator. Paper is submitted for IROS 2018. Code is written in C++ and also provides a wrapper for usage in Python.

![alt text](https://github.com/julimueller/disparity-sliding-window/blob/master/images/principle.png)

### Contents
1. [Requirements:](#requirements)
2. [Installation:](#installation)
3. [Demo:](#demo)
4. [Citation:](#citation)

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
cd disparity-sliding-window/python
python dsw_test.py
```
Results should be close to the following:

![alt text](https://github.com/julimueller/disparity-sliding-window/blob/master/images/detections_kitti.png)

### Citation:
Of course we would be happy to be cited in your work. Use the following BibTeX entry:
```
@article{DBLP:journals/corr/abs-1805-06830,
  author    = {Julian M{\"{u}}ller and
               Andreas Fregin and
               Klaus Dietmayer},
  title     = {Disparity Sliding Window: Object Proposals From Disparity Images},
  journal   = {CoRR},
  volume    = {abs/1805.06830},
  year      = {2018},
  url       = {http://arxiv.org/abs/1805.06830},
  archivePrefix = {arXiv},
  eprint    = {1805.06830},
  timestamp = {Tue, 05 Jun 2018 18:50:11 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1805-06830},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
