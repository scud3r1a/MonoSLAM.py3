## MonoSLAM.py3
A prototype implementation of a monocular SLAM algorithm in Python 3

### Usage
```bash
export REVERSE=1   # Hack for reverse video
export F=500       # Focal length (in px)

./slam.py <video.mp4>

# good example
F=525 ./slam.py videos/test_freiburgxyz525.mp4

# ground truth
F=525 ./slam.py videos/test_freiburgrpy525.mp4 videos/test_freiburgrpy525.npz

# kitti example
REVERSE=1 F=984 ./slam.py videos/test_kitti984_reverse.mp4

# extract ground truth
tools/parse_ground_truth.py videos/groundtruth/freiburgrpy.txt videos/test_freiburgrpy525.npz 
```

### Libraries
* SDL2 for 2-D display
* cv2 for feature extraction
* pangolin for 3-D display
* g2opy for optimization (soon!)


### Classes
* Frame -- An image with extracted features
* Point -- A 3-D point in the Map and it's 2-D Frame correspondences
* Map -- A collection of points and frames
* Display2D -- SDL2 display of the current image
* Display3D -- Pangolin display of the current map

### References
- https://github.com/MagicLeapResearch/SuperPointPretrainedNetwork/blob/master/demo_superpoint.py

### License
All code is MIT licensed. Videos and libraries follow their respective licenses.