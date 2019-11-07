# lanes_detection_objects_fusion
Lanes detection (Deep Learning), Objects detection and Object-Lane fusion

### Lane detection uses Deep Learning vs. CV methods by default
  - Lanenet DL detection model
  - Use polyfit (^2) by default
  - Curvature, CenterDeviation, Heading Angle Error

### Objects detection uses autoware.ai perception methods by default
  - DetectedObjectArray
  - Object Label, Distance Vector, Velocity Vector (TODO)

### Object-Lane fusion
  - Detected Objects to lanes relative to ego lane
  - Register lane id according to ADAS map and localization (TODO GNSS and Visual Cue)

### Software configuration:
  - Host: Ubuntu 16.04/18.04 LTS
  - Docker CE 19.03 (--gpus=all) or 18.09 (--runtime=nvidia)
  - CUDA: Ubuntu 16.04 (xenial): 10.0, Driver: 410.48
          Ubuntu 18.04 (bionic): 10.1, Driver: 418.87
  - CuDNN: 7.6.x
  - OpenCV: 4.0.0 (3.4+) CUDA enabled
  - TensorFlow: tensorflow-gpu 1.13.1
  - Python: 2.7.16
  - Autoware: 1.12
  - ROS:  melodic
  - CARLA: 0.9.5+

### Performance:
  - i7-8750H CPU @ 2.20GHz, 1 GTX1070 MaxQ, 32GB RAM: 10~15 FPS overall @800x600:
	- Object detections [DLCV '+' LiDAR]
	- Lane detections
	- Lane fitting
	- Lane fusion

### Future Improvement:
  - Photo geometry
  - Radar support 2 DetectedObject
  - Spline, etc. advanced fittings
  - Stereo/Monostereo/OFlow vision
  - Localization: GNSS/Vision/ADAS MAP
  - Add Qualcomm SNPE/DLC
  - TensorRT optimization

### Disclaimer:
  - Work is compiled together with OSS and reinvented code, free of use.
  - Contribution and improvement is encouraged.
