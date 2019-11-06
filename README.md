# lanes_detection_objects_fusion
Lanes detection (Deep Learning), Objects detection and Object-Lane fusion

### Lane detection uses Deep Learning vs. CV methods by default
  - Use polyfit (^2) by default
  - Curvature, CenterDeviation, Heading Angle Error
### Objects detection uses autoware.ai perception methods by default
  - DetectedObjectArray
  - Object Label, Distance Vector, Velocity Vector (TODO)
### Object-Lane fusion
  - Detected Objects to lanes relative to ego lane
  - Register lane id according to ADAS map and localization (GNSS and Visual)

