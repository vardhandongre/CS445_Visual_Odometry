# CS445_Visual_Odometry
Final Project for CS 445 (2020)

## SIFT_plus_RANSAC
@ Details of Implementation:
- Modfied for Own datset and SIFT Tracking
- Better Outlier rejection 
- Imporved flow tracking - minimized dynamic object keypoint selection 
- Scene -based masking
- Semantic segmentation maskign (remaining)

# Setup and excecution :
1. **DatasetReaderKITTI** is responsible for loading frames from [KITTI Visual Odometry Dataset](http://www.cvlibs.net/datasets/kitti/eval_odometry.php)
2. Run keypoint detection using SIFT and then track these keypoints with **FeatureTracker** that uses OpenCV Sparse optical flow
3. Load calib.txt and poses.txt for the corresponding sequecne from [KITTI Visual Odometry Dataset], and Update Trajectory path while execution.
4. Scene-based masking: Mask sky region (1/4th of top frame) and edges of frame
5. Tune RANSAC theshold (if required) 

# own-dataset
1. Convert video to frames (resize if needed)
2. Setup data_directory to folder with video_frames

## SuperPoint_plus_RANSAC
ipython notebook 

## KITTI dataset
### Raw data structure
- Download odometry data (grey) synced + rectified from [here](http://www.cvlibs.net/datasets/kitti/eval_odometry.php).
- Copy the ground truth poses from [here] (http://www.cvlibs.net/download.php?file=data_odometry_poses.zip)

```
`-- KITTI_data (raw data, odometry sequences, GT poses)
|   |-- data_odometry_gray
|   |   |-- 00
|   |   |   |-- image_00/
|   |   |   `-- ...
|   |   |-- ...
|   |   `-- 01
|   |   |   |-- image_00/
|   |   |   `-- ...
|   |   |-- ...
|   |-- data_odometery_pose
|   |   |-- dataset
|   |   |   |-- poses
|   |   |   |   |--00.txt
|   |   |-- |-- |...
|   |-- data_odometery_calib
|   |   |-- dataset
|   |   |   |-- sequences
|   |   |   |   |--00/
|   |   |   |   |  |--calib.txt
|   |   |   |   `-- ...


