# CS445_Visual_Odometry
Final Project for CS 445 (2020)

## SIFT_plus_RANSAC

## SuperPoint_plus_RANSAC
ipython notebook 

## KITTI dataset
### Raw data structure
- Download odometry data (grey) synced + rectified from [here](http://www.cvlibs.net/datasets/kitti/eval_odometry.php).
- Copy the ground truth poses from [here] (http://www.cvlibs.net/download.php?file=data_odometry_poses.zip)

- Setup the directory structure for SuperPoint + RANSAC as follows:
```
`-- KITTI_data (data_odometry_gray, data_odometery_pose, data_odometery_calib)
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


