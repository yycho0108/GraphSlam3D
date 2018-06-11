# GraphSlam3

Last Modified by Yoonyoung Cho @ June 11th, 2018

This project implements Online/Offline Graph Slam in 3-Dimensions, including rotations.

The implementation is independent from, but follows the theory as described in [A Tutorial on Graph-Based Slam](https://ieeexplore.ieee.org/document/5681215/) by Giorgio Grisetti.

Adaptation from its original version to the online (incremental) variant closely followed Sebastian Thrun's method introduced in [CS373 Udacity Course](https://classroom.udacity.com/courses/cs373), Artificial Intelligence for Robotics. namely, the the new information matrix and the coefficients are computed through the [block-matrix inverse](https://en.wikipedia.org/wiki/Block_matrix#Block_matrix_inversion).

Each Landmark is identified by its fully qualified pose, such as would be the case for april tags, for instance.

Orientations are represented by quaternions in the state-form, and are treated with minimal parametrization with `T(q) = h v`.

To run, simply type the following in the terminal:

```bash
python main.py
```

Which would produce an output such as:

```bash
======== Ground Truth ========
final pose
(array([-60.5253, -63.891 ,  12.3384]), array([-0.7555,  0.379 ,  0.1496,  0.513 ]))
landmarks
[[array([ 17.4111, -16.5466, -10.9947])
  array([-0.2441, -0.1224,  0.0396,  0.9612])]
 [array([  1.7009, -23.3336,  -1.7232])
  array([-0.0876, -0.0553,  0.3559,  0.9288])]
 [array([10.3241, -7.3791, 13.1882])
  array([0.3553, 0.8674, 0.2175, 0.272 ])]
 [array([  6.2393, -10.2777,  15.8645])
  array([ 0.2804, -0.1007, -0.4581,  0.8375])]]
--------              --------


======== Raw Results ========
final pose
[-44.1774 -71.2953   2.4658  -0.756    0.1418   0.3163   0.5552]
landmarks
[Not available at this time]
--------             --------


======== Online ========
final pose
[-67.3803 -64.57     8.5176  -0.7206   0.3921   0.1738   0.5448]
landmarks
[[ 17.3581 -16.3457 -12.3706  -0.2125  -0.1368  -0.0331   0.967 ]
 [  1.7459 -22.9052  -3.2887  -0.0831  -0.1383   0.2638   0.951 ]
 [ 11.0464  -8.0334  13.4041   0.4288   0.7506   0.3377   0.3725]
 [  7.2658 -11.3247  14.3272   0.3098  -0.1079  -0.4167   0.8478]]
--------        --------


======== Offline ========
final pose
[-58.7092 -65.6913  12.2601  -0.7402   0.3844   0.1879   0.5187]
landmarks
[[ 16.8471 -15.723  -13.4576  -0.2169  -0.0942   0.0511   0.9703]
 [  1.849  -23.2313  -3.5829  -0.054   -0.0349   0.365    0.9288]
 [ 11.1996  -7.926   11.5698   0.3628   0.8719   0.2306   0.2345]
 [  7.338  -11.013   14.3555   0.2871  -0.06    -0.4634   0.8362]]
--------         --------
```

Visualization in ROS with RViz had been supported in the past, and will be revived in the future.
