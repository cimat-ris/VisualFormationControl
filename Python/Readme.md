Development code for formation control by means of vision
=====================================
Patricia Tavares.


1. *GroundTruth.py*

Consensus implemented with **relative positions and relative rotations with the known poses in the world frame**. It is contained in *GroundTruth.py*. You can select how many cameras you want in line: 37 and what formation (line_formation or circle_formation) in line: 39. The consensus is described in file *Functions/Control.py*, in the function named GTC. 
To run the program: 

```console
python GroundTruth.py
```

2. *HomographyDecompositionNoScale.py*

Uses the **homography** to update velocities. It is a **position based control** contained in *HomographyDecompositionNoScale.py*. It doesn't estimate the scale. You can select how many cameras do you want in line: 43 and what formation (line_formation or circle_formation) in line: 45. It verifies if the formation is infinitesimally rigid. The control is contained in *Functions/Control.py*, in the function named HDC. To run the program: 

```console
python HomographyDecompositionNoScale.py
```

3. *EssentialDecompositionNoScale.py* 

The next method is a consensus with n cameras using the information given from the **essential matrix**. It **decomposes the essential matrix** and doesn't estimate the scale. To change the amount of cameras go to line 43, and for changing the radius for the communication graph, go to line 50. It verifies if the formation is infinitesimally rigid. The control is contained in *Functions/control.py*, in the function named  EDC. Run: 

```console
python EssentialDecompositionNoScale.py 
```

Always run first the rand.py,


4. *HomographyDecompositionScale.py* 

Uses the **homography** to update velocities. It is a **position based control** contained in *HomographyDecompositionScale.py*. It estimates the scale. You can select how many cameras do you want in line: 43 and what formation (line_formation or circle_formation) in line: 45. It verifies if the formation is infinitesimally rigid. The control is contained in *Functions/Control.py*, in the function named HDC. To run the program: 

```console
python HomographyDecompositionNoScale.py
```

5. *EssentialDecompositionScale.py* 

The next part is a consensus with n cameras using the information given from the **essential matrix**. It **decomposes the essential matrix** and doesn't estimate the scale. To change the amount of cameras go to line 43, and for changing the radius for the communication graph, go to 55. It verifies if the formation is infinitesimally rigid. The control is contained in *Functions/control.py*, in the function named EDC. Run: 

```console
python EssentialDecompositionScale.py 
```
Always run first the rand.py


6. *RigidityHomography.py* 

Consensus with n cameras using the information **given from the homography matrix**. It **decomposes the homography matrix** and uses Schiano control to compute velocities. To change the amount of cameras go to line 44, and for changing the radius for the communication graph, go to 51. The control is contained in *Functions/Control.py*, in the function named RMF. Run: 

```console
python RigidityHomography.py
```

7. *RigidityEssential.py* 

Consensus with n cameras using the information given from the **essential matrix**. It decomposes the **essential matrix and uses Schiano control** to compute velocities. To change the amount of cameras go to line 44, and for changing the radius for the communication graph, go to 51. The control is contained in *Functions/Control.py*, in the function named RMF. Run: 

```console
python RigidityEssential.py
```

Extra implementations 
=====================================

1. *EssentialOneCamera.py*

Uses the essential matrix for one camera. It uses **opencv3 functions**. It needs a points cloud that can be obtained using the script in the cloud folder. To obtain a new point cloud run python *rand.py*, always run it first. You can change the target position in lines 25-34 and initial position for the current camera in 36-45. To run the program: 

```console
python EssentialOneCamera.py
```
