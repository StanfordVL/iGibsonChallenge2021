Parameters and Implementation Details
=============


Parameters that used in this challenge is specified as below:
| <b>AGENT</b>                 |                                              |
|----------------------------|----------------------------------------------|
| HEIGHT (m)                 | 0.88m                                         |
| MASS (kg)                  | 9.2                                           |
| RADIUS                     | 0.18m                                         |
| SENSORS                    | [RGB_SENSOR, DEPTH_SENSOR, POINTGOAL_SENSOR, VELOCITIES_SENSOR] |
| POSSIBLE_ACTIONS           | [LINEAR_VELOCITY, ANGULAR_VELOCITY]           |
| MAX_EPISODE_STEPS          | 500                                          |
|                            |                                              |
| <b>DEPTH_SENSOR</b>              |                                              |
| HEIGHT                     | 180                                          |
| WIDTH                      | 320                                          |
| Horizontal FOV             | 69.4                                         |
| ORIENTATION (Euler angles) | [0, 0.3490659, 0] # 20 degrees               |
| MAX_DEPTH                  | 10                                           |
| MIN_DEPTH                  | 0.1                                          |
| NORMALIZE_DEPTH*            | TRUE                                         |
| POSITION                   | [0, 0, 0.88]                                 |
|                            |                                              |
| <b>RGB_SENSOR</b>              |                                              |
| HEIGHT                     | 180                                          |
| WIDTH                      | 320                                          |
| Horizontal FOV             | 69.4                                         |
| ORIENTATION (Euler angles) | [0, 0.3490659, 0] # 20 degrees               |
| POSITION                   | [0, 0, 0.88]                                 |
|                            |                                              |
| <b>POINTGOAL_SENSOR</b>  |                                              |
| GOAL_FORMAT:               | POLAR                                        |
|                            |                                              |
| <b>SUCCESS CRITERIA </b>           |                                              |
| SUCCESS_GEODESIC_DISTANCE  | 0.36m                                         |

<b>*</b> Depth Normalization:

We do depth normalization as the following (pesudo code):

```
depth[isnan(depth)] = 0
depth[depth > high] = 0
depth[depth < low] = 0
normalized_depth = depth / high
```
