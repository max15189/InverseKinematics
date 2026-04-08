"""
ViperX-300 6-DOF robot arm parameters.

All physical constants for the robot — joint limits, home configuration,
and space-frame screw axes. Import from here whenever you need robot geometry.

Spec: https://docs.trossenrobotics.com/interbotix_xsarms_docs/specifications/vx300s.html
"""

import numpy as np

# Joint limits in radians, shape (6, 2): [[min, max], ...]
JOINT_LIMITS = np.deg2rad([
    [-180,  180],   # waist
    [-101,  101],   # shoulder
    [-101,   92],   # elbow
    [-180,  180],   # forearm roll
    [-107,  130],   # wrist angle
    [-180,  180],   # wrist rotate
])

# Home configuration — end-effector transform when all joints = 0
M_HOME = np.array([
    [1, 0, 0, 0.536494],
    [0, 1, 0, 0       ],
    [0, 0, 1, 0.42705 ],
    [0, 0, 0, 1       ],
], dtype=float)

# Space-frame screw axes — each COLUMN is [w; v] for one joint (shape 6×6)
SLIST = np.array([
    [0,  0,  1,  0,        0,       0      ],  # joint 1 (waist)
    [0,  1,  0, -0.12705,  0,       0      ],  # joint 2 (shoulder)
    [0,  1,  0, -0.42705,  0,       0.05955],  # joint 3 (elbow)
    [1,  0,  0,  0,        0.42705, 0      ],  # joint 4 (forearm roll)
    [0,  1,  0, -0.42705,  0,       0.35955],  # joint 5 (wrist angle)
    [1,  0,  0,  0,        0.42705, 0      ],  # joint 6 (wrist rotate)
]).T  # shape (6, 6)
