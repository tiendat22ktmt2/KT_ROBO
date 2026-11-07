from robot.model import Planar3R


def forward_kinematics(robot: Planar3R, q):

    return robot.fk(q)