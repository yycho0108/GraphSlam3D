from geometry_msgs.msg import Pose, PoseStamped, PoseArray

def pose(p, q):
    msg = Pose()
    msg.position.x = p[0]
    msg.position.y = p[1]
    msg.position.z = p[2]
    msg.orientation.x = q[0]
    msg.orientation.y = q[1]
    msg.orientation.z = q[2]
    msg.orientation.w = q[3]
    return msg

def msg1(p, q, t):
    msg = PoseStamped()
    msg.pose = pose(p,q)
    msg.header.frame_id = 'map'
    msg.header.stamp = t
    return msg

def msgn(pqs, t):
    msg = PoseArray()
    msg.header.frame_id = 'map'
    msg.header.stamp = t
    msg.poses = [pose(p,q) for (p,q) in pqs]
    return msg

