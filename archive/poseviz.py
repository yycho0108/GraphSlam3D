import pickle
import rospy
import tf
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

def main():
    rospy.init_node('poseviz', anonymous=True)
    tfb = tf.TransformBroadcaster()

    pub_bl = rospy.Publisher('base_link', PoseStamped, queue_size=10)
    pub_ble = rospy.Publisher('base_link_est', PoseStamped, queue_size=10)
    pub_z = rospy.Publisher('landmark', PoseArray, queue_size=10)
    pub_ze = rospy.Publisher('landmark_est', PoseArray, queue_size=10)
    pub_zr = rospy.Publisher('landmark_raw', PoseArray, queue_size=10)

    with open('/tmp/data.pkl', 'r') as f:
        data = pickle.load(f)

    rate = rospy.Rate(10)

    for (p,q,zs,ep,eq,ezs,rzs) in data:
        t = rospy.Time.now()
        pub_bl.publish( msg1(p,q,t))
        pub_ble.publish( msg1(ep,eq,t))
        pub_z.publish(msgn(zs, t))
        pub_ze.publish(msgn(ezs, t))
        pub_zr.publish(msgn(rzs, t))

        #tfb.sendTransform(
        #        translation=p,
        #        rotation=q,
        #        time=t,
        #        child='base_link', 
        #        parent='map')

        #print p
        #for _ in range(10):

        now = t#rospy.Time.now()
        # ground truth
        tfb.sendTransform(p, q, now, 'base_link', 'map')
        for zi, (zp,zq) in enumerate(zs):
            tfb.sendTransform(zp, zq, now, 'landmark_{}'.format(zi), 'map')

        tfb.sendTransform(ep, eq, now, 'base_link_est', 'map')
        for zi, (zp,zq) in enumerate(ezs):
            tfb.sendTransform(zp, zq, now, 'landmark_{}_est'.format(zi), 'map')


        rate.sleep()

        if rospy.is_shutdown():
            return

if __name__ == "__main__":
    main()
