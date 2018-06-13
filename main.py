import numpy as np
from gen_data import DataGenerator

from utils import qmath_np
from utils.parse_g2o import parse_g2o
from slam import GraphSlam3
import pickle
from matplotlib import pyplot as plt

def cat(a,b):
    return np.concatenate([a,b], axis=-1)

class Report():
    """ Provides separation between print calls"""
    def __init__(self, name):
        self._name = name
        self._title = ' {} '.format(self._name)

    def __enter__(self):
        print ''
        print '=' * 8 + self._title  + '=' * 8

    def __exit__(self, *args):
        print '-' * 8 + ' '*len(self._title) + '-' * 8
        print ''

class Print1(object):
    """ Print something once at iteration n. """
    def __init__(self, n=0):
        self._i = 0
        self._n = n
    def __call__(self, *args):
        if (self._i == self._n):
            print(args)
        self._i += 1

import rospy
from geometry_msgs.msg import PoseArray
from utils.cvt_pose import pose, msg1, msgn

def subsample_graph(nodes, edges, n):
    idx = np.random.choice(len(nodes), n, replace=False).tolist()
    idx = sorted(idx)

    nodes_2 = []
    edges_2 = []

    for n in nodes:
        if not n[0] in idx:
            continue
        nodes_2.append([idx.index(n[0]), n[1]])

    mx = 0
    for e in edges:
        if not (e[0] in idx and e[1] in idx):
            continue
        e0 = idx.index(e[0])
        e1 = idx.index(e[1])
        mx = max(e0, mx)
        mx = max(e1, mx)
        edges_2.append([e0, e1, e[2], e[3]])

    edges_2 = sorted(edges_2, key=lambda e:e[0])

    return nodes_2, edges_2

#def g2o_main():
#    rospy.init_node('poseviz', anonymous=True)
#    pub_x0 = rospy.Publisher('node0', PoseArray, queue_size=10)
#    pub_x1 = rospy.Publisher('node1', PoseArray, queue_size=10)
#
#    filename = '/home/jamiecho/Downloads/sphere_bignoise_vertex3.g2o'
#    #filename = '/home/jamiecho/Downloads/tinyGrid3D.g2o'
#    nodes, edges = parse_g2o(filename)
#
#    # subsample ...
#    nodes, edges = subsample_graph(nodes, edges, 200)
#    #idx = np.random.choice(len(nodes), 200, replace=False)
#    #nodes = [nodes[i] for i in idx]
#    #edges = [e for e in edges if (e[0] in idx and e[1] in idx)]
#
#    slam = GraphSlam3(None, l=0.1)
#    slam.initialize_n(nodes)
#    x0 = [qmath_np.x2pq(n[1]) for n in nodes]
#    x1 = [qmath_np.x2pq(n) for n in slam.run(edges, max_nodes=len(nodes), n_iter=100, debug=True, tol=1e-4)]
#
#    r = rospy.Rate(10)
#    while not rospy.is_shutdown():
#        t = rospy.Time.now()
#        pub_x0.publish(msgn(x0, t))
#        pub_x1.publish(msgn(x1, t))
#        r.sleep()

def main():
    # configure parameters ...

    ## main params ##
    n_t = 200         # number of timesteps
    n_l = 4           # number of landmarks
    p_obs = 0.2       # probability of landmark observation
    seed = None # set the seed here, for repeatable experiments, or None
    v    = 4.0
    map_scale = 100.0 # landmark/pose initialization
    marquadt  = 100.0   # marquadt smoothing; TODO : tune
    n_ofl_it  = 100   # number of offline slam iterations
    #################

    ## noise params ##
    # when s= 1.0 = 1m = 57.2 deg.
    s = 0.05
    dx_p = s # motion position noise
    dx_q = s # motion orientation noise
    dz_p = s # landmark position noise
    dz_q = s # landmark orientation noise
    ##################

    # fix the state for both offline and online slam
    # TODO : use one data for offline/online
    # rather than specifying DataGenerator(stepwise=True)

    np.random.seed(seed)
    rs = np.random.get_state() # random state

    gen  = DataGenerator(n_t=n_t, n_l=n_l, scale=map_scale)

    np.set_printoptions(precision=4)
    with np.errstate(invalid='raise'):
        max_nodes = n_t + n_l
        slam = GraphSlam3(n_l, l=marquadt)

        # V2 : Online Version
        np.random.set_state(rs)
        xs, zs, obs = gen(
                dx_p,dx_q,dz_p,dz_q,
                p_obs=p_obs,
                stepwise=True
                )

        # store ...
        gt_p, gt_q = zip(*[x for x in xs[1:]])

        with Report('Ground Truth'):
            print 'final pose'
            print xs[-1]
            print 'landmarks'
            print np.asarray(zs)

        # compute raw results ...
        xes_raw = []
        x_raw = cat(*xs[0])
        for dx, o, z in obs:
            x_raw = qmath_np.xadd_rel(x_raw, dx, T=False)
            xes_raw.append(x_raw.copy())

        with Report('Raw Results'):
            print 'final pose'
            print x_raw
            print 'landmarks'
            print '[Not available at this time]'

        # store ...
        raw_p, raw_q = zip(*[qmath_np.x2pq(x) for x in xes_raw])

        dp = np.subtract(gt_p, raw_p)
        dq = [qmath_np.T(qmath_np.qmul(q1, qmath_np.qinv(q0))) for (q1,q0) in zip(gt_q, raw_q)]
        delta  = np.concatenate([dp,dq], axis=-1)
        delta  = np.linalg.norm(delta, axis=-1)
        plt.plot(delta)

        # online slam ...
        slam.initialize(cat(*xs[0]))
        xes_onl = []
        zes_onl = []
        for dx, o, z in obs:
            es = slam.step(dx, o, z)
            xes_onl.append(es[1].copy())
            zes_onl.append(np.copy(es[2:]))

        with Report('Online'):
            print 'final pose'
            print es[1]
            print 'landmarks'
            print np.asarray(es[2:])

        onl_p, onl_q= zip(*[qmath_np.x2pq(x) for x in xes_onl])
        dp = np.subtract(gt_p, onl_p)
        dq = [qmath_np.T(qmath_np.qmul(q1, qmath_np.qinv(q0))) for (q1,q0) in zip(gt_q, onl_q)]
        delta  = np.concatenate([dp,dq], axis=-1)
        delta  = np.linalg.norm(delta, axis=-1)
        plt.plot(delta)

        # offline slam ...
        if n_t < 1000:
            # try not to do this when matrix will get too big
            np.random.set_state(rs)
            xs, zs, obs = gen(
                    dx_p,dx_q,dz_p,dz_q,
                    p_obs=p_obs,
                    stepwise=False
                    )

            slam._nodes = {}
            slam.initialize(cat(*xs[0]))
            xes_ofl = slam.run(obs, max_nodes=max_nodes, n_iter=10)

            with Report('Offline'):
                print 'final pose'
                print xes_ofl[n_t-1]
                print 'landmarks'
                print np.asarray(es[-n_l:])

            ofl_p, ofl_q= zip(*[qmath_np.x2pq(x) for x in xes_ofl[1:n_t]])
            dp = np.subtract(gt_p, ofl_p)
            dq = [qmath_np.T(qmath_np.qmul(q1, qmath_np.qinv(q0))) for (q1,q0) in zip(gt_q, ofl_q)]
            delta  = np.concatenate([dp,dq], axis=-1)
            delta  = np.linalg.norm(delta, axis=-1)

            plt.plot(delta)

        plt.legend(['raw','slam-onl', 'slam-ofl'])
        plt.title('Estimated Error Over Time')
        plt.show()

        zs     = [zs for _ in range(n_t)] # repeat zs in time
        zes    = [[qmath_np.x2pq(e) for e in tmp] for tmp in zes_onl] # z estimates
        #(p,q,zs,ep,eq,ezs,rzs)
        with open('/tmp/data.pkl', 'w+') as f:
            #ofl_p, ofl_q, ofl_z_p, ofl_z_q should be incorporated later
            pickle.dump(zip(*[
                gt_p,gt_q,zs,
                onl_p, onl_q,
                raw_p, raw_q, zes]), f)


if __name__ == "__main__":
    main()
    #g2o_main()
