import numpy as np

def parse_g2o(filename):
    tui = np.triu_indices(6)
    with open(filename, 'r') as f:
        nodes = []
        edges = []
        for l in f.readlines():
            fields = l.strip().split(' ')
            if fields[0] == 'VERTEX_SE3:QUAT':
                v0 = np.int32(fields[1])
                x  = np.float32(fields[2:2+7])
                nodes.append( (v0, x) )
            if fields[0] == 'EDGE_SE3:QUAT':
                v0, v1 = np.int32(fields[1:3])
                x      = np.float32(fields[3:3+7])
                O = np.zeros((6,6))
                O[tui] = fields[3+7:3+7+21]
                O = O + O.T - np.diag(O.diagonal())
                edges.append( (v0, v1, x, O) )
    return nodes, edges

if __name__ == "__main__":
    filename = '/home/jamiecho/Downloads/sphere_bignoise_vertex3.g2o'
    parse_g2o(filename)
