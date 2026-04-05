
import numpy as np
import modern_robotics as mr

def IKinSpace_with_iters(Slist, M, T, thetalist0, eomg=1e-2, ev=1e-3):
    '''
    neumerical inverse kinematics taken from modern robotics library 
    added the number of iteratoin to the return values 
    '''
    thetalist = np.array(thetalist0).copy()
    i = 0
    maxiterations = 50
    Tsb = mr.FKinSpace(M,Slist, thetalist)
    Vs = np.dot(mr.Adjoint(Tsb),mr.se3ToVec(mr.MatrixLog6(np.dot(mr.TransInv(Tsb), T))))
    err = np.linalg.norm([Vs[0], Vs[1], Vs[2]]) > eomg or np.linalg.norm([Vs[3], Vs[4], Vs[5]]) >  ev
    while err and i < maxiterations:
        thetalist = thetalist + np.dot(np.linalg.pinv(mr.JacobianSpace(Slist,thetalist)), Vs)
        i = i + 1
        Tsb = mr.FKinSpace(M, Slist, thetalist)
        Vs = np.dot(mr.Adjoint(Tsb), \
                    mr.se3ToVec(mr.MatrixLog6(np.dot(mr.TransInv(Tsb), T))))
        err = np.linalg.norm([Vs[0], Vs[1], Vs[2]]) > eomg \
              or np.linalg.norm([Vs[3], Vs[4], Vs[5]]) > ev
    return (thetalist, not err,i)