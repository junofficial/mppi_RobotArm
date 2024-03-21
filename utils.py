import numpy as np
from sys_params import SYS_PARAMS

params = SYS_PARAMS()
m1 = params['m1']
m2 = params['m2']
l1 = params['l1']
l2 = params['l2']
lc1 = params['lc1']
lc2 = params['lc2']
g = params['g']


def Arm_Dynamic(q, dq, u):
    M11 = m1 * lc1 ** 2 + l1 + m2 * \
        (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * np.cos(q[1])) + l2
    M22 = m2 * lc2 ** 2 + l2
    M12 = m2 * l1 * lc2 * np.cos(q[1]) + m2 * lc2 ** 2 + l2
    M21 = m2 * l1 * lc2 * np.cos(q[1]) + m2 * lc2 ** 2 + l2
    M = np.array([[M11, M12], [M21, M22]])
    h = m2 * l1 * lc2 * np.sin(q[1])
    g1 = m1 * lc1 * g * np.cos(q[0]) + m2 * g * \
        (lc2 * np.cos(q[0] + q[1]) + l1 * np.cos(q[0]))
    g2 = m2 * lc2 * g * np.cos(q[0] + q[1])
    G = np.array([g1, g2])
    C = np.array([[-h * dq[1], -h * dq[0] - h * dq[1]], [h * dq[0], 0]])
    ddq = np.linalg.inv(M).dot(u - C.dot(dq) - G)

    return ddq


def Forward_Kinemetic(q):
    x1 = l1 * np.cos(q[0])
    y1 = l1 * np.sin(q[0])
    x2 = l1 * np.cos(q[0]) + l2 * np.cos(q[0] + q[1])
    y2 = l1 * np.sin(q[0]) + l2 * np.sin(q[0] + q[1])

    return x1, y1, x2, y2


def Inverse_Kinemetic(Theta):
    l1 = 1
    l2 = 1

    XE = 0.8 + 0.6 * np.cos(Theta)
    YE = 0.8 + 0.6 * np.sin(Theta)
    if 2 * np.pi - 0.2 <= Theta <= 2 * np.pi + 0.2:
        XE = 1.4
        YE = 0.8
    if Theta > 2 * np.pi + 0.2:
        XE = 2
        YE = 0

    term = np.sqrt(
        -XE ** 4 - 2 * XE ** 2 * YE ** 2 + 2 * XE ** 2 * l1 ** 2 + 2 * XE ** 2 * l2 ** 2 - YE ** 4 + 2 * YE ** 2 * l1 ** 2 + 2 * YE ** 2 * l2 ** 2 - l1 ** 4 + 2 * l1 ** 2 * l2 ** 2 - l2 ** 4)
    x1d = 2 * np.arctan((2 * YE * l1 + term) / (XE ** 2 +
                        2 * XE * l1 + YE ** 2 + l1 ** 2 - l2 ** 2))
    x2d = 2 * np.arctan((2 * YE * l1 - term) / (XE ** 2 +
                        2 * XE * l1 + YE ** 2 + l1 ** 2 - l2 ** 2))

    r = np.array([x1d, x2d - x1d])
    return r, XE, YE


def Feedback_linearization(q, dq, v):

    M11 = m1 * lc1 ** 2 + l1 + m2 * \
        (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * np.cos(q[1])) + l2
    M22 = m2 * lc2 ** 2 + l2
    M12 = m2 * l1 * lc2 * np.cos(q[1]) + m2 * lc2 ** 2 + l2
    M21 = m2 * l1 * lc2 * np.cos(q[1]) + m2 * lc2 ** 2 + l2
    M = np.array([[M11, M12], [M21, M22]])

    h = m2 * l1 * lc2 * np.sin(q[1])
    g1 = m1 * lc1 * g * np.cos(q[0]) + m2 * g * \
        (lc2 * np.cos(q[0] + q[1]) + l1 * np.cos(q[0]))
    g2 = m2 * lc2 * g * np.cos(q[0] + q[1])
    G = np.array([g1, g2])

    C = np.array([[-h * dq[1], -h * dq[0] - h * dq[1]], [h * dq[0], 0]])

    u = np.dot(M, v) + np.dot(C, dq) + G

    return u


def Controller(q, dq, r, dr, ddr):
    KD = 20
    KP = 100
    TH = q - r
    dTH = dq - dr
    v = ddr - (KD*(dTH)) - (KP*(TH))
    return v
    
