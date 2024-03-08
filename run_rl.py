import os
import pdb
import time
import datetime
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#
from functools import partial

from reference_tracking import Tracking
from control import PID, MPC


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_video', type=bool, default=False,
                        help="Save the animation as a video file")
    return parser


def run_mpc(env, pid):
    obs = env.reset()
    t, n = 0, 0
    t0 = time.time()

    x_rec = []
    r_rec = []
    u_rec = []
    t_rec = []
    while t < env.sim_T:
        t = env.sim_dt * n
        print("t: ", t)
        action = pid.get_action(obs)
        print(action)
        obs, reward, done, info = env.step(action)
        t_now = time.time()
        # print(t_now - t0)

        t0 = time.time()
        #
        n += 1
        x_rec.append(obs[:3])
        r_rec.append(obs[6:9])
        u_rec.append(action)
        t_rec.append(t)
        # print("done: ", done)

        update = False
        if t >= env.sim_T:
            update = True
        # yield [info, t, update]
    x_rec = np.array(x_rec)
    r_rec = np.array(r_rec)
    u_rec = np.array(u_rec)
    t_rec = np.array(t_rec)

    fig, axs = plt.subplots(3, 1)
    axs[0].plot(t_rec, r_rec[:, 0], 'r-', label='ref')
    axs[0].plot(t_rec, x_rec[:, 0], 'b-.', label='uav')
    axs[0].set_title('x(1)')
    axs[0].grid(True)
    axs[0].legend(loc='upper right', ncol=2, fontsize=14)

    axs[1].plot(t_rec, r_rec[:, 1], 'r-', label='ref')
    axs[1].plot(t_rec, x_rec[:, 1], 'b-.', label='uav')
    axs[1].set_title('x(2)')
    axs[1].grid(True)
    axs[1].legend(loc='upper right', ncol=2, fontsize=14)

    axs[2].plot(t_rec, r_rec[:, 2], 'r-', label='ref')
    axs[2].plot(t_rec, x_rec[:, 2], 'b-.', label='uav')
    axs[2].set_title('x(3)')
    axs[2].grid(True)
    axs[2].legend(loc='upper right', ncol=2, fontsize=14)

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(t_rec, u_rec[:, 0])
    axs[0, 0].set_title('u(1)')
    axs[0, 0].grid(True)

    axs[0, 1].plot(t_rec, u_rec[:, 1])
    axs[0, 1].set_title('u(2)')
    axs[0, 1].grid(True)

    axs[1, 0].plot(t_rec, u_rec[:, 2])
    axs[1, 0].set_title('u(3)')
    axs[1, 0].grid(True)

    axs[1, 1].plot(t_rec, u_rec[:, 3])
    axs[1, 1].set_title('u(4)')
    axs[1, 1].grid(True)
    plt.tight_layout()
    plt.show()


def main():
    #
    args = arg_parser().parse_args()
    sim_T = 20.0
    sim_dt = 0.01
    # pid = PID(sim_T, sim_dt)
    mpc = MPC(sim_T, sim_dt)
    env = Tracking(mpc, sim_T, sim_dt)
    run_mpc(env, mpc)

    # sim_visual = SimVisual(env)


if __name__ == "__main__":
    main()
