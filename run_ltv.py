import os
import pdb
import time
import datetime
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


from functools import partial

from task import Tracking_LPV

from control import MPC_PY, MPC_CA


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_video",
        type=bool,
        default=False,
        help="Save the animation as a video file",
    )
    return parser


def run_mpc(env, ctr):
    obs = env.reset()
    t, n = 0, 0

    x_rec = []
    r_rec = []
    u_rec = []
    t_rec = []
    while t < env.sim_T:
        t = env.sim_dt * n
        print("t: ", t)
        obs, reward, done, info = env.step()

        n += 1
        x_rec.append(info["sys_state"])
        r_rec.append(info["traj_state"])
        u_rec.append(info["action"])
        t_rec.append(t)

        update = False
        if t >= env.sim_T:
            update = True
        # yield [info, t, update]

    x_rec = np.array(x_rec)
    r_rec = np.array(r_rec)
    u_rec = np.array(u_rec)
    t_rec = np.array(t_rec)
    # pdb.set_trace()

    fig, axs = plt.subplots(3, 1)
    axs[0].plot(t_rec, r_rec[:, 0], "r-", label="r(1)")
    axs[0].plot(t_rec, x_rec[:, 0], "b-.", label="x(1)")
    axs[0].set_title("x(1)")
    axs[0].grid(True)
    axs[0].legend(loc="upper right", ncol=2, fontsize=14)

    axs[1].plot(t_rec, r_rec[:, 1], "r-", label="r(2)")
    axs[1].plot(t_rec, x_rec[:, 1], "b-.", label="x(2)")
    axs[1].set_title("x(2)")
    axs[1].grid(True)
    axs[1].legend(loc="upper right", ncol=2, fontsize=14)

    axs[2].plot(t_rec, u_rec[:, 0], "r-", label="u")
    axs[2].set_title("u(1)")
    axs[2].grid(True)
    axs[2].legend(loc="upper right", ncol=2, fontsize=14)
    plt.tight_layout()
    plt.show()


def main():
    #
    args = arg_parser().parse_args()
    sim_T = 20.0
    mpc_T = 10
    sim_dt = 0.1
    ctr = MPC_CA(mpc_T, sim_dt)
    env = Tracking_LPV(ctr, sim_T, sim_dt)
    run_mpc(env, ctr)

    # sim_visual = SimVisual(env)


if __name__ == "__main__":
    main()
