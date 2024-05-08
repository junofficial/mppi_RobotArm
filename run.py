import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sys_params import SYS_PARAMS
from utils import *
from control import MPPIControllerForPathTracking
import os
params = SYS_PARAMS()
sim_time = 10 
dt = 0.003
iter = 1500

#q = np.array([0.1,0.0])
q = np.array([1.152198236517471885e+00,-1.266101672070702344e+00])
dq = np.array([0.0, 0.0])
isDesturbance = 0

ref_path = np.loadtxt('xydq_circle.txt')
ref_path = ref_path[:,0:4]
print(ref_path.shape)
sim_steps = 10 # [steps]

state = [q[0],q[1],dq[0],dq[1]]

mppi = MPPIControllerForPathTracking(
    delta_t = dt * 2, # [s]
    ref_path = ref_path, # ndarray, size is <num_of_waypoints x 2>
    horizon_step_T = 30, # [steps]
    number_of_samples_K = 100, # [samples] # 500개의 가닥
    param_exploration = 0.0,
    param_lambda = 100.0,
    param_alpha = 0.98,
    sigma = np.array([[20.0, 0.0], [0.0, 20.0]]),
    stage_cost_weight = np.array([0.50, 0.50, 5.0, 5.0]), # weight for [q1, q2, dq1, dq2]
    terminal_cost_weight = np.array([5.0, 5.0, 50.0, 50.0]), # weight for [q1, q2, dq1, dq2]
    visualze_sampled_trajs = True
)

rq_rec = np.zeros((int(iter)+1, 2))
rx_rec = np.zeros((int(iter)+1, 2))
ry_rec = np.zeros((int(iter)+1, 2))
x_rec = np.zeros((int(iter)+1, 2))
y_rec = np.zeros((int(iter)+1, 2))
q_rec = np.zeros((int(iter)+1, 2))
u_rec = np.zeros((int(iter)+1, 2))
t_rec = np.zeros(int(iter)+1)

for k in range(1, int(iter) + 1): # 여기에서 
    u, optimal_input_sequence, optimal_traj, sampled_traj_list = mppi.calc_control_input(
            observed_x = state
        )
        
    dq += dt * Arm_Dynamic(q, dq, u) # dq에 ddq를 더하는 거 업데이트 하는거
    #dq = np.clip(dq,-0.4,0.4)
    q += dt * dq

    x1, y1, x2, y2 = Forward_Kinemetic(q)
    
    state = np.concatenate((q, dq))
    print(f"{k}     state = {state}")
    print(f"{k}     position = {x1,y1,x2,y2}")
    if k == 1:
        continue
    rq_rec[k, :] = q
    rx_rec[k, :] = ref_path[k,0:1]
    ry_rec[k, :] = ref_path[k,1:2]
    x_rec[k, :] = [x1, x2]
    y_rec[k, :] = [y1, y2]
    q_rec[k, :] = q
    u_rec[k, :] = u
    t_rec[k] = k

    if False:
        Joint_1 = [0, 0]
        Joint_2 = [1, 0]
        Joint_3 = [2, 0]    
        min_alpha_value = 0.25
        max_alpha_value = 0.35
        fig, ax = plt.subplots()  # create figure and axis objects
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(0, 1.5)
        ax.set_ylim(0, 1.5)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_title('Sampled Trajectories')
        Robot_arm_1, = ax.plot([0, x1], [0, y1], 'k', linewidth=4)
        Robot_arm_2, = ax.plot([x1, x2], [y1, y2], 'k', linewidth=4)
        for idx, sampled_traj in enumerate(sampled_traj_list):
            # draw darker for better samples
            alpha_value = (1.0 - (idx + 1) / len(sampled_traj_list)) * (max_alpha_value - min_alpha_value) + min_alpha_value
        
            #sampled_traj_x_offset = np.ravel(sampled_traj[:, 0]) - np.full(sampled_traj.shape[0], x2)
            sampled_traj_x_offset = np.ravel(sampled_traj[:, 0]) 
            #sampled_traj_y_offset = np.ravel(sampled_traj[:, 1]) - np.full(sampled_traj.shape[0], y2)
            sampled_traj_y_offset = np.ravel(sampled_traj[:, 1]) 
            x_sample = [1 * np.cos(q1) + 1 * np.cos(q1 + q2) for (q1, q2) in zip(sampled_traj_x_offset,sampled_traj_y_offset)]
            y_sample = [1 * np.sin(q1) + 1 * np.sin(q1 + q2) for (q1, q2) in zip(sampled_traj_x_offset,sampled_traj_y_offset)]
            ax.plot(x_sample, y_sample, color='gray', linestyle="solid", linewidth=0.2, zorder=4, alpha=alpha_value)
        x1 = [1 * np.cos(q1) + 1 * np.cos(q1 + q2) for q1,q2 in zip(optimal_traj[:,0],optimal_traj[:,1])]
        y1 = [1 * np.sin(q1) + 1 * np.sin(q1 + q2) for q1,q2 in zip(optimal_traj[:,0],optimal_traj[:,1])]
        #Joint_1 = np.array([0, 0])
        #Joint_2 = np.array([0.4, 0.9])
        #Joint_3 = np.array([1.4, 0.8])

        #Robot_arm_1 = ax.plot([Joint_1[0], Joint_2[0]], [Joint_1[1], Joint_2[1]], 'k',linewidth=4)
        #Robot_arm_2 = ax.plot([Joint_2[0], Joint_3[0]], [Joint_2[1], Joint_3[1]], 'k',linewidth=4)
        ax.plot(x1, y1, color='red', linestyle="solid", linewidth=1, zorder=4)
        #ax.plot(x_sample[0], y_sample[0], marker='o', linestyle='-', color='b')
        ax.plot(ref_path[:,0], ref_path[:,1], '--b')
        ax.set_aspect('equal', adjustable='box')  # set equal aspect ratio
        ax.set_xlabel('X Label')  # set x-axis label
        ax.set_ylabel('Y Label')  # set y-axis label
        ax.set_title('Sampled Trajectories')  # set title
        #plt.show()  # show the plot
        i = str(k)
        my_path = os.path.abspath(r'/home/jnu/Desktop/Control/example/MPPI_ARM_QDQ/image')
        plt.savefig(my_path + '/' + i, dpi=300, bbox_inches='tight')
        #plt.pause(0.1)
        
plt.figure(1)

plt.subplot(2, 2, 1)
plt.plot(t_rec, 180/np.pi*q_rec[:, 0], 'k', t_rec,
         180/np.pi*rq_rec[:, 0], '--b', linewidth=1.2)
plt.title('Theta 1 Input & Output')
plt.xlabel('Time(s)')
plt.ylabel('Theta (Deg)')
plt.axis([0, 10, -10, 160])
plt.legend(['Theta 1 Output', 'Theta 1 Input'])
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(t_rec, 180/np.pi*q_rec[:, 1], 'k', t_rec,
         180/np.pi*rq_rec[:, 1], '--b', linewidth=1.2)
plt.title('Theta 2 Input & Output')
plt.xlabel('Time(s)')
plt.ylabel('Theta (Deg)')
plt.axis([0, 10, -160, 10])
plt.legend(['Theta 2 Output', 'Theta 2 Input'])
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(t_rec, x_rec[:, 1], 'k', t_rec, rx_rec[:, 0], '--b', linewidth=1.2)
plt.title('X(end point) Input & Output')
plt.xlabel('Time(s)')
plt.ylabel('X (m)')
plt.axis([0, 10, -1, 4])
plt.legend(['X output', 'X input'])
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(t_rec, y_rec[:, 1], 'k', t_rec, ry_rec[:, 0], '--b', linewidth=1.2)
plt.title('Y(end point) Input & Output')
plt.xlabel('Time(s)')
plt.ylabel('Y (m)')
plt.axis([0, 10, -2, 4])
plt.legend(['Y output', 'Y input'])
plt.grid(True)

############################
plt.figure(2)

plt.subplot(2, 1, 1)
plt.plot(t_rec, u_rec[:, 0], 'k', linewidth=1.2)
plt.title('u(1)')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(t_rec, u_rec[:, 1], 'k', linewidth=1.2)
plt.title('u(2)')
plt.grid(True)

plt.show()
        
