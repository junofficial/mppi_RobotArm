import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sys_params import SYS_PARAMS
from utils import *
from control import MPPIControllerForPathTracking
params = SYS_PARAMS()
sim_time = 10 # 
dt = 0.02
#dt = params['Ts']
#iter = sim_time/0.2
iter = 100
q = np.array([1.152197950636272505e+00, -1.266103672779498979e+00])
#q = np.array([0.0,0.0])
dq = np.array([0.5, 0.5])

trajName = 'circle'
isDesturbance = 0

ref_path = np.loadtxt('trajectory1.txt')
#ref_path = ref_path[:,2:4]

print(ref_path.shape)
delta_t = 0.02 # [sec]
sim_steps = 10 # [steps]
#print(f"ref_path = {ref_path}")
#print(f"ref_path = {ref_path[:,0]}")
#print(f"ref_path = {ref_path[:,2:4]}")
#print(f"ref_path = {ref_path[:,2]}")
#print(f"ref_path = {ref_path[:,3]}")

state = [q[0],q[1],dq[0],dq[1]]

mppi = MPPIControllerForPathTracking(
    delta_t = delta_t*2, # [s]
    ref_path = ref_path, # ndarray, size is <num_of_waypoints x 2>
    horizon_step_T = 20, # [steps]
    number_of_samples_K = 500, # [samples]
    param_exploration = 0.0,
    param_lambda = 100.0,
    param_alpha = 0.98,
    sigma = np.array([[0.4, 0.0], [0.0, 0.4]]),
    stage_cost_weight = np.array([1.0, 1.0, 1.0, 1.0]), # weight for [x, y, yaw, v]
    terminal_cost_weight = np.array([2.0, 2.0, 2.0, 2.0]), # weight for [x, y, yaw, v]
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
    
    t = k * dt

    u, optimal_input_sequence, optimal_traj, sampled_traj_list = mppi.calc_control_input(
            observed_x = state
        )

    dq += dt * Arm_Dynamic(q, dq, u) # dq에 ddq를 더하는 거 업데이트 하는거

    q += dt * dq

    x1, y1, x2, y2 = Forward_Kinemetic(q)
    
    
    state = np.concatenate((q, dq))
    print(f"{k}     state = {state}")
    print(f"{k}     position = {x1,y1,x2,y2}")
    if k == 1:
        continue

    x_rec[k, :] = [x1, x2]
    y_rec[k, :] = [y1, y2]
    q_rec[k, :] = q
    u_rec[k, :] = u
    t_rec[k] = t
    #if sampled_traj_list.any():
    if True:     
        min_alpha_value = 0.25
        max_alpha_value = 0.35
        fig, ax = plt.subplots()  # create figure and axis objects
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
        ax.plot(x1, y1, color='red', linestyle="solid", linewidth=1, zorder=4)
        ax.plot(x_sample[0], y_sample[0], marker='o', linestyle='-', color='b')
        ax.set_aspect('equal', adjustable='box')  # set equal aspect ratio
        ax.set_xlabel('X Label')  # set x-axis label
        ax.set_ylabel('Y Label')  # set y-axis label
        ax.set_title('Sampled Trajectories')  # set title
        plt.show()  # show the plot

############################
Joint_1 = [0, 0]
Joint_2 = [1, 0]
Joint_3 = [2, 0]
fig, ax = plt.subplots()
ax.axis('equal')
ax.set_xlim(-5, 5) # x,y축 범위 설정
ax.set_ylim(-5, 5)
ax.grid(True)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title('Robot Movement')

Robot_arm_1, = ax.plot([Joint_1[0], Joint_2[0]], [
                       Joint_1[1], Joint_2[1]], 'k', linewidth=4)
Robot_arm_2, = ax.plot([Joint_2[0], Joint_3[0]], [
                       Joint_2[1], Joint_3[1]], 'k', linewidth=4)
Robot_path, = ax.plot([Joint_3[0]-0.01, Joint_3[0]],
                      [Joint_3[1]-0.01, Joint_3[1]], 'r.', linewidth=0.5)
Target_path, = ax.plot(ref_path[:, 2], ref_path[:, 3], '--b')

path_x, path_y = [], []


def update(frame):
    Robot_X1 = x_rec[frame, 0]
    Robot_Y1 = y_rec[frame, 0]
    Robot_X2 = x_rec[frame, 1]
    Robot_Y2 = y_rec[frame, 1]

    Robot_arm_1.set_data([Joint_1[0], Robot_X1], [Joint_1[1], Robot_Y1])
    Robot_arm_2.set_data([Robot_X1, Robot_X2], [Robot_Y1, Robot_Y2])
    Robot_path.set_data(Robot_X2, Robot_Y2)

    path_x.append(Robot_X2)
    path_y.append(Robot_Y2)
    Robot_path.set_data(path_x, path_y)

    return Robot_arm_1, Robot_arm_2, Robot_path


ani = animation.FuncAnimation(fig, update, frames=range(
    0, int(iter)+1, 10), blit=True, interval=5, repeat=False)
plt.show()

############################
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

