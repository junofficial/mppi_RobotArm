import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from matplotlib import patches
from matplotlib.animation import ArtistAnimation
from IPython import display
from scipy.ndimage import median_filter
from sys_params import SYS_PARAMS
import copy
params = SYS_PARAMS()
m1 = params['m1']
m2 = params['m2']
l1 = params['l1']
l2 = params['l2']
lc1 = params['lc1']
lc2 = params['lc2']
g = params['g']

class MPPIControllerForPathTracking():
    def __init__(
            self,
            delta_t: float = 0.01,
            ref_path: float = 0,
            horizon_step_T: int = 20,
            number_of_samples_K: int = 500,
            param_exploration: float = 0.0,
            param_lambda: float = 50.0,
            param_alpha: float = 1.0,
            sigma: np.ndarray = np.array([[10.0,10.0], [100.0, 100.0]]), 
            stage_cost_weight: np.ndarray = np.array([10.0, 10.0, 10.0, 10.0]), # weight for [x, y, q1, q2]
            terminal_cost_weight: np.ndarray = np.array([10.0, 10.0, 10.0, 10.0]), # weight for [x, y, yaw, v]
            visualize_optimal_traj = True,  # if True, optimal trajectory is visualized
            visualze_sampled_trajs = False, # if True, sampled trajectories are visualized
    ) -> None:
        """initialize mppi controller for path-tracking"""
        # mppi parameters
        self.dim_x = 4 # dimension of system state vector
        self.dim_u = 2 # dimension of control input vector
        self.T = horizon_step_T # prediction horizon
        self.K = number_of_samples_K # number of sample trajectories
        self.param_exploration = param_exploration  # constant parameter of mppi
        self.param_lambda = param_lambda  # constant parameter of mppi
        self.param_alpha = param_alpha # constant parameter of mppi
        self.param_gamma = self.param_lambda * (1.0 - (self.param_alpha))  # constant parameter of mppi
        self.Sigma = sigma # deviation of noise
        self.stage_cost_weight = stage_cost_weight
        self.terminal_cost_weight = terminal_cost_weight
        self.visualize_optimal_traj = visualize_optimal_traj
        self.visualze_sampled_trajs = visualze_sampled_trajs

        # vehicle parameters
        self.delta_t = delta_t #[s]
        self.ref_path = ref_path
        self.l1 = 1
        self.l2 = 1

        # mppi variables
        self.u_prev = np.array([[10.0, -2.0] for i in range(self.T)])

        #self.u_prev = np.full((self.T, self.dim_u),0.0)
        #self.u_prev[:,1] = 0.0

        # ref_path info
        self.prev_waypoints_idx = 0

    def calc_control_input(self, observed_x: np.ndarray) -> Tuple[float, np.ndarray]:
        """calculate optimal control input"""
        # 이전 제어입력을 가져와서
        u = self.u_prev
        # 현재의 state를 넣어줌
        x0 = observed_x #q1,q2,dq1,dq2

        # get the waypoint closest to current vehicle position 
        self._get_nearest_waypoint(x0[0], x0[1], update_prev_idx=True)
        if self.prev_waypoints_idx >= self.ref_path.shape[0]-1:
            print("[ERROR] Reached the end of the reference path.")
            raise IndexError

        # prepare buffer
        S = np.zeros((self.K)) # state cost list

        # sample noise
        epsilon = self._calc_epsilon(self.Sigma, self.K, self.T, self.dim_u) # size is self.K x self.T
        #print(epsilon)

        # prepare buffer of sampled control input sequence
        v = np.zeros((self.K, self.T, self.dim_u)) # control input sequence with noise

        # loop for 0 ~ K-1 samples
        for k in range(self.K):         
            # set initial(t=0) state x i.e. observed state of the vehicle
            x = x0 # q1,q2,dq1,dq2
            # loop for time step t = 1 ~ T
            for t in range(1, self.T+1):

                # get control input with noise
                if k < (1.0-self.param_exploration)*self.K:
                    v[k, t-1] = u[t-1] + epsilon[k, t-1] # sampling for exploitation
                else:
                    v[k, t-1] = epsilon[k, t-1] # sampling for exploration

                # update x
                x = self._F(x, self._g(v[k, t-1])) #q1,q2,dq1,dq2
                # add stage cost
                S[k] += self._c(x) + self.param_gamma * u[t-1].T @ np.linalg.inv(self.Sigma) @ v[k, t-1] # 아님

            # add terminal cost
            S[k] += self._phi(x) # 아님
        #print(f"1     S = {S}")
        # compute information theoretic weights for each sample
        w = self._compute_weights(S)
        #print(f"2     w = {w}")
        # calculate w_k * epsilon_k
        w_epsilon = np.zeros((self.T, self.dim_u))
        for t in range(self.T): # loop for time step t = 0 ~ T-1
            for k in range(self.K):
                w_epsilon[t] += w[k] * epsilon[k, t]
        #print(f"3     epsilon = {epsilon}")
        # apply moving average filter for smoothing input sequence
        #print(f"3     w_epsilon = {w_epsilon}")
        w_epsilon = self._moving_median_filter(xx=w_epsilon, window_size=10)
        #print(f"3-1     w_epsilon = {w_epsilon}")
        # update control input sequence
        #u = w_epsilon
        u += w_epsilon
        #print(f"4     u = {u}")
        # calculate optimal trajectory
        optimal_traj = np.zeros((self.T, self.dim_x))
        if self.visualize_optimal_traj:
            x = x0 # q1,q2,dq1,dq2
            for t in range(self.T):
                x = self._F(x, self._g(u[t-1]))
                optimal_traj[t] = x

        # calculate sampled trajectories
        sampled_traj_list = np.zeros((self.K, self.T, self.dim_x))
        sorted_idx = np.argsort(S) # sort samples by state cost, 0th is the best sample
        if self.visualze_sampled_trajs:
            for k in sorted_idx:
                x = x0 # q1,q2,dq1,dq2
                for t in range(self.T):
                    x = self._F(x, self._g(v[k, t-1]))
                    #print(x)
                    sampled_traj_list[k, t] = x

        # update privious control input sequence (shift 1 step to the left)
        self.u_prev[:-1] = u[1:]
        self.u_prev[-1] = u[-1]

        # return optimal control input and input sequence
        return u[0], u, optimal_traj, sampled_traj_list

    def _calc_epsilon(self, sigma: np.ndarray, size_sample: int, size_time_step: int, size_dim_u: int) -> np.ndarray:
        """sample epsilon"""
        # check if sigma row size == sigma col size == size_dim_u and size_dim_u > 0
        if sigma.shape[0] != sigma.shape[1] or sigma.shape[0] != size_dim_u or size_dim_u < 1:
            print("[ERROR] sigma must be a square matrix with the size of size_dim_u.")
            raise ValueError

        # sample epsilon
        mu = np.full((size_dim_u),0.0) # set average as a zero vector
        epsilon = np.random.multivariate_normal(mu, sigma, (size_sample, size_time_step))
        return epsilon

    def _g(self, v: np.ndarray) -> float:
        """clamp input"""
        ##############################
        # 제한두는 거인데 일단 꺼놓음
        #v[0] = np.clip(v[0], -0.8, 0.8) # limit steering input
        #v[1] = np.clip(v[1], -0.8, 0.8) # limit acceleraiton input
        return v

    def _c(self, x_t: np.ndarray) -> float:
        """calculate stage cost"""
        # parse x_t
        q1,q2,dq1,dq2 = x_t
        x = self.l1 * np.cos(q1) + self.l2 * np.cos(q1 + q2)
        y = self.l1 * np.sin(q1) + self.l2 * np.sin(q1 + q2)
        #yaw = ((yaw + 2.0*np.pi) % (2.0*np.pi)) # normalize theta to [0, 2*pi]

        # calculate stage cost
        _, ref_x, ref_y, ref_dq1, ref_dq2 = self._get_nearest_waypoint(q1, q2)
        stage_cost = self.stage_cost_weight[0]*(x-ref_x)**2 + self.stage_cost_weight[1]*(y-ref_y)**2 + self.stage_cost_weight[2]*(dq1-ref_dq1)**2 + self.stage_cost_weight[3]*(dq2-ref_dq2)**2
        return stage_cost * 10000

    def _phi(self, x_T: np.ndarray) -> float:
        """calculate terminal cost"""
        # parse x_T
        q1,q2,dq1,dq2 = x_T
        x = self.l1 * np.cos(q1) + self.l2 * np.cos(q1 + q2)
        y = self.l1 * np.sin(q1) + self.l2 * np.sin(q1 + q2)
        #yaw = ((yaw + 2.0*np.pi) % (2.0*np.pi)) # normalize theta to [0, 2*pi]

        # calculate stage cost
        _, ref_x, ref_y, ref_dq1, ref_dq2 = self._get_nearest_waypoint(q1, q2)
        terminal_cost = self.terminal_cost_weight[0]*(x-ref_x)**2 + self.terminal_cost_weight[1]*(y-ref_y)**2 + self.terminal_cost_weight[2]*(dq1-ref_dq1)**2 + self.terminal_cost_weight[3]*(dq2-ref_dq2)**2
        return terminal_cost * 10000

    def _get_nearest_waypoint(self, q1: float, q2: float, update_prev_idx: bool = False):
        """search the closest waypoint to the vehicle on the reference path"""
        
        SEARCH_IDX_LEN = 30 # [points] forward search range
        prev_idx = self.prev_waypoints_idx
        x = self.l1 * np.cos(q1) + self.l2 * np.cos(q1 + q2)
        y = self.l1 * np.sin(q1) + self.l2 * np.sin(q1 + q2)
        #print(x,y)
        dx = [x - ref_x for ref_x in self.ref_path[prev_idx:(prev_idx + SEARCH_IDX_LEN), 0]]
        dy = [y - ref_y for ref_y in self.ref_path[prev_idx:(prev_idx + SEARCH_IDX_LEN), 1]]
        #print(f"dx = {dx}")
        #print(f"dy = {dy}")
        d = [(idx ** 2 + idy ** 2)*100 for (idx, idy) in zip(dx, dy)]
        min_d = min(d)
        #print(f"d = {d}")
        nearest_idx = d.index(min_d) + prev_idx
        #print(f"0     min(d) = {min_d}")
        #print(f"0     prev_idx = {prev_idx}")
        #print(f"0     nearest_idx = {nearest_idx}")
        # get reference values of the nearest waypoint
        ref_x = self.ref_path[nearest_idx,0]
        ref_y = self.ref_path[nearest_idx,1]
        ref_dq1 = self.ref_path[nearest_idx,2]
        ref_dq2 = self.ref_path[nearest_idx,3]

        # update nearest waypoint index if necessary
        if update_prev_idx:
            print(f"0     prev_idx = {prev_idx}")
            print(f"0     nearest_idx = {nearest_idx}")
            print("======================updated=======================")
            self.prev_waypoints_idx = nearest_idx 

        return nearest_idx, ref_x, ref_y, ref_dq1, ref_dq2  
              
    def _F(self, x_t: np.ndarray, v_t: np.ndarray) -> np.ndarray:
        q = x_t[0:2]
        dq = x_t[2:4]
        u = v_t
        #print(f"q = {q}")
        #print(f"dq = {dq}")
        dt = self.delta_t
        M11 = m1 * lc1 ** 2 + l1 + m2 * \
            (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * np.cos(q[1])) + l2
        M22 = m2 * lc2 ** 2 + l2
        M12 = m2 * l1 * lc2 * np.cos(q[1]) + m2 * lc2 ** 2 + l2
        M21 = m2 * l1 * lc2 * np.cos(q[1]) + m2 * lc2 ** 2 + l2
        M = np.array([[M11, M12], [M21, M22]])
        h = m2 * l1 * lc2 * np.sin(q[1])
        g1 = m1 * lc1 * g * np.cos(q[0]) + m2 * g * (lc2 * np.cos(q[0] + q[1]) + l1 * np.cos(q[0]))
        g2 = m2 * lc2 * g * np.cos(q[0] + q[1])
        G = np.array([g1, g2])
        C = np.array([[-h * dq[1], -h * dq[0] - h * dq[1]], [h * dq[0], 0]])
        ddq = np.linalg.inv(M).dot(u - C.dot(dq) - G)
        #print(f"u = {u}")
        #print(f"before ddq = {ddq}")
        #print(f"before dq = {dq}")
        dq = dq + ddq * dt
        #print(f"after dq = {dq}")
        #print(f"before q = {q}")
        q = q + dq * dt
        #print(f"after q = {q}")
        x_updated = np.concatenate((q, dq))
        #print(f"x_updated = {x_updated}")
        return x_updated    
        
    def _F1(self, x_t: np.ndarray, v_t: np.ndarray) -> np.ndarray:
        q = x_t[0:2]
        dq = x_t[2:4]
        #print(f"q = {q}")
        #print(f"dq = {dq}")
        dt = self.delta_t
        M11 = m1 * lc1 ** 2 + l1 + m2 * \
            (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * np.cos(q[1])) + l2
        M22 = m2 * lc2 ** 2 + l2
        M12 = m2 * l1 * lc2 * np.cos(q[1]) + m2 * lc2 ** 2 + l2
        M21 = m2 * l1 * lc2 * np.cos(q[1]) + m2 * lc2 ** 2 + l2
        M = np.array([[M11, M12], [M21, M22]])
        h = m2 * l1 * lc2 * np.sin(q[1])
        #g1 = m1 * lc1 * g * np.cos(q[0]) + m2 * g * (lc2 * np.cos(q[0] + q[1]) + l1 * np.cos(q[0]))
        #g2 = m2 * lc2 * g * np.cos(q[0] + q[1])
        g1,g2=0.0,0.0
        G = np.array([g1, g2])
        C = np.array([[-h * dq[1], -h * dq[0] - h * dq[1]], [h * dq[0], 0]])
        u = np.dot(M, v_t) + np.dot(C, dq) + G
        ddq = np.linalg.inv(M).dot(u - C.dot(dq) - G)
        #print(f"u = {u}")
        #print(f"before ddq = {ddq}")
        #print(f"before dq = {dq}")
        dq = dq + ddq * dt
        #print(f"after dq = {dq}")
        #print(f"before q = {q}")
        q = q + dq * dt
        #print(f"after q = {q}")
        x_updated = np.concatenate((q, dq))
        #print(f"x_updated = {x_updated}")
        return x_updated

    def _compute_weights(self, S: np.ndarray) -> np.ndarray:
        """compute weights for each sample"""
        # prepare buffer
        w = np.zeros((self.K))

        # calculate rho
        rho = S.min()

        # calculate eta
        eta = 0.0
        for k in range(self.K):
            eta += np.exp( (-1.0/self.param_lambda) * (S[k]-rho) )

        # calculate weight
        for k in range(self.K):
            w[k] = (1.0 / eta) * np.exp( (-1.0/self.param_lambda) * (S[k]-rho) )
            #w[k] = (1.0 / eta) * np.exp( (-1.0/self.par am_lambda) * (S[k]-rho) )*5
        return w

    import numpy as np


    def _moving_median_filter(self, xx: np.ndarray, window_size: int) -> np.ndarray:
        """Apply moving median filter for smoothing input sequence"""
        dim = xx.shape[1]
        xx_smoothed = np.zeros(xx.shape)

        for d in range(dim):
            xx_smoothed[:, d] = median_filter(xx[:, d], size=window_size, mode='reflect')

        return xx_smoothed
        
    def _moving_average_filter(self, xx: np.ndarray, window_size: int) -> np.ndarray:
        """apply moving average filter for smoothing input sequence
        Ref. https://zenn.dev/bluepost/articles/1b7b580ab54e95
        """
        b = np.ones(window_size)/window_size
        dim = xx.shape[1]
        xx_mean = np.zeros(xx.shape)

        for d in range(dim):
            xx_mean[:,d] = np.convolve(xx[:,d], b, mode="same")
            n_conv = math.ceil(window_size/2)
            xx_mean[0,d] *= window_size/n_conv
            for i in range(1, n_conv):
                xx_mean[i,d] *= window_size/(i+n_conv)
                xx_mean[-i,d] *= window_size/(i + n_conv - (window_size % 2)) 
        return xx_mean
