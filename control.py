import pdb
import numpy as np
from os import system

from cvxpy import *
from scipy import sparse

import casadi as ca

from sys_params import SYS_PARAMS


class PID(object):
    """
    PID control for quadrotor
    """
    def __init__(self, dt):
        self.params = SYS_PARAMS()
        self.dt = dt
        # action dimensions (thrust, wx, wy, wz)
        self._u_dim = 4
        
        # 피드백 게인은 위치 오차를 줄이고 원하는 위치에 QUADROTOR를 안정적으로 유지
        self.KpF = np.array([4.55e0, 4.55e0, 2.55e0])
        # 미분 게인은 속도 오차를 줄이고 원하는 속도로 QUADROTOR를 안정적으로 유지
        self.KdF = np.array([8.75e0, 8.75e0, 2.75e0])
        # 각도 오차를 줄이기 위한 피드백 게인
        self.KpM = np.array([3e2, 3e2, 3e2])
        # 각속도 오차를 줄이기 위한 미분 게인
        self.KdM = np.array([3e1, 3e1, 3e1])
        # 이전 QUADROTOR 위치와 목표 위치를 저장하는 배열을 초기화
        self.old_pos = np.zeros(6) # 이전 위치
        self.old_pos_d = np.zeros(6) #이전 DESIRED한 위치

    def get_action(self, obs): # 현재상태(OBS)를 입력으로 받아 행동을 반환
        """
        obs: np.array

        """
        quad_pos = obs[:6] # OBS를 통한 현재 위치
        traj_pos = obs[6:] # OBS를 통한 목표 위치
        # pdb.set_trace()
   
        # 이전의 POS와 현재의 POS를 빼서 시간 간격으로 나눔(속도를 구함)
        vx = (quad_pos[0] - self.old_pos[0]) / self.dt
        vy = (quad_pos[1] - self.old_pos[1]) / self.dt
        vz = (quad_pos[2] - self.old_pos[2]) / self.dt
        vp = (quad_pos[3] - self.old_pos[3]) / self.dt
        vq = (quad_pos[4] - self.old_pos[4]) / self.dt
        vr = (quad_pos[5] - self.old_pos[5]) / self.dt
        # pdb.set_trace()
        
        # 이전의 DESIRED POS와 현재의 DESIRED POS를 빼서 시간 간격으로 나눔(속도를 구함)
        vx_d = (traj_pos[0] - self.old_pos_d[0]) / self.dt
        vy_d = (traj_pos[1] - self.old_pos_d[1]) / self.dt
        vz_d = (traj_pos[2] - self.old_pos_d[2]) / self.dt
        vp_d = (traj_pos[3] - self.old_pos_d[3]) / self.dt
        vq_d = (traj_pos[4] - self.old_pos_d[4]) / self.dt
        vr_d = (traj_pos[5] - self.old_pos_d[5]) / self.dt

        # X,Y,Z 축 방향의 제어 입력 
        u1x = -self.KpF[0] * (quad_pos[0] - traj_pos[0]) - self.KdF[0] * (vx - vx_d)
        u1y = -self.KpF[1] * (quad_pos[1] - traj_pos[1]) - self.KdF[1] * (vy - vy_d)
        u1z = (
            -self.KpF[2] * (quad_pos[2] - traj_pos[2])
            - self.KdF[2] * (vz - vz_d)
            + self.params["grav"]
        )
        
        # ROLL 속도 DESIRED값
        p_d = np.arctan2(
            np.sin(traj_pos[5]) * np.cos(traj_pos[5]) * u1x
            - np.cos(traj_pos[5]) * np.cos(traj_pos[5]) * u1y,
            u1z,
        )
        
        # PITCH 속도 DESIRED값
        q_d = np.arcsin(
            (
                np.cos(traj_pos[5]) * np.cos(traj_pos[5]) * u1x
                + np.sin(traj_pos[5]) * np.cos(traj_pos[5]) * u1y
            )
            / u1z
        )
        # YAW 속도 DESIRED값
        r_d = traj_pos[5]
    
        # 목표하는 ROLL, PITCH, YAW를 ANGLE_DESIRED로 NUMPY배열 생성
        ang_d = np.array([p_d, q_d, r_d])
        rate_d = np.array([0, 0, 0])

        # Z축 방향의 힘을 계산
        Fz = u1z / (np.cos(p_d) * np.cos(r_d))
        
        # 관성 모멘트 행렬을 생성. 
        I = np.diag([self.params["Ixx"], self.params["Iyy"], self.params["Izz"]])
        
        # 모멘트 계산
        Mr = np.dot(
            I,
            self.KdM * (rate_d - np.array([vp, vq, vr]))
            + self.KpM * (ang_d - quad_pos[3:6]),
        )

        # 
        self.old_pos = np.array(quad_pos)
        self.old_pos_d = np.array(traj_pos)

        # z축 방향의 힘과 모멘트를 리턴
        return np.hstack((Fz, Mr))


class MPC_PY(object):
    """TODO, Maybe discrete update"""

    def __init__(self, mpc_T, dt):
        self.mpc_T = mpc_T
        self.dt = dt
        self._N = int(self.mpc_T / self.dt)
        
        # 상태 x와 입력 u의 크기 
        self.nx = 2
        self.nu = 1

        # x와 u의 최대 최소값 설정
        self.umin = np.array([-5.0])
        self.umax = np.array([5.0])
        # self.umin = np.array([-np.inf, -np.inf, -np.inf, -np.inf])
        # self.umax = np.array([np.inf, np.inf, np.inf, np.inf])
        self.xmin = np.array([-np.inf, -np.inf])
        self.xmax = np.array(
            [
                np.inf,
                np.inf,
            ]
        )

        # 목적 함수(제어 입력과 상태 변수의 차이를 최소화하기 위함)
        # Q : 상태 변수의 가중치 행렬, 대각 행렬로써 상태 변수가 목표값과의 차이에 대해 얼마나 중요한지 보임
        self.Q = sparse.diags([10.0, 10.0])
        # 마지막 단계의 가중치 행렬
        self.QN = self.Q
        # R : 제어 입력의 가중치 행렬
        self.R = 1.0 * sparse.eye(self.nu)
        # 이산화된 상태 공간 모델의 상태 및 입력 행렬을 설정
        self.Ad = np.array([[0.0, 1.0], [1.0, 1.0]])
        self.Bd = np.array([[0.0], [1.0]])

    def get_action(self, obs):
    
        # 시스템 상태 변수와 trj 목표 상태 변수를 받음 
        sys_pos = obs[:2]
        trj_pos = obs[2:]
        
        # 초기 상태 변수와 목표 상태 변수 값을 설정
        x_init = np.array([0.0, 0.0])
        xr = np.array(trj_pos.tolist())
        
        # 최적 제어 문제를 해결하기 위한 것으로 (입력의 수, 입력의 길이)
        u = Variable((self.nu, self._N))
        x = Variable((self.nx, self._N + 1))
        
        # 목적함수 및 제약 조건 초기화(reset)
        objective = 0
        constraints = [x[:, 0] == x_init]

        # MPC 돌아가는 부분
        for k in range(self._N):
            # quad_form : 행렬과 벡터사이의 제곱 항을 계산하는 데 사용
            # 목적함수를 계산, 상태변수와 제어 입력의 차이에 대한 가중치를 사용하여 정의
            objective += quad_form(x[:, k] - xr, self.Q) + quad_form(u[:, k], self.R)
            
            # 각 단계에서 다음 상태 변수를 현재 상태 변수와 제어 입력을 사용하여 예측
            constraints += [x[:, k + 1] == self.Ad @ x[:, k] + self.Bd @ u[:, k]]
            # 상태변수 및 제어 입력이 허용 범위내에 있어야 하는 조건
            constraints += [self.xmin <= x[:, k], x[:, k] <= self.xmax]
            constraints += [self.umin <= u[:, k], u[:, k] <= self.umax]
        # 최종 목적 함수 및 제약조건 설정하여 최적화 문제를 정의
        objective += quad_form(x[:, self._N] - xr, self.QN)
        prob = Problem(Minimize(objective), constraints)
        # 최적화 문제를 해결하고, MPC를 통해 계산된 제어 입력을 반환
        prob.solve(solver=OSQP, warm_start=True)
        u_mpc = u[:, 0].value
        # pdb.set_trace()
        return u_mpc


class MPC_CA(object):
    def __init__(self, mpc_T, dt, so_path="./nmpc.so"):
        """
        Nonlinear MPC for quadrotor control
        """
        self.so_path = so_path

        # Time constant
        self.mpc_T = mpc_T
        self._dt = dt
        self._N = int(self.mpc_T / self._dt)

        # constants
        self._u_min = -20.0
        self._u_max = 20.0

        self._Q_x = np.diag([10, 0.1])
        self._Q_u = np.diag([0.1])

        # initial state and control action
        self._sys_s0 = [0.0, 0.0]
        self._sys_u0 = [0.0]

        self._initDynamics()

    def _initDynamics(
        self,
    ):
        # state symbolic variables
        px, vx = ca.SX.sym("px"), ca.SX.sym("vx")
        self._x = ca.vertcat(px, vx)
        self.nx = self._x.numel()

        pu = ca.SX.sym("pu")
        self._u = ca.vertcat(pu)
        self.nu = self._u.numel()

        # x_dot = ca.vertcat(vx, px + vx)
        # self.f = ca.Function("f", [self._x, self._u], [x_dot], ["x", "u"], ["ode"])

        A = ca.DM([[0, 1], [1, 1]])
        B = ca.DM([[0], [1]])
        RHS = A @ self._x + B @ self._u
        self.f = ca.Function("f", [self._x, self._u], [RHS], ["x", "u"], ["ode"])

        # loss function
        # placeholder for the quadratic cost function
        Delta_s = ca.SX.sym("Delta_s", self.nx)
        Delta_u = ca.SX.sym("Delta_u", self.nu)

        cost_s = Delta_s.T @ self._Q_x @ Delta_s
        cost_u = Delta_u.T @ self._Q_u @ Delta_u

        f_cost_s = ca.Function("cost_s", [Delta_s], [cost_s])
        f_cost_u = ca.Function("cost_u", [Delta_u], [cost_u])

        # Non-linear Optimization
        self.nlp_w = []  # nlp variables
        self.nlp_w0 = []  # initial guess of nlp variables
        self.lbw = []  # lower bound of the variables, lbw <= nlp_x
        self.ubw = []  # upper bound of the variables, nlp_x <= ubw

        self.mpc_obj = 0  # objective
        self.nlp_g = []  # constraint functions
        self.lbg = []  # lower bound of constrait functions, lbg < g
        self.ubg = []  # upper bound of constrait functions, g < ubg

        u_min = [self._u_min]
        u_max = [self._u_max]
        x_bound = ca.inf
        x_min = [-x_bound for _ in range(self.nx)]
        x_max = [+x_bound for _ in range(self.nx)]

        g_min = [0 for _ in range(self.nx)]
        g_max = [0 for _ in range(self.nx)]

        # P = ca.SX.sym("P", self._s_dim + (self._s_dim + 3) * self._N + self._s_dim)
        P = ca.SX.sym("P", self.nx + self.nx)
        X = ca.SX.sym("X", self.nx, self._N + 1)
        U = ca.SX.sym("U", self.nu, self._N)

        F = self.sys_dynamics(self._dt)
        fMap = F.map(self._N, "openmp")  # parallel
        X_next = fMap(X[:, : self._N], U)

        # "Lift" initial conditions
        self.nlp_w += [X[:, 0]]
        self.nlp_w0 += self._sys_s0
        self.lbw += x_min
        self.ubw += x_max

        # # starting point.
        self.nlp_g += [X[:, 0] - P[0 : self.nx]]
        self.lbg += g_min
        self.ubg += g_max

        for k in range(self._N):
            self.nlp_w += [U[:, k]]
            self.nlp_w0 += self._sys_u0
            self.lbw += u_min
            self.ubw += u_max

            delta_s_k = X[:, k + 1] - P[self.nx :]
            cost_s_k = f_cost_s(delta_s_k)
            delta_u_k = U[:, k]
            cost_u_k = f_cost_u(delta_u_k)

            self.mpc_obj = self.mpc_obj + cost_s_k + cost_u_k

            # New NLP variable for state at end of interval
            self.nlp_w += [X[:, k + 1]]
            self.nlp_w0 += self._sys_s0
            self.lbw += x_min
            self.ubw += x_max

            # Add equality constraint
            self.nlp_g += [X_next[:, k] - X[:, k + 1]]
            self.lbg += g_min
            self.ubg += g_max

        # nlp objective
        nlp_dict = {
            "f": self.mpc_obj,
            "x": ca.vertcat(*self.nlp_w),
            "p": P,
            "g": ca.vertcat(*self.nlp_g),
        }

        ipopt_options = {
            "verbose": False,
            "ipopt.tol": 1e-4,
            "ipopt.acceptable_tol": 1e-4,
            "ipopt.max_iter": 100,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.print_level": 0,
            "print_time": False,
        }

        self.solver = ca.nlpsol("solver", "ipopt", nlp_dict, ipopt_options)

    def solve(self, ref_states):
        system_state = ca.vertcat(self.nlp_w0[:2])
        traj_state = ca.vertcat(ref_states[:2])

        self.sol = self.solver(
            x0=self.nlp_w0,
            lbx=self.lbw,
            ubx=self.ubw,
            lbg=self.lbg,
            ubg=self.ubg,
            # p=ca.vertcat(self.nlp_w0[:2], ref_states),
            p=ca.vertcat(system_state, traj_state),
        )

        sol_x0 = self.sol["x"].full()
        opt_u = sol_x0[self.nx : self.nx + self.nu]

        # Warm initialization
        self.nlp_w0 = list(sol_x0[self.nx + self.nu : 2 * (self.nx + self.nu)]) + list(
            sol_x0[self.nx + self.nu :]
        )

        x0_array = np.reshape(sol_x0[: -self.nx], newshape=(-1, self.nx + self.nu))

        # return optimal action, and a sequence of predicted optimal trajectory.
        return opt_u, x0_array

    def sys_dynamics(self, dt):
        X0 = ca.SX.sym("X", self.nx)
        U = ca.SX.sym("U", self.nu)

        X = X0
        k1 = dt * self.f(X, U)
        k2 = dt * self.f(X + 0.5 * k1, U)
        k3 = dt * self.f(X + 0.5 * k2, U)
        k4 = dt * self.f(X + k3, U)

        X = X + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        F = ca.Function("F", [X0, U], [X])
        return F


# class MPC_CA_UAV(object):
#     def __init__(self, mpc_T, dt, so_path="./nmpc.so"):
#         """
#         Nonlinear MPC for quadrotor control
#         """
#         self.so_path = so_path
#         self.params = SYS_PARAMS()

#         # Time constant
#         self.mpc_T = mpc_T
#         self._dt = dt
#         self._N = int(self.mpc_T / self._dt)

#         # constants
#         self._u_min = 0.0  # -np.inf
#         self._u_max = 1000.0  # np.inf

#         self._Q_x = np.diag([100, 100, 100, 10, 10, 10, 100, 100, 100, 10, 10, 10])
#         self._Q_u = np.diag([0.1, 0.1, 0.1, 0.1])

#         # initial state and control action
#         self._sys_s0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
#         self._sys_u0 = [0.0, 0.0, 0.0, 0.0]

#         self._initDynamics()

#     def _initDynamics(
#         self,
#     ):
#         # state symbolic variables
#         px, py, pz = ca.SX.sym("px"), ca.SX.sym("py"), ca.SX.sym("pz")
#         vx, vy, vz = ca.SX.sym("vx"), ca.SX.sym("vy"), ca.SX.sym("vz")
#         pp, pq, pr = ca.SX.sym("pp"), ca.SX.sym("pq"), ca.SX.sym("pr")
#         vp, vq, vr = ca.SX.sym("vp"), ca.SX.sym("vq"), ca.SX.sym("vr")
#         self._x = ca.vertcat(px, py, pz, vx, vy, vz, pp, pq, pr, vp, vq, vr)
#         self.nx = self._x.numel()

#         wrr, wrl, wlr, wll = (
#             ca.SX.sym("wrr"),
#             ca.SX.sym("wrl"),
#             ca.SX.sym("wlr"),
#             ca.SX.sym("wll"),
#         )
#         self._u = ca.vertcat(wrr, wrl, wlr, wll)
#         self.nu = self._u.numel()

#         ######################################################################################
#         total_thrust = self.params["b"] / self.params["mass"] * (wrr + wrl + wlr + wll)
#         x_dot = ca.vertcat(
#             vx,
#             vy,
#             vz,
#             (np.cos(pp) * np.sin(pq) * np.cos(pr) + np.sin(pp) * np.sin(pr))
#             * total_thrust,
#             (np.cos(pp) * np.sin(pq) * np.sin(pr) - np.sin(pp) * np.cos(pr))
#             * total_thrust,
#             -self.params["grav"] + np.cos(pp) * np.cos(pq) * total_thrust,
#             vp,
#             vq,
#             vr,
#             (
#                 self.params["Lxx"] * self.params["b"] / self.params["Ixx"] * (wll - wrl)
#                 + ((self.params["Iyy"] - self.params["Izz"]) / self.params["Ixx"])
#                 * vp
#                 * vq
#             ),
#             (
#                 self.params["Lyy"] * self.params["b"] / self.params["Iyy"] * (wrr - wlr)
#                 + ((self.params["Izz"] - self.params["Ixx"]) / self.params["Iyy"])
#                 * vr
#                 * vp
#             ),
#             (
#                 self.params["d"] / self.params["Izz"] * (-wrr + wrl - wlr + wll)
#                 + ((self.params["Ixx"] - self.params["Iyy"]) / self.params["Izz"])
#                 * vp
#                 * vq
#             ),
#         )
#         self.f = ca.Function("f", [self._x, self._u], [x_dot], ["x", "u"], ["ode"])

#         ######################################################################################
#         # # 상태 벡터 x와 입력 벡터 u의 크기 정의
#         # # x = ca.MX.sym('x', 12, 1)
#         # # u = ca.MX.sym('u', 4, 1)

#         # # 시스템 행렬 A와 B 정의
#         # A = ca.DM.eye(12)
#         # B = ca.DM.ones(12, 4)

#         # # 시스템의 동역학을 정의
#         # RHS = A @ self._x + B @ self._u

#         # # 함수 f 생성
#         # self.f = ca.Function("f", [self._x, self._u], [RHS], ["x", "u"], ["ode"])
#         ######################################################################################

#         # loss function
#         # placeholder for the quadratic cost function
#         Delta_s = ca.SX.sym("Delta_s", self.nx)
#         Delta_u = ca.SX.sym("Delta_u", self.nu)

#         cost_s = Delta_s.T @ self._Q_x @ Delta_s
#         cost_u = Delta_u.T @ self._Q_u @ Delta_u

#         f_cost_s = ca.Function("cost_s", [Delta_s], [cost_s])
#         f_cost_u = ca.Function("cost_u", [Delta_u], [cost_u])

#         # Non-linear Optimization
#         self.nlp_w = []  # nlp variables
#         self.nlp_w0 = []  # initial guess of nlp variables
#         self.lbw = []  # lower bound of the variables, lbw <= nlp_x
#         self.ubw = []  # upper bound of the variables, nlp_x <= ubw

#         self.mpc_obj = 0  # objective
#         self.nlp_g = []  # constraint functions
#         self.lbg = []  # lower bound of constrait functions, lbg < g
#         self.ubg = []  # upper bound of constrait functions, g < ubg

#         u_min = [self._u_min, self._u_min, self._u_min, self._u_min]
#         u_max = [self._u_max, self._u_max, self._u_max, self._u_max]
#         x_bound = ca.inf
#         x_min = [-x_bound for _ in range(self.nx)]
#         x_max = [+x_bound for _ in range(self.nx)]

#         g_min = [0 for _ in range(self.nx)]
#         g_max = [0 for _ in range(self.nx)]

#         # P = ca.SX.sym("P", self._s_dim + (self._s_dim + 3) * self._N + self._s_dim)
#         P = ca.SX.sym("P", self.nx + self.nx)
#         X = ca.SX.sym("X", self.nx, self._N + 1)
#         U = ca.SX.sym("U", self.nu, self._N)

#         F = self.sys_dynamics(self._dt)
#         fMap = F.map(self._N, "openmp")  # parallel
#         X_next = fMap(X[:, : self._N], U)

#         # "Lift" initial conditions
#         self.nlp_w += [X[:, 0]]
#         self.nlp_w0 += self._sys_s0
#         self.lbw += x_min
#         self.ubw += x_max

#         # # starting point.
#         self.nlp_g += [X[:, 0] - P[0 : self.nx]]
#         self.lbg += g_min
#         self.ubg += g_max

#         for k in range(self._N):
#             self.nlp_w += [U[:, k]]
#             self.nlp_w0 += self._sys_u0
#             self.lbw += u_min
#             self.ubw += u_max

#             delta_s_k = X[:, k + 1] - P[self.nx :]
#             cost_s_k = f_cost_s(delta_s_k)
#             delta_u_k = U[:, k]
#             cost_u_k = f_cost_u(delta_u_k)

#             self.mpc_obj = self.mpc_obj + cost_s_k + cost_u_k

#             # New NLP variable for state at end of interval
#             self.nlp_w += [X[:, k + 1]]
#             self.nlp_w0 += self._sys_s0
#             self.lbw += x_min
#             self.ubw += x_max

#             # Add equality constraint
#             self.nlp_g += [X_next[:, k] - X[:, k + 1]]
#             self.lbg += g_min
#             self.ubg += g_max

#         # nlp objective
#         nlp_dict = {
#             "f": self.mpc_obj,
#             "x": ca.vertcat(*self.nlp_w),
#             "p": P,
#             "g": ca.vertcat(*self.nlp_g),
#         }

#         ipopt_options = {
#             "verbose": False,
#             "ipopt.tol": 1e-4,
#             "ipopt.acceptable_tol": 1e-4,
#             "ipopt.max_iter": 100,
#             "ipopt.warm_start_init_point": "yes",
#             "ipopt.print_level": 0,
#             "print_time": False,
#         }

#         self.solver = ca.nlpsol("solver", "ipopt", nlp_dict, ipopt_options)

#     def solve(self, ref_states):

#         # quad_state = ca.vertcat(self.nlp_w0[:3], self.nlp_w0[:6])
#         quad_state = ca.vertcat(self.nlp_w0[:12])
#         traj_state = ca.vertcat(
#             ref_states[:3],
#             np.array([0.0, 0.0, 0.0]),
#             ref_states[3:6],
#             np.array([0.0, 0.0, 0.0]),
#         )
#         # pdb.set_trace()
#         self.sol = self.solver(
#             x0=self.nlp_w0,
#             lbx=self.lbw,
#             ubx=self.ubw,
#             lbg=self.lbg,
#             ubg=self.ubg,
#             p=ca.vertcat(quad_state, traj_state),
#         )

#         sol_x0 = self.sol["x"].full()
#         opt_u = sol_x0[self.nx : self.nx + self.nu]

#         # Warm initialization
#         self.nlp_w0 = list(sol_x0[self.nx + self.nu : 2 * (self.nx + self.nu)]) + list(
#             sol_x0[self.nx + self.nu :]
#         )

#         x0_array = np.reshape(sol_x0[: -self.nx], newshape=(-1, self.nx + self.nu))

#         # return optimal action, and a sequence of predicted optimal trajectory.
#         return opt_u, x0_array

#     def sys_dynamics(self, dt):
#         X0 = ca.SX.sym("X", self.nx)
#         U = ca.SX.sym("U", self.nu)

#         X = X0
#         k1 = dt * self.f(X, U)
#         k2 = dt * self.f(X + 0.5 * k1, U)
#         k3 = dt * self.f(X + 0.5 * k2, U)
#         k4 = dt * self.f(X + k3, U)

#         X = X + (k1 + 2 * k2 + 2 * k3 + k4) / 6
#         F = ca.Function("F", [X0, U], [X])
#         return F


class MPC_CA_UAV(object):
    def __init__(self, mpc_T, dt, so_path="./nmpc.so"):
        """
        Nonlinear MPC for quadrotor control
        """
        self.so_path = so_path

        # Time constant
        self.mpc_T = mpc_T
        self._dt = dt
        self._N = int(self.mpc_T / self._dt)

        self._gz = 9.81

        # Quadrotor constant
        self._w_max_yaw = 6.0
        self._w_max_xy = 6.0
        self._thrust_min = 0.0
        self._thrust_max = 20.0

        self._Q_x = np.diag([100, 100, 100, 0, 0, 0, 0, 0.1, 0.1, 0.1])
        self._Q_u = np.diag([0.1, 0.1, 0.1, 0.1])

        # initial state and control action
        self._sys_s0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self._sys_u0 = [0.0, 0.0, 0.0, 0.0]

        self._initDynamics()

    def _initDynamics(
        self,
    ):
        # state symbolic variables
        px, py, pz = ca.SX.sym("px"), ca.SX.sym("py"), ca.SX.sym("pz")
        qw, qx, qy, qz = (
            ca.SX.sym("qw"),
            ca.SX.sym("qx"),
            ca.SX.sym("qy"),
            ca.SX.sym("qz"),
        )
        vx, vy, vz = ca.SX.sym("vx"), ca.SX.sym("vy"), ca.SX.sym("vz")
        self._x = ca.vertcat(px, py, pz, qw, qx, qy, qz, vx, vy, vz)
        self.nx = self._x.numel()

        thrust, wx, wy, wz = (
            ca.SX.sym("thrust"),
            ca.SX.sym("wx"),
            ca.SX.sym("wy"),
            ca.SX.sym("wz"),
        )
        self._u = ca.vertcat(thrust, wx, wy, wz)
        self.nu = self._u.numel()

        x_dot = ca.vertcat(
            vx,
            vy,
            vz,
            0.5 * (-wx * qx - wy * qy - wz * qz),
            0.5 * (wx * qw + wz * qy - wy * qz),
            0.5 * (wy * qw - wz * qx + wx * qz),
            0.5 * (wz * qw + wy * qx - wx * qy),
            2 * (qw * qy + qx * qz) * thrust,
            2 * (qy * qz - qw * qx) * thrust,
            (qw * qw - qx * qx - qy * qy + qz * qz) * thrust - self._gz,
        )
        self.f = ca.Function("f", [self._x, self._u], [x_dot], ["x", "u"], ["ode"])

        # loss function
        # placeholder for the quadratic cost function
        Delta_s = ca.SX.sym("Delta_s", self.nx)
        Delta_u = ca.SX.sym("Delta_u", self.nu)

        cost_s = Delta_s.T @ self._Q_x @ Delta_s
        cost_u = Delta_u.T @ self._Q_u @ Delta_u

        f_cost_s = ca.Function("cost_s", [Delta_s], [cost_s])
        f_cost_u = ca.Function("cost_u", [Delta_u], [cost_u])

        # Non-linear Optimization
        self.nlp_w = []  # nlp variables
        self.nlp_w0 = []  # initial guess of nlp variables
        self.lbw = []  # lower bound of the variables, lbw <= nlp_x
        self.ubw = []  # upper bound of the variables, nlp_x <= ubw

        self.mpc_obj = 0  # objective
        self.nlp_g = []  # constraint functions
        self.lbg = []  # lower bound of constrait functions, lbg < g
        self.ubg = []  # upper bound of constrait functions, g < ubg

        u_min = [self._thrust_min, -self._w_max_xy, -self._w_max_xy, -self._w_max_yaw]
        u_max = [self._thrust_max, self._w_max_xy, self._w_max_xy, self._w_max_yaw]
        x_bound = ca.inf
        x_min = [-x_bound for _ in range(self.nx)]
        x_max = [+x_bound for _ in range(self.nx)]

        g_min = [0 for _ in range(self.nx)]
        g_max = [0 for _ in range(self.nx)]

        P = ca.SX.sym("P", self.nx + self.nx)
        X = ca.SX.sym("X", self.nx, self._N + 1)
        U = ca.SX.sym("U", self.nu, self._N)

        F = self.sys_dynamics(self._dt)
        fMap = F.map(self._N, "openmp")  # parallel
        X_next = fMap(X[:, : self._N], U)

        # "Lift" initial conditions
        self.nlp_w += [X[:, 0]]
        self.nlp_w0 += self._sys_s0
        self.lbw += x_min
        self.ubw += x_max

        # # starting point.
        self.nlp_g += [X[:, 0] - P[0 : self.nx]]
        self.lbg += g_min
        self.ubg += g_max

        for k in range(self._N):
            self.nlp_w += [U[:, k]]
            self.nlp_w0 += self._sys_u0
            self.lbw += u_min
            self.ubw += u_max

            delta_s_k = X[:, k + 1] - P[self.nx :]
            cost_s_k = f_cost_s(delta_s_k)
            delta_u_k = U[:, k]
            cost_u_k = f_cost_u(delta_u_k)

            self.mpc_obj = self.mpc_obj + cost_s_k + cost_u_k

            # New NLP variable for state at end of interval
            self.nlp_w += [X[:, k + 1]]
            self.nlp_w0 += self._sys_s0
            self.lbw += x_min
            self.ubw += x_max

            # Add equality constraint
            self.nlp_g += [X_next[:, k] - X[:, k + 1]]
            self.lbg += g_min
            self.ubg += g_max

        # nlp objective
        nlp_dict = {
            "f": self.mpc_obj,
            "x": ca.vertcat(*self.nlp_w),
            "p": P,
            "g": ca.vertcat(*self.nlp_g),
        }

        ipopt_options = {
            "verbose": False,
            "ipopt.tol": 1e-4,
            "ipopt.acceptable_tol": 1e-4,
            "ipopt.max_iter": 100,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.print_level": 0,
            "print_time": False,
        }

        self.solver = ca.nlpsol("solver", "ipopt", nlp_dict, ipopt_options)

    def solve(self, ref_states):

        # quad_state = ca.vertcat(self.nlp_w0[:3], self.nlp_w0[:6])
        quad_state = ca.vertcat(self.nlp_w0[:10])
        traj_state = ca.vertcat(
            ref_states[:3],
            self.EulerToQuat(ref_states[3:6]),
            np.array([0.0, 0.0, 0.0]),
        )
        # pdb.set_trace()
        self.sol = self.solver(
            x0=self.nlp_w0,
            lbx=self.lbw,
            ubx=self.ubw,
            lbg=self.lbg,
            ubg=self.ubg,
            p=ca.vertcat(quad_state, traj_state),
        )

        sol_x0 = self.sol["x"].full()
        opt_u = sol_x0[self.nx : self.nx + self.nu]

        # Warm initialization
        self.nlp_w0 = list(sol_x0[self.nx + self.nu : 2 * (self.nx + self.nu)]) + list(
            sol_x0[self.nx + self.nu :]
        )

        x0_array = np.reshape(sol_x0[: -self.nx], newshape=(-1, self.nx + self.nu))

        # return optimal action, and a sequence of predicted optimal trajectory.
        return opt_u, x0_array

    def sys_dynamics(self, dt):
        X0 = ca.SX.sym("X", self.nx)
        U = ca.SX.sym("U", self.nu)

        X = X0
        k1 = dt * self.f(X, U)
        k2 = dt * self.f(X + 0.5 * k1, U)
        k3 = dt * self.f(X + 0.5 * k2, U)
        k4 = dt * self.f(X + k3, U)

        X = X + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        F = ca.Function("F", [X0, U], [X])
        return F

    def EulerToQuat(self, euler):
        """
        Convert Quaternion to Euler Angles
        """
        roll, pitch, yaw = euler[0], euler[1], euler[2]
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)

        quat_w = cr * cp * cy + sr * sp * sy
        quat_x = sr * cp * cy - cr * sp * sy
        quat_y = cr * sp * cy + sr * cp * sy
        quat_z = cr * cp * sy - sr * sp * cy
        return [quat_w, quat_x, quat_y, quat_z]
