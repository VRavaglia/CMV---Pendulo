import numpy as np
import cv2

from InvertedPendulum import InvertedPendulum

from scipy.integrate import solve_ivp
import control
import code
import matplotlib.pyplot as plt


class MyLinearizedSystem:
    def __init__(self):
        g = 9.8
        L = 1.5
        m = 1.0
        M = 5.0
        b = 1

        # Pendulum up (linearized eq)
        # Eigen val of A : array([[ 1.        , -0.70710678, -0.07641631,  0.09212131] )

        self.A = np.array([ \
            [0, 1, 0, 0], \
            [0, -b / M, m * g / M, 0], \
            [0, 0, 0, 1.], \
            [0, -b / (M * L), (m + M) * g / (M * L), 0]])

        self.B = np.expand_dims(np.array([0, 1.0 / M, 0., 1 / (M * L)]), 1)  # 4x1

    def compute_K(self, desired_eigs=[-0.1, -0.2, -0.3, -0.4]):
        print('[compute_K] desired_eigs=', desired_eigs)
        self.K = control.place(self.A, self.B, desired_eigs)

    def get_K(self):
        return self.K


# This will be our LQR Controller.
# LQRs are more theoritically grounded, they are a class of optimal control algorithms.
# The control law is u = KY. K is the unknown which is computed as a solution to minimization problem.
def u(t, y):
    u_ = -np.matmul(ss.K, y)
    # print('u()', 't=', t, 'u_=', u_)
    # code.interact(local=dict(globals(), **locals()))
    # return 0.1
    return u_[0]


# Pendulum and cart system (non-linear eq). The motors on the cart turned at fixed time. In other words
# The motors are actuated to deliver forward x force from t=t1 to t=t2.
# Y : [ x, x_dot, theta, theta_dot]
# Return \dot(Y)
def y_dot(t, y):
    g = 9.8
    L = 1.5
    m = 1.0
    M = 5.0
    b = 1

    #print('y=', y)
    F = u(t, y)

    x_ddot = m * g * np.cos(y[2]) * np.sin(y[2]) - m * L * (y[3] ** 2) * np.sin(y[2]) + F - b * y[1]
    x_ddot = x_ddot / (M + m * (1 - np.cos(y[2]) ** 2))

    theta_ddot = (M + m) * g * np.sin(y[2]) - m * L * (y[3] ** 2) * np.sin(y[2]) * np.cos(y[2]) + (
                F - b * y[1]) * np.cos(y[2])
    theta_ddot = theta_ddot / (L * (M + m * (1 - np.cos(y[2]) ** 2)))

    return [y[1], x_ddot, y[3], theta_ddot]


if __name__ == "__1main__":
    # Verifying the correctness of linearization by evaluating y_dot
    # in two ways a) non-linear b) linear approximation

    at_y = np.array([0, 0, 0.1, 0.002])
    print('non-linear', y_dot(1.0, at_y))
    print('linearized ', np.matmul(ss.A, at_y + ss.B.T * u(1, at_y)))
    code.interact(local=dict(globals(), **locals()))

    # See also analysis_of_linearization to know more.

# Both cart and the pendulum can move.
if __name__ == "__main__":

    # Global Variables
    ss = MyLinearizedSystem()

    # Arbitrarily set Eigen Values
    # ss.compute_K(desired_eigs = np.array([-.1, -.2, -.3, -.4])*3. ) # Arbitarily set desired eigen values

    # Eigen Values set by LQR
    parameters = []
    Q = np.diag([1, 1, 1, 1.])
    R1 = np.diag([1.])
    R = R1

    
    K, S, E = control.lqr(ss.A, ss.B, Q, R)
    ss.compute_K(desired_eigs=E)  # Arbitarily set desired eigen values
    #ss.K = np.matrix('-10.0000  -34.3268  553.5394  379.0879')
        
    sol = solve_ivp(y_dot, [0, 20], [-1, 0, -0.4, 0], t_eval=np.linspace(0, 20, 200))
    syst = InvertedPendulum()

    recalculated_u = [];
    # We will need to recalculate u as it is not available in sol
    for input_index in range(len(sol.t)):
        recalculated_u.append(u(sol.t[input_index].item(0), sol.y[:, input_index]).item(0))
    
    for i, t in enumerate(sol.t):
        rendered = syst.step( [sol.y[0,i], sol.y[1,i], sol.y[2,i], sol.y[3,i] ], t )
        cv2.imshow( 'im', rendered )
        cv2.moveWindow( 'im', 100, 100 )

        if cv2.waitKey(30) == ord('q'):
            break











 