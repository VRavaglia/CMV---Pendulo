import numpy as np
import cv2


from InvertedPendulum import InvertedPendulum

from scipy.integrate import solve_ivp


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

   
  
# Global Variables
ss = MyLinearizedSystem()


ss.K = np.matrix('-10.0000  -34.3268  553.5394  379.0879')

def u( t , y ):
    u_ = -np.matmul( ss.K , y )  
    return u_[0]



# Y : [ x, x_dot, theta, theta_dot]
def func3( t, y):
    g = 9.8 # Gravitational Acceleration
    L = 1.5 # Length of pendulum

    m = 1.0 #mass of bob (kg)
    M = 5.0  # mass of cart (kg)
    
    b = 1.0 

    F = u(t, y)

    x_ddot = m * g * np.cos(y[2]) * np.sin(y[2]) - m * L * (y[3] ** 2) * np.sin(y[2]) + F - b * y[1]
    x_ddot = x_ddot / (M + m * (1 - np.cos(y[2]) ** 2))

    theta_ddot = (M + m) * g * np.sin(y[2]) - m * L * (y[3] ** 2) * np.sin(y[2]) * np.cos(y[2]) + (
                F - b * y[1]) * np.cos(y[2])
    theta_ddot = theta_ddot / (L * (M + m * (1 - np.cos(y[2]) ** 2)))

   
    return [ y[1], x_ddot , y[3], theta_ddot ]


# Both cart and the pendulum can move.
if __name__=="__main__":
    # We need to write the Euler-lagrange equations for the both the
    # systems (bob and cart). The equations are complicated expressions. Although
    # it is possible to derive with hand. The entire notes are in media folder or the
    # blog post for this entry. Otherwse in essense it is very similar to free_fall_pendulum.py
    # For more comments see free_fall_pendulum.py
    sol = solve_ivp(func3, [0, 20], [ -1.0, 0., -0.4, 0. ],   t_eval=np.linspace( 0, 20, 300)  )


    syst = InvertedPendulum()
    


    for i, t in enumerate(sol.t):
        rendered = syst.step( [sol.y[0,i], sol.y[1,i], sol.y[2,i], sol.y[3,i] ], t )
        cv2.imshow( 'im', rendered )
        cv2.moveWindow( 'im', 100, 100 )

        if cv2.waitKey(30) == ord('q'):
            break

            