from c4dynamics import state, d2r
import numpy as np


class helicopter(state):

    theta: float
    psi: float
    dtheta: float
    dpsi: float

    # parameters
    # Helicopter physical parameters
    Jp  = 0.0215          # pitch moment of inertia          [kg·m²]
    Jy  = 0.0237          # yaw moment of inertia            [kg·m²]
    Dp  = 0.0171          # pitch viscous damping            [N/V]
    Dy  = 0.232           # yaw viscous damping              [N/V]
    Kpp =  0.0015         # thrust gain: pitch←pitch prop    [N·m/V]
    Kpy =  0.0021         # thrust gain: pitch←yaw prop      [N·m/V]
    Kyp = -0.0027         # thrust gain: yaw←pitch prop      [N·m/V]
    Kyy =  0.0014         # thrust gain: yaw←yaw prop        [N·m/V]
    Lcm =  0.0071         # centre-of-mass arm length        [m]
    M   =  1.075          # total helicopter mass            [kg]
    g   =  9.81           # gravitational acceleration       [m/s²]


    def __init__(self, theta=0.0, psi=0.0, dtheta=0.0, dpsi=0.0):

        super().__init__(theta=theta, psi=psi, dtheta=dtheta, dpsi=dpsi)


    def F(self, X = None):

        theta, psi, dtheta, dpsi = self.X if X is None else X

        beta1  = self.Jp + self.M * self.Lcm**2
        beta2  = (- self.M * self.g * self.Lcm * np.cos(theta)
                - self.Dp * dtheta
                - self.M * self.Lcm**2 * dpsi**2 * np.sin(theta) * np.cos(theta))

        gamma1 = self.Jy + self.M * self.Lcm**2 * np.cos(theta)**2
        gamma2 = (- self.Dy * dpsi
                + 2.0 * self.M * self.Lcm**2 * dpsi * dtheta * np.sin(theta) * np.cos(theta))

        F = np.array([beta2 / beta1,
                    gamma2 / gamma1])

        return F



    def G(self, X = None):

        theta, psi, dtheta, dpsi = self.X if X is None else X

        beta1  = self.Jp + self.M * self.Lcm**2
        gamma1 = self.Jy + self.M * self.Lcm**2 * np.cos(theta)**2

        G = np.array([[self.Kpp / beta1,  self.Kpy / beta1],
                    [self.Kyp / gamma1, self.Kyy / gamma1]])

        return G


    def dynamics(self, t: float, y: np.ndarray, u: np.ndarray):
        x1, x2 = self.split(y)

        # F, G  = helicopter_dynamics_matrices(x1, x2)
        dx1   = x2
        dx2   = self.F(y) + self.G(y) @ u
        return np.concatenate([dx1, dx2])


    def reference(self, t: float):

        xd      = np.array([ 10 * np.sin(t),
                            15 * np.cos(t)]) * d2r # degrees to readians

        xd_d    = np.array([ 10 * np.cos(t),
                            -15 * np.sin(t)]) * d2r

        xd_dd   = np.array([-10 * np.sin(t),
                            -15 * np.cos(t)]) * d2r

        return xd, xd_d, xd_dd
    
    
    def split(self, X=None):

        x = self.X if X is None else X
        n = len(x) // 2
        return x[:n], x[n:]


