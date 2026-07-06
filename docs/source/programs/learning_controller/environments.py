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
        """Initialize the helicopter state.

        Args:
            theta: Initial pitch angle, in radians.
            psi: Initial yaw angle, in radians.
            dtheta: Initial pitch rate, in radians per second.
            dpsi: Initial yaw rate, in radians per second.

        Returns:
            None.
        """

        super().__init__(theta=theta, psi=psi, dtheta=dtheta, dpsi=dpsi)


    def F(self, X = None,
          mass = None, lcm = None,
          jy = None, jp = None
        ):
        """Compute the nonlinear drift dynamics.

        Args:
            X: Optional state vector ordered as
                ``[theta, psi, dtheta, dpsi]``. If omitted, the current
                object state is used.
            mass: Optional mass parameter. If omitted, the default class parameter is used.
            lcm: Optional centre-of-mass arm length parameter. If omitted, the default class
                parameter is used.
            jy: Optional yaw moment of inertia parameter. If omitted, the default class parameter
                is used.
            jp: Optional pitch moment of inertia parameter. If omitted, the default class parameter
                is used.

        Returns:
            A two-element array containing the uncontrolled pitch and yaw
            angular accelerations.
        """

        M = self.M if mass is None else mass
        Lcm = self.Lcm if lcm is None else lcm
        Jy = self.Jy if jy is None else jy
        Jp = self.Jp if jp is None else jp

        theta, psi, dtheta, dpsi = self.X if X is None else X

        beta1  = Jp + M * Lcm**2
        beta2  = (- M * self.g * Lcm * np.cos(theta)
                - self.Dp * dtheta
                - M * Lcm**2 * dpsi**2 * np.sin(theta) * np.cos(theta))

        gamma1 = Jy + M * Lcm**2 * np.cos(theta)**2
        gamma2 = (- self.Dy * dpsi
                + 2.0 * M * Lcm**2 * dpsi * dtheta * np.sin(theta) * np.cos(theta))

        F = np.array([beta2 / beta1,
                    gamma2 / gamma1])

        return F



    def G(self, X = None,
          mass = None, lcm = None,
          jy = None, jp = None,
          kpy = None, kyy = None
        ):
        """Compute the control-effectiveness matrix.

        Args:
            X: Optional state vector ordered as
                ``[theta, psi, dtheta, dpsi]``. If omitted, the current
                object state is used.
            mass: Optional mass parameter. If omitted, the default class parameter is used.
            lcm: Optional centre-of-mass arm length parameter. If omitted, the default class
                parameter is used.
            jy: Optional yaw moment of inertia parameter. If omitted, the default class parameter
                is used.
            jp: Optional pitch moment of inertia parameter. If omitted, the default class parameter
                is used.
            kpy: Optional thrust gain parameter for pitch←yaw prop. If omitted, the default class
                parameter is used.
            kyy: Optional thrust gain parameter for yaw←yaw prop. If omitted, the default class
                parameter is used.

        Returns:
            A 2-by-2 matrix that maps the control input vector to pitch and
            yaw angular accelerations.
        """
        M = self.M if mass is None else mass
        Lcm = self.Lcm if lcm is None else lcm
        Jy = self.Jy if jy is None else jy
        Jp = self.Jp if jp is None else jp
        Kpy = self.Kpy if kpy is None else kpy
        Kyy = self.Kyy if kyy is None else kyy

        theta, psi, dtheta, dpsi = self.X if X is None else X

        beta1  = Jp + M * Lcm**2
        gamma1 = Jy + M * Lcm**2 * np.cos(theta)**2

        G = np.array([[self.Kpp / beta1, Kpy / beta1],
                    [self.Kyp / gamma1, Kyy / gamma1]])

        return G


    def dynamics(self, t: float, y: np.ndarray, u: np.ndarray):
        """Evaluate the helicopter state derivative for integration.

        Args:
            t: Current simulation time, in seconds.
            y: State vector ordered as ``[theta, psi, dtheta, dpsi]``.
            u: Two-element control input vector.

        Returns:
            The time derivative of ``y`` as ``[dtheta, dpsi, ddtheta, ddpsi]``.
        """

        x1, x2 = self.split(y)

        # F, G  = helicopter_dynamics_matrices(x1, x2)
        dx1   = x2
        dx2   = self.F(y) + self.G(y) @ u
        return np.concatenate([dx1, dx2])


    def reference(self, t: float):
        """Evaluate the desired pitch and yaw reference trajectory.

        Args:
            t: Current simulation time, in seconds.

        Returns:
            A tuple ``(xd, xd_d, xd_dd)`` containing desired angles, desired
            angular rates, and desired angular accelerations.
        """

        xd      = np.array([ 10 * np.sin(t),
                            15 * np.cos(t)]) * d2r # degrees to readians

        xd_d    = np.array([ 10 * np.cos(t),
                            -15 * np.sin(t)]) * d2r

        xd_dd   = np.array([-10 * np.sin(t),
                            -15 * np.cos(t)]) * d2r

        return xd, xd_d, xd_dd


    def split(self, X=None):
        """Split a full state vector into position and velocity components.

        Args:
            X: Optional full state vector. If omitted, the current object
                state is used.

        Returns:
            A tuple ``(x1, x2)`` where ``x1`` contains angle states and ``x2``
            contains angular-rate states.
        """

        x = self.X if X is None else X
        n = len(x) // 2
        return x[:n], x[n:]


