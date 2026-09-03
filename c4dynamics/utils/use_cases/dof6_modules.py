import numpy as np


class control_system:

    Gn = 250  # gain factor relating aoa of ctrl surface to acc cmd per unit dynamic pressure, [rad*Pa/(m/s)^2]
    afp = 0
    afy = 0
    tau = 0.04
    dt = 0

    def __init__(self, dt, **kwargs):
        self.dt = dt
        self.__dict__.update(kwargs)

    def update(self, ab_cmd, Q):
        afp = -self.Gn * ab_cmd[2] / Q
        afy = self.Gn * ab_cmd[1] / Q

        self.afp = self.afp * np.exp(-self.dt / self.tau) + afp * (1 - np.exp(-self.dt / self.tau))
        self.afy = self.afy * np.exp(-self.dt / self.tau) + afy * (1 - np.exp(-self.dt / self.tau))

        if abs(self.afp) > np.deg2rad(20):
            self.afp = np.sign(self.afp) * np.deg2rad(20)
        if abs(self.afy) > np.deg2rad(20):
            self.afy = np.sign(self.afy) * np.deg2rad(20)

        return self.afp, self.afy


class aerodynamics:

    s = 0.0127
    d = 0.127
    xref = 1.35

    mach_table = (0, 0.8, 1.14, 1.75, 2.5, 3.5)
    cD0_table = (0.8, 0.8, 1.2, 1.15, 1.05, 0.94)
    cLa_table = (38, 39, 56, 55, 40, 33)
    cMa_table = (-160, -170, -185, -235, -190, -150)
    cMd_table = (180, 250, 230, 130, 80, 45)
    cMqcMadot_table = (-6000, -13000, -16000, -13500, -10000, -6000)
    k_table = (0.0255, 0.0305, 0.0361, 0.0441, 0.0540, 0.0665)

    alt_table = (0, 2000, 4000, 6000)
    pressure_table = (101325, 79501, 61660, 47217)
    density_table = (1.225, 1.0066, 0.81935, 0.66011)
    speed_of_sound_table = (340.29, 332.53, 324.59, 316.45)

    def f_coef(self, mach, alpha_total):
        cLa = np.interp(mach, self.mach_table, self.cLa_table)
        cD0 = np.interp(mach, self.mach_table, self.cD0_table)
        k = np.interp(mach, self.mach_table, self.k_table)

        # lift and drag
        cL = cLa * alpha_total
        cD = cD0 + k * cL**2

        return cL, cD

    def m_coef(self, mach, alpha, beta, d_pitch, d_yaw, xcm, Q, v, fAby, fAbz, q, r):

        cNb = cMa = np.interp(mach, self.mach_table, self.cMa_table)
        cNd = cMd = np.interp(mach, self.mach_table, self.cMd_table)
        cNrcNbdot = cMqcMadot = np.interp(mach, self.mach_table, self.cMqcMadot_table)

        # pitch and yaw moments
        # yb, zb normal force aero coefficient
        cNy = fAby / Q / self.s
        cNz = fAbz / Q / self.s
        cMref = cMa * alpha + cMd * d_pitch
        cNref = cNb * beta + cNd * d_yaw

        # to center of mass
        cM = cMref - cNz * (xcm - self.xref) / self.d + self.d / (2 * v) * cMqcMadot * q
        cN = cNref - cNy * (xcm - self.xref) / self.d + self.d / (2 * v) * cNrcNbdot * r

        return cM, cN

    @staticmethod
    def m2idx(m):
        return np.argmin(np.abs(aerodynamics.mach_table - m))

    @staticmethod
    def alt2atmo(alt):
        p = np.interp(alt, aerodynamics.alt_table, aerodynamics.pressure_table)
        rho = np.interp(alt, aerodynamics.alt_table, aerodynamics.density_table)
        vs = np.interp(alt, aerodynamics.alt_table, aerodynamics.speed_of_sound_table)
        return p, rho, vs


class engine:

    # tbo = 5.6       # t burnout
    pref = 101314  # reference ambient pressure
    Ae = 0.011  # exit area of rocket nozzle
    Isp = 2224  # specific impulse

    def __init__(self):
        # sec
        self.times = np.array(
            [
                0,
                0.01,
                0.04,
                0.05,
                0.08,
                0.1,
                0.2,
                0.3,
                0.6,
                1,
                1.5,
                2.5,
                3.5,
                3.8,
                4,
                4.1,
                4.3,
                4.5,
                4.7,
                4.9,
                5.2,
                5.6,
            ]
        )
        # Newton
        self.thrust = np.array(
            [
                0,
                450,
                17800,
                23100,
                21300,
                20000,
                18200,
                17000,
                15000,
                13800,
                13300,
                13800,
                14700,
                14300,
                12900,
                11000,
                7000,
                4500,
                2900,
                1500,
                650,
                0,
            ]
        )

    def update(self, t, pa):
        # return thrust force at time t and pressure pa
        # pa pressure at altitude h
        thrust_ref = np.interp(t, self.times, self.thrust)  # thrust at time t
        thrust_atm = thrust_ref + (self.pref - pa) * self.Ae  # correction for atmosphere conditions
        return thrust_atm, thrust_ref
