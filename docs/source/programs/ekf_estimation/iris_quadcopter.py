import numpy as np

class IrisQuadcopterParams:
    """
    Unified parameter datasheet for a 40-50cm quadcopter.
    Values are extracted directly from the official PX4/Gazebo 'iris.sdf' file.

    This class contains all mechanical, geometric, inertial, and aerodynamic
    constants needed to solve the Newton-Euler equations of motion.
    """
    def __init__(self):
        # --- 1. Rigid Body Physical Properties ---
        self.mass = 1.5000  # Total takeoff weight (kg)
        self.g = 9.80665    # Acceleration due to gravity (m/s^2)

        # --- 2. Inertia Tensor Matrix (kg * m^2) ---
        # Extracted directly from the <inertial> block of the base_link
        self.Ixx = 0.0347563  # Roll axis inertia
        self.Iyy = 0.0458929  # Pitch axis inertia
        self.Izz = 0.0977000  # Yaw axis inertia

        # Construct the complete full 3x3 inertia tensor matrix
        self.inertia_matrix = np.diag([self.Ixx, self.Iyy, self.Izz])

        # --- 3. Rotor / Propeller Aerodynamics ---
        self.motor_constant = 8.5485e-06  # Thrust coefficient (kt) in N/(rad/s)^2
        self.moment_constant = 1.6000e-07 # Drag moment coefficient (kd) in N*m/(rad/s)^2
        self.rotor_inertia = 6.1300e-05   # Individual rotor bell + prop inertia (Jm) in kg*m^2

        # --- 4. Actuator Limits & Dynamics ---
        self.max_rot_velocity = 838.0     # Maximum motor speed (omega_max) in rad/s (~8000 RPM)
        self.time_constant_up = 0.0125    # First-order motor spin-up lag (seconds)
        self.time_constant_down = 0.025   # First-order motor spin-down lag (seconds)

        # --- 5. Geometric Multi-Rotor Layout (X-Configuration) ---
        # Explicit [X, Y, Z] coordinate locations of the motor centers relative to the CoM (meters)
        # Sourced from the relative spatial <pose> offsets inside the SDF file.
        self.rotor_positions = {
            'rotor_0': np.array([ 0.13, -0.22, 0.023]),  # Front-Right (CCW)
            'rotor_1': np.array([-0.13,  0.20, 0.023]),  # Rear-Left (CCW)
            'rotor_2': np.array([ 0.13,  0.22, 0.023]),  # Front-Left (CW)
            'rotor_3': np.array([-0.13, -0.20, 0.023])   # Rear-Right (CW)
        }

        # Direction of rotation for torque mapping (+1 = CCW, -1 = CW)
        self.rotor_directions = {
            'rotor_0':  1,
            'rotor_1':  1,
            'rotor_2': -1,
            'rotor_3': -1
        }

    @property
    def arm_radius(self):
        """Calculates geometric distance from CoM to the first motor (r) in meters."""
        pos = self.rotor_positions['rotor_0']
        return float(np.sqrt(pos[0]**2 + pos[1]**2))

    @property
    def hover_motor_speed(self):
        """
        Computes the theoretical motor speed (rad/s) required to hover.
        w_hover = sqrt( (m * g) / (4 * kt) )
        """
        required_total_thrust = self.mass * self.g
        thrust_per_motor = required_total_thrust / 4.0
        return float(np.sqrt(thrust_per_motor / self.motor_constant))

    def to_dict(self):
        """Exports the internal object attributes as a raw standard Python dictionary."""
        return {
            "mass": self.mass,
            "gravity": self.g,
            "inertia_tensor": self.inertia_matrix.tolist(),
            "aerodynamics": {
                "thrust_coefficient_kt": self.motor_constant,
                "drag_coefficient_kd": self.moment_constant,
                "prop_inertia_jm": self.rotor_inertia
            },
            "limits": {
                "omega_max": self.max_rot_velocity,
                "omega_hover": self.hover_motor_speed
            },
            "geometry": {
                "diagonal_arm_radius": self.arm_radius,
                "offsets": {k: v.tolist() for k, v in self.rotor_positions.items()}
            }
        }

# --- Quick Verification Check ---
if __name__ == "__main__":
    quad_params = IrisQuadcopterParams()
    print(f"--- Iris Quadcopter Parameters Initialized ---")
    print(f"Total Takeoff Mass: {quad_params.mass} kg")
    print(f"Calculated Arm Radius: {quad_params.arm_radius:.4f} m (Wheelbase: {quad_params.arm_radius*2*100:.1f} cm)")
    print(f"Required Hover Speed: {quad_params.hover_motor_speed:.2f} rad/s")
