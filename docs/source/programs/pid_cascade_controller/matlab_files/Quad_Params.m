%% ========================================================================
%  QUADCOPTER CASCADE PID — WORKSPACE PARAMETER INITIALIZATION
%  Updated to include figure-8 reference trajectory
%  Updated to include FF_X and FF_Y feedforward gains
%
%  Run this script BEFORE starting the Simulink simulation.
%  Type in Command Window:  quad_params
%% ========================================================================

clear; clc;
fprintf('Initializing Quadcopter Cascade PID Parameters...\n');

%% ========================================================================
%  1. PHYSICAL PARAMETERS (Table 1 from report)
%% ========================================================================

g        = 9.81;          % gravitational acceleration [m/s^2]
m        = 0.468;         % quadcopter mass [kg]
L        = 0.225;         % arm length center to propeller [m]
K_thrust = 2.98e-6;       % thrust coefficient [N/(rad/s)^2]
B_torque = 0.0382;        % torque coefficient [N.m/(rad/s)^2]
IR       = 3.357e-5;      % rotor inertia [kg.m^2]

%% ========================================================================
%  2. INERTIA MATRIX
%% ========================================================================

IXX = 4.856e-3;           % roll axis inertia  [kg.m^2]
IYY = 4.856e-3;           % pitch axis inertia [kg.m^2]
IZZ = 8.801e-3;           % yaw axis inertia   [kg.m^2]

%% ========================================================================
%  3. AERODYNAMIC DRAG COEFFICIENTS
%% ========================================================================

Ax = 0.30;                % translational drag X [N.s/m]
Ay = 0.30;                % translational drag Y [N.s/m]
Az = 0.25;                % translational drag Z [N.s/m]
Ar = 0.20;                % rotational drag      [N.m.s/rad]

%% ========================================================================
%  4. INITIAL CONDITIONS
%  Quadcopter starts on ground, stationary, flat
%% ========================================================================

X0     = 0;   Y0     = 0;   Z0     = 0;
U0     = 0;   V0     = 0;   W0     = 0;
Phi0   = 0;   Theta0 = 0;   Psi0   = 0;
P0     = 0;   Q0     = 0;   R0     = 0;

IC = [P0; Q0; R0; Phi0; Theta0; Psi0; U0; V0; W0; X0; Y0; Z0];

%% ========================================================================
%  5. FIXED SETPOINTS
%  Yaw stays constant throughout mission
%  X, Y, Z are now time-varying (figure-8) — see Section 12 below
%% ========================================================================

Psid    = 0.0;            % desired yaw angle [rad] — fixed
Z_hover = 3.0;            % hover altitude [m]

% These are kept for compatibility but will be
% overridden by From Workspace blocks in Simulink
Xd = 0;
Yd = 0;
Zd = Z_hover;

%% ========================================================================
%  6. PID GAINS — INNER LOOP (Angular Rate)
%  Fastest loop — 200 Hz
%% ========================================================================

KP_P = 0.80;   KI_P = 0.0001;  KD_P = 0.010;  % Roll rate
KP_Q = 0.80;   KI_Q = 0.0001;  KD_Q = 0.010;  % Pitch rate
KP_R = 0.60;   KI_R = 0.0001;  KD_R = 0.008;  % Yaw rate
N_rate = 50;                                 % Derivative filter

%% ========================================================================
%  7. PID GAINS — MIDDLE LOOP (Attitude)
%  Medium loop — 100 Hz
%% ========================================================================

KP_Phi   = 6.0;  KI_Phi   = 0.0001;  KD_Phi   = 0.80;  % Roll
KP_Theta = 6.0;  KI_Theta = 0.0001;  KD_Theta = 0.80;  % Pitch
KP_Psi   = 4.0;  KI_Psi   = 0.5;     KD_Psi   = 0.40;  % Yaw
N_att    = 20;                                        % Derivative filter

AW_Phi   = 0.5;  % Anti-windup limit [N.m]
AW_Theta = 0.5;
AW_Psi   = 0.3;

%% ========================================================================
%  8. PID GAINS — OUTER LOOP (Position)
%  Slowest loop — 50 Hz
%% ========================================================================

KP_Z = 10.0;  KI_Z = 10.0;   KD_Z = 1.50;   % Altitude
KP_X = 0.80;  KI_X = 0.00;  KD_X = 0.50;  % X position
KP_Y = 1.00;  KI_Y = 0.00;  KD_Y = 0.70;  % Y position
N_pos = 10;

AW_Z = 3.0;   % Anti-windup limit Z
AW_X = 0.5;   % Anti-windup limit X — tight to prevent windup
AW_Y = 0.5;   % Anti-windup limit Y — tight to prevent windup

%% ========================================================================
%  9. FEEDFORWARD GAINS — NEW ADDITION
%
%  WHAT IS FEEDFORWARD:
%    Normal PID only reacts AFTER error builds up.
%    Feedforward predicts where reference is going
%    and acts BEFORE error builds up.
%    This reduces phase lag — the cyan path
%    follows the yellow reference more closely.
%
%  HOW IT WORKS IN Outer_Position_PID.m:
%    Reference velocity is computed using finite difference:
%      Xd_dot = (Xd_now - Xd_previous) / Ts
%      Yd_dot = (Yd_now - Yd_previous) / Ts
%    These velocities are rotated to body frame using Psi.
%    Then added directly to attitude commands:
%      Theta_d = PID_output + FF_X * Xd_dot_body
%      Phi_d   = PID_output - FF_Y * Yd_dot_body
%
%  TUNING GUIDE:
%    FF_X = FF_Y = 0.0  →  no feedforward (pure PID)
%    FF_X = FF_Y = 0.3  →  light feedforward
%    FF_X = FF_Y = 0.5  →  medium (recommended start value)
%    FF_X = FF_Y = 0.7  →  strong feedforward
%    FF_X = FF_Y = 1.0  →  very strong (risk of oscillation)
%
%    Cyan path BEHIND yellow  →  INCREASE FF_X, FF_Y
%    Cyan path AHEAD of yellow →  DECREASE FF_X, FF_Y
%    More oscillation appears  →  DECREASE FF_X, FF_Y
%% ========================================================================

FF_X = 0.35;   % feedforward gain for X axis → adds to Theta_d (pitch)
FF_Y = 0.40;   % feedforward gain for Y axis → adds to Phi_d   (roll)

fprintf('  Feedforward gains loaded: FF_X=%.2f  FF_Y=%.2f\n', FF_X, FF_Y);

%% ========================================================================
%  10. SATURATION LIMITS
%% ========================================================================

att_cmd_limit  = 18 * pi/180;  % max desired roll/pitch [rad] = 15 degrees
yaw_rate_limit = 1.0;          % max yaw rate [rad/s]

omega_max    = 1000;           % max rotor speed [rad/s]
T_max        = 4 * K_thrust * omega_max^2;
T_min        = 0;
M_max        = L * K_thrust * omega_max^2;
omega_sq_min = 0;
omega_sq_max = omega_max^2;

%% ========================================================================
%  11. SIMULATION SETTINGS
%% ========================================================================

t_sim    = 90;            % total simulation time [s]
Ts_inner = 0.005;          % inner loop sample time [s]  200 Hz
Ts_mid   = 0.010;          % middle loop sample time [s] 100 Hz
Ts_outer = 0.020;          % outer loop sample time [s]  50 Hz
Ts_out   = 0.001;          % output logging sample time [s]

%% ========================================================================
%  12. FIGURE-8 REFERENCE TRAJECTORY GENERATION
%
%  Three phases:
%  PHASE 1 — Takeoff:   Z rises 0 to 3m, X=Y=0  (0 to 8 seconds)
%  PHASE 2 — Figure-8:  X=A*sin(wt), Y=B*sin(2wt), Z=3m
%  PHASE 3 — Landing:   Z drops 3 to 0m, X=Y go to 0
%
%  These timeseries are read by From Workspace blocks in Simulink.
%% ========================================================================

fprintf('  Generating figure-8 reference trajectory...\n');

% Figure-8 parameters
A_fig8 = 4.0;    % X amplitude [m]
B_fig8 = 2.0;    % Y amplitude [m]
w_fig8 = 0.10;   % angular frequency [rad/s]
                 % One full cycle = 2*pi/0.10 = 62.8 seconds

% Phase durations
t_takeoff_duration = 8.0;    % seconds to rise to hover altitude
t_land_duration    = 8.0;    % seconds to descend

% Time vector (fine resolution for smooth interpolation)
n_points = 10000;
t_ref    = linspace(0, t_sim, n_points)';

% Pre-allocate trajectory arrays
Xd_traj = zeros(n_points, 1);
Yd_traj = zeros(n_points, 1);
Zd_traj = zeros(n_points, 1);

% Time when figure-8 ends and landing begins
t_land_start = t_sim - t_land_duration;

for k = 1:n_points
    tk = t_ref(k);

    if tk <= t_takeoff_duration
        % ---- PHASE 1: TAKEOFF ----
        % Smooth S-curve rise: starts and ends with zero velocity
        frac        = tk / t_takeoff_duration;
        smooth_frac = 3*frac^2 - 2*frac^3;
        Xd_traj(k)  = 0;
        Yd_traj(k)  = 0;
        Zd_traj(k)  = Z_hover * smooth_frac;

    elseif tk <= t_land_start
        % ---- PHASE 2: FIGURE-8 TRACKING ----
        tau        = tk - t_takeoff_duration;
        Xd_traj(k) = A_fig8 * sin(w_fig8 * tau);
        Yd_traj(k) = B_fig8 * sin(2 * w_fig8 * tau);
        Zd_traj(k) = Z_hover;

    else
        % ---- PHASE 3: LANDING ----
        frac        = (tk - t_land_start) / t_land_duration;
        smooth_frac = 3*frac^2 - 2*frac^3;

        % Position at the moment landing begins
        tau_land     = t_land_start - t_takeoff_duration;
        X_land_start = A_fig8 * sin(w_fig8 * tau_land);
        Y_land_start = B_fig8 * sin(2 * w_fig8 * tau_land);

        % Smoothly return to origin
        Xd_traj(k) = X_land_start * (1 - smooth_frac);
        Yd_traj(k) = Y_land_start * (1 - smooth_frac);
        Zd_traj(k) = Z_hover * (1 - smooth_frac);
    end
end

% Create timeseries objects for From Workspace blocks in Simulink
Xd_ts      = timeseries(Xd_traj, t_ref);
Yd_ts      = timeseries(Yd_traj, t_ref);
Zd_ts      = timeseries(Zd_traj, t_ref);
Xd_ts.Name = 'Xd';
Yd_ts.Name = 'Yd';
Zd_ts.Name = 'Zd';

fprintf('  Trajectory generated: %.0f seconds total\n', t_sim);
fprintf('  Phase 1 Takeoff:  0 to %.0f s\n',            t_takeoff_duration);
fprintf('  Phase 2 Figure-8: %.0f to %.0f s\n',         t_takeoff_duration, t_land_start);
fprintf('  Phase 3 Landing:  %.0f to %.0f s\n',         t_land_start, t_sim);

%% ========================================================================
%  13. PREVIEW TRAJECTORY PLOTS
%% ========================================================================

figure('Name','Reference Trajectory Preview','Color','w');

subplot(3,1,1);
plot(t_ref, Xd_traj, 'b', 'LineWidth', 1.5);
ylabel('Xd [m]'); title('Reference Trajectory'); grid on;
xline(t_takeoff_duration, 'r--', 'Takeoff End');
xline(t_land_start,       'g--', 'Landing Start');

subplot(3,1,2);
plot(t_ref, Yd_traj, 'r', 'LineWidth', 1.5);
ylabel('Yd [m]'); grid on;
xline(t_takeoff_duration, 'r--');
xline(t_land_start,       'g--');

subplot(3,1,3);
plot(t_ref, Zd_traj, 'g', 'LineWidth', 1.5);
ylabel('Zd [m]'); xlabel('Time [s]'); grid on;
xline(t_takeoff_duration, 'r--');
xline(t_land_start,       'g--');

figure('Name','3D Reference Trajectory Preview','Color','w');
plot3(Xd_traj, Yd_traj, Zd_traj, 'b-', 'LineWidth', 2);
hold on;
plot3(0, 0, 0,       'g^', 'MarkerSize', 12, 'MarkerFaceColor', 'g');
plot3(0, 0, Z_hover, 'r*', 'MarkerSize', 12);
grid on;
xlabel('X [m]'); ylabel('Y [m]'); zlabel('Z [m]');
title('3D Reference Trajectory (Takeoff + Figure-8 + Landing)');
legend('Reference Path', 'Start', 'Hover Altitude');
view(45, 30);

%% ========================================================================
%  14. DISPLAY COMPLETE SUMMARY
%% ========================================================================

fprintf('\n========================================\n');
fprintf('  ALL PARAMETERS LOADED SUCCESSFULLY\n');
fprintf('  Simulation time:  %d s\n',             t_sim);
fprintf('  Mass:  %.3f kg\n',                     m);
fprintf('  Arm:   %.3f m\n',                      L);
fprintf('  IXX=IYY=%.4f  IZZ=%.4f\n',            IXX, IZZ);
fprintf('  T_max: %.2f N\n',                      T_max);
fprintf('  Hover altitude:  %.1f m\n',            Z_hover);
fprintf('  Figure-8:  X=%.1f m  Y=%.1f m\n',     A_fig8, B_fig8);
fprintf('  Frequency: w=%.2f rad/s\n',            w_fig8);
fprintf('  Period:    %.1f s per cycle\n',        2*pi/w_fig8);
fprintf('----------------------------------------\n');
fprintf('  FEEDFORWARD:\n');
fprintf('  FF_X = %.2f  FF_Y = %.2f\n',           FF_X, FF_Y);
fprintf('  Increase if cyan lags yellow\n');
fprintf('  Decrease if cyan overshoots yellow\n');
fprintf('----------------------------------------\n');
fprintf('  OUTER LOOP:\n');
fprintf('  X: KP=%.2f KI=%.2f KD=%.2f\n',        KP_X, KI_X, KD_X);
fprintf('  Y: KP=%.2f KI=%.2f KD=%.2f\n',        KP_Y, KI_Y, KD_Y);
fprintf('  Z: KP=%.1f KI=%.1f KD=%.1f\n',        KP_Z, KI_Z, KD_Z);
fprintf('----------------------------------------\n');
fprintf('  MIDDLE LOOP:\n');
fprintf('  Phi:   KP=%.1f KI=%.1f KD=%.2f\n',    KP_Phi,   KI_Phi,   KD_Phi);
fprintf('  Theta: KP=%.1f KI=%.1f KD=%.2f\n',    KP_Theta, KI_Theta, KD_Theta);
fprintf('  Psi:   KP=%.1f KI=%.1f KD=%.2f\n',    KP_Psi,   KI_Psi,   KD_Psi);
fprintf('----------------------------------------\n');
fprintf('  INNER LOOP:\n');
fprintf('  P: KP=%.1f KI=%.2f KD=%.3f\n',        KP_P, KI_P, KD_P);
fprintf('  Q: KP=%.1f KI=%.2f KD=%.3f\n',        KP_Q, KI_Q, KD_Q);
fprintf('  R: KP=%.1f KI=%.2f KD=%.3f\n',        KP_R, KI_R, KD_R);
fprintf('========================================\n\n');
fprintf('SIMULINK STEPS:\n');
fprintf('  1. From Workspace blocks: Xd_ts Yd_ts Zd_ts\n');
fprintf('  2. Add Constant blocks for FF_X and FF_Y\n');
fprintf('     FF_X → Outer PID Port 32\n');
fprintf('     FF_Y → Outer PID Port 33\n');
fprintf('  3. Run simulation\n');
fprintf('  4. Run animate_quad\n');
fprintf('========================================\n');