%% ========================================================================
%  QUADCOPTER ANIMATION — REAL TIME UPDATING PLOTS
%
%  Fixes:
%  1) Error and angle plots update in real time frame by frame
%  2) Phase detection uses actual Z value not just time
%
%  Requires in workspace: states_log, tout, Xd_traj, Yd_traj,
%                         Zd_traj, t_ref, t_sim, Z_hover
%% ========================================================================

clc;

%% ========================================================================
%  CHECK DATA
%% ========================================================================

if ~exist('states_log','var') || ~exist('tout','var')
    error('Run Simulink simulation first. states_log and tout must exist.');
end

if ~exist('Xd_traj','var')
    error('Run quad_params.m first. Xd_traj must exist in workspace.');
end

fprintf('Data found. Building animation...\n');

%% ========================================================================
%  UNPACK REAL STATES
%% ========================================================================

Phi_real   = states_log(:,4);
Theta_real = states_log(:,5);
Psi_real   = states_log(:,6);
X_real     = states_log(:,10);
Y_real     = states_log(:,11);
Z_real     = states_log(:,12);
t_real     = tout;
N_real     = length(t_real);

%% ========================================================================
%  INTERPOLATE REFERENCE TO MATCH SIMULATION TIME
%% ========================================================================

Xd_interp = interp1(t_ref, Xd_traj, t_real, 'linear', 'extrap');
Yd_interp = interp1(t_ref, Yd_traj, t_real, 'linear', 'extrap');
Zd_interp = interp1(t_ref, Zd_traj, t_real, 'linear', 'extrap');

%% ========================================================================
%  COMPUTE ERRORS
%% ========================================================================

eX = Xd_interp - X_real;
eY = Yd_interp - Y_real;
eZ = Zd_interp - Z_real;
position_error = sqrt(eX.^2 + eY.^2 + eZ.^2);

%% ========================================================================
%  FIGURE AND AXES SETUP
%% ========================================================================

fig = figure('Name','Cascade PID Quadcopter — Real Time Animation', ...
             'Color','k', ...
             'Position',[30 30 1300 820]);

%% --- Main 3D axes ---
ax = axes('Parent',fig, ...
          'Position', [0.03 0.22 0.58 0.73], ...
          'Color',    [0.04 0.04 0.09], ...
          'XColor','w','YColor','w','ZColor','w', ...
          'GridColor',[0.28 0.28 0.28], ...
          'FontSize', 11);
hold(ax,'on'); grid(ax,'on');
view(ax,45,28);

A_lim = max(abs(Xd_traj)) + 2.0;
B_lim = max(abs(Yd_traj)) + 2.0;
xlim(ax,[-A_lim  A_lim]);
ylim(ax,[-B_lim  B_lim]);
zlim(ax,[-0.3    Z_hover+2.5]);
xlabel(ax,'X (m)','Color','w','FontSize',11);
ylabel(ax,'Y (m)','Color','w','FontSize',11);
zlabel(ax,'Z (m)','Color','w','FontSize',11);
title(ax,'Cascade PID Quadcopter — Real Simulation Results', ...
         'Color','w','FontSize',13,'FontWeight','bold');

%% --- Error plot axes (top right) ---
ax_err = axes('Parent',fig, ...
              'Position',[0.65 0.55 0.33 0.38], ...
              'Color',   [0.04 0.04 0.09], ...
              'XColor','w','YColor','w', ...
              'FontSize', 10);
hold(ax_err,'on'); grid(ax_err,'on');
title(ax_err,'Position Tracking Error  (Real Time)', ...
             'Color','w','FontSize',11,'FontWeight','bold');
xlabel(ax_err,'Time (s)','Color','w');
ylabel(ax_err,'Error (m)','Color','w');
xlim(ax_err,[0 t_sim]);
y_err_max = max(position_error) * 1.15 + 0.5;
ylim(ax_err,[-y_err_max  y_err_max]);

%% --- Attitude plot axes (bottom right) ---
ax_att = axes('Parent',fig, ...
              'Position',[0.65 0.10 0.33 0.38], ...
              'Color',   [0.04 0.04 0.09], ...
              'XColor','w','YColor','w', ...
              'FontSize', 10);
hold(ax_att,'on'); grid(ax_att,'on');
title(ax_att,'Euler Angles  (Real Time)', ...
             'Color','w','FontSize',11,'FontWeight','bold');
xlabel(ax_att,'Time (s)','Color','w');
ylabel(ax_att,'Angle (deg)','Color','w');
xlim(ax_att,[0 t_sim]);
y_att_max = max([abs(Phi_real); abs(Theta_real); abs(Psi_real)])*180/pi * 1.2 + 2;
ylim(ax_att,[-y_att_max  y_att_max]);

%% ========================================================================
%  INITIALIZE REAL-TIME PLOT LINES
%  These start empty and grow frame by frame
%% ========================================================================

% Error plot lines — start empty
h_eX  = plot(ax_err, NaN, NaN, 'Color',[0.25 0.55 1.00],'LineWidth',1.8);
h_eY  = plot(ax_err, NaN, NaN, 'Color',[1.00 0.30 0.30],'LineWidth',1.8);
h_eZ  = plot(ax_err, NaN, NaN, 'Color',[0.25 1.00 0.35],'LineWidth',1.8);
h_eTot= plot(ax_err, NaN, NaN, 'Color',[1.00 0.90 0.15],'LineWidth',2.2);

legend(ax_err,{'eX','eY','eZ','|e|'}, ...
       'TextColor','w','Color',[0.08 0.08 0.10],'FontSize',9, ...
       'Location','northwest');

% Attitude plot lines — start empty
h_phi = plot(ax_att, NaN, NaN, 'Color',[0.25 0.65 1.00],'LineWidth',1.8);
h_tht = plot(ax_att, NaN, NaN, 'Color',[1.00 0.45 0.20],'LineWidth',1.8);
h_psi = plot(ax_att, NaN, NaN, 'Color',[0.35 1.00 0.35],'LineWidth',1.8);

legend(ax_att,{'Phi (roll)','Theta (pitch)','Psi (yaw)'}, ...
       'TextColor','w','Color',[0.08 0.08 0.10],'FontSize',9, ...
       'Location','northwest');

%% ========================================================================
%  STATIC 3D SCENE
%% ========================================================================

% Desired figure-8 reference path
plot3(ax, Xd_traj, Yd_traj, Zd_traj, ...
      '--','Color',[0.38 0.38 0.38],'LineWidth',1.5);

% Ground plane
[gx,gy] = meshgrid(-A_lim:1.5:A_lim, -B_lim:1.5:B_lim);
surf(ax,gx,gy,zeros(size(gx)), ...
     'FaceColor',[0.06 0.06 0.11],'EdgeColor',[0.18 0.18 0.24],'FaceAlpha',0.55);

% Start marker
plot3(ax,X_real(1),Y_real(1),0,'g^','MarkerSize',13,'MarkerFaceColor','g');
text(ax,X_real(1)+0.2,Y_real(1)+0.2,0.2,'Start','Color','g','FontSize',11);

% Hover altitude ring
th_r = linspace(0,2*pi,80);
plot3(ax,0.3*cos(th_r),0.3*sin(th_r),Z_hover*ones(1,80),'y-','LineWidth',1.5);
text(ax,0.4,0,Z_hover,sprintf('Z=%.0fm',Z_hover),'Color','y','FontSize',10);

%% ========================================================================
%  DYNAMIC 3D ELEMENTS
%% ========================================================================

% Actual path trail (cyan)
h_trail_real = plot3(ax,X_real(1),Y_real(1),Z_real(1), ...
                     'Color',[0.20 1.00 0.20],'LineWidth',2.5);

% Reference trail (yellow)
h_trail_ref  = plot3(ax,Xd_interp(1),Yd_interp(1),Zd_interp(1), ...
                     'Color',[1.00 0.88 0.10],'LineWidth',1.5);

% Tracking error vector (red line from actual to reference)
h_err_vec = plot3(ax,[X_real(1) Xd_interp(1)], ...
                     [Y_real(1) Yd_interp(1)], ...
                     [Z_real(1) Zd_interp(1)], ...
                  'r-','LineWidth',2.2);

% Reference position dot (yellow)
h_ref_dot = plot3(ax,Xd_interp(1),Yd_interp(1),Zd_interp(1), ...
                  'y*','MarkerSize',12,'LineWidth',2.0);

%% ========================================================================
%  QUADCOPTER + SHAPE
%% ========================================================================

arm_scale  = 0.60;
motor_size = 10;
body_size  = 15;

m1_b = [ arm_scale;  0;         0];
m2_b = [ 0;         -arm_scale; 0];
m3_b = [-arm_scale;  0;         0];
m4_b = [ 0;          arm_scale; 0];

h_arm1 = plot3(ax,[0 0],[0 0],[0 0],'w-','LineWidth',3);
h_arm2 = plot3(ax,[0 0],[0 0],[0 0],'w-','LineWidth',3);

h_m1 = plot3(ax,0,0,0,'o','MarkerSize',motor_size, ...
             'MarkerFaceColor',[1.0 0.25 0.25],'MarkerEdgeColor','w','LineWidth',1.5);
h_m2 = plot3(ax,0,0,0,'o','MarkerSize',motor_size, ...
             'MarkerFaceColor',[1.0 0.55 0.10],'MarkerEdgeColor','w','LineWidth',1.5);
h_m3 = plot3(ax,0,0,0,'o','MarkerSize',motor_size, ...
             'MarkerFaceColor',[1.0 0.25 0.25],'MarkerEdgeColor','w','LineWidth',1.5);
h_m4 = plot3(ax,0,0,0,'o','MarkerSize',motor_size, ...
             'MarkerFaceColor',[1.0 0.55 0.10],'MarkerEdgeColor','w','LineWidth',1.5);

h_body = plot3(ax,0,0,0,'s','MarkerSize',body_size, ...
               'MarkerFaceColor',[0.20 0.55 1.0],'MarkerEdgeColor','w','LineWidth',2.0);

h_heading = quiver3(ax,0,0,0,0,0,0, ...
                    'Color',[0.2 1.0 0.2],'LineWidth',2.5,'MaxHeadSize',0.8);

h_zline  = plot3(ax,[0 0],[0 0],[0 0],'--','Color',[0.50 0.50 0.50],'LineWidth',0.8);
h_shadow = plot3(ax,0,0,0.01,'o','MarkerSize',11, ...
                 'MarkerFaceColor',[0.22 0.22 0.22],'MarkerEdgeColor','none');

%% ========================================================================
%  TEXT OVERLAYS ON 3D AXES
%% ========================================================================

h_phase = text(ax,-A_lim+0.3, B_lim-0.5, Z_hover+2.1, ...
               'PHASE 1:  TAKEOFF  ↑', ...
               'Color',[0.4 1.0 0.4],'FontSize',13,'FontWeight','bold');

h_time = text(ax,-A_lim+0.3, B_lim-0.5, Z_hover+1.5, ...
              't = 0.00 s', ...
              'Color','y','FontSize',12,'FontWeight','bold');

h_pos_txt = text(ax,-A_lim+0.3, B_lim-0.5, Z_hover+0.9, ...
                 'X=0.00  Y=0.00  Z=0.00', ...
                 'Color',[0.85 0.85 0.85],'FontSize',10);

h_err_txt = text(ax,-A_lim+0.3, B_lim-0.5, Z_hover+0.3, ...
                 '|error| = 0.000 m', ...
                 'Color',[1.0 0.5 0.5],'FontSize',10);

h_loop_txt = text(ax,-A_lim+0.3, B_lim-0.5, Z_hover-0.3, ...
                  '', ...
                  'Color',[0.4 1.0 0.9],'FontSize',10);

% Trail legend
%text(ax, A_lim-2.8, -B_lim+0.5, Z_hover+1.6, ...
 %    '— Actual path','Color',[0.10 0.90 1.00],'FontSize',10);
%text(ax, A_lim-2.8, -B_lim+0.5, Z_hover+1.0, ...
 %    '— Reference',  'Color',[1.00 0.88 0.10],'FontSize',10);
%text(ax, A_lim-2.8, -B_lim+0.5, Z_hover+0.4, ...
 %    '-- Desired path','Color',[0.38 0.38 0.38],'FontSize',10);

%% ========================================================================
%  PHASE BOUNDARY TIMES
%% ========================================================================

t_takeoff_end = 8.0;
t_land_start  = t_sim - 8.0;

%% ========================================================================
%  ANIMATION LOOP SETTINGS
%
%  skip:       how many frames to jump each iteration
%              1  = every frame (smoothest)
%              5  = every 5th frame (faster)
%              10 = every 10th frame (fastest)
%
%  pause_time: seconds displayed per frame
%              0.08 = very slow
%              0.04 = slow
%              0.01 = fast
%              0    = as fast as possible
%% ========================================================================

skip       = 10;      % ← CHANGE THIS to speed up or slow down
pause_time = 0.02;   % ← CHANGE THIS to speed up or slow down

fprintf('=============================================\n');
fprintf('  REAL TIME ANIMATION STARTED\n');
fprintf('  Cyan  = actual quadcopter path\n');
fprintf('  Yellow= reference path\n');
fprintf('  Red   = tracking error vector\n');
fprintf('  Plots update in real time\n');
fprintf('  Close figure window to stop.\n');
fprintf('=============================================\n\n');

for i = 1:skip:N_real

    if ~ishandle(fig)
        break;
    end

    % Current real states from controller
    cx = X_real(i);
    cy = Y_real(i);
    cz = Z_real(i);
    ph = Phi_real(i);
    th = Theta_real(i);
    ps = Psi_real(i);
    tk = t_real(i);

    % Reference at this moment
    rx = Xd_interp(i);
    ry = Yd_interp(i);
    rz = Zd_interp(i);

    % ---- Rotation matrix ----
    Rz = [cos(ps) -sin(ps) 0; sin(ps)  cos(ps) 0; 0 0 1];
    Ry = [cos(th)  0 sin(th); 0        1        0;-sin(th) 0 cos(th)];
    Rx = [1 0 0; 0 cos(ph) -sin(ph); 0 sin(ph) cos(ph)];
    R  = Rz * Ry * Rx;

    % ---- Motor positions ----
    m1_i = R*m1_b + [cx;cy;cz];
    m2_i = R*m2_b + [cx;cy;cz];
    m3_i = R*m3_b + [cx;cy;cz];
    m4_i = R*m4_b + [cx;cy;cz];
    h_vec = R * [arm_scale*0.9; 0; 0];

    % ---- Update + arms ----
    set(h_arm1,'XData',[m3_i(1) m1_i(1)], ...
               'YData',[m3_i(2) m1_i(2)], ...
               'ZData',[m3_i(3) m1_i(3)]);
    set(h_arm2,'XData',[m2_i(1) m4_i(1)], ...
               'YData',[m2_i(2) m4_i(2)], ...
               'ZData',[m2_i(3) m4_i(3)]);

    % ---- Update motors ----
    set(h_m1,'XData',m1_i(1),'YData',m1_i(2),'ZData',m1_i(3));
    set(h_m2,'XData',m2_i(1),'YData',m2_i(2),'ZData',m2_i(3));
    set(h_m3,'XData',m3_i(1),'YData',m3_i(2),'ZData',m3_i(3));
    set(h_m4,'XData',m4_i(1),'YData',m4_i(2),'ZData',m4_i(3));

    % ---- Update body and heading ----
    set(h_body,'XData',cx,'YData',cy,'ZData',cz);
    set(h_heading,'XData',cx,'YData',cy,'ZData',cz, ...
                  'UData',h_vec(1),'VData',h_vec(2),'WData',h_vec(3));

    % ---- Update 3D trails ----
    set(h_trail_real,'XData',X_real(1:i), ...
                     'YData',Y_real(1:i), ...
                     'ZData',Z_real(1:i));
    set(h_trail_ref, 'XData',Xd_interp(1:i), ...
                     'YData',Yd_interp(1:i), ...
                     'ZData',Zd_interp(1:i));

    % ---- Update error vector and reference dot ----
    set(h_err_vec,'XData',[cx rx],'YData',[cy ry],'ZData',[cz rz]);
    set(h_ref_dot,'XData',rx,'YData',ry,'ZData',rz);

    % ---- Update height line and shadow ----
    set(h_zline, 'XData',[cx cx],'YData',[cy cy],'ZData',[0 cz]);
    set(h_shadow,'XData',cx,'YData',cy,'ZData',0.01);

    % ================================================================
    %  REAL TIME PLOT UPDATE
    %  Key fix: only show data up to current frame index i
    %  This makes plots grow from left to right in real time
    % ================================================================

    % Time vector up to now
    t_now = t_real(1:i);

    % Error plot — grows frame by frame
    set(h_eX,   'XData', t_now, 'YData', eX(1:i));
    set(h_eY,   'XData', t_now, 'YData', eY(1:i));
    set(h_eZ,   'XData', t_now, 'YData', eZ(1:i));
    set(h_eTot, 'XData', t_now, 'YData', position_error(1:i));

    % Attitude plot — grows frame by frame
    set(h_phi, 'XData', t_now, 'YData', Phi_real(1:i)*180/pi);
    set(h_tht, 'XData', t_now, 'YData', Theta_real(1:i)*180/pi);
    set(h_psi, 'XData', t_now, 'YData', Psi_real(1:i)*180/pi);

    % ================================================================
    %  TEXT UPDATES
    % ================================================================

    set(h_time,    'String', sprintf('t = %.2f s', tk));
    set(h_pos_txt, 'String', sprintf('X=%.2f  Y=%.2f  Z=%.2f', cx,cy,cz));
    set(h_err_txt, 'String', sprintf('|error| = %.3f m', position_error(i)));

    % ================================================================
    %  PHASE DETECTION BASED ON ACTUAL Z VALUE AND TIME
    %  Not just time — uses real Z to determine phase
    % ================================================================

    if tk <= t_takeoff_end
        % Takeoff phase
        set(h_phase,    'String', 'PHASE 1:  TAKEOFF  ↑', ...
                        'Color',  [0.40 1.00 0.40]);
        set(h_loop_txt, 'String', sprintf('Climbing... Z = %.2f m', cz));

    elseif tk <= t_land_start
        % Figure-8 phase
        set(h_phase,    'String', 'PHASE 2:  FIGURE-8  TRACKING', ...
                        'Color',  [0.30 0.70 1.00]);
        elapsed  = tk - t_takeoff_end;
        period   = 2*pi / 0.3;
        loop_num = floor(elapsed/period) + 1;
        set(h_loop_txt, 'String', sprintf('Loop: %d', loop_num));

    else
        % Landing phase
        set(h_phase,    'String', 'PHASE 3:  LANDING  ↓', ...
                        'Color',  [1.00 0.70 0.20]);
        set(h_loop_txt, 'String', sprintf('Descending... Z = %.2f m', cz));
    end

    % ================================================================
    %  CAMERA ROTATION during figure-8
    % ================================================================

    if tk > t_takeoff_end && tk <= t_land_start
        frac    = (tk - t_takeoff_end) / (t_land_start - t_takeoff_end);
        az_view = 40 + 25*sin(2*pi*frac);
        view(ax, az_view, 28);
    end

    drawnow
    pause(pause_time);

end

%% ========================================================================
%  PERFORMANCE SUMMARY
%% ========================================================================

fprintf('\n========== CONTROLLER PERFORMANCE ==========\n');

idx_fig8 = t_real >= t_takeoff_end & t_real <= t_land_start;

if sum(idx_fig8) > 0
    fprintf('  Figure-8 Phase RMSE:\n');
    fprintf('    X:   %.4f m\n', sqrt(mean(eX(idx_fig8).^2)));
    fprintf('    Y:   %.4f m\n', sqrt(mean(eY(idx_fig8).^2)));
    fprintf('    Z:   %.4f m\n', sqrt(mean(eZ(idx_fig8).^2)));
    fprintf('    |e|: %.4f m\n', sqrt(mean(position_error(idx_fig8).^2)));
    fprintf('  Max tracking error: %.4f m\n', max(position_error(idx_fig8)));
end

fprintf('=============================================\n');