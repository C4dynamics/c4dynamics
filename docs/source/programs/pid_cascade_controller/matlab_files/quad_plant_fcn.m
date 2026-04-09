function quadplant(block)
setup(block);
function setup(block)
 block.NumInputPorts = 4 ;
 block.NumOutputPorts = 12;
 
 for i = 1:4; % These are the motor inputs
 block.InputPort(i).Dimensions = 1;
 block.InputPort(i).DirectFeedthrough = false;
 block.InputPort(i).SamplingMode = 'Sample';
 end

 for i = 1:12;
 block.OutputPort(i).Dimensions = 1;
 block.OutputPort(i).SamplingMode = 'Sample';
 end

 % Register the parameters.
 block.NumDialogPrms = 0; %fromtemplate
 
 % Set up the continuous states.
block.NumContStates = 12; %notintemplate
 block.SampleTimes = [0 0];
 
 block.SetAccelRunOnTLC(false); 
 block.SimStateCompliance = 'DefaultSimState';
 
 block.RegBlockMethod('InitializeConditions', @InitializeConditions);
 
 block.RegBlockMethod('Outputs', @Outputs);
 block.RegBlockMethod('Derivatives', @Derivatives);
 block.RegBlockMethod('Terminate', @Terminate); % Required

function InitializeConditions(block)
% P, Q, R are in rad/s
P=0; Q=0; R=0;
% Phi, The, Psi are in rads

Phi=0; The=0; Psi=0;
U=0; V=0; W=0;
X=0; Y=0; Z=0;
init = [P,Q,R,Phi,The,Psi,U,V,W,X,Y,Z];

for i=1:12
block.OutputPort(i).Data = init(i);
block.ContStates.Data(i) = init(i);
end

function Outputs(block)
for i = 1:12;
 block.OutputPort(i).Data = block.ContStates.Data(i);
end

function Derivatives(block)
% P Q R in units of rad/sec
P = block.ContStates.Data(1);
Q = block.ContStates.Data(2);
R = block.ContStates.Data(3);
% Phi The Psi in radians
Phi = block.ContStates.Data(4);
The = block.ContStates.Data(5);
Psi = block.ContStates.Data(6);
% U V W in units of m/s
U = block.ContStates.Data(7);
V = block.ContStates.Data(8);
W = block.ContStates.Data(9);
% X Y Z in units of m
X = block.ContStates.Data(10);
Y = block.ContStates.Data(11);
Z = block.ContStates.Data(12);
% w values in rev/min! NOT radians/s!!!!
w1 = block.InputPort(1).Data;
w2 = block.InputPort(2).Data;
w3 = block.InputPort(3).Data;
w4 = block.InputPort(4).Data; 
w = [w1; w2; w3; w4];
% CALCULATE MOMENT AND THRUST FORCES
%find k,d,l
k=2.98e-06; d=.0382; l=0.225;
%find m,Ixx,Iyy,Izz,Ir
m=0.468; Ixx=4.856e-03;Iyy=4.856e-03;Izz=8.801e-03;Ir=3.357e-05;
Ax=.3; Ay=0.3; Az=0.25; Ar=0.2;
T1= k*w1^2;
T2= k*w2^2;
T3= k*w3^2;
T4= k*w4^2;

T = T1+T2+T3+T4; %total thrust
Mphi= l*(T4-T2); %torques
Mthe= l*(T3-T1);
Mpsi= d*(-T1+T2-T3+T4);
Omega=w1-w2+w3-w4;
dP= ((Iyy-Izz)/Ixx)*Q*R - Ir/Ixx * Q*Omega + Mphi/Ixx - Ar/Ixx*P;
dQ= ((Izz-Ixx)/Iyy)*P*R + Ir/Iyy * P*Omega + Mthe/Iyy - Ar/Iyy*Q;
dR= ((Ixx-Iyy)/Izz)*P*Q + Mpsi/Izz -Ar/Izz*R;
dPhi= P+ sin(Phi)*tan(The)*Q + cos(Phi)*tan(The)*R;
dTheta= cos(Phi)*Q - sin(Phi)*R;
dPsi= sin(Phi)/cos(The)*Q + cos(Phi)/cos(The)*R;
dU= ( sin(Phi)*sin(Psi) + cos(Phi)*sin(The)*cos(Psi) )*T/m - Ax/m*U;
dV= ( -sin(Phi)*cos(Psi) + cos(Phi)*sin(The)*sin(Psi) )*T/m - Ay/m*V;
dW= -9.8 + cos(Phi)*cos(The)*T/m - Az/m*W;
vb = [U;V;W];
dX = U;
dY = V;
dZ = W;
 
f = [dP dQ dR dPhi dTheta dPsi dU dV dW dX dY dZ].';
 %This is the state derivative vector
block.Derivatives.Data = f;
function Terminate(block)
%endfunction