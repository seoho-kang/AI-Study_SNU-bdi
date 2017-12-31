M=64;
Fs=32000; %sampling frequncy=32000


f2=2000; %w=pi/8
A2=1.5; %original frequncy=2000
omega2=2*pi*f2/Fs;
n2=0:M-1;
xn2=A2*cos(omega2.*n2);
figure(2),subplot(311), stem(n2,xn2);
axis([ 0 30 -3 3] ); grid; xlabel('Time[n]'); ylabel('x[n]');
title('x[n]= Acos(\omegan+\theta), \pi/8, A=1.5'); 

f3=4000; % w= pi/4 
A3=2; %original frequncy=4000
omega3=2*pi*f3/Fs;
n3=0:M-1;
xn3=A3*cos(omega3.*n3);
subplot(312), stem(n3,xn3);
axis([ 0 30 -3 3] ); grid; xlabel('Time[n]'); ylabel('x[n]');
title('x[n]= Acos(\omegan+\theta), \pi/4, A=2'); 

f4=6000; %w= pi/8
A4=2.5; %original frequncy=6000
omega4=2*pi*f4/Fs;
n4=0:M-1;
xn4=A4*cos(omega4.*n4);
subplot(313), stem(n4,xn4);
axis([ 0 30 -3 3] );
grid;
xlabel('Time[n]');
ylabel('x[n]');
title('x[n]= Acos(\omegan+\theta), 3\pi/8, A=2'); 