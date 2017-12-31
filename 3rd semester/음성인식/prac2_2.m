N = 16;
Ts = 1;
f0 = 0.25;

t = 0:0.001:N-1;
x = cos(2*pi*f0*t);

n = 0:N-1;
x1 = cos(2*pi*f0*n*Ts);

figure(1), subplot(311), plot(t, x, 'r'), hold on, stem(n, x1), hold off, ...
    title('Sinusoid Sampled at 4 the sampling rate'), ...
    xlabel('Time(samples)'), ylabel('Amplitude of x(Ts)');

Ts2 = 4;
n2 = 0:Ts2:N-1;
x12 = cos(2*pi*f0*n2*Ts2);
subplot(312), plot(t, x, 'r'), hold on, stem(n2, x12), hold off, ...
    title('Sinusoid Sampled at 1 the sampling rate'), ...
    xlabel('Time(samples)'), ylabel('Amplitude of x(Ts)');

Ts4 = 2;
n4 = 0:Ts4:N-1;
%x = cos(2*pi*f0*t*Ts3):
x14 = cos(2*pi*f0*n4*Ts);
subplot(313), plot(t, x, 'r'), hold on, stem(n4, x14), hold off, ...
    title('Sinusoid Sampled at 2 the sampling rate'), ...
    xlabel('Time(samples)'), ylabel('Amplitude of x(Ts)');
n4