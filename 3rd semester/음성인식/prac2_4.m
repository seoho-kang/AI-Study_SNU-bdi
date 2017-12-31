f1 = 2000;
Fs1 = 32000;
A1 = 1.5;
M = 64;
t = 0:1/32000:M-1;
n1 = 0:M-1;

x1 = A1*cos(2*pi*f1*t);
x11 = A1*cos(2*pi*f1/Fs1*n1);

figure(1), subplot(211), plot(t, x1, 'r'), hold on, stem(n1, x11), hold off

DFT_X1 = fft(x1, M);
f = (0:(M-1))*Fs1/M;
subplot(212), plot(f, abs(DFT_X1), 'r');
xlim([0 Fs1/2]);
ylabel('DFT Values');
xlabel('Frequency (Hz)');