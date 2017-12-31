% [y,Fs] = audioread('she.wav'); %y는 샘플된 점 개수
% % sound(y, Fs); %Fs는 주파수고 주파수가 44100이라는 뜻
% plot(y)
% t = linspace(0, length(y)*(1/Fs), length(y));
% plot(t, y);
% xlabel('time(s)');
% ylabel('amplitude');
% l = length(y)
% a = (1/Fs)
% al = length(y)*(1/Fs)
% samples = [1, 3.2*Fs]
% 
% x(t) = cos2*pi]
freq_c = 261.63;
freq_d = 293.66;
freq_e = 329.63;
Fs = 8000;
C4 = [cos(2*pi*freq_c/Fs*[0:8000])];
D4 = [cos(2*pi*freq_d/Fs*[0:8000])];
E4 = [cos(2*pi*freq_e/Fs*[0:8000])];

function [a, b] = playsound(x)
a = sound(x, 8000);
b = pause(0.5);
end

