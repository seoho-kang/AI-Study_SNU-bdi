[sound, rate] = audioread('wsj.wav');

preempsound = filter([1 -0.97], 1, sound);

windowed_sound = zeros(400, 553);
i = 1;
for n = 1:553
    windowed_sound(:, n) = preempsound(i:i+(rate*0.025)-1);
    i = i+rate*0.01;
end
figure(1), subplot(211), plot(windowed_sound(:, 1)), hold on, plot(sound(1:400), 'r'), hold off;
%sound(sound, rate);

%hamming window
wHamm = hamming(length(windowed_sound(:, 1)));
Hammed = zeros(400, 553);
for n = 1:553
    Hammed(:, n) = wHamm.*windowed_sound(:, n);
end
subplot(212), plot(Hammed(:,1), 'r'), hold on, plot(windowed_sound(:, n)), hold off; 

fftsound = zeros(512, 553);
for n = 1:553
    fftsound(:, n) = fft(Hammed(:, n), 512);
end
figure(2), subplot(211), stem(fftsound(:, 1))

ensound = zeros(512, 553);
for n = 1:553
    ensound(:, n) = abs(fftsound(:, n)).^2;
end
subplot(212), stem(ensound(:, 1))

fb = load('fb.mat');
fb = struct2array(fb);
fb_sound = zeros(26, 553);

for n = 1:553
    for i = 1:26
        fb_sound(i, n) = log(sum(ensound(1:257, n)'.*fb(i, :)));
    end
end
figure(3), stem(fb_sound(:, 1))

inverse_sound = zeros(26, 553);
for n = 1:553
    inverse_sound(:, n) = ifft(fb_sound(:, n));
end
