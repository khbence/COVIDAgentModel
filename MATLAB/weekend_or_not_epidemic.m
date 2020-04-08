function y=weekend_or_not_epidemic(t)

% y=1 ha hetvege van, 0 ha nem

%egy nap ugye 144 idolepes
% egy het 1008, az 5 munkanap 720

if mod(t,1008) > 720
    y=1;
else
    y=0;
end