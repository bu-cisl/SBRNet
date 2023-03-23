tic
clear all; clc
rng('shuffle')
for i = 1:1500
    i
    a = value2D(600);
    value1 = imgaussfilt(a,rand*7+13);
    imwrite(value1,['value_',num2str(i),'.png'])
end
toc

function s = value2D(m)
  s = zeros([m,m]);     % Prepare output image (size: m x m)
  w = m;
  i = 0;
  while w > 3
    i = i + 1;
    d = interp2(single(randn([m,m])), i-1, 'spline');
    d = d(1:m,1:m);
    s = s + i * d;%(1:m, 1:m);
    w = w - ceil(w/2 - 1);
  end
  s = (s - min(min(s(:,:)))) ./ (max(max(s(:,:))) - min(min(s(:,:))));
end

