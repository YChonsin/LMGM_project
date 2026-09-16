
data = complete;

% data = missing;

% data = reconstructed;

% data = difference;


line_idx = 2;

dt=0.008;  % 采样时间间隔  45shots
dx=20; % 替换为道间距   45shots

[S,f,k] = fk(data(:, :, line_idx), dt, dx, 9); 

figure('Position', [100, 100, 700, 600]);
imagesc(k,f,log(S));
% imagesc(k,f,S);
set(gca, 'FontSize', 32);
xlabel('Wavenumber (c/m)', 'FontName', 'Arial', 'FontWeight','bold');
ylabel('Frequency (Hz)', 'FontName', 'Arial', 'FontWeight','bold');



% 设置色图范围

min_value = 8.2; % 替换为你希望的最小值  
max_value = 12;  % 替换为你希望的最大值 
% min_value = 50000; % 替换为你希望的最小值
% max_value = 200000;  % 替换为你希望的最大值 
clim([min_value, max_value]);
colormap('turbo');
% colorbar;

% ylim([0, 125]);
xlim([-0.025, 0.025]); 
xticks([-0.025,  0, 0.025])
