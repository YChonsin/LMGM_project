

data = complete;

% data = missing;

% data = reconstructed; 

% data = difference;


line = 2;   % the line you want to display, 45shots: 2
dt = 0.008;  %  45shots: 0.008

data = data(:, :, line);
[nt, nx] = size(data);

t = (0:nt-1)*dt;               % 时间轴
x = 1:nx;                      % 道号


figure('Color','w','Position',[100 100 640 800]);

imagesc(x,t,data);
colormap(cseis);      

set(gca,'FontSize',32,'LineWidth',1.2);

set(gca,'YDir','reverse');   % 时间向下（地震常用显示方式）
set(gca,'XAxisLocation','top')  % x轴到上面
xlabel('Trace', 'FontWeight', 'bold');
ylabel('Time (s)', 'FontWeight', 'bold');

yticks([0, 1, 2, 3, 4, 4.99]);  
yticklabels({'0', '1','2','3','4','5'})
xticks([1, 50, 100, 150, 201]);

set(get(gca,'Children'),'Interpolation','bilinear'); % 平滑
% colorbar;
pbaspect([4 5 5])
clim([-10, 10]) 

% axis off