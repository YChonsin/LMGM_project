

data = complete;

% data = missing;

% data = reconstructed;

% data = difference;

% removed = reconstructed - missing;
% data = removed;


dt = 0.008;  % 45shots


figure('Position',[400 400 755 700], 'Color','w');


[tline, xxline, inline] = size(data);

drr_plot3d(data, dt, [130, xxline-101, 101]); % 45shots

set(gca,'Linewidth',2,'Fontsize',36, 'FontName', 'Arial');
pbaspect([3 5 5]);
colormap(cseis);

clim([-20,20]);
xticks([0, 100, 201])
yticks([0, 100, 201])
zticks([0, 2.5, 4.99])
zticklabels({'0','2.5','5.0'})

shading interp; 
% axis off;