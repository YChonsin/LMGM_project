
function [map]=cseis()

% map = [[0.5*ones(1,40),linspace(0.5,1,88),linspace(1,0,88),zeros(1,40)]',[0.25*ones(1,40),linspace(0.25,1,88),linspace(1,0,88),zeros(1,40)]',[zeros(1,40),linspace(0.,1,88),linspace(1,0,88),zeros(1,40)]'];
    seismic_color = load('blue_white_red.mat');
%     seismic_color = load('colorbar/fk.mat');
%     seismic_color = load('colorbar/red_white_black_1.mat');
%     seismic_color = load('colorbar/red_white_black_2.mat');
%     seismic_color = load('colorbar/blue_white_red.mat');
%     seismic_color = load('colorbar/grey.mat');
    seismic_color = struct2cell(seismic_color);
    map = cell2mat(seismic_color);
return