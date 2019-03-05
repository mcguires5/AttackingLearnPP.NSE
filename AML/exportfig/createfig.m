function [fig opts]=createfig(fontsize, w, h)
%load figure settings
options
opts.fontsize = fontsize;
fig=figure('InvertHardcopy','off','Color',[1 1 1]);
axes('Parent',fig,'FontSize',fontsize,'LineWidth',1.5);
set(fig, 'PaperPositionMode', 'manual');
set(fig, 'PaperUnits', 'centimeters');
set(fig, 'PaperPosition', [2 2 opts.width opts.height]);