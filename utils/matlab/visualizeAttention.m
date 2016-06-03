function [] = visualizeAttention(sCell, wCell, colormap, saveDir, num)
%for i=2230:size(sCell,2)
if num == -1 || num > size(sCell,2)
    num = size(sCell,2)
end

for i=1:num
    heatmap(wCell{i},sCell{i}, '', [], 'TickAngle', 45, 'ShowAllTicks', true, 'Colormap', colormap./255); 
    set(gcf,'PaperPositionMode', 'auto');
    set(gcf,'units','points','position',[10,200,size(wCell{i},2)*30,250]);
    saveas(gcf,[saveDir,num2str(i)],'eps');
    close;
end
end
