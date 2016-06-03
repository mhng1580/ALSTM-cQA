function [] = run(sPath, wPath, saveDir)
    [sCell, wCell] = loadData(sPath, wPath);
    cStruct = load('/data/sls/scratch/wnhsu/ANLP/ANLP_final/rnn_enc/utils/matlab/colormap.mat');
    colormap = cStruct.colormap;
    visualizeAttention(sCell, wCell, colormap, saveDir, 20);
end
