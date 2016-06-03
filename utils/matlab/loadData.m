function [sCell, wCell] = loadData(sPath, wPath)
sFile = fopen(sPath);
wFile = fopen(wPath);
count = 1;
sCell = {};
wCell = {};
sLine = fgetl(sFile);
wLine = fgetl(wFile);
while ischar(sLine)
    sCell{count} = strsplit(sLine);
    wCell{count} = str2num(wLine);
    count = count + 1;
    sLine = fgetl(sFile);
    wLine = fgetl(wFile);
end