v = version();
di = find(v=='.');
v = str2num(v(1:di(2)-1));

if strcmp(computer(),'GLNXA64') || strcmp(computer(), 'PCWIN64')
    mex -g  -largeArrayDims GCMex.cpp graph.cpp GCoptimization.cpp LinkedBlockList.cpp maxflow.cpp
else
    mex -g GCMex.cpp graph.cpp GCoptimization.cpp LinkedBlockList.cpp maxflow.cpp
end    
