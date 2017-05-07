clear;clc;
addpath('gcmex-1.0');
addpath('Data');

load('prob_map.mat')
load('Indian_pines_gt.mat')

Label = indian_pines_gt;
label = Label(:);
prob_estimatesM = double(prob_map);
H = 145;
W = 145;
B = 220;
nclasses = 16;
    
load('TestIndex.mat')
load('TrainIndex.mat')

TestIndex = TestIndex + 1;
TrainIndex  =TrainIndex + 1;

nTest = length(TestIndex);
TrainLabels = label(TrainIndex);
TestLabels = label(TestIndex);

TrainingMap = zeros(H,W);
TrainingMap(TrainIndex) = TrainLabels;
TestMap = Label - TrainingMap;

% Compute accuracy before MRF
[maxp,class] = max(prob_map,[],2);
predict_label = class(TestIndex);
predict_label = bestMap(TestLabels,predict_label);
Predict_label = Label;
Predict_label(TestIndex) = predict_label;

[OA_class,kappa_class,AA_class,CA_class] = calcError( TestLabels-1, predict_label-1, 1: nclasses);
CA_class
[OA_class,AA_class,kappa_class]

%% segmentation
Dc = reshape((log(prob_map+eps)),[H,W,nclasses]);
Sc = ones(nclasses) - eye(nclasses);
beta  = 20;  % spatial prior parameter

% Expantion Algorithm
gch = GraphCut('open', -Dc, beta*Sc);
[gch seg] = GraphCut('expand',gch);
gch = GraphCut('close', gch);

seg_label = seg(TestIndex) + 1;
seg_label = bestMap(TestLabels,seg_label);
Seg_label = Label;
Seg_label(TestIndex) = seg_label;

[OA_seg,kappa_seg,AA_seg,CA_seg] = calcError( TestLabels-1, seg_label-1, 1: nclasses);
CA_seg
[OA_seg,AA_seg,kappa_seg]


% display
figure;
subplot(1,5,1);imagesc(Label); axis image;set(gca,'xtick',[],'ytick',[]);title('Groundtruth Map');
subplot(1,5,2);imagesc(TrainingMap); axis image;set(gca,'xtick',[],'ytick',[]);title('Training Map');
subplot(1,5,3);imagesc(TestMap); axis image;set(gca,'xtick',[],'ytick',[]);title('Test Map');
subplot(1,5,4);imagesc(Predict_label);axis image;set(gca,'xtick',[],'ytick',[]);
st = sprintf('The classification OA = %2.2f', OA_class);title(st,'Fontsize', 10);
subplot(1,5,5);imagesc(Seg_label);axis image;set(gca,'xtick',[],'ytick',[]);
st = sprintf('The segmentation OA = %2.2f', OA_seg);title(st,'Fontsize', 10);