close all, clear all, clc;
%% Cuda info of PC:
% g=gpuDevice;R
% reset(g)
% disp(g.FreeMemory)
% cudaDeviceSynchronize();
% auto err = cudaGetLastError();
% if (err ~= cudaSuccess) 
%     mexPrintf("CUDA error: %s\n", cudaGetErrorString(err));
% end

%% Load Data
% unzip('v_200x400.zip');  
imds = imageDatastore('data_rgb', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames'); 
%% Split Data
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.8);

%% Load Pretrained Net
net = googlenet;
analyzeNetwork(net);
inputSize = net.Layers(1).InputSize;
%% Extract the layer graph from the trained network. 
% 'loss3-classifier' and 'output' in GoogLeNet, contain information on 
% how to combine the features that the network extracts into class probabilities, 
% a loss value, and predicted labels. To retrain a pretrained network to classify new images, 
% replace these two layers with new layers adapted to the new data set.
if isa(net,'SeriesNetwork') 
  lgraph = layerGraph(net.Layers); 
else
  lgraph = layerGraph(net);
end 
[learnableLayer,classLayer] = findLayersToReplace(lgraph);
% 
numClasses = numel(categories(imdsTrain.Labels));
if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end
lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);  
figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
plot(lgraph)
ylim([0,10])

%% Freezing layers and reconnecting
layers = lgraph.Layers;
connections = lgraph.Connections;

layers(1:82) = freezeWeights(layers(1:82));
lgraph = createLgraphUsingConnections(layers,connections);

%% Aumenta el training set
pixelRange = [-30 30];
scaleRange = [0.9 1.1];
rotation_scale = [0 360];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange, ...
    'RandXScale',scaleRange, ...
    'RandYScale',scaleRange, ...
    'RandRotation', rotation_scale );
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);
%% Aumenta el test set
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);
%% Parametros de entrenamiento
% Linea 82 debe eliminarse para que use GPU por defecto o seleccionar
% cpu/gpu
options = trainingOptions('sgdm', ...
    'MiniBatchSize',256, ...
    'MaxEpochs',50, ...
    'InitialLearnRate',1e-4, ...
    'Momentum', 0.9, ...
    'Shuffle','every-epoch', ...
    'ExecutionEnvironment', 'auto', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',1, ...
    'Verbose',true, ...
    'ExecutionEnvironment','parallel', ...
    'Plots','training-progress');
%% Entrenamiento
net = trainNetwork(augimdsTrain,lgraph,options);
%% Validacion
[YPred,probs] = classify(net,augimdsValidation);
accuracy = mean(YPred == imdsValidation.Labels)
%% Testeo
idx = randperm(numel(imdsValidation.Files),4);
figure
for i = 1:4
    subplot(2,2,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label) + ", " + num2str(100*max(probs(idx(i),:)),3) + "%");
end

save('modelo1', 'net')