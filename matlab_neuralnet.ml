T = readtable('C:\Users\crist\Documents\AES tests\tstnew1_output_short4.csv');


XTrain = cell(length(T.TEXT), 1)
YTrain = cell(length(T.TEXT), 1)
for j=1:length(XTrain)
    aa = double(T.TEXT{j})
    w = zeros(1, length(T.TEXT{j}))
    z = zeros(1, length(T.TEXT{j}))
    for i=1:length(T.TEXT{j})
        substr = extractBetween(T.ENCRYPTED{j},i*2-1,i*2)
        z(1,i) = hex2dec(substr{1})
        w(1,i) = aa(1,i)
    end
    XTrain{j} = w
    YTrain{j} = z 
end





numFeatures = size(YTrain{1},1)
numResponses = size(XTrain{1},1);
numHiddenUnits = 200;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
    fullyConnectedLayer(128)
    dropoutLayer(0.5)
    fullyConnectedLayer(256)
    dropoutLayer(0.5)
    fullyConnectedLayer(256)
    dropoutLayer(0.5)
    fullyConnectedLayer(128)
    dropoutLayer(0.5)
    fullyConnectedLayer(numResponses)
    regressionLayer];

maxEpochs = 60;
miniBatchSize = 20;

options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',0.01, ...
    'GradientThreshold',1, ...
    'Shuffle','never', ...
    'Plots','training-progress',...
    'Verbose',0);



net = trainNetwork(YTrain,XTrain,layers,options);



YPred = predict(net,YTrain,'MiniBatchSize',1);


thr = 150;
idx = randperm(numel(YPred),4);
figure
for i = 1:numel(idx)
    subplot(2,2,i)
    
    plot(XTrain{idx(i)},'--')
    hold on
    plot(YPred{idx(i)},'.-')
    hold off
    
    ylim([0 thr + 25])
    title("Test Observation " + idx(i))
    xlabel("Time Step")
    ylabel("RUL")
end
legend(["Test Data" "Predicted"],'Location','southeast')


for i = 1:numel(XTrain)
    YTestLast(i) = XTrain{i}(end);
    YPredLast(i) = YPred{i}(end);
end
figure
rmse = sqrt(mean((YPredLast - YTestLast).^2))
histogram(YPredLast - YTestLast)
title("RMSE = " + rmse)
ylabel("Frequency")
xlabel("Error")



