function [ Wlr, blr, Wnn1, Wnn2, bnn1, bnn2, trainErr, testCorr, testWrong, ErrPercent] = runLogisticAndNeural()

    x = loadImages('train-images.idx3-ubyte');
    T = loadLabels('train-labels.idx1-ubyte');
    testx = loadImages('t10k-images.idx3-ubyte');
    testT = loadLabels('t10k-labels.idx1-ubyte');
    
    target = zeros(10, size(T, 1));
    for n = 1: size(T, 1)
        target(T(n) + 1, n) = 1;
    end;   
    w = LogisticRegression(x', target', testx', testT');
    
    blr = w(size(x,1)+1,:);
    Wlr = w(1:size(x,1),:);
    logisticended = 1
    
   
   [Wnn1, Wnn2, bnn1, bnn2, trainErr, testCorr, testWrong] = neuralNetwork( x, target, testx,testT);
    
   ErrPercent = 100*(testWrong/(testCorr+testWrong));

end

function [Wnn1, Wnn2, bnn1, bnn2, error, correct, wrong] = neuralNetwork( x, target, testx, testT)
    
    HidUnits = 400;
    learnRate = 0.1;
    batchSz = 100;
    
    % number of columns = 60,000
    trainSz = size(x, 2);
    % making it 60,000 X 785
    x = [ones(1,size(x,2));x];

    features = size(x, 1);
    % 785 X 60,000
    
    outDim = size(target, 1);
    
    hidW = rand(HidUnits, features)/(features);
    outW = rand(outDim, HidUnits+1)/(HidUnits+1);
    
    n = zeros(batchSz);
    
    error = 1;
    count = 1;
    i = 0;
    figure; hold on;
    while error >= 0.04
        count = count + 1;
        n = randperm(trainSz);
        n = n(1:batchSz);
        for k = 1: batchSz
            
            inputVector = x(:, n(k));
            hidIn = hidW * inputVector;
            hidIn = [ ones(size(hidIn,2),1); hidIn ];
            hidActiv =  1./(1 + exp(- hidIn ));
            hIn = outW * hidActiv;
            activOut =  1./(1 + exp(- hIn ));
            
            activHid = 1./(1 + exp(-hIn));
            dactivHid = activHid.*(1 - activHid);
            outputDelta = dactivHid.*( activOut - target(:, n(k)) );
           
            activHid = 1./(1 + exp(-hidIn));
            dactivHid = activHid.*(1 - activHid);
            hiddenDelta = dactivHid.*(outW'*outputDelta);
            
            outW = outW - learnRate.*outputDelta*hidActiv';
            delta = hiddenDelta*inputVector';
            delta = delta(2:size(delta,1),:);
            hidW = hidW - learnRate.*delta;
        
        end;
        
        error = 0;
        ow = outW(:,2:size(outW,2));
        
        for k = 1: batchSz
            activHid = 1./(1 + exp(-hidW*x(:, n(k))));
            hidActive = activHid;
            outIn = ow*hidActive;
            activHid = 1./(1 + exp(-outIn));
            error = error + norm(activHid - target(:, n(k)), 2);
        end;
        error = error/batchSz
        plot(i, error,'*'); 
        i = i + 1;
    end;
    Wnn1 = hidW(:,2:size(hidW,2));
    bnn1 = hidW(:,1)';
    Wnn2 = outW(:,2:size(outW,2));
    bnn2 = outW(:,1)';
    testSetSize = size(testx, 2);
    correct = 0;
    count
    for n = 1: testSetSize
        datapt = testx(:, n);
        activHid = 1./(1 + exp(-Wnn1*datapt));
        activOut = 1./(1 + exp(-Wnn2*activHid));
        [~,class] = max(activOut);
        correct = correct + ((class-1) == testT(n));
    end;
    wrong = testSetSize - correct;
    Wnn1 = Wnn1';
    Wnn2 = Wnn2';
end

function w = LogisticRegression( x, T,testx,testT )
    miniBSz = 100;
    iterations = 20;

    feature = size(x,2); 
    classes = size(T,2);  

    bias = 1;
    phi = [x, ones(size(x,1),bias)];
    testphi = [testx, ones(size(testx,1),bias)];

    w = randn(feature+1,classes)*0.1; 
    testN = size(testphi,1); 
    N = size(phi,1);  
      
    indices = randperm(N);  
    pass = 1;          
    learnRate = 1/pass;
    strtMiniB = 1;
    
    figure; hold on;
    while(pass < iterations)
        stopMiniB   = min(N,strtMiniB+miniBSz-1);
        miniIndi = indices(strtMiniB:stopMiniB);
        miniphi = phi(miniIndi,:);
        miniT = T(miniIndi,:);
        miniN = size(miniphi,1);
        classes = size(miniT,2);

        dw = zeros(size(w));
        for n = 1:miniN
            val = exp(w'*miniphi(n,:)');
            y = val/sum(val);
            
            for i=1:classes
                dw(:,i) = dw(:,i) + (1/miniN)*(y(i) - miniT(n,i))*miniphi(n,:)';
            end
        end
        w = w - learnRate*dw;
        strtMiniB = strtMiniB+miniBSz;
        if(strtMiniB>N)
            for i=1:testN
                [~, learntT(i)] = max(w'*testphi(i,:)');
                learntT(i) = learntT(i)-1;
            end
            success =  sum(learntT(:)==testT(:));
            wrong = testN - success;
            error = (wrong/testN)*100;
            strtMiniB = 1;     
            pass = pass+1;
            learnRate = 1/pass;
            indices = randperm(N);
            plot(pass, error,'*');
        end
    end
end
