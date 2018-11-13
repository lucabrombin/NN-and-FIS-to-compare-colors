clear;
clc;
% carico il dataset 
load IS_dataset.mat;
% moltiplico per 100 gli spettri per standardizzare i dati per la libreria 
spectra=spectra*100;
coordinates = coordinates(:,1:4:1269);
spectra = spectra(:,1:4:1269);
% Wavelength
wlRange = 380:1:800;  
% numero di copie
numcopies = 10; 
% creo il dataset che mi servirà per la fuzzy e per la NN
[masterCopy,DE,copyLabs,masterLabs,fuzzyInputs,lch,masterCopyRGB] = prepareInitialSets(spectra,numcopies,wlRange,coordinates);
%plotNoise(masterCopyPairs);
%%%%%%%%%FUZZY SYSTEM%%%%%%%%%
mamdaniFuzzysystem = readfis('fuzzysystem_V12.fis');
% Correggo la differenza dei colori con la fuzzy
correctDiff = evalfis(fuzzyInputs,mamdaniFuzzysystem)';
%showColors(masterCopyRGB',DE,correctDiff);
% plot delle MP utilizzate
%plotFuzzyMF(mamdaniFuzzysystem);
%%%%%%%%%FEATURES EXTRACTION%%%%%%%%%
k=10;
masterCopyExtracted = extractFeatures(masterCopy,k);
%%%%%%%%%FEATURES SELECTION%%%%%%%%%
opt = statset('display','off');
fprintf("3) FEATURES SELECTION \n");
[fs,hst]=sequentialfs(@fitFs,masterCopyExtracted',correctDiff','cv','none','opt',opt,'nfeatures',6);
% parsing delle features con il risultato della sequentialFS
masterCopySelected = masterCopyExtracted(fs,:);
%%%%%%%%%NEURAL NETWORK%%%%%%%%%
neuralNetwork(masterCopySelected,correctDiff);

%%%%%%%%%%%%%%%%FUNCTIONS%%%%%%%%%%%%%%%%
function [masterCopyPairs,DE,copyLabs,masterLabs,fuzzyInputs,lch,masterCopyRGB] = prepareInitialSets(spectra,numcopies,wlRange,coordinates)
    fprintf("1) CREO IL DATASET \n");
    optsetpref('cwf','D65/2'); 
    cmax = 127;
    spectraSizes = size(spectra);
    numSamples=spectraSizes(2);
    numWL = spectraSizes(1);
    %creo matrici temporanee composte da zeri
    masterCopyPairs = zeros([numWL*2 numSamples*numcopies],'double');
    DE = zeros([1 numSamples*numcopies],'double');
    fuzzyInputs = zeros([4 numSamples*numcopies],'double');
    
    masterLabs = coordinates(4:6,:)';
    masterCopyRGB = [];
    masterRGB = roo2rgb(spectra','srgb',wlRange);
    %setto il seed 
    rng(11)
    for i=1:numcopies
        low = (i-1)*numSamples+1; 
        upp = numSamples*i;
        
        %creo il rumore
        copySpectra = spectra;
        noise1 = random('unif',1.01,1.15);
        noise2 = random('unif',1.01,1.13);

        copySpectra(1:300,:) = copySpectra(1:300,:)*noise1;
        copySpectra(301:421,:) = copySpectra(301:421,:)*noise2;
                
        %insert into input set
        masterCopyPairs(:,low:upp) = [spectra; copySpectra]; 
        %calcolo le coordinate RGB per la copia 
        copyRGB = roo2rgb(copySpectra','srgb',wlRange);
        %converto lo spettro in coordinate lab
        copyLabs = roo2lab(copySpectra',[],wlRange); 
        %calcolo la differenza tra master e copia
        diff = de(masterLabs,copyLabs); 
        
        %trasformo C in una percentuale in modo da standardizzare tutti
        %valori lungo L (gli slice hanno diverso raggio)
        %converto le coordinate lab in lch
        lch = lab2lch(masterLabs); 
        
        %applico l'equazione dell'ellisse per trovare la relazione tra c e
        %l, potremmo poi calcolare la percentiale
        maxcs = cmax*sqrt(1-((lch(:,1)-50)/50).^2); 
        %sostituisco il valore precendento di C con la percentuale
        lch(:,2) = 100*lch(:,2)./maxcs; 
        
        %creo il dataset per la fuzzy, composta dalla differenza non
        %corretta e gli lch
        fuzzyInputs(:,low:upp) = ([lch diff])'; 
        %differenza non corretta
        DE(:,low:upp) = diff; 
        %matrice contente le coordinate RGB
        masterCopyRGB = [masterCopyRGB; masterRGB copyRGB];            
    end
end


function ret = extractFeatures(matrix,k)
    fprintf("2) FEATURES EXTRACTION \n");
    %numero delle features totali 
    N = 421*2;
    %numero di sub set che voglio craare
    P = k;
    sizeM = size(matrix);
    numColors = sizeM(2)
    for i = 1:numColors
        arr = [];
        X =  matrix(:,i);                       %selezziono la riga nella matrice degli spetri (scelgo il colore)
        r = diff(fix(linspace(0, N, P+1)));     %calcolo quanti elementi devono avere ogni sub set
        C = mat2cell(X, r, 1) ;                 %creo i sub set del vettore delle features
        for j = 1:P
            meanValue = mean(C{j});
            %varianza = var(C{j});
            massimo = max(C{j});
            arr = [arr meanValue massimo];        
        end
        if i == 1
            featuresExtracted = arr;
        else
            featuresExtracted = [featuresExtracted; arr];
        end
    end
    featuresExtracted = zscore(featuresExtracted);
    ret = featuresExtracted';
end

%nn per la features selection
function mse = fitFs(inputs,targets)
    inputs = inputs';
    targets = targets';
    net = fitnet(4);
    net.trainParam.showWindow=0;
    %suddivisione degli insiemi: train, validation e test
    net.divideParam.trainRatio = 75/100;
    net.divideParam.valRatio = 15/100;
    net.divideParam.testRatio = 15/100;
    [net,trainingInfo] = train(net,inputs,targets);
    trIndex = trainingInfo.testInd;
    outputs = net(inputs(:,trIndex));
    mse = immse(outputs,targets(trIndex));
end

%Neural network
function neuralNetwork(masterCopyPairSelected,DE)
    fprintf("4) CREAZIONE DELLA NN \n");
    
    x = masterCopyPairSelected;
    t = DE;
    
    trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.

    % Create a Fitting Network
    hiddenLayerSize = 2;
    net = fitnet(hiddenLayerSize,trainFcn);

    % Setup Division of Data for Training, Validation, Testing
    net.divideParam.trainRatio = 70/100;
    net.divideParam.valRatio = 15/100;
    net.divideParam.testRatio = 15/100;

    % Train the Network
    [net,tr] = train(net,x,t);

    % Test the Network
    y = net(x);
    e = gsubtract(t,y);
    performance = perform(net,t,y)

    % View the Network
    view(net)

    % Plots
    % Uncomment these lines to enable various plots.
    %figure, plotperform(tr)
    %figure, plottrainstate(tr)
    %figure, ploterrhist(e)
    figure, plotregression(t,y)
    %figure, plotfit(net,x,t)
end

%plot del master con le copie
function showColors(masterCopyRGB,DE,correctDE)
    coordinateMaster = masterCopyRGB(1:3,:)';
    coordinateCopy = masterCopyRGB(4:6,:)';
    
    min = 100;
    colore = 0;
%     for i=1:12690
%         if (DE(i) < min) && (DE(i) > 1.32)
%             min = DE(i);
%             colore = i;
%             temp = i;
%         end
%     end
    
    temp = 3
    %DE(5435);
    DE(temp)
    colore = 935;
        
    x1 = [0 1 1 0];
    y1 = [0 0 1 1];
    patch(y1,x1,coordinateMaster(colore,:),'LineStyle','none');
    x1 = [0 1 1 0];
    y1 = [1 1 2 2];
    patch(y1,x1,coordinateCopy(colore+1269*1,:),'LineStyle','none');
    
    fprintf("Difefrenza con la %d copia: %d - %d\n",1,DE(colore),correctDE(colore));
    
    for k=1:9
        x1 = [k k k k];
        y1 = [0 0 2 2];
        patch(y1,x1,coordinateCopy(colore+1269*k,:));
        
        x1 = [k k+1 k+1 k];
        y1 = [0 0 1 1];
        patch(y1,x1,coordinateMaster(colore,:),'LineStyle','none');
        x1 = [k k+1 k+1 k];
        y1 = [1 1 2 2];
        patch(y1,x1,coordinateCopy(colore+1269*k,:),'LineStyle','none');
        
         fprintf("Difefrenza con la %d copia: %d - %d\n",k+1,DE(colore+1269*k),correctDE(colore+1269*k));
    end
  
end

%plot delle membership function utilizzate
function plotFuzzyMF(fis)
    fis = readfis('fuzzysystem_V12.fis');
    %plotmf(fis,'input',1); 
    %saveas(gca,'./plot/mf_L.png');
    %plotmf(fis,'input',2); 
    %saveas(gca,'./plot/mf_c.png');
    plotmf(fis,'input',1); 
    %saveas(gca,'./plot/mf_h.png');
    %plotmf(fis,'input',4); 
    %saveas(gca,'./plot/mf_de.png');
end

%plot del rumore applicato al segnale
function plotNoise(masterCopyPairs)
    n=1;
    master = masterCopyPairs(1:421,:);
    copy = masterCopyPairs(422:842,:);

    figure;
    ax1 = subplot(2,1,1); 
    plot(ax1,master(:,n),'r')
    title(ax1,'Master')
    xlabel(ax1,'wavelength (nm)')
    ylabel(ax1,'% of reflection')

    ax2 = subplot(2,1,2);
    plot(ax2,copy(:,n),'b')
    title(ax2,'Copy')
    xlabel(ax2,'wavelength (nm)')
    ylabel(ax2,'% of reflection')
    print('-clipboard','-dmeta')

    figure 
    plot(copy(:,n),'r');hold on;
    plot(master(:,n),'b');             
    title('Comparison')
    legend('master','copy')
    xlabel('wavelength (nm)')
    ylabel('% of reflection')
end