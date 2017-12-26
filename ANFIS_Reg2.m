%ADAPTATIVE NEURO-FUZZY INFERENCE SYSTEM (ANFIS) for Regression
%José Maia Neto - Federal University of Minas Gerais
%30.07.2017

%Description: Implementation of ANFIS using Stochastic Gradient Descent
%for regression problems with instantaneous visualization of the evolution
%of the output, error and membership functions along the epochs. 

close all
clear all
clc

%The parameter K defines the number of clusters of fuzzy c-means algorithm
%and consequently the number of fuzzy rules of ANFIS) 

%define number of clusters (and rules)
K = 10;
%Define learning rate
alpha = 0.0001;
%Define the maximum number of epochs
maxEpocas = 500; 



%dados = load('gasfurnace.txt');
%X1 = dados(:,1);
%Y1 = dados(:,2);
%X1 = [X1;X1];
%Y1 = [Y1;Y1];
X1 = (0:0.05:6*pi)';
Y1 = sin(X1) + 0.3*randn(size(X1));

%X_aug = lagmatrix(X1,[0,1,2,3,4]);
%Y_aug = lagmatrix(Y1,[0,1]);
 
%X_aug2 = [X_aug(5:end,:),Y_aug(5:end,2)];
%X1 = X_aug2;
%X1 = zscore(X1);
%Y1 = Y1(5:end,:);
y1 = Y1;

tam = round(0.7 * size(X1,1));

idxTreino = [ones(tam,1);zeros(size(X1,1)-tam,1)];
idxTest = [zeros(tam,1);ones(size(X1,1)-tam,1)];

trIdx = logical(idxTreino);
teIdx = logical(idxTest);

X_train = X1(trIdx,:);
Y_train = Y1(trIdx,:);

X_test = X1(teIdx,:);
Y_test = Y1(teIdx,:);

y1 = y1(trIdx);

ErroVal = [];
ErroTreino = [];

X = X_train;
Y = Y_train;
    
    
% Fuzzy K-Means Algorithm
%Initialization of the membership matrix.
n = size(X,1);
d = size(X,2);
U = rand(n,K);    
idx = zeros(n,1);
U_bin = zeros(n,K);

for i=1:n
    
    U(i,:)   = U(i,:)/sum(U(i,:));
    [val,cl] = max(U(i,:));
    idx(i)   = cl;
    U_bin(i,idx(i))= 1;

end
initData = [U,idx];
idx_init = idx;

%Initialization of centroids
m = d;
Centroids = zeros(K,d); 
oldIdx = idx;
iter = 1;

display('press Ctrl + C to stop')
%Start Looping
while(1)
 
    for i=1:K    
        indexes            = find(idx==i);
        Cusj               = X(indexes,:);
        Centroids(i,:) = ((U(:,i).^m)' * X)./sum(U(:,i).^m);    
    end
    
    %Calculate distance between points and centroids
    for i=1:K
        for j=1:n            
            d(j,i) = 1 / norm(X(j,:) - Centroids(i,:));            
        end
    end
    
    %update membership matrix
    for i=1:K
        for j=1:n            
            U(j,i) = d(j,i).^2 / sum(d(j,:).^2);            
        end
    end    
    
    %assign each point to the group with it has max membership    
    for i=1:n        
        [val,cl] = max(U(i,:));
        idx(i)   = cl;        
    end
    
    if isequal(idx,oldIdx),
        break;
    else
        oldIdx = idx;
    end;
    
     iter = iter + 1;
    
end

 
Xmax = max(X);
Xmin = min(X);

%Parameter initialization
nRegras = K; %Number of rules

m = nRegras;

%n = nObs e d = nDimensions
[n,d] = size(X);

%Notation
%j = columns of X,(dimension of X);
%i = rows of X, (number of obs);
%k = number of rules in the ANFIS

%c - Center of the Gaussian MFs, dimension j x k (mxd).
%Sigma - Spread of Gaussian MFs, dimension j x k.
%p     - parameters of the consequent Z, dimensões j x k (mxd).
%q     - bias term of the consequent Z, dimensões j x k (mxd).

%Parameter initialization

c = Centroids';
Sigma  = unifrnd(0.5,1,d,m);
p = unifrnd(-1,1,d,m);
q = unifrnd(-1,1,1,m);

mu = zeros(d,m);
w  = zeros(1,m);
z  = zeros(1,m);

%Initialization of the partial derivatives
 DJ_dyest = 0;
 Dyest_dw = zeros(size(w));
 Dw_dmu   = zeros(size(mu));
 Dmu_dc   = zeros(size(c));
 Dmu_ds   = zeros(size(Sigma));
 Dz_dq    = zeros(size(q));
 Dz_dp    = zeros(d,1);
 Dyest_dz = zeros(size(z));
 
 %Define training parameters
 erroEpoca = [];
 erro = 0;
 tol = 1e-7; 
 
 xrange1 = (Xmin(1)-0.5):0.01:(Xmax(1)+0.5); 
 gauss1 = zeros(size(Sigma,2),size(xrange1,2));
 gauss2 = zeros(size(Sigma,2),size(X_train,1));

figure('units','normalized','outerposition',[0 0 1 1])
 for g = 1:maxEpocas 
    for i = 1:n     
      for j=1:d
          for k=1:m              
              mu(j,k) = exp(-0.5*(((X(i,j) - c(j,k))^2)/(Sigma(j,k)^2)));              
          end
      end
      
      for j=1:d
          for k=1:m 
              if mu(j,k) == 0
                 mu(j,k) = 1e-50;
              end               
          end
      end
      
      %calculate  w (rule activation)
      for k=1:m          
          w(k) = prod(mu(:,k));           
      end
      
      %Calculate z's (rule consequent)      
          for k=1:m               
              z(k) = sum(X(i,:) .* p(:,k)')+q(k);   
          end
      
      %Calculate estimated output      
      y_estimado(i) = sum(w.*z)/sum(w);
            
      %Calculate the cost
      J(i) = 0.5*((Y(i)- y_estimado(i))^2);      
      
      %Estimate gradient
      %DJ/Dy_estimado
      DJ_dyest = -(Y(i) - y_estimado(i));
      
      %Dy_estimado/Dw
      for k=1:m
          Dyest_dw(k) = (z(k) - y_estimado(i))/(sum(w)); 
      end     
      
      
      %Dw/Dmu
      for j=1:d
          for k=1:m              
              Dw_dmu(j,k) = w(k)/mu(j,k);                
          end
      end
      
      %Dmu/Dc
      for j=1:d
          for k=1:m              
              Dmu_dc(j,k) = mu(j,k) * ((X(i,j)-c(j,k))/(Sigma(j,k)^2));              
          end
      end
      
      %Dz/Dq 
      Dz_dq = ones(size(q));
      
      %Dz/Dp
      Dz_dp = X(i,:)';        
      
      %Dmu/Dsigma
      for j=1:d
          for k=1:m              
              Dmu_ds(j,k) = mu(j,k) * ((X(i,j)-c(j,k))^2/(Sigma(j,k)^3));   
          end
      end
      
      %Dyest/Dz      
      for k=1:m              
          Dyest_dz(k) = w(k) / sum(w);        
      end            

      %Chain rule
      DJdc = DJ_dyest .* repmat(Dyest_dw,d,1) .* Dw_dmu .* Dmu_dc ;
      DJds = DJ_dyest .* repmat(Dyest_dw,d,1) .* Dw_dmu .* Dmu_ds ;
      DJdp = DJ_dyest .* repmat(Dyest_dz,d,1) .* repmat(Dz_dp,1,m);
      DJdq = DJ_dyest .* Dyest_dz  .* Dz_dq ;
      
      %update parameters
      c     = c - (alpha .* DJdc);
      Sigma = Sigma - (alpha .* DJds);
      p     = p - (alpha .* DJdp);
      q     = q - (alpha .* DJdq); 
      
      erro = erro + J(i); 

    end
    
    erroEpoca = [erroEpoca; sqrt(mean(J))];
    
    
   
    %figure
    subplot(2,2,1)
    hplot4 = plot(X_train,sin(X_train))
    title('True function')
    mu2 = zeros(d,m);
    
    for i = 1:n     
      for j=1:d
          for k=1:m              
              mu2(j,k) = exp(-0.5*(((X_train(i,j) - c(j,k))^2)/(Sigma(j,k)^2)));              
          end
      end
      
      for j=1:d
          for k=1:m 
              if mu2(j,k) == 0
                 mu2(j,k) = 1e-10;
              end               
          end
      end  
      for k=1:m          
          w1(k) = prod(mu2(:,k));           
      end
      for k=1:m               
          z1(k) = sum(X_train(i,:) .* p(:,k)')+q(k);   
      end
      y_plot(i) = sum(w1.*z1)/sum(w1);
    end
    
    
    
    %for ii = 1:size(Sigma,2)    
    %    gauss2(ii,:) = gaussmf(X_train, [Sigma(ii), c(ii)]);         
    %end
    
    
      
      %calculate  w (rule activation)
      
      
      %Calculate z's (rule consequent)
      %for jj=1:size(X_train,1)
          
          %Calculate estimated output      
          
      %end
      
     %figure
    subplot(2,2,2)
    hplot = plot(X_train,y_plot, 'b',  'LineWidth', 2);
    hold on;
    plot(X_train, Y_train, 'ro');
    hold on
    xlabel('t')
    ylabel('y')
    title('Regression')
      
      
    subplot(2,2,3)
    hplot2 = plot(erroEpoca,'bo-');
    xlabel('Epoch')
    ylabel('Error')
    title('Error per Epoch')
    
    for ii = 1:size(Sigma,2)    
        gauss1(ii,:) = gaussmf(xrange1, [Sigma(ii), c(ii)]);   
   end
subplot(2,2,4)
 hplot3 = plot(xrange1,gauss1);
 title('Membership functions')

    drawnow;
    delete(hplot);
    delete(hplot2);
    delete(hplot3);
    delete(hplot4);

 end
 
%test Error

for i = 1:length(X_test)    
      
      z  = zeros(1,m);
      
      for j=1:d
          for k=1:m              
              mu(j,k) = exp(-0.5 * (((X_test(i,j) - c(j,k))^2)/(Sigma(j,k)^2)));              
          end
      end
      
      %calculate w
      for k=1:m          
          w(k) = prod(mu(:,k));          
      end
      
      %Calculate z's
      
      for k=1:m 
          for j = 1:d
              z(k) = z(k) + X_test(i,j)* p(j,k);              
          end
          z(k) = z(k)+q(1,k);
      end
      
      %Calculate estimated output      
      y_estimado4(i) = sum(w*z')/sum(w);
            
end


ErroTeste = sqrt(sum((Y_test - y_estimado4').^2)/length(Y_test));

figure
plot(Y_test,'ro', 'LineWidth', 2)
hold on
plot(y_estimado4,'b', 'LineWidth', 2)
xlabel('t')
ylabel('Y')
title('Test Data:Desired x Estimated - K = 16 Regras','FontSize', 15)
legend('Desired', 'Estimated')
grid on

RMSE_treinoMedio = erroEpoca;

figure
plot(RMSE_treinoMedio,'bo-', 'LineWidth', 2)
xlabel('Epoch')
ylabel('RMSE')
title('RMSE  x Epoch - K = 16 ','FontSize', 15)
grid on


RMSE_medio_treino = mean(erroEpoca');
RMSE_medio_teste  = ErroTeste;

display('Number of Rules:')
K
display('----------------------------')
display('RMSE_average_train:')
RMSE_medio_treino
display('----------------------------')
display('RMSE_average_test:')
RMSE_medio_teste
