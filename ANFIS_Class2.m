
close all
clear all
clc

%Para alterar o número de Regras basta alterar 
%o valor da variável K (número de clusters do
%fuzzy kmeans) 

%dados = load('gasfurnace.txt');
load('dataset_2d.mat')

%define number of clusters e de REGRAS
K = 4;
maxEpocas = 100;
tol = 1e-7; 
%Define a taxa de aprendizado
alpha = 0.01;


idx = randperm(size(x,1));
X1 = x(idx,:);
Y1 = y(idx,:);
Y1(Y1==0) = -1;

idxTreino = [ones(70,1);zeros(30,1)];
idxTest = [zeros(70,1);ones(30,1)];

trIdx = logical(idxTreino);
teIdx = logical(idxTest);

X_train = X1(trIdx,:);
Y_train = Y1(trIdx,:);

X_test = X1(teIdx,:);
Y_test = Y1(teIdx,:);


ErroVal = [];
ErroTreino = [];

X = X_train;
Y = Y_train;

x1_min = min(X_train(:,1)) - 0.5;
x1_max = max(X_train(:,1)) + 0.5;

x2_min = min(X_train(:,2)) - 0.5;
x2_max = max(X_train(:,2)) + 0.5;

x1_plot = x1_min:0.1:x1_max;
x2_plot = x2_min:0.1:x2_max;

n_x1plot = length(x1_plot);
n_x2plot = length(x2_plot);
    
% Fuzzy K-Means Algorithm


%Inicialização da matriz U.
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

%Inicialização dos centroids
m = d;
Centroids = zeros(K,d); 
oldIdx = idx;
iter = 1;

%Start Looping

while(1)
 
    for i=1:K    
        indexes            = find(idx==i);
        Cusj               = X(indexes,:);
        Centroids(i,:) = ((U(:,i).^m)' * X)./sum(U(:,i).^m);    
    end
    
    %Cálculo das distancias entre pontos e centros
    for i=1:K
        for j=1:n            
            d(j,i) = 1 / norm(X(j,:) - Centroids(i,:));            
        end
    end
    
    %atualização da matriz U
    for i=1:K
        for j=1:n            
            U(j,i) = d(j,i).^2 / sum(d(j,:).^2);            
        end
    end    
    
    %Atualiza os indices dos pontos pertencentes a cada grupo    
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

%Inicializa os parâmetros
nRegras = K; %Numero de regras

%m = nRegras
m = nRegras;

%n = nObs e d = nDimensões
[n,d] = size(X);

%Convenções
%j = colunas de X,(dimensão de X);
%i = linhas de X, (número de observações);
%k = número de regras do sistema de inferência fuzzy

%c - Centro da Gaussiana, dimensões j x k (mxd).
%Sigma - Dispersão da Gaussiana, dimensões j x k.
%p     - Coeficientes da saída Z, dimensões j x k (mxd).
%q     - variável independente da saída Z, dimensões j x k (mxd).

%Inicialização dos Parâmetros


%c = unifrnd(Xmin,Xmax,d,m);
c = Centroids';

Sigma  = unifrnd(0.5,1,d,m);

p = unifrnd(-1,1,d,m);

q = unifrnd(-1,1,1,m);

mu = zeros(d,m);
w  = zeros(1,m);
z  = zeros(1,m);

%Inicializa derivadas parciais
 DJ_dyest = 0;
 Dyest_dw = zeros(size(w));
 Dw_dmu   = zeros(size(mu));
 Dmu_dc   = zeros(size(c));
 Dmu_ds   = zeros(size(Sigma));
 Dz_dq    = zeros(size(q));
 Dz_dp    = zeros(d,1);
 Dyest_dz = zeros(size(z));
 
 %Define parâmetros de trienamento
 erroEpoca = [];
 erro = 0;
 
 xrange1 = (Xmin(1)-0.5):0.01:(Xmax(1)+0.5);
 xrange2 = (Xmin(2)-0.5):0.01:(Xmax(2)+0.5);
 gauss1 = zeros(size(Sigma,2),size(xrange1,2));
 gauss2 = zeros(size(Sigma,2),size(xrange2,2));
 
 figure('units','normalized','outerposition',[0 0 1 1])

 for g = 1:maxEpocas 
    for i = 1:n         
        
      %Calcula a saída y_estimado     
      %Calcula mu (pertinencias)
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
      
      %calcula o w (ativação da regra)
      for k=1:m          
          w(k) = prod(mu(:,k));           
      end
      
      %Calcula os z's (saida de cada regra)      
          for k=1:m               
              z(k) = sum(X(i,:) .* p(:,k)')+q(k);   
          end
      
      %Calcula a saída estimada      
      y_estimado(i) = (sum(w.*z)/sum(w));
            
      %Calcula o custo
      J(i) = 0.5*((Y(i)- y_estimado(i))^2);
      
      
      %Calcula a estimativa dos gradientes
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
          Dyest_dz(k) = (w(k) / sum(w));        
      end      
      
      %Cálculo das derivadas parciais de J em função dos parâmetros
      DJdc = DJ_dyest .* repmat(Dyest_dw,d,1) .* Dw_dmu .* Dmu_dc ;
      DJds = DJ_dyest .* repmat(Dyest_dw,d,1) .* Dw_dmu .* Dmu_ds ;
      DJdp = DJ_dyest .* repmat(Dyest_dz,d,1) .* repmat(Dz_dp,1,m);
      DJdq = DJ_dyest .* Dyest_dz  .* Dz_dq ;
      
      %atualiza os parâmetros
      c     = c - (alpha .* DJdc);
      Sigma = Sigma - (alpha .* DJds);
      p     = p - (alpha .* DJdp);
      q     = q - (alpha .* DJdq); 
      
      %calcula a função de custo      
      erro = erro + J(i);  
       
    end
    %g
    %mean(J)
    erroEpoca = [erroEpoca; sqrt(mean(J))];
    
    %Verificar se a tolerância de erro mínimo foi atendida
    if(erroEpoca(g) < tol)
        break;
    end
    
    
    %Plota a superfície de decisão
for i=1:n_x1plot
    for j=1:n_x2plot
        
        x_plot = [x1_plot(i), x2_plot(j)];
        
         %Calcula a saída y_estimado     
      %Calcula mu
      z  = zeros(1,m);
      
      for jj=1:d
          for k=1:m              
              mu(jj,k) = exp(-0.5 * (((x_plot(jj) - c(jj,k))^2)/(Sigma(jj,k)^2)));              
          end
      end
      
      %calcula o w
      for k=1:m          
          w(k) = prod(mu(:,k));          
      end
      
      %Calcula os z's
      
      for k=1:m 
          for jj = 1:d
              z(k) = z(k) + x_plot(jj)* p(jj,k);              
          end
          z(k) = z(k)+q(1,k);
      end
      
      %Calcula a saída estimada      
      zplot(i,j) = double((sum(w*z')/sum(w))>=0);
      if(zplot(i,j) == 0 )
          zplot(i,j) = -1;
      end        
    end
end

%figure
subplot(2,2,1)
hplot = mesh(x1_plot, x2_plot, zplot', 'FaceAlpha',1);
 title('Classification Surface')

%view(15,90)%view de cima
%view(-45,85) %view superficie
view(-25,55)
hold on;
plot3(X_train(Y_train==-1,1),X_train(Y_train==-1,2), 0*ones(length(X_train(Y_train==-1,2))), 'ro', 'MarkerFaceColor','r');
plot3(X_train(Y_train==1,1),X_train(Y_train==1,2), 0*ones(length(X_train(Y_train==1,2))), 'bo', 'MarkerFaceColor','b')
subplot(2,2,2)
hplot2 = plot(erroEpoca,'bo-');
 title('Error per epoch')


for ii = 1:size(Sigma,2)    
   gauss1(ii,:) = gaussmf(xrange1, [Sigma(1,ii), c(1,ii)]);
   gauss2(ii,:) = gaussmf(xrange2, [Sigma(2,ii), c(2,ii)]);
end
subplot(2,2,3)
 hplot3 = plot(xrange1,gauss1);
 title('MFs x1')
 subplot(2,2,4)
 hplot4 = plot(xrange2,gauss2);
 title('MFs x2')

drawnow;
delete(hplot);
delete(hplot2);
delete(hplot3);
delete(hplot4);



 end
 
%Erro de teste

for i = 1:length(X_test)    
      %Calcula a saída y_estimado     
      %Calcula mu
      z  = zeros(1,m);
      
      for j=1:d
          for k=1:m              
              mu(j,k) = exp(-0.5 * (((X_test(i,j) - c(j,k))^2)/(Sigma(j,k)^2)));              
          end
      end
      
      %calcula o w
      for k=1:m          
          w(k) = prod(mu(:,k));          
      end
      
      %Calcula os z's
      
      for k=1:m 
          for j = 1:d
              z(k) = z(k) + X_test(i,j)* p(j,k);              
          end
          z(k) = z(k)+q(1,k);
      end
      
      %Calcula a saída estimada      
      y_estimado4(i) = sum(w*z')/sum(w);    
            
end


ErroTeste = sqrt(sum((Y_test - y_estimado4').^2)/length(Y_test));

figure
plot(Y_test,'b', 'LineWidth', 2)
hold on
plot(y_estimado4,'r', 'LineWidth', 2)
xlabel('t')
ylabel('Y')
title('Dados de Teste:Saída Real x Estimada - K = 16 Regras','FontSize', 15)
legend('Saída Real', 'Saída Estimada')
grid on

RMSE_treinoMedio = erroEpoca;

figure
plot(RMSE_treinoMedio,'bo-', 'LineWidth', 2)
xlabel('Época')
ylabel('RMSE médio')
title('RMSE médio x Época - K = 16 Regras','FontSize', 15)
grid on


RMSE_medio_treino = mean(ErroTreino');
RMSE_medio_teste  = ErroTeste;

display('Número de Regras:')
K
display('----------------------------')
display('RMSE_medio_treino:')
RMSE_medio_treino
display('----------------------------')
display('RMSE_medio_teste:')
RMSE_medio_teste




%plota a superfície aprendida

%Plota a superfície de decisão
x1_min = min(X_train(:,1)) - 0.5;
x1_max = max(X_train(:,1)) + 0.5;

x2_min = min(X_train(:,2)) - 0.5;
x2_max = max(X_train(:,2)) + 0.5;

x1_plot = x1_min:0.01:x1_max;
x2_plot = x2_min:0.01:x2_max;

n_x1plot = length(x1_plot);
n_x2plot = length(x2_plot);


for i=1:n_x1plot
    for j=1:n_x2plot
        
        x_plot = [x1_plot(i), x2_plot(j)];
        
         %Calcula a saída y_estimado     
      %Calcula mu
      z  = zeros(1,m);
      
      for jj=1:d
          for k=1:m              
              mu(jj,k) = exp(-0.5 * (((x_plot(jj) - c(jj,k))^2)/(Sigma(jj,k)^2)));              
          end
      end
      
      %calcula o w
      for k=1:m          
          w(k) = prod(mu(:,k));          
      end
      
      %Calcula os z's
      
      for k=1:m 
          for jj = 1:d
              z(k) = z(k) + x_plot(jj)* p(jj,k);              
          end
          z(k) = z(k)+q(1,k);
      end
      
      %Calcula a saída estimada      
      zplot(i,j) = double(sum(w*z')/sum(w) >=0);
      if(zplot(i,j) == 0)
          
          zplot(i,j) = -1;
      end
      
        
    end
end



figure
mesh(x1_plot, x2_plot, zplot')
hold on
plot3(X_train(Y_train==-1,1),X_train(Y_train==-1,2), 0*ones(length(X_train(Y_train==-1,2))), 'ro')
plot3(X_train(Y_train==1,1),X_train(Y_train==1,2), 0*ones(length(X_train(Y_train==1,2))), 'bo')

figure
plot(X_train(Y_train==-1,1),X_train(Y_train==-1,2),'ro')
hold on
plot(X_train(Y_train==1,1),X_train(Y_train==1,2),'bo')
contour(x1_plot, x2_plot,zplot')


for ii = 1:size(Sigma,2)    
   gauss1(ii,:) = gaussmf(xrange1, [Sigma(1,ii), c(1,ii)]);
   gauss2(ii,:) = gaussmf(xrange2, [Sigma(2,ii), c(2,ii)]);
end
figure
 plot(xrange1,gauss1);
 figure
 plot(xrange2,gauss2);
