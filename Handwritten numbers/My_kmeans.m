function [means,Nmeans,IDX] = My_kmeans(X,K,maxerr)



[Ndata, dims] = size(X);
dist = zeros(1,K);

for i=1:K-1
   means(i,:) = X(i,:);
end
means(K,:) = mean(X(K:Ndata,:));

cmp = 1 + maxerr;
while (cmp > maxerr)
   
   class = zeros(K,dims);
   Nclass = zeros(K,1);
k=1;
   
   for i=1:Ndata
      for j=1:K
        
         dist(j) = norm(X(i,:)-means(j,:))^2;
      end
      
      index_min = find(~(dist-min(dist)));
     
      index_min = index_min(ceil(length(index_min)*rand));
      class(index_min,:) = class(index_min,:) + X(i,:);
t=0;

      while(t<=K)
          if(t==index_min)
              IDX(k)=t;
              k=k+1;
          end
          t=t+1;
      end
           
      Nclass(index_min) = Nclass(index_min) + 1;
      
   end
   for i=1:K
      class(i,:) = class(i,:) / Nclass(i);
   end

   
   cmp = 0;
   for i=1:K
      cmp = norm(class(i,:)-means(i,:)); 
   end

  
   means = class;
end

Nmeans = Nclass;
IDX=IDX';
    