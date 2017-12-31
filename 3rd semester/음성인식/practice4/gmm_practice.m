close all;

%data loading
load('point.mat');

%% using the result of kmeans for initializing
%the number of clustering 
k=3 ;
iteration=50;

% Random permutation of data
[dim, number]= size(point);
random = randperm(number);

% assigning arbitary centers
c= zeros(dim,k);
for i = 1:k 
  c(:,i) = point(:,random(i));
end

% index function
index_n = zeros(dim,number);

for j = 1:iteration
  for n = 1: number
     min_index=1;
     minValue = abs(c(:,min_index)-point(:,n));
      
      for m =1:k
         distance= abs(c(:,m)-point(:,n));
         if distance < minValue
             min_index = m;
             minValue  = distance;             
         end    
     end
      
     index_n(:,n) = min_index;       
  end            
  %re-computation of centers
      for q= 1:k
         c(:,q) = sum(point(:, find(index_n == q))); 
         c(:,q) = c(:,q)/ length(find(index_n == q)); 
      end    
end

% 
% nbins=300;
% % h1= histogram(point(:, find(index_n == 1)),nbins,'LineStyle','none');
% % hold on;
% % h2= histogram(point(:, find(index_n == 2)),nbins,'LineStyle','none');
% % hold on;
% % h3= histogram(point(:, find(index_n == 3)),nbins,'LineStyle','none');
% % hold on;
% % h4= histogram(point(:, find(index_n == 4)),nbins,'LineStyle','none');


%% %initialization

%initial average & variance
means= zeros(dim,k);
variance=zeros(dim,k); 
mixture_weight = zeros(dim,k);

for i= 1:k 
 
temp_mean = mean(point(:, find(index_n == i)));
temp_variance = var(point(:, find(index_n == i)));
temp_weight = numel(point(:, find(index_n == i)));

means(dim, i) = temp_mean;
variance(dim, i) = temp_variance;
mixture_weight(dim, i) = temp_weight./numel(point)
 
end

%% iteration
score=zeros(k, numel(point));

 for t=1:101

%% expectation process

 for i = 1: numel(point)
    x = point(dim, i);
    partial_sum = 0;
    for j = 1 : k
        y = mvnpdf(x, means(:, j), variance(:, j));
        partial_sum = partial_sum + y * mixture_weight(:, j);
    end
    
    for h = 1 : k
        y_nominator = mvnpdf(x, means(:, h), variance(:, h));
        y_nominator_with_weight = y_nominator.*mixture_weight(:, h);
        
        score(h, i) = y_nominator_with_weight/partial_sum;
    end
 end
 
     
 
 
 
 
 
 
 
 
%%  maxmization process

 



 %%%%%%%%%%%%Fix me!!!!!
 






 end        %iteration end
 
 
 %% visualize 
 
 %sorting for visualize
 point2= sort(point);
% visualizing depending on the value of k 
if k == 1
   p1 = mixture_weight(:,1) * normpdf(point2 , means(:,1), sqrt(variance(:,1)));
   p_final=p1;
   figure(2), plot(point2,p_final);   
elseif k==2
   p1 = mixture_weight(:,1) * normpdf(point2 , means(:,1), sqrt(variance(:,1)));
   p2 = mixture_weight(:,2) * normpdf(point2, means(:,2), sqrt(variance(:,2)));  
   p_final=p1+p2;
   figure(2), plot(point2,p_final); 
elseif k==3
   p1 = mixture_weight(:,1) * normpdf(point2 , means(:,1), sqrt(variance(:,1)));
   p2 = mixture_weight(:,2) * normpdf(point2, means(:,2), sqrt(variance(:,2))); 
   p3 = mixture_weight(:,3) * normpdf(point2, means(:,3), sqrt(variance(:,3)));
   p_final=p1+p2+p3;
   figure(2), plot(point2,p_final);
    
elseif k==4
    p1 = mixture_weight(:,1) * normpdf(point2 , means(:,1), sqrt(variance(:,1)));
    p2 = mixture_weight(:,2) * normpdf(point2, means(:,2), sqrt(variance(:,2))); 
    p3 = mixture_weight(:,3) * normpdf(point2, means(:,3), sqrt(variance(:,3)));
    p4 = mixture_weight(:,4) * normpdf(point2, means(:,4), sqrt(variance(:,4)));
  
   p_final=p1+p2+p3+p4;
   figure(2), plot(point2,p_final);
 
end
 
 
 
 
 
 