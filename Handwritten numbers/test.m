tic;
fid = fopen('train-images.idx3-ubyte', 'r', 'b');
header = fread(fid, 1, 'int32');
count = fread(fid, 1, 'int32');
h = fread(fid, 1, 'int32');
w = fread(fid, 1, 'int32');
offset=0;
readDigits=60000;
imgs = zeros([h w readDigits]);
for i=1:readDigits
    for y=1:h
        imgs(y,:,i) = fread(fid, w, 'uint8');
    end
end

fod=fopen('train-labels.idx1-ubyte', 'r', 'b');
header1 = fread(fod, 1, 'int32');
count1 = fread(fod, 1, 'int32');

labels=zeros(1,readDigits);
for i=1:readDigits
    
     labels(i) = fread(fod, 1, 'uint8');
end

X=reshape(imgs,h*w,readDigits);
T=X(:,1:60000);
[~,Nmeans,idx]=My_kmeans(T',10,0);

Y=T(:,idx==1);
aY=(mean(Y'))';
aaY=[aY];
figure(1);
imagesc(reshape(aY,28,28));

for j=2:10

Y=T(:,idx==j);
d=size(Y,2);
sq=round(sqrt(d))+1;

figure(j);
aY=(mean(Y'))';
aaY=[aaY,aY];
imagesc(reshape(aY,28,28));
end

join=[5,0,4,8,7,2,9,3,1,6];
join1=[3,0,4,8,7,2,9,5,1,6];
for k=1:10
    freq(k)=sum((idx==k)==1 & (labels'==join(k))==1)/sum(labels'==join(k));
end

frekvencije=[join;freq]
figure(11)
bar(frekvencije(1,:),frekvencije(2,:),0.8,'red')
set(gca,'xlim',[-1 10])
title 'BarChart frekvencija rukom pisanih brojeva za moju impl. kmeans algoritma'
xlabel 'Brojevi 0-9'
ylabel 'Frekvencije'




fed = fopen('t10k-images.idx3-ubyte', 'r', 'b');
header2 = fread(fed, 1, 'int32');
count2 = fread(fed, 1, 'int32');
h2 = fread(fed, 1, 'int32');
w2 = fread(fed, 1, 'int32');
offset2=0;
readDigits=10000;
imgs1 = zeros([h2 w2 readDigits]);
for i=1:readDigits
    for y=1:h2
        imgs1(y,:,i) = fread(fed, w2, 'uint8');
    end
end

fad=fopen('t10k-labels.idx1-ubyte', 'r', 'b');
header3 = fread(fad, 1, 'int32');
count3 = fread(fad, 1, 'int32');

labels1=zeros(1,readDigits);
for i=1:readDigits
    
     labels1(i) = fread(fad, 1, 'uint8');
end

Xx=reshape(imgs1,h2*w2,readDigits);
Te=Xx(:,1:10000);
group=1:10;
group=group';
Class = knnclassify(Te',aaY', group);





for k=1:10
  freq_test(k)=sum((Class==k)==1 & (labels1'==join(k))==1)/sum(labels1'==join(k));
end



frekvencije_test=[join;freq_test]

figure(12)
bar(frekvencije_test(1,:),frekvencije_test(2,:),0.8,'red')
set(gca,'xlim',[-1 10])
title 'BarChart frekvencija klasifikacije rukom pisanih brojeva za alg. k-najbližih susjeda'
xlabel 'Brojevi 0-9'
ylabel 'Frekvencije'
toc;



