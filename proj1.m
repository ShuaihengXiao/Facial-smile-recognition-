clear
close all
%load smile and nonsmile images sets and preprocess
srcDir=uigetdir('smile/');
cd(srcDir);
allnames=struct2cell(dir('*.jpg'));
[k,len]=size(allnames);
x=[];
for i=1:len
    name=allnames{1,i};
    a = imread(name);
    a=imresize(a,[50,50]);% resize all images with same pixel (50*50)
    a=rgb2gray(a);
    a=reshape(a,[],1); %2500*1
    x=[x a];
end
srcDir=uigetdir('nonsmile/');
cd(srcDir);
allnames=struct2cell(dir('*.jpg'));
[k,len]=size(allnames);
y=[];
for i=1:len
    name=allnames{1,i};
    a = imread(name);
    a=imresize(a,[50,50]);% resize all images with same pixel (50*50)
    a=rgb2gray(a);
    a=reshape(a,[],1); %2500*1
    y=[y a];
end

% mean image
mu1 = mean(x,2);
mu2 = mean(y,2);
mu_show1 = reshape(mu1,50,50);
mu_show2 = reshape(mu2,50,50);
figure(1);
subplot(2,1,1);
imshow(mat2gray(mu_show1));% mat2gray: convert matrix to grayscale image
title('smile mean face');
subplot(2,1,2);
imshow(mat2gray(mu_show2));
title('nonsmile mean face');

% PCA get eigenface
x=double(x);
y=double(y);
scatterMatrix1=zeros(2500,1);
scatterMatrix2=zeros(2500,1);
for i=1:101
     s1=(x(:,i)-mu1)*(x(:,i)-mu1)';
     s2=(y(:,i)-mu2)*(y(:,i)-mu2)';
     scatterMatrix1=scatterMatrix1+s1;
     scatterMatrix2=scatterMatrix2+s2;
end
[V1,D1]=eig(scatterMatrix1);
[V2,D2]=eig(scatterMatrix2);
D1=D1(:,2491:end); % keep top 10 eigenvalues
D2=D2(:,2491:end);
V1=V1(:,2491:end); % keep top 10 eigenvectors
V2=V2(:,2491:end);
figure(2); % plot 10 eigenface
for i=1:10
    subplot(2,5,i);
    imshow(mat2gray(reshape(V1(:,i),50,50))); % V1 stands for smile eigenfaces
end
figure(3);
for i=1:10
    subplot(2,5,i);
    imshow(mat2gray(reshape(V2(:,i),50,50))); % V2 stands for nonsmile eigenfaces
end

% neural network
data=[V1 V2];
target_matrix=[ones(1,10) -ones(1,10);-ones(1,10) ones(1,10)];
theta=0.06;
step_size=0.00001;
a=1.716;
b=2/3;
Ni=2500; %num of input nodes + 1 bias 
Nh=2500; % num of hidden nodes + 1 bias
No=2;  % num of output nodes
maxIter=10000;
%standardized input data
std_matrix=std(data);
mean_matrix=mean(data,2);
data_=(data-mean_matrix)./std_matrix; % standardize data
data_t=[data_;target_matrix]; % combine data and target matrix
w_ji=-1/sqrt(Ni)+(1/sqrt(Ni)+1/sqrt(Ni)).*rand(2501,2500); % wji 2501,2500
w_kj=-1/sqrt(Nh)+(1/sqrt(Nh)+1/sqrt(Nh)).*rand(2501,2); % wkj 2501,2

for i=1:maxIter
    random=data_t(:,unidrnd(20)); %random choose from data_t set
    x=random(1:2500,:);                     %get random x
    target=random(2501:2502,:);                %get random target
    net_j=w_ji'*[x;1];          % add the bias x=1 at the bottom  (2500*1)
    y_j=a.*tanh(b.*net_j);     % y_j=f(net_j) (2500*1)
    net_k=w_kj'*[y_j;1];        % add the bias y=1 at the bottom (2*1)
    z=a.*tanh(b.*net_k);       % z=f(net_k) (2*1)
    error_=norm(target-z);
    if error_<=theta
        break
    end
    delta_k=(target-z).*(a.*b.*sech(b.*net_k).^2); % delta_k=(t-z)f'(net_k) (2*1)
    delta_j=a*b.*sech(b.*net_j).^2*(sum(w_kj*delta_k)); % delta_j=f'(net_j)* sum(w_kj*delta_k) (2500*1)
    delta_w_kj=step_size.*delta_k'.*[y_j;1];   % delta_w_kj=step_size*delta_k*y_j (2501*2)
    delta_w_ji=step_size.*[x;1]*delta_j';      % delta_w_ji=step_size*x*delta_j  (2501*2500)
    w_ji=w_ji+delta_w_ji;
    w_kj=w_kj+delta_w_kj;     % updata w
    
end
% test smile face
load('opt_parameters.mat');
srcDir=uigetdir('test_smile/');
cd(srcDir);
allnames=struct2cell(dir('*.jpg'));
[k,len]=size(allnames);
test1=[];
for i=1:len
    name=allnames{1,i};
    a = imread(name);
    a=imresize(a,[50,50]);% resize all images with same pixel (50*50)
    a=rgb2gray(a);
    a=reshape(a,[],1); %2500*1
    test1=[test1 a];
end
test1=double(test1);
mean_test1=mean(test1,2);
std_test1=std(test1);
test1_=(test1-mean_test1)./std(test1);
a=1.716;
b=2/3;
class=[];
for i=1:40
    net_j=w_ji'*[test1_(:,i);1];
    y=a.*tanh(b.*net_j).^2;
    net_k=w_kj'*[y;1];
    z=a.*tanh(b.*net_k).^2;
    if z(1,:)>z(2,:)
        class=[class 1];
    else
        class=[class 0];
    end
    class1=sum(class);
end
accuracy=class1/40