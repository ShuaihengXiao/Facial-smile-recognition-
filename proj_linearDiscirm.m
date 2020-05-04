% linear discriminant function
% design phi function y=[1,x1,x2,....,x1+x2+...]
one=ones(1,10);
V1_sum=sum(V1,1);
V2_sum=sum(V2,1);
y1=[one;V1;V1_sum];
y2=-[one;V2;V2_sum]; %design y
y_all=[y1 y2];
%    initialization
next_a=sum(y_all,2);
theta=1;
step_size=1;
maxIters=10000;
%    main steps
for i=1:maxIters
    curr_a=next_a;
    mis_set=[];
    result=curr_a'*y_all;
    for ii=1:20
        if result(ii)<0  % if correctly classify, remove it from mis_set(misclassify_Set)
           mis_set=[mis_set y_all(:,ii)];
        end
    end 
    if size(mis_set)==[0 0] % test if mis_set is empty
        break
    end
    next_a=curr_a+(step_size.*sum(mis_set,2));
    if sqrt(abs(curr_a'*mis_set-next_a'*mis_set))<theta
        break
    end
end
% test 
test1=imread('/Users/xiaoshuaiheng/Desktop/CPE 646 pattern recognition /final proj/test_nonsmile/469.jpg');
test1=imresize(test1,[50,50]);
test1=rgb2gray(test1);
test1=reshape(test1,[],1); 
test1=double(test1);
test1=[1;test1;sum(test1,1)];
test_result1=curr_a'*test1
