tic;
%% training set
Ratio = 0.8;%trainning ratio
renlianku_root = '»À¡≥ø‚';
renlianku_list = dir(renlianku_root);
train_set_root = '—µ¡∑ºØ';
train_set_list = dir(train_set_root);
testing_set_root = '≤‚ ‘ºØ';
testing_set_list = dir(testing_set_root);
for i=3:length(renlianku_list);
    imlist = dir(fullfile(renlianku_root,renlianku_list(i).name));
    imlist_train = dir(fullfile(train_set_root,train_set_list(i).name));
    imlist_test = dir(fullfile(testing_set_root,testing_set_list(i).name));
     for j=3:ceil(length(imlist))
        delete(strcat(fullfile(train_set_root,train_set_list(i).name,imlist(j).name)));
        delete(strcat(fullfile(testing_set_root,testing_set_list(i).name,imlist(j).name)));
     end
end
for i=3:length(renlianku_list);
    imlist = dir(fullfile(renlianku_root,renlianku_list(i).name));
    imlist_train = dir(fullfile(train_set_root,train_set_list(i).name));
    for j=3:ceil(Ratio*length(imlist))
        im = imread(fullfile(renlianku_root,renlianku_list(i).name,imlist(j).name));
%         if(length(size(im))) == 2
%             im1(:,:,1) = im;
%             im1(:,:,2) = im;
%             im1(:,:,3) = im;
%             im = im1;
%         end
%         if imlist(j).name(1)=='0'||imlist(j).name(1)=='1'||imlist(j).name(1)=='2'||imlist(j).name(1)=='3'
%             delete(strcat(fullfile(renlianku_root,renlianku_list(i).name,imlist(j).name)));
%         end
%         Resize_path = [fullfile(renlianku_root,renlianku_list(i).name,imlist(j).name)];
%         im = imresize(im,[Pixel,Pixel]);
%         imwrite(im,Resize_path)        
        trainning_set_path = [fullfile(train_set_root,train_set_list(i).name,imlist(j).name)];
        imwrite(im,trainning_set_path)
    end
    for j = ceil(Ratio*length(imlist))+1:length(imlist)
        im = imread(fullfile(renlianku_root,renlianku_list(i).name,imlist(j).name));        
        testing_set_path = [fullfile(testing_set_root,testing_set_list(i).name,imlist(j).name)];
        imwrite(im,testing_set_path)
    end
    
end
distinguish_cost_time = toc
