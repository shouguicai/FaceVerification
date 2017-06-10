clc
clear all
close all
tic;
Pixel = 182;
renlianku_root = '»À¡≥ø‚';
renlianku_list = dir(renlianku_root);
for i=3:length(renlianku_list);
    i
    imlist = dir(fullfile(renlianku_root,renlianku_list(i).name))
    for j=3:ceil(length(imlist))
        im = imread(fullfile(renlianku_root,renlianku_list(i).name,imlist(j).name));
        if(length(size(im))) == 2
            im1(:,:,1) = im;
            im1(:,:,2) = im;
            im1(:,:,3) = im;
            im = im1;
        end
        if imlist(j).name(1)=='0'||imlist(j).name(1)=='1'||imlist(j).name(1)=='2'||imlist(j).name(1)=='3'
            delete(strcat(fullfile(renlianku_root,renlianku_list(i).name,imlist(j).name)));
        end
        im = imresize(im,[Pixel,Pixel]);
        imwrite(im,fullfile(renlianku_root,renlianku_list(i).name,imlist(j).name))
    end
end
resize_timecost = toc
distinguish
get_labels