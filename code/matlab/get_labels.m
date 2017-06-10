tic
testing_set_root = '测试集';
testing_set_list = dir(testing_set_root);
fid=fopen('labels.txt','r+');   
tline = fgetl(fid);    
pos = ftell(fid);     %获取当前指针位置
fseek(fid,pos,'bof');  %将指针移入开始位置
for i=3:length(testing_set_list)
    imlist_test = dir(fullfile(testing_set_root,testing_set_list(i).name));
    for j=3:(length(imlist_test))        
        im = imread(fullfile(testing_set_root,testing_set_list(i).name,imlist_test(j).name));  
        for m = j:length(imlist_test)
            if (imlist_test(j).name(1)) ~= '0'
                non_zero_j = 1;
            else if (imlist_test(j).name(1) == '0')&&(imlist_test(j).name(2)~='0')
                    non_zero_j = 2;
                else
                    non_zero_j = 3;
                end
            end
            if (imlist_test(m).name(1)) ~= '0'
                non_zero_m = 1;
            else if (imlist_test(m).name(1) == '0')&&(imlist_test(m).name(2)~='0')
                    non_zero_m = 2;
                else
                    non_zero_m = 3;
                end
            end
            fprintf(fid,[testing_set_list(i).name,' ',(imlist_test(j).name(non_zero_j:3)),' ',(imlist_test(m).name(non_zero_m:3)),'\n']); 
        end
    end
    for jj = 3:(length(imlist_test))
        if i+1<=length(testing_set_list)
            for n = i+1:length(testing_set_list)
                imlist_test_next = dir(fullfile(testing_set_root,testing_set_list(n).name));
                for p = 3:length(imlist_test_next)  
                    if (imlist_test(jj).name(1)) ~= '0'
                        non_zero_jj = 1;
                    else if (imlist_test(jj).name(1) == '0')&&(imlist_test(jj).name(2)~='0')
                            non_zero_jj = 2;
                        else
                            non_zero_jj = 3;
                        end
                    end
                    if (imlist_test_next(p).name(1)) ~= '0'
                        non_zero_p = 1;
                    else if (imlist_test_next(p).name(1) == '0')&&(imlist_test_next(p).name(2)~='0')
                            non_zero_p = 2;
                        else
                            non_zero_p = 3;
                        end
                    end
                    fprintf(fid,[testing_set_list(i).name,' ',imlist_test(jj).name(non_zero_jj:3),' ',testing_set_list(n).name,' ',imlist_test_next(p).name(non_zero_p:3),'\n']);
                end
            end
        end
    end
end
fclose(fid)  
get_labels_timecost = toc