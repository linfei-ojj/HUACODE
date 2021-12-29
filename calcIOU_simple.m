clc
clear all
close all


whose_data = 'D:\file_zcl\zcl_UNET1\';
which_data = 'Dataset_BUSI';
zcl_classer = 'ori';


out_file = strcat(zcl_classer,'_img_dataset');
zhe = '3';
zhe_file = strcat(out_file,'_',zhe);

root_path = strcat('D:\file_zcl\zcl_data\',which_data);
local_dir = char(strcat(whose_data,'results\', which_data,'\results\',out_file,'\',zhe_file,'\','predict_result1'));
gt_dir = strcat(root_path,'\', zcl_classer , "_mask");
imlist = dir([local_dir,char(strcat('\*.png'))]);

s = {};
s{1,1} = 'image_id';s{1,2} = 'ACC';s{1,3} ='DICE';s{1,4} = 'IOU';s{1,5} = 'M_IOU';
s{1,6} = 'TP';s{1,7} ='TN';s{1,8} = 'FP';s{1,9} ='FN';
s{1,10} = 'Hausdorff';     
            
for i = 1:length(imlist)
    [~,image_id] = fileparts(imlist(i).name); 
    local = imread(strcat(imlist(i).folder,'\',imlist(i).name));
    if size(local,3)
       local = local(:,:,1);
    end
                
    ground_truth = imread(strcat(gt_dir,'\',imlist(i).name));
    
    if size(ground_truth,3)
        ground_truth = ground_truth(:,:,1);
    end
    
    p = z_caculate(local, ground_truth);
    h = z_hausdorff(local, ground_truth);

    s{i + 1,1} = image_id;
    s{i + 1,2} = p(1);
    s{i + 1,3} = p(2);
    s{i + 1,4} = p(3);
    s{i + 1,5} = p(4);
    s{i + 1,6} = p(5);
    s{i + 1,7} = p(6);
    s{i + 1,8} = p(7);
    s{i + 1,9} = p(8);
    s{i + 1,10} = h;
end           
            
s{length(imlist) + 2,1} = 'ave';
s{length(imlist) + 2,2} = roundn(sum(cell2mat(s(2:length(imlist) + 1,2)))/(length(imlist)),-4);
s{length(imlist) + 2,3} = roundn(sum(cell2mat(s(2:length(imlist) + 1,3)))/(length(imlist)),-4);
s{length(imlist) + 2,4} = roundn(sum(cell2mat(s(2:length(imlist) + 1,4)))/(length(imlist)),-4);
s{length(imlist) + 2,5} = roundn(sum(cell2mat(s(2:length(imlist) + 1,5)))/(length(imlist)),-4);
s{length(imlist) + 2,6} = roundn(sum(cell2mat(s(2:length(imlist) + 1,6)))/(length(imlist)),-4);
s{length(imlist) + 2,7} = roundn(sum(cell2mat(s(2:length(imlist) + 1,7)))/(length(imlist)),-4);
s{length(imlist) + 2,8} = roundn(sum(cell2mat(s(2:length(imlist) + 1,8)))/(length(imlist)),-4);
s{length(imlist) + 2,9} = roundn(sum(cell2mat(s(2:length(imlist) + 1,9)))/(length(imlist)),-4);
s{length(imlist) + 2,10} = roundn(sum(cell2mat(s(2:length(imlist) + 1,10)))/(length(imlist)),-4);

s{length(imlist) + 3,1} = 'std';
s{length(imlist) + 3,2} = roundn(std(cell2mat(s(2:length(imlist) + 1 ,2))),-4);
s{length(imlist) + 3,3} = roundn(std(cell2mat(s(2:length(imlist) + 1 ,3))),-4);
s{length(imlist) + 3,4} = roundn(std(cell2mat(s(2:length(imlist) + 1 ,4))),-4);
s{length(imlist) + 3,5} = roundn(std(cell2mat(s(2:length(imlist) + 1 ,5))),-4);
s{length(imlist) + 3,6} = roundn(std(cell2mat(s(2:length(imlist) + 1 ,6))),-4);
s{length(imlist) + 3,7} = roundn(std(cell2mat(s(2:length(imlist) + 1 ,7))),-4);
s{length(imlist) + 3,8} = roundn(std(cell2mat(s(2:length(imlist) + 1,8))),-4);
s{length(imlist) + 3,9} = roundn(std(cell2mat(s(2:length(imlist) + 1 ,9))),-4);
s{length(imlist) + 3,10} = roundn(std(cell2mat(s(2:length(imlist) + 1 ,10))),-4);

save_dir = strcat(local_dir , "\value",".xls");
xlswrite(save_dir, s);


function x = z_caculate(l_pic, gt_pic)
    TP = 0;
    TN = 0;
    FP = 0;
    FN = 0;
    
    l_pic(find(l_pic > 127)) = 255;l_pic(find(l_pic < 128)) = 0;
    if max(gt_pic , [] , 'all') == 1
        gt_pic = gt_pic .* 255;
    end
    f_l_pic = uint8(ones(size(l_pic))) * 255 - l_pic;
    f_gt_pic = uint8(ones(size(l_pic))) * 255 - gt_pic;    
    s_TP = l_pic .* gt_pic;TP = sum(sum(s_TP./255));
    s_FN = f_l_pic .* gt_pic;FN = sum(sum(s_FN./255));
    s_FP = l_pic .* f_gt_pic;FP = sum(sum(s_FP./255));
    s_TN = f_l_pic .* f_gt_pic;TN = sum(sum(s_TN./255));

    if TP == 0
        ACC = (TP + TN)/(TP + TN + FN + FP);
        DICE = 0;
        ONE_IOU = 0;  
        ZERO_IOU = TN/(FP + FN + TN);
        M_IOU = (ONE_IOU + ZERO_IOU)/2;
    else
        ACC = (TP + TN)/(TP + TN + FN + FP);
        DICE = (2 * TP) / (2 * TP + FP + FN);

        ONE_IOU = TP/(TP + FP + FN);
        ZERO_IOU = TN/(FP + FN + TN);
        
        M_IOU = (ONE_IOU + ZERO_IOU)/2;
    end

    
    x = [ACC, DICE,ONE_IOU,M_IOU,TP,TN,FP,FN];
end


function h = z_hausdorff(l_pic, gt_pic)

gt_pic_bd = z_cal_coordinate(gt_pic);

if sum(l_pic,'all') == 0
    h = sqrt(size(l_pic,1).^2 + size(l_pic,2).^2);    
else
    l_pic_bd = z_cal_coordinate(l_pic);
    d = zeros(length(l_pic_bd),1);
    for i=1:length(l_pic_bd)
        t =  gt_pic_bd-l_pic_bd(i,:);
        d(i) = min(sqrt(t(:,1).^2+t(:,2).^2));    
    end
    hab = max(d);

    d = zeros(length(gt_pic_bd),1);
    for i=1:length(gt_pic_bd)
        t = l_pic_bd - gt_pic_bd(i,:);
        d(i) = min(sqrt(t(:,1).^2+t(:,2).^2));    
    end

    hba = max(d);
    h = max([hab hba]);
end
end

function bd = z_cal_coordinate(pic)
pic1 = edge(pic,'sobel');
[b,~] = bwboundaries(pic1,'noholes');

bd = [];
for i = 1:length(b)
    b1 = b{i}; 
    bd = [bd;b1];
end
bd(:,[1,2]) = fliplr(bd(:,[1,2]));
end
