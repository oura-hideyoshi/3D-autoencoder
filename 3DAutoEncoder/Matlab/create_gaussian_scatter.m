%% Parameter
% 点を分布させ、ガウシアンフィルタをかけることで球状にさせる。

save_path = "./gaussian_scatter";
vol_size = [128,128,32];
probably = 0.0001; % 球の存在確率
sample_n = 128; % データセット数

mkdir(save_path)

%% create & save dataset
for i = 1 : sample_n
    voxel = rand(vol_size) < probably;
    voxel = cast(voxel, 'double');
    voxel = imgaussfilt3(voxel, 1.0);
    save(strcat(save_path, "\", string(i), ".mat"), 'voxel')
    
end

volshow(voxel)
