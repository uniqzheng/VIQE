%% SLEEQ by Zheng Qi %%
close all; 
clear;
addpath(genpath('../include'));
 load modelparameters.mat;
%% parameters
algo_name = 'nss34_tp_everyframe';  %haar, db2, bior22, spatial

data_name = 'Youtube-UGC';
data_path = '../../YouTube-UGC/original_videos_h264/';
write_file = true;

video_tmp = '../tmp';
if ~exist(video_tmp, 'dir'), mkdir(video_tmp); end
feat_path = '../features';
mos_filename = fullfile(feat_path, [data_name,'_metadata.csv']);
filelist = readtable(mos_filename);
num_videos = size(filelist,1);
out_mat_tp_name = fullfile(feat_path, [data_name,'_',algo_name,'_tp_feats.mat']);

for idx = 1:num_videos
   temporal_score=[];
    % get video file name
    
%     video_name = fullfile(data_path,[num2str(filelist.Filename(i)),'.mp4']);
%     fprintf('Computing features for %d sequence: %s\n', i, video_name);
%     strs = split(filelist.File{idx},'.');
%     video_name = fullfile(data_path,filelist.File{idx});
%     fprintf('Computing features for %d sequence: %s\n', idx, video_name);
%     % get video meta data
%     width = filelist.width(idx);
%     height = filelist.height(idx);
%     framerate = round(filelist.framerate(idx));
%     
%     %decode video and store in video_tmp dir
%     yuv_name = fullfile(video_tmp, [strs{1}, '.yuv']);
%     cmd = ['ffmpeg -loglevel error -y -r ', num2str(framerate), ...
%         ' -i ', video_name, ' -pix_fmt ', filelist.pixfmt{idx}, ...
%         ' -s ', [num2str(width),'x',num2str(height)], ' -vsync 0 ', yuv_name];
%     system(cmd);
% get video file name
    % num2str(filelist.Filename(i))
    %strs = strsplit(filelist.File{i}, '.');
    video_name = fullfile(data_path,[filelist.filename{idx}, '_crf_10_ss_00_t_20.0.mp4']);
    fprintf('Computing features for %d sequence: %s\n', idx, video_name);
    
    % get video meta data
    resolution = filelist.resolution(idx);
    switch (resolution)
        case {360}
            width = 480; 
            height = 360; 
        case {480}
            width = 640;
            height = 480;
        case {720}
            width = 1280;
            height = 720;
        case {1080}
            width = 1920;
            height = 1080;
        case {2160}
            width = 3840;
            height = 2160;
    end
    %width = filelist.width(i);
    %height = filelist.height(i);
    framerate = 30;
    pixfmt = 'yuv420p';
    
    % decode video and store in video_tmp dir
    yuv_name = fullfile(video_tmp, [filelist.filename{idx}, '.yuv']);
    cmd = ['ffmpeg -loglevel error -y -r ', num2str(framerate), ...
        ' -i ', video_name, ' -pix_fmt ', pixfmt, ...
        ' -s ', [num2str(width),'x',num2str(height)], ' -vsync 0 ', yuv_name];
    system(cmd);
	temporal_score = calc_temporal_NSS34(yuv_name,width, height, framerate, pixfmt);
    delete(yuv_name);
    if write_file
          save(strcat('../features/Youtube-UGC_nss34/',filelist.filename{idx},'.mat'),'temporal_score');
    end
end

if write_file
%     save(out_mat_sp_name, 'feats_mat');
%     save(out_mat_sp_name, 'spatial_score');
    save(strcat('../features/Youtube-UGC_nss34/',filelist.filename{idx},'.mat'),'temporal_score');
end

