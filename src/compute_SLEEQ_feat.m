close all; 
clear;
addpath(genpath('../include'));

%% parameters
algo_name = 'SLEEQ18_persec';  % or SLEEQ1


data_name = 'LIVE_VQC';
data_path = '../../LIVE_VQC/';
write_file = true;

video_tmp = '../tmp';
if ~exist(video_tmp, 'dir'), mkdir(video_tmp); end
feat_path = '../features';
mos_filename = fullfile(feat_path, [data_name,'_metadata.csv']);
filelist = readtable(mos_filename);
num_videos = size(filelist,1);
out_mat_name = fullfile(feat_path, [data_name,'_',algo_name,'_param_feats.mat']);
feats_mat = zeros(num_videos,18);

blocksizerow=72;
blocksizecol=72;
blockrowoverlap=0;
blockcoloverlap=0;

for i = 1:num_videos
    % get video file name
    strs = split(filelist.File{i},'.');
    video_name = fullfile(data_path,filelist.File{i});
    fprintf('Computing features for %d sequence: %s\n', i, video_name);
    %yuv_name = video_name;
    % get video meta data
    width = filelist.width(i);
    height = filelist.height(i);
    framerate = round(filelist.framerate(i));
    Gaus_sigma = round((width/768)*(height/432)*1.16);
    sh_th=((ceil(width/768)*ceil(height/432)-1)*5+10); 
    %decode video and store in video_tmp dir
    yuv_name = fullfile(video_tmp, [strs{1}, '.yuv']);
    cmd = ['ffmpeg -loglevel error -y -r ', num2str(framerate), ...
        ' -i ', video_name, ' -pix_fmt ', filelist.pixfmt{i}, ...
        ' -s ', [num2str(width),'x',num2str(height)], ' -vsync 0 ', yuv_name];
    system(cmd);
%     % get video meta data   
%     video_name = fullfile(data_path,[filelist.filename{i}, '_crf_10_ss_00_t_20.0.mp4']);
%     %video_name = fullfile(data_path,filelist.File{i});
%     fprintf('Computing features for %d sequence: %s\n', i, video_name);
%     
%     resolution = filelist.resolution(i);
%     switch (resolution)
%         case {360}
%             width = 480; 
%             height = 360; 
%         case {480}
%             width = 640;
%             height = 480;
%         case {720}
%             width = 1280;
%             height = 720;
%         case {1080}
%             width = 1920;
%             height = 1080;
%         case {2160}
%             width = 3840;
%             height = 2160;
%     end
% %   width = filelist.width(i);
% %   height = filelist.height(i);
% %   framerate = round(filelist.framerate(i));
% %   pixfmt = filelist.pixfmt{i};
%     framerate = 30;
%     pixfmt = 'yuv420p';
%     Gaus_sigma = round((width/768)*(height/432)*1.16);
%     switch(width)
%         case(3840)
%             sh_th = ((ceil(1920/768)*ceil(1080/432)-1)*5+10);
%         otherwise
%             sh_th = ((ceil(width/768)*ceil(height/432)-1)*5+10); 
%     end
% %     sh_th=((ceil(width/768)*ceil(height/432)-1)*5+10); 
%     % decode video and store in video_tmp dir
%     yuv_name = fullfile(video_tmp, [filelist.filename{i}, '.yuv']);
%     cmd = ['ffmpeg -loglevel error -y -r ', num2str(framerate), ...
%         ' -i ', video_name, ' -pix_fmt ', pixfmt, ...
%         ' -s ', [num2str(width),'x',num2str(height)], ' -vsync 0 ', yuv_name];
%     system(cmd);


%   compute SLEEQ18 or SLEEQ1
	feats_mat(i,:) = calc_SLEEQ18_Score(yuv_name,width, height, framerate, filelist.pixfmt{i}, ...
	                       Gaus_sigma, blocksizerow, blocksizecol, blockrowoverlap, blockcoloverlap, sh_th);
    delete(yuv_name);
   
    if write_file
        save(out_mat_name, 'feats_mat');
    end
end

if write_file
    save(out_mat_name, 'feats_mat');
end

