clc;
clear all;
close all;
addpath('../include/');

data_name = 'KoNVid';
data_path = '/home/zhengqi/VQA/KoNViD_1k_videos/';
video_tmp = './tmp';
if ~exist(video_tmp, 'dir'), mkdir(video_tmp); end
feat_path = './features';
mos_filename = fullfile(feat_path, [data_name,'_metadata.csv']);
filelist = readtable(mos_filename);
num_videos = size(filelist,1);
for i = 1:num_videos
%     % get video file name
%     video_name = fullfile(data_path,[filelist.filename{i}, '_crf_10_ss_00_t_20.0.mp4']);
%     %video_name = fullfile(data_path,filelist.File{i});
%     fprintf('Computing features for %d sequence: %s\n', i, video_name);
%     
%     % get video meta data
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
%     
%     % decode video and store in video_tmp dir
%     yuv_name = fullfile(video_tmp, [filelist.filename{i}, '.yuv']);
%     cmd = ['ffmpeg -loglevel error -y -r ', num2str(framerate), ...
%         ' -i ', video_name, ' -pix_fmt ', pixfmt, ...
%         ' -s ', [num2str(width),'x',num2str(height)], ' -vsync 0 ', yuv_name];
%     system(cmd);
    % get video file name
    %strs = split(filelist.File{i},'.');
    %video_name = fullfile(data_path,filelist.File{i});
    video_name = fullfile(data_path,[num2str(filelist.Filename(i)),'.mp4']);
    fprintf('Computing features for %d sequence: %s\n', i, video_name);
    width = filelist.width(i);
    height = filelist.height(i);
    framerate = round(filelist.framerate(i));
    pixfmt = filelist.pixfmt{i};
    % decode video and store in video_tmp dir
    yuv_name = fullfile(video_tmp, [num2str(filelist.Filename(i)), '.yuv']);
    cmd = ['ffmpeg -loglevel error -y -r ', num2str(framerate), ...
        ' -i ', video_name, ' -pix_fmt ', pixfmt, ...
        ' -s ', [num2str(width),'x',num2str(height)], ' -vsync 0 ', yuv_name];
    system(cmd);
    test_file = fopen(yuv_name,'r');
    if test_file == -1
        fprintf('Test YUV file not found.');
        return;
    end
    fseek(test_file, 0, 1);
    file_length = ftell(test_file);
    nb_frames = floor(file_length/width/height/1.5); % for 8 bit
    fprintf('nb_frames: %d\n',nb_frames);
    niqe_scores = [];
    LGN_features_level6 =[];
    for fr = 1:1:nb_frames
        this_YUV_frame = YUVread(test_file,[width height],fr,'yuv420p');
        frameGray = this_YUV_frame(:,:,1);
        %imshow(frameGray,[]);
        [y l] = frame_LGN_features(frameGray);
        LGN_features_level6(fr,:) = y{6}(:);
    end
    save(strcat('../features/LGN6_KoNVid_new/',filelist.filename{i},'.mat'),'LGN_features_level6');
    fclose(test_file);
    delete(yuv_name);
     
end
% Read one frame from YUV file
function YUV = YUVread(f,dim,frnum,type)

    % This function reads a frame #frnum (0..n-1) from YUV file into an
    % 3D array with Y, U and V components
    if strcmp(type, 'yuv420p')
        %% Start a file pointer
        fseek(f,(frnum-1)*1.5*dim(1)*dim(2), 'bof'); % Frame read for 8 bit ; bof == -1
        %Read Y-component
        Y=fread(f,dim(1)*dim(2),'uchar');
        % Read U-component
        U=fread(f,dim(1)*dim(2)/4,'uchar');    
		% Read V-component
        V=fread(f,dim(1)*dim(2)/4,'uchar');
    else
        fseek(f,(frnum-1)*3.0*dim(1)*dim(2), 'bof'); % Frame read for 10 bit
		%Read Y-component
        Y=fread(f,dim(1)*dim(2),'uint16');
        % Read U-component
        U=fread(f,dim(1)*dim(2)/4,'uint16');    
		% Read V-component
        V=fread(f,dim(1)*dim(2)/4,'uint16');
    end
    %fseek(f,dim(1)*dim(2)*1.5*frnum,'bof');
    
    % Read Y-component
    if length(Y)<dim(1)*dim(2)
        YUV = [];
        return;
    end
    Y=cast(reshape(Y,dim(1),dim(2)),'double');
	
    % Read U-component
    if length(U)<dim(1)*dim(2)/4
        YUV = [];
        return;
    end
    U=cast(reshape(U,dim(1)/2,dim(2)/2),'double');
    U=imresize(U,2.0);
    
    % Read V-component
    if length(V)<dim(1)*dim(2)/4
        YUV = [];
        return;
    end    
    V=cast(reshape(V,dim(1)/2,dim(2)/2),'double');
    V=imresize(V,2.0);
    
    % Combine Y, U, and V
    YUV(:,:,1)=Y';
    YUV(:,:,2)=U';
    YUV(:,:,3)=V';
end
