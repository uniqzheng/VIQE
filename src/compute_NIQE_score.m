 load modelparameters.mat
 addpath('../include/');
 blocksizerow    = 96;
 blocksizecol    = 96;
 blockrowoverlap = 0;
 blockcoloverlap = 0;

% read 
algo_name = 'NIQE';
data_name = 'LIVE_VQC';
data_path = '../../LIVE_VQC/';
video_tmp = '../tmp';
if ~exist(video_tmp, 'dir'), mkdir(video_tmp); end
feat_path = '../features';
mos_filename = fullfile(feat_path, [data_name,'_metadata.csv']);
filelist = readtable(mos_filename);
num_videos = size(filelist,1);
quality = zeros(1,585);
for i = 1:num_videos
    % get video file name
    strs = strsplit(filelist.File{i}, '.');
    video_name = fullfile(data_path,filelist.File{i});
    fprintf('Computing features for %d sequence: %s\n', i, video_name);
    
    % get video meta data
    width = filelist.width(i);
    height = filelist.height(i);
    framerate = round(filelist.framerate(i));
    
    % decode video and store in video_tmp dir
    yuv_name = fullfile(video_tmp, [strs{1}, '.yuv']);
    cmd = ['ffmpeg -loglevel error -y -r ', num2str(framerate), ...
        ' -i ', video_name, ' -pix_fmt ', filelist.pixfmt{i},  ...
       ' -s ', [num2str(width),'x',num2str(height)], ' -vsync 0 ', yuv_name];
    system(cmd);
    
    % read YUV frame (credit: Dae Yeol Lee)  
    fp_input = fopen(yuv_name, 'r');
    fseek(fp_input, 0, 1);
    file_length = ftell(fp_input);
    nb_frames = floor(file_length/width/height/1.5); % for 8 bit
    uv_width = width/2; 
    uv_height = height/2;
    feats = [];
    sub_quality = [];
%     sub_quality = zeros(1,10);
    % compute frame-level features and average
    for fr = floor(framerate/2):framerate:nb_frames-3  %取每秒中间的图片
        YUV = YUVread(fp_input, [width height], fr, 'yuv420p');
        try
        sub_quality(end+1) = computequality(YUV(:,:,1),blocksizerow,blocksizecol,blockrowoverlap,blockcoloverlap, ...
    mu_prisparam,cov_prisparam);
        catch
            continue;
        end
    end
    fclose(fp_input);
    delete(yuv_name);
    quality(i) = nanmean(sub_quality);
    save('../features/LIVE_VQC_NIQE_quality.mat','quality');
end
save('../features/LIVE_VQC_NIQE_quality.mat','quality');

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
        fseek(f,(frnum-1)*1.5*dim(1)*dim(2), 'bof'); % Frame read for 10 bit
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
