function tp = calc_temporal_NSS34(test_video,width, height, framerate, pixfmt)
% Try to open test_video; if cannot, return
    tp = [];
    test_file = fopen(test_video,'r');
    if test_file == -1
        fprintf('Test YUV file not found.');
        tp = [];
        return;
    end
    % Open test video file
    fseek(test_file, 0, 1);
    file_length = ftell(test_file);
    % get frame number
    nb_frames = floor(file_length/width/height/1.5); % for 8 bit
    fprintf('nb_frames: %d\n',nb_frames);
%     for fr = floor(framerate/2):framerate:nb_frames-2
      for fr = 1:nb_frames
        this_YUV_frame = YUVread(test_file,[width height],fr,'yuv420p');
        tp(end+1,:) = computefeature34(this_YUV_frame(:,:,1));
    end
    fclose(test_file);
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