function Q_final = calc_SLEEQ18_Score(test_video,width, height, framerate, pixfmt, sigma, blocksizerow, blocksizecol, blockrowoverlap, blockcoloverlap, sh_th)
    Q_final = [];
    % Try to open test_video; if cannot, return
    test_file = fopen(test_video,'r');
    if test_file == -1
        fprintf('Test YUV file not found.');
        Q_final = [];
        return;
    end
    % Open test video file
    fseek(test_file, 0, 1);
    file_length = ftell(test_file);
    % get frame number
%     if strcmp(pixfmt, 'yuv420p')
        nb_frames = floor(file_length/width/height/1.5); % for 8 bit
%     else
%         nb_frames = floor(file_length/width/height/3.0); % for 10 bit
%     end
    fprintf('nb_frames: %d\n',nb_frames);
    itr_scale = 1;
    Q_allframes = zeros(1, nb_frames);
    patch_box = [];
    sharpness_box = [];
% 	for fr = 1:2:nb_frames-1
    for fr = floor(framerate/2):framerate:nb_frames-2
        
        Q_patch = [];
        this_YUV_frame = YUVread(test_file,[width height],fr,'yuv420p');
		next_YUV_frame = YUVread(test_file,[width height],fr+1,'yuv420p');
		org_frame = this_YUV_frame(:,:,1);
        dif_frame = next_YUV_frame(:,:,1);
		dif_frame = dif_frame - org_frame;
		org_blr_frame = imgaussfilt(org_frame,sigma);
		dif_blr_frame = imgaussfilt(dif_frame,sigma);

        [row col]        = size(org_frame);
        block_rownum     = floor(row/blocksizerow);
        block_colnum     = floor(col/blocksizecol);
        
        org_frame          = org_frame(1:block_rownum*blocksizerow, ...
                           1:block_colnum*blocksizecol); 
        dif_frame          = dif_frame(1:block_rownum*blocksizerow, ...
                           1:block_colnum*blocksizecol); 
        org_blr_frame      = org_blr_frame(1:block_rownum*blocksizerow, ...
                           1:block_colnum*blocksizecol);
        dif_blr_frame      = dif_blr_frame(1:block_rownum*blocksizerow, ...
                           1:block_colnum*blocksizecol);  
	
        org_patch_param       = blkproc(org_frame,[blocksizerow/itr_scale blocksizecol/itr_scale], ...
                               [blockrowoverlap/itr_scale blockcoloverlap/itr_scale], ...
                               @computefeature18);
%         org_patch_shape       = org_patch_param(1:2:end,:);
        org_patch_shape       = org_patch_param;
        org_patch_sharpness   = org_patch_param(2:18:end,:);
        dif_patch_param       = blkproc(dif_frame,[blocksizerow/itr_scale blocksizecol/itr_scale], ...
                               [blockrowoverlap/itr_scale blockcoloverlap/itr_scale], ...
                               @computefeature18);
        dif_patch_shape       = dif_patch_param;                  
                               
        org_blr_patch_param   = blkproc(org_blr_frame,[blocksizerow/itr_scale blocksizecol/itr_scale], ...
                               [blockrowoverlap/itr_scale blockcoloverlap/itr_scale], ...
                               @computefeature18);
        org_blr_patch_shape = org_blr_patch_param;
        org_blr_patch_sharpness = org_blr_patch_param(2:18:end,:);

        dif_blr_patch_param   = blkproc(dif_blr_frame,[blocksizerow/itr_scale blocksizecol/itr_scale], ...
                               [blockrowoverlap/itr_scale blockcoloverlap/itr_scale], ...
                               @computefeature18);
        dif_blr_patch_shape   = dif_blr_patch_param;
        
        spatial_delta = abs(org_patch_shape-org_blr_patch_shape);
        temporal_delta = abs(dif_patch_shape-dif_blr_patch_shape);
        averg_dif_patch = blkproc(abs(dif_frame),[blocksizerow/itr_scale blocksizecol/itr_scale], ...
                        [blockrowoverlap/itr_scale blockcoloverlap/itr_scale], ...
                        @computemean);
        flatteneddata = averg_dif_patch(:)';
        mappeddata = mapminmax(flatteneddata,0,1);
        norm_averg_dif_patch = reshape(mappeddata,size(averg_dif_patch));
        matrix_one = ones(block_rownum*18,block_colnum);
        proc_norm_averg_dif_patch = zeros(size(norm_averg_dif_patch,1)*18,size(norm_averg_dif_patch,2));
        for ii = 1:size(norm_averg_dif_patch,1)
            proc_norm_averg_dif_patch((ii-1)*18+1:(ii-1)*18+18,:) = repmat(norm_averg_dif_patch(ii,:),18,1);
        end
        Q_patch = (matrix_one-proc_norm_averg_dif_patch).*spatial_delta + proc_norm_averg_dif_patch.*temporal_delta;
        
        sharpness_delta = abs(org_patch_sharpness-org_blr_patch_sharpness);
        Q_patch = reshape(Q_patch,1,block_rownum*block_colnum*18);
        patch_box(end+1,:) = Q_patch;
        sharpness_delta = reshape(sharpness_delta,1,block_rownum*block_colnum);
        sharpness_box(end+1,:) = sharpness_delta;
	end
    % select patches in the video
    flat_patchbox = reshape(patch_box',1,size(patch_box,1)*size(patch_box,2));
    flat_sharpbox = reshape(sharpness_box',1,size(sharpness_box,1)*size(sharpness_box,2));
    sigTresh = prctile(flat_sharpbox(:),sh_th);
    IX = find(flat_sharpbox(:)>sigTresh);
    proc_flat_patchbox = zeros(size(IX,1),18);
    for ii = 1:size(IX,1)
        proc_flat_patchbox(ii,:) = flat_patchbox(1+(ii-1)*18:18+(ii-1)*18);
    end
    Q_final = nanmean(proc_flat_patchbox);
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