clear;
addpath('./Spatial_Quality/');
data_name = 'Youtube-UGC';
% data_path = '../../database/LIVE_VQC/VideoDatabase/';
video_tmp = './tmp';
if ~exist(video_tmp, 'dir'), mkdir(video_tmp); end
feat_path = '../features';
mos_filename = fullfile(feat_path, [data_name,'_metadata.csv']);
filelist = readtable(mos_filename);
num_videos = size(filelist,1);
feature_RMSE11 = [];
feature_RMSE22 = [];
Mos_Scores = filelist.MOSFull(:);
for idx = 1:num_videos
    %strs = split(filelist.File{idx},'.');
    name_folder = strcat(filelist.filename{idx},'.mat');
    fprintf('process NO. %d video %s\n',idx,filelist.filename{idx});

    name_tp = strcat('./LGN6_youtube_new/',name_folder);
    load(name_tp);
    framerate = 30;
%     name_sp = strcat('./NIQE_youtube_new/',name_folder);
%     load(name_sp);

    % Temporal feature process
    % 
    %---------------------------------------------------------------------------------
    LGN_features_level6(isnan(LGN_features_level6)|isinf(LGN_features_level6))=0;
    TrainData = LGN_features_level6';
    TrainMean = mean(TrainData,2); % Total mean of the training set
    n = size(LGN_features_level6,2);
    TotalTrainSamples = size(LGN_features_level6,1);
    Gt=zeros([ n n]);
    for i=1:TotalTrainSamples
        Temp = TrainData(:,i)- TrainMean;
        Gt = Gt + Temp'*Temp;
    end
    Gt=Gt/TotalTrainSamples; 

    [EigVect1,EigVal1]=eig_decomp(Gt);
    EigVect=EigVect1(:,1:10); 
    Ytrain = [];
    for i=1:TotalTrainSamples
        Ytrain(:,i)=(TrainData(:,i)'*EigVect)';
    end
    LGN6_features =Ytrain';
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % last1,2,3 for every frame
    C_mse11 = [];
    Ytrain11 =LGN6_features;
    %for oo = floor(framerate/2):framerate:size(Ytrain11,1)
    for oo = 1:5:size(Ytrain11,1)-3
    try
        xx11 = Ytrain11(oo:oo+2,:);
        yy11 = Ytrain11(oo+3,:);
        model11 = fitlm(xx11',yy11');
        C_mse11 = [C_mse11;model11.RMSE]; 
    catch
        continue;
    end
    end
    feature_RMSE11 = [feature_RMSE11;nanmean(C_mse11)];
    
    % last2,4,6 for every frame
    C_mse22 = [];
    Ytrain22 =LGN6_features;
    for oo = 1:5:size(Ytrain22,1)-6
    %for oo = floor(framerate/2):framerate:size(Ytrain22,1)
        try
        xx22 = [Ytrain22(oo,:);Ytrain22(oo+2,:);Ytrain22(oo+4,:)];
        yy22 = Ytrain22(oo+6,:);
        model22 = fitlm(xx22',yy22');
        C_mse22 = [C_mse22;model22.RMSE]; 
        catch
            continue;
        end
    end
    feature_RMSE22 = [feature_RMSE22;nanmean(C_mse22)];
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    
end

feats_mat = [log(feature_RMSE11),log(feature_RMSE22)];
save('./Youtube-UGC_LGN6_last3_ev1ev2_every5frame_feats.mat','feats_mat');



