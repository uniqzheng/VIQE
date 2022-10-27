clear;
data_name = 'LIVE_VQC';
% data_path = '../../database/LIVE_VQC/VideoDatabase/';
video_tmp = './tmp';
if ~exist(video_tmp, 'dir'), mkdir(video_tmp); end
feat_path = '../features';
mos_filename = fullfile(feat_path, [data_name,'_metadata.csv']);
filelist = readtable(mos_filename);
num_videos = size(filelist,1);
feature_RMSE11 = [];
feature_RMSE22 = [];
Mos_Scores = filelist.MOS(:);
for idx = 1:num_videos
    strs = split(filelist.File{idx},'.');
    name_folder = strcat(strs{1},'.mat');
    fprintf('process NO. %d video %s\n',idx,filelist.File{idx});

    name_tp = strcat('../features/LIVE_VQC_nss34/',name_folder);
    load(name_tp);
    
    temporal_score(isnan(temporal_score)|isinf(temporal_score))=0;
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % last1,2,3 for every frame
    C_mse11 = [];
    Ytrain11 =temporal_score;
    for oo = 1:5:size(Ytrain11,1)-3
        xx11 = Ytrain11(oo:oo+2,:);
        yy11 = Ytrain11(oo+3,:);
        model11 = fitlm(xx11',yy11');
        C_mse11 = [C_mse11;model11.RMSE]; 
    end
    feature_RMSE11 = [feature_RMSE11;nanmean(C_mse11)];
    
    % last2,4,6 for every frame
    C_mse22 = [];
    Ytrain22 =temporal_score;
    for oo = 1:5:size(Ytrain22,1)-6
        xx22 = [Ytrain22(oo,:);Ytrain22(oo+2,:);Ytrain22(oo+4,:)];
        yy22 = Ytrain22(oo+6,:);
        model22 = fitlm(xx22',yy22');
        C_mse22 = [C_mse22;model22.RMSE]; 
    end
    feature_RMSE22 = [feature_RMSE22;nanmean(C_mse22)];
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

end

feats_mat = [log(feature_RMSE11),log(feature_RMSE22)];
save('./LIVE_VQC_nss34_last3_ev1ev2_every5frame_feats.mat','feats_mat');



