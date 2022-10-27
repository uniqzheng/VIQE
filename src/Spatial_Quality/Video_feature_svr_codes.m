%%%%
clc;
clear all;
close all;
%%%%%%%%%%%%%
addpath('./All_videos/');
FULL_PATH = dir(fullfile('./videomatfileskonvid_level/*.mat'));
MOS_table = readtable('KoNViD_1k_mos.csv');
%FULL_PATH(5).name
graph_frequencies_shape =[];
graph_frequencies_scale =[];
width_video = [];
Mos_scores = [];
ids = table2array(MOS_table(:,1))
mos_scores = table2array(MOS_table(:,2));
features1 =[];
features2 =[];
features3 =[];
features4 =[];
Mos_scores = [];
for k=1:1:944
    name_folder = FULL_PATH(k).name;
    %file_name = 'video_level_graph_frequencies.mat';
    %file_name_mat = strcat('./graph_frequencies_konvid/',name_folder,'/',file_name);
    load(name_folder);
    yy = tsne(LGN_features);
%figure;
% gscatter(yy(:,1),yy(:,2))
% saveas(gcf,strcat('./plots4/4',DistVideoName1,'scatter.png'));
 LGN_features2 = yy;
%%%%%%%%%%% Angle calucation and curvature %%%%%%%%%%
cos_Alphas2_diff = [];
cos_Alphas1_diff = [];
cos_Alphas2_normal = [];
cos_Alphas1_normal = [];
length = size(LGN_features,1);
for i= 2:1:length-1
    prev = LGN_features2(i-1,:);
    current = LGN_features2(i,:);
    next = LGN_features2(i+1,:);
    numerator = dot((next - current),(current - prev));
    size(numerator);
    denominator = norm(next-current)*norm(current-prev);
    cos_alpha = acos(numerator/denominator);
    cos_Alphas2_diff = [cos_Alphas2_diff,cos_alpha];
    prev = LGN_features(i-1,:);
    current = LGN_features(i,:);
    next = LGN_features(i+1,:);
    numerator = dot((next - current),(current - prev));
    size(numerator);
    denominator = norm(next-current)*norm(current-prev);
    cos_alpha = acos(numerator/denominator);
    cos_Alphas1_diff = [cos_Alphas1_diff,cos_alpha];
    prev = LGN_features(i-1,:);
    current = LGN_features(i,:);
    %next = LGN_features(i+1,:);
    numerator = dot((prev),(current));
    size(numerator);
    denominator = norm(prev)*norm(current);
    cos_alpha = acos(numerator/denominator);
    cos_Alphas1_normal = [cos_Alphas1_normal,cos_alpha];
    prev = LGN_features2(i-1,:);
    current = LGN_features2(i,:);
    %next = LGN_features(i+1,:);
    numerator = dot((prev),(current));
    size(numerator);
    denominator = norm(prev)*norm(current);
    cos_alpha = acos(numerator/denominator);
    cos_Alphas2_normal = [cos_Alphas2_normal,cos_alpha];
end
final_curvature1_normal = mean(cos_Alphas1_normal);
final_curvature2_normal = mean(cos_Alphas2_normal);
final_curvature1_diff = mean(cos_Alphas1_diff);
final_curvature2_diff = mean(cos_Alphas2_diff);
    features1 = [features1;final_curvature1_normal];
    features2 = [features2;final_curvature2_normal];
    features3 = [features3;final_curvature1_diff];
    features4 = [features4;final_curvature2_diff];
    %graph_frequencies = -1*video_level(:);
    %indices = find(graph_frequencies);
    %graph_frequencies_non_zero = graph_frequencies(indices);
    %[shape,scale] = gamfitMoM(graph_frequencies_non_zero,0);
    %mean_graph = var(graph_frequencies_non_zero);
    %graph_frequencies_shape = [graph_frequencies_shape,shape];
    %graph_frequencies_scale = [graph_frequencies_scale,scale];
    %%%%%%%%%%%%%%%%
    %width = max(graph_frequencies_non_zero);
    finding_id = (split(name_folder,'.'));
    finding_id_string = finding_id{1};
    k_id = find(strcmp(ids,finding_id_string)); 
    %feature_vector = [width,shape,scale];
    %if(k == 3)
      %  features = feature_vector;
    %else
      %  features
      %  feature_vector
      %  features = vertcat(features,feature_vector);
   
    Mos_score_video = mos_scores(k_id);
    %Mos_score_video,shape,scale;
    Mos_scores = [Mos_scores;str2num(Mos_score_video{1})];
    %k = find(ids==13)

end
TF = isnan(real(features1));
indices = find(TF);
features_mod = real(features1);
Mos_scores2 = Mos_scores;
features_mod(indices) = [];
Mos_scores2(indices) = [];
calculatepearsoncorr(features_mod,Mos_scores2)
TF = isnan(real(features2));
indices = find(TF);
features_mod = real(features2);
Mos_scores2 = Mos_scores;
features_mod(indices) = [];
Mos_scores2(indices) = [];
calculatepearsoncorr(features_mod,Mos_scores2)
TF = isnan(real(features3));
indices = find(TF);
features_mod = real(features3);
Mos_scores2 = Mos_scores;
features_mod(indices) = [];
Mos_scores2(indices) = [];
calculatepearsoncorr(features_mod,Mos_scores2)
TF = isnan(real(features4));
indices = find(TF);
features_mod = real(features4)*180/3.14;videomatfileskonvid_level
Mos_scores2 = Mos_scores;
features_mod(indices) = [];
Mos_scores2(indices) = [];
calculatepearsoncorr(features_mod,Mos_scores2)
%plot(Mos_scores,'r');
%hold on;
%plot(graph_frequencies_shape,'g');
%hold on;
%plot(graph_frequencies_scale,'b');
%%temp = calculatepearsoncorr(Mos_scores(1,:)',graph_frequencies_shape(1,:)')
%temp = calculatepearsoncorr(Mos_scores(1,:)',width_video(1,:)')
%temp = calculatepearsoncorr(Mos_scores(1,:)',graph_frequencies_scale(1,:)')