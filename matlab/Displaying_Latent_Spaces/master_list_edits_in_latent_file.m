%%%%master_list_edits_in_latent_file.m
%%% Run this script in the same folder with the file name
close all
clear all

file_name='latent_embeddings_3d_auto_MATLAB_merged.mat';
file_name='/Volumes/Thode_AI_Working_Disk/Bowhead_DL_Project/Networks_And_LatentSpaceRuns.dir/LD32/Autoencoder_v13_100E_32LD_32C_AutoManual_Combined_100K_Date20260416-180022.dir/MATLAB/latent_embeddings_3d_auto_MATLAB.mat';

data=load(file_name);
reviewer='AH';


I_changed=find(data.features.type~=data.features.type_org);
fprintf('%i images have changed in this file...\n',reviewer,length(I_changed));

I_reviewed=contains(data.reviewer,reviewer);
fprintf('%s has edited %i images...\n',reviewer,sum(I_reviewed));
dates_adjusted=sort(data.date_adjusted(I_reviewed));
fprintf('First date is %s\n',dates_adjusted(1));
fprintf('Latest date is %s\n',dates_adjusted(end));


I_both=find(data.features.type(I_reviewed)~=data.features.type_org(I_reviewed));
fprintf('%s has changed %i labels total...\n',reviewer,length(I_both));
