function [latent_space_dir,image_dir,review_initials,manual_file,gsi_dir] = setUpDatabasePaths
% set up paths depending on what system I'm using
% latent_space_dir: base directory where all network weights and latent
%   space samples are stored
% image_dir:  directory where unsupervised image databases are stored.
% gsi_dir:    directory where the audio files are stored...
% manual_file:  Where the MATLAB-converted Greeneridge manual data file
%           'All_manual_results.mat' is located--full-path file name
% gitpath: directory of GIT repo sio_research


gsi_dir=[];image_dir=[];latent_space_dir=[];gitpath=[];manual_file=[];
success=false;


[~,hostname] = system('hostname');
[~,user_name]=system('whoami');

%hostname =
%
%'Angels-MacBook-Air-6.local
%'
%
%user_name
%
%5user_name =
%'angel

if strcmpi(deblank(user_name),'thode')
    review_initials='AT';
elseif strcmpi(deblank(user_name),'oboulais')
    review_initials='OB';
elseif strcmpi(deblank(user_name),'angel')
    review_initials='AH';
elseif contains(deblank(user_name),'shelby')
    review_initials='SV'; %Shelby Villescaz-Best
elseif contains(deblank(user_name),'raquel')
    review_initials='RR'; %Raquel Ramos
else
    review_initials='JT';
end



switch hostname(1:end-1)

    case 'ishmael.ucsd.edu'

        if strcmpi(deblank(user_name),'thode')
            fprintf('Using direct drive on AaronThode''s ishmael account\n')
            gitpath = '/Users/thode/Desktop/ThodeLab';
            latent_space_dir = '/Users/oboulais/Public/Bowhead_DL_Project/';

            %%%Check if GSI data available
            gsi_dir='~/mnt/jonah3/Shared/Data';
            if ~exist(gsi_dir,'dir')==7
                disp('No GSI directory found, cannot link DASARs or play sounds');
            end

            success=true;
        end
    otherwise

        if contains(deblank(user_name),'shelby')

            fprintf('Using Shelby VB local laptop...\n');

            external_dir=['D:' filesep 'Bowhead_DL_Project'];
            %%%Check if GSI data available
            gsi_dir='/Volumes/Shared/Data/';

            if ~exist(gsi_dir,'dir')==7
                disp('No GSI directory found, cannot link DASARs or play sounds');
            end

            %manual_file='/Volumes/Thode_AI_Working_Disk/Bowhead_DL_Project/Shell_Manual_Results/';
            manual_file=[external_dir filesep 'Shell_Manual_Results' filesep];

            manual_file=[manual_file filesep 'All_manual_results.mat'];
            disp('Using external disk for latent space and images')
            latent_space_dir=[external_dir filesep 'Networks_And_LatentSpaceRuns.dir' filesep 'LD32' filesep];
            image_dir=[external_dir filesep 'BCB_Whale_Datasets' filesep];
            success=true;

        elseif contains(deblank(user_name),'raquel')
            fprintf('Using Raquel R local laptop...\n');

            external_dir=['D:' filesep 'Bowhead_DL_Project'];
            %%%Check if GSI data available
            gsi_dir='/Volumes/Shared/Data/';

            if ~exist(gsi_dir,'dir')==7
                disp('No GSI directory found, cannot link DASARs or play sounds');
            end

            manual_file=[external_dir filesep 'Shell_Manual_Results' filesep];

            manual_file=[manual_file filesep 'All_manual_results.mat'];
            disp('Using external disk for latent space and images')
            latent_space_dir=[external_dir filesep 'Networks_And_LatentSpaceRuns.dir' filesep 'LD32' filesep];
            image_dir=[external_dir filesep 'BCB_Whale_Datasets' filesep];
            success=true;

        elseif strcmpi(deblank(user_name),'angel')
            fprintf('Using Angel''s local laptop\n');

            %%%Check if GSI data available
            gsi_dir='/Volumes/Shared/Data/';

            if ~exist(gsi_dir,'dir')==7
                disp('No GSI directory found, cannot link DASARs or play sounds');
            end

            %manual_file='/Volumes/Thode_AI_Working_Disk/Bowhead_DL_Project/Shell_Manual_Results/';
            manual_file='/Volumes/Thode_AI_Working_Disk/Shell_Manual_Results/';

            manual_file=[manual_file filesep 'All_manual_results.mat'];
            disp('Using external Thode_AI_Working_Disk for latent space and images')
            latent_space_dir='/Volumes/Thode_AI_Working_Disk/Bowhead_DL_Project/Networks_And_LatentSpaceRuns.dir/LD32/';
            image_dir='/Volumes/Thode_AI_Working_Disk/Bowhead_DL_Project/BCB_Whale_Datasets/';
            success=true;

        elseif strcmpi(deblank(user_name),'thode')
            fprintf('Using Aaron''s local laptop\n')

            manual_file='/Users/thode/Projects/Greeneridge_bowhead_detection/DeepLearningNPRB_Project/Shell_Manual_Results';
            manual_file=[manual_file filesep 'All_manual_results.mat'];

            %%%Check if GSI data available
            gsi_dir='/Volumes/Shared/Data/';

            if ~exist(gsi_dir,'dir')==7
                disp('No GSI directory found, cannot link DASARs or play sounds');
            end

            %%%Check if external drive.  If exists use it.
            if exist('/Volumes/Thode_AI_Working_Disk/','dir')==7
                disp('Using external Thode_AI_Working_Disk for latent space files and images')
                latent_space_dir='/Volumes/Thode_AI_Working_Disk/Bowhead_DL_Project/Networks_And_LatentSpaceRuns.dir/LD32/';
                image_dir='/Volumes/Thode_AI_Working_Disk/Bowhead_DL_Project/BCB_Whale_Datasets/';

            else
                disp('Using internal laptop storage for latent space.')
                latent_space_dir='/Users/thode/Projects/Greeneridge_bowhead_detection/DeepLearningNPRB_Project/Bowhead_DL_Project/LD32/';

                %%%Check if external server mounted for images
                if exist('/Volumes/Bowhead_DL_Project/','dir')==7
                    image_dir='/Volumes/Bowhead_DL_Project/BCB_Whale_Datasets/';
                else
                    disp('WARNING!  CANNOT DISPLAY IMAGES....')

                end
            end
            success=true;


        end


end

if ~success
    error('No valid database paths found');
    return
end


%