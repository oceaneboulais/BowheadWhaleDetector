function [data_basedir,procdata_basedir,gitpath] = setUpDatabasePaths
% set up paths depending on what system I'm using
% data_basedir: base directory where all drifter data is hosted
% procdata_basedir: base directory where all analysis products are saved
% gitpath: directory of GIT repo sio_research


success=false;

[~,hostname] = system('hostname');
[~,user_name]=system('whoami');
switch hostname(1:end-1)

    case 'ishmael.ucsd.edu'

        if strcmpi(deblank(user_name),'thode')
            fprintf('Using direct drive on AaronThode''s ishmael account\n')
            gitpath = '/Users/thode/Desktop/ThodeLab';
            data_basedir = '/Users/oboulais/Public/Bowhead_DL_Project/';

            procdata_basedir = data_basedir;
            % envdir = '/Volumes/Shared/Databases';
            success=true;
        end
    otherwise
        fprintf('Using Aaron''s local laptop\n')
        gitpath = '/Users/thode/Desktop/ThodeLab';
        %data_basedir = '/Volumes/Shared/ONR_DRIFTER'; %%%Jonah2
        data_basedir='/Volumes/Bowhead_DL_Project/';

        procdata_basedir=data_basedir;
        success=true;
end


%