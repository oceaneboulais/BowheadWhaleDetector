%%%%%master_create_tdrift_table.m%%%
%  Goes through GSI files and copies tdrift information
%  to allow future use by WAV file versions.
%

close all
clear


strr='ABCDEFGHIJKLM';
GSI_file_dir='/Volumes/Shared/Data/';

year_want={'08','09','10','11','12','13','14'};
Site={'2','3','4','5'};

if exist(GSI_file_dir,'dir')==0
    error('GSI_file_dir not present')
end


for Iyear=1:length(year_want)
    for Isite=1:length(Site)
        for I=1:length(strr)
            % DASAR_list{I}=['S314' strr(I) '0'];
            DASAR_list{I}=sprintf('S%s%s%s0',Site{Isite},year_want{Iyear},strr(I));
        end
        ctmin=0;
        ctmax=Inf;

        %%%Loop through dates and create a selection file for each DASAR and day
        create_folder_flag=true;  %Flag to check for output directory structure when a new year-site combo is started

        for Id=1:length(strr)  %For each DASAR

            %keyboard
            fprintf('DASAR %s\n',DASAR_list{Id});

            %%%Placeholder to read in clock drift information for GSI
            %%%file for this day...

            dir_want=sprintf('%s/Shell20%s_GSI_Data/S%s%sgsif/S%s%s%s0', ...
                GSI_file_dir,year_want{Iyear},Site{Isite},year_want{Iyear}, ...
                Site{Isite},year_want{Iyear},strr(Id));
            %fs=head.Fs*(1+head.tdrift/86400);
            if exist(dir_want,'dir')~=7
                dir_want(end)='1';
                if exist(dir_want,'dir')~=7
                    disp('Data do not exist')
                    continue
                end
            end
            file_names{Iyear,Isite,Id}=dir([dir_want '/*gsi']);
            %for Ifile=1:length(file_names{Iyear,Isite,Id})
            head{Iyear,Isite,Id}=readgsif_header([dir_want filesep file_names{Iyear,Isite,Id}(1).name]);
            %end

        end %DASAR
        fprintf('Finished exporting this site and year.... \n\n\n')
    end %Isite
end %Iyear

save GSI_header_table file_names head year_want Site strr

