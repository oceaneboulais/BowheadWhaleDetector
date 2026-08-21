%%%%master_index_database.m%%%%%
%
%   RUN THIS SCRIPT AFTER master_create_datasets.m BUT BEFORE  master_assemble_unsupervised_database.m
%       ALSO REVIEW AND CLEAN THE DATABASE SO THAT SAME DATES ARE PRESENT AT
%       ALL SITES FOR A GIVEN YEAR.  THE PREVIOUS SCRIPT WILL SOMETIMES CREATE
%       EXTRA DAYS AT A SITE IF MANUAL CALLS ARE VERY CLOSE TO THE START OR END
%       OF A DAY.
%   This script counts all the files in the new database and makes two
%   indexing variables that are stored at the top-level of database in
%   'Database_index.mat'
%
%   Furthermore, this script uses consistency of timing and bearing to flag
%   likely airgun signals in the index variables.
%
%   Databases has 'D*.dir' folders at lowest level to ensure number of
%   folders items is not so large that UNIX commands do not work.
%
%   Variable formats and organization:
%       index{Iyear,Isite,Iday,Ifold,I_d_directory};  Note that results NOT
%               broken down by DASAR
%       file_fraction.manual_all=zeros(Iyear,Isite,Iday,Idasar,I_d_directory);  %Last index is maximum number of 'D' folders expected.
%       file_fraction.auto_all= file counts including possible airguns
%       file_fraction.auto_noairgun= file counts after possible airgun
%                   removal

close all
clear

database_folder='../../Spectrogram_Image_Database.dir';
eval(sprintf('!mkdir %s/Trash.dir',database_folder));

%%%These variables should match those used for master_create_datasets.m
%%%  or at least span a larger domain.
%%%
%%%  Note that each year/site combination may cover different days.
year_want={'08','09','10','11','12','13','14'};
Site={'2','3','4','5'};
DASAR_strings='ABCDEFG';  %%No real harm if not all of these exists

year_want={'08','10','12','14'};
Site={'3','5'};
DASAR_strings='ADG';  %%No real harm if not all of these exists

folder_names={'Event_sounds.dir','Manually_selected_bowhead_calls.dir'};
SNR_gram_dims=[121 104];  %%%Dimensions expected to ensure consistentcy of data fed into autoencoder.

%%%Interval airgun detector parameters.
param.interval_remove.debug=false;
param.interval_remove.ICItol=[0.5 1.00];  %How close does a candidate have to be to predicted ICI time? (sec)
param.interval_remove.ICI_range=[7 10;
    10 42];
param.interval_remove.Ndet=20;
param.interval_remove.Nmiss=12;
param.interval_remove.tol_feature=15;

param.interval_remove.ICI_std=0.4;  %How close do adjacent ICI estimates have to be (sec)
param.interval_remove.Nstd=7;      % How many ICIs must pass the 'ICI_std' test?

test_file='S314A0T20140818T064828_Type0.mat';

DASARs_counted=false;
max_D_dirs=0;
max_DASAR_strings='';
for Iyear=1:length(year_want)
    disp(year_want{Iyear});
    for Isite=1:length(Site)
        disp(Site{Isite});

        dir_string=sprintf('%s/20%s/Site%s/Day_*', ...
            database_folder,year_want{Iyear},Site{Isite});

        %%%Identify the unique days for this Year/Site combo
        temp_dir_names=dir(dir_string);
        fprintf('Days in Year %s\n',year_want{Iyear});

        clear day_want
        for Iday=1:length(temp_dir_names)
            day_want{Iday}=temp_dir_names(Iday).name(9:12);
            disp(day_want{Iday});

        end
        disp('Working though day list for this Site/year combo...')
        %%%Derives the unique day list
        for Iday=1:length(day_want)
            disp(day_want{Iday});
            for Ifold=1:length(folder_names)  %Event or Manual detection?
                disp(folder_names{Ifold});
                dir_string=sprintf('%s/20%s/Site%s/Day_20%s%sT000000/%s', ...
                    database_folder,year_want{Iyear},Site{Isite},year_want{Iyear},day_want{Iday},folder_names{Ifold});
                DD_dirs=dir([dir_string '/D*dir']);
                max_D_dirs=max([max_D_dirs length(DD_dirs)]);  %Identifies the maximum number of D*.dir directories for any Year/Site/Day combo.
                for Idd=1:length(DD_dirs)
                    fnames=dir([dir_string '/' DD_dirs(Idd).name '/S*mat']);
                    Nfiles=length(fnames);

                    if Nfiles==0
                        disp([dir_string '/' DD_dirs(Idd).name ' has no files.']);
                        continue
                    end

                    %%%Initialize primary index variable
                    str_length=length(fnames(1).name);
                    index{Iyear,Isite,Iday,Ifold,Idd}.fname=zeros(Nfiles,str_length);
                    index{Iyear,Isite,Iday,Ifold,Idd}.bearing=zeros(1,Nfiles);
                    index{Iyear,Isite,Iday,Ifold,Idd}.tabs=zeros(1,Nfiles);
                    index{Iyear,Isite,Iday,Ifold,Idd}.type=zeros(1,Nfiles);


                    for Ifile=1:Nfiles
                        temp=load([fnames(Ifile).folder filesep fnames(Ifile).name]);
                        % if contains(fnames(Ifile).name,test_file)
                        %    keyboard
                        % end
                        %%%Check that spectrogram sizes are consistent...if
                        %%%not, move to a Trash folder for investigation.
                        if ~all(SNR_gram_dims==size(temp.SNR_gram))
                            disp([ fnames(Ifile).name ' SNR_gram not correct size:\n' size(temp.SNR_gram)]);
                            keyboard
                            eval(sprintf('!mv %s/%s %s/Trash.dir',fnames(Ifile).folder,fnames(Ifile).name,database_folder));
                            continue
                        end
                        index{Iyear,Isite,Iday,Ifold,Idd}.fname(Ifile,:)=fnames(Ifile).name;
                        index{Iyear,Isite,Iday,Ifold,Idd}.bearing(Ifile)=temp.bearing;
                        %index{Iyear,Isite,Iday,Ifold,Idd}.tabs(Ifile)=temp.tabs_mid;
                        index{Iyear,Isite,Iday,Ifold,Idd}.tabs(Ifile)=temp.tabs_tstartt;
                    end

                    %%%Remove bad files from index variable
                    Igood=find(index{Iyear,Isite,Iday,Ifold,Idd}.tabs>0);
                    index{Iyear,Isite,Iday,Ifold,Idd}.fname=index{Iyear,Isite,Iday,Ifold,Idd}.fname(Igood,:);
                    index{Iyear,Isite,Iday,Ifold,Idd}.bearing=index{Iyear,Isite,Iday,Ifold,Idd}.bearing(Igood);
                    index{Iyear,Isite,Iday,Ifold,Idd}.tabs=index{Iyear,Isite,Iday,Ifold,Idd}.tabs(Igood);
                    index{Iyear,Isite,Iday,Ifold,Idd}.is_airgun=zeros(1,length(Igood));  %New airgun flag variable.

                    %%%Identify unique DASARs present in this directory
                    DASAR_ID=char(unique(index{Iyear,Isite,Iday,Ifold,Idd}.fname(:,5)));

                    if length(DASAR_ID)>length(max_DASAR_strings)
                        fprintf('new DASAR string is %s\n',DASAR_ID);
                        max_DASAR_strings=DASAR_ID;
                    end

                    %%%%Initialize file_fraction variables once
                    if ~DASARs_counted
                        DASARs_counted=true;
                        file_fraction.manual_all=zeros(length(year_want),length(Site),length(day_want),length(DASAR_strings),3);  %Last index is maximum number of 'D' folders expected.
                        file_fraction.auto_all=file_fraction.manual_all;
                        file_fraction.auto_noairgun=file_fraction.manual_all;
                    end

                    %%%Record number of files present from a specific DASAR in this directory
                    %folder_names={'Event_sounds.dir','Manually_selected_bowhead_calls.dir'};
                    for Idasar=1:length(DASAR_ID)
                        Nfiles_DASAR=sum(double(DASAR_ID(Idasar))==index{Iyear,Isite,Iday,Ifold,Idd}.fname(:,5));
                        Idasar_index=find(DASAR_ID(Idasar)==DASAR_strings);
                        if Ifold==1  %auto
                            file_fraction.auto_all(Iyear,Isite,Iday,Idasar_index,Idd)=Nfiles_DASAR;
                        else  %manual calls
                            file_fraction.manual_all(Iyear,Isite,Iday,Idasar_index,Idd)=Nfiles_DASAR;
                        end
                    end %Idasar

                    if Ifold==2  %%If a certified bowhead call from a manual annotation, skip airgun analysis.
                        continue
                    end

                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    %%%Optional search for airgun signals
                    disp('Starting airgun detection');tic

                    for Idasar=1:length(DASAR_ID)
                        Iltr=find(double(DASAR_ID(Idasar))==index{Iyear,Isite,Iday,Ifold,Idd}.fname(:,5));
                        param.interval_remove.titstr=fnames(Iltr(1)).name;  %Optional title string for when debug flag active
                        ICI=estimate_airgun_interval(index{Iyear,Isite,Iday,Ifold,Idd}.tabs(Iltr),index{Iyear,Isite,Iday,Ifold,Idd}.bearing(Iltr),param.interval_remove);
                        index{Iyear,Isite,Iday,Ifold,Idd}.is_airgun(Iltr)=ICI;
                        Nfiles_DASAR=sum(ICI<1);  %Count files not flagged as possible airgun
                        Idasar_index=find(DASAR_ID(Idasar)==DASAR_strings);
                        file_fraction.auto_noairgun(Iyear,Isite,Iday,Idasar_index,Idd)=Nfiles_DASAR;
                    end %Idasar

                    for Ifile=1:Nfiles
                        temp=load([fnames(Ifile).folder filesep fnames(Ifile).name],'features');
                        features=temp.features;
                        features.ICI=index{Iyear,Isite,Iday,Ifold,Idd}.is_airgun(Ifile);
                        save([fnames(Ifile).folder filesep fnames(Ifile).name],'features',"-append");
                    end

                    disp('Finished airgun detection');toc
                end %Idd
            end %Ifold
        end %Iday
    end %Isite
end %Iyear

%file_fraction.auto_all=file_fraction.auto_all/sum(file_fraction.auto_all(:));
%file_fraction.manual_all=file_fraction.manual_all/sum(file_fraction.manual_all(:));
%file_fraction.auto_noairgun=file_fraction.auto_noairgun/sum(file_fraction.auto_noairgun(:));
max_DASAR_strings=max_DASAR_strings.';
save([database_folder '/Database_index.mat'],'index','file_fraction','year_want','Site','day_want','folder_names','SNR_gram_dims','DASAR_strings','max_DASAR_strings');
