%%%%master_assemble_unsupervised_database.m%%%%%
%%%%  Create an unsupervised dataset from a complete dataset
%%%%  RUN THIS SCRIPT AFTER master_index_database.m
%
%  You can control what specific years, sites, and DASARs are incorporated.
%   However, all days from a particular year/site/DASAR are included, since
%   the specific dates are specific to a given year/site/DASAR combo.

close all
clear

database_folder='../../Spectrogram_Image_Database.dir';
data=load([database_folder '/Database_index.mat']);
eval(sprintf('!mkdir %s/Unsupervised_database.dir',database_folder));
eval(sprintf('!rm %s/Unsupervised_database.dir/*.mat',database_folder));

Nsamples=100000;  %%Total data samples wanted in unsupervised database
manual_fraction=0;  %%Fraction of samples that come from manual annotations
include_airguns=true;  %%If true include spectrograms of likely airgun signals

year_want={'08','10','12','14'};%Note, must be in numerical order
Site={'3','5'};
DASAR_strings={'A','D','G'};
folder_names={'Event_sounds.dir','Manually_selected_bowhead_calls.dir'};

%index{Iyear,Isite,Iday,Ifold,I_d_directory};  Note that results NOT
%               broken down by DASAR
%       file_fraction.manual_all=zeros(Iyear,Isite,Iday,Idasar,I_d_directory);  %Last index is maximum number of 'D' folders expected.
%     

for I=1:length(data.DASAR_strings)
    data.DASAR_strings_cell{I}=data.DASAR_strings(I);
end
folder_fractions=[1-manual_fraction manual_fraction];

%%%Trim down database indicies to match requested subsets above
Isite=contains(data.Site,Site);
Iyear=contains(data.year_want,year_want);
%Iday=contains(data.day_want,day_want);
Idasar=contains(data.DASAR_strings_cell,DASAR_strings);

data_sub.index=data.index(Iyear,Isite,:,:,:);
data_sub.file_fraction.manual=data.file_fraction.manual_all(Iyear,Isite,:,Idasar,:);
if include_airguns
    data_sub.file_fraction.auto=data.file_fraction.auto_all(Iyear,Isite,:,Idasar,:);
else
    data_sub.file_fraction.auto=data.file_fraction.auto_noairgun(Iyear,Isite,:,Idasar,:);
end

%%%Normalize the file_fraction variables so that contents show the fraction
%%%of the total each database folder will provide.
data_sub.file_fraction.auto=data_sub.file_fraction.auto/sum(data_sub.file_fraction.auto(:));
data_sub.file_fraction.manual=data_sub.file_fraction.manual/sum(data_sub.file_fraction.manual(:));


%%%%%%%Enter each database folder and copy subset of files to Unsupervised
%%%%%%%folder.
file_name_list=[];file_name_flag=true;
is_airgun_list=false(1,Nsamples);
Icount=1;

Nsamples_running_count=0;
for Iyear=1:length(year_want)
    disp(year_want{Iyear});
    for Isite=1:length(Site)
        disp(Site{Isite});
        dir_string=sprintf('%s/20%s/Site%s/Day_*', ...
            database_folder,year_want{Iyear},Site{Isite});

        %%%Store specific days available in this year/site
        temp_dir_names=dir(dir_string);
        for Iday=1:length(temp_dir_names)
            day_want{Iday}=temp_dir_names(Iday).name(9:12);
        end
        %%%Derives the unique day list

        for Iday=1:length(day_want)
            disp(day_want{Iday});
            for Ifold=1:length(folder_names)
                if folder_fractions(Ifold)==0
                    continue
                end
                disp(folder_names{Ifold});
                dir_string=sprintf('%s/20%s/Site%s/Day_20%s%sT000000/%s', ...
                    database_folder,year_want{Iyear},Site{Isite},year_want{Iyear},day_want{Iday},folder_names{Ifold});
                if Ifold==1
                    file_fraction=data_sub.file_fraction.auto;
                else
                    file_fraction=data_sub.file_fraction.manual;
                end

                %%%%How many samples are wanted from each folder
                %  row is DASAR, column is 'D*.dir' folder
                Nsamples_want=floor(folder_fractions(Ifold)*Nsamples*squeeze(file_fraction(Iyear,Isite,Iday,:,:)));
                Nsamples_running_count=Nsamples_running_count+sum(Nsamples_want(:));
                Ndd=size(Nsamples_want,2);
                for Idd=1:Ndd  %%subfolders
                    if sum(Nsamples_want(:,Idd))==0  %if this D*.dir directory does not exist, exit.
                        %disp("directory does not exist")
                        continue
                    end

                    fnames=data_sub.index{Iyear,Isite,Iday,Ifold,Idd}.fname;
                    if file_name_flag
                        file_name_list=zeros(Nsamples,size(fnames,2));
                        file_name_flag=false;
                    end
                    airgun_index=data_sub.index{Iyear,Isite,Iday,Ifold,Idd}.is_airgun;
                    %%%Remove airgun signals if desired
                    if ~include_airguns
                        fnames=fnames(data_sub.index{Iyear,Isite,Iday,Ifold,Idd}.is_airgun<1,:);  %A value <1 is not an airgun (ICI is 0 or -1)
                        airgun_index=airgun_index(data_sub.index{Iyear,Isite,Iday,Ifold,Idd}.is_airgun<1);
                    end
                    for Idasar=1:length(DASAR_strings)
                        disp(DASAR_strings{Idasar});
                        Igood=find(fnames(:,5)==DASAR_strings{Idasar});
                        Nfiles_local=length(Igood);
                        if Nfiles_local==0
                            %disp('No DASAR here')
                            continue
                        end
                        %Two situations can occur.  First, the number
                        %of files in the folder is greater than the
                        %number of requested samples from this folder.

                        if Nfiles_local>=Nsamples_want(Idasar,Idd)
                            Ichoose=Igood(randperm(length(Igood),Nsamples_want(Idasar,Idd)));  %%Select number of random files needed.

                        else  %%if we are requesting more samples than files, sample with replacement.
                           % Ichoose=Igood(randperm(Nfiles_local));
                            %NN=ceil(Nsamples_want(Idasar,Idd)/Nfiles_local);
                            %Ichoose=repmat(Ichoose,1,NN);Ichoose=Ichoose(:);
                            %Ichoose=Ichoose(1:Nsamples_want(Idasar,Idd));

                            Ichoose=Igood(randi(Nfiles_local,1,Nsamples_want(Idasar,Idd)));
                        end

                       % disp(char(fnames(Ichoose,:)));
                       % pause

                       %%%Copy specific files to destination folder
                        for Ifile=1:length(Ichoose)
                            copy_file_str=sprintf('%s/D%i.dir/%s',dir_string,Idd,fnames(Ichoose(Ifile),:));
                            if exist(copy_file_str,'file')==0
                                keyboard
                            end
                            eval(sprintf('!cp %s %s/Unsupervised_database.dir',copy_file_str,database_folder));
                            file_name_list(Icount,:)=fnames(Ichoose(Ifile),:);
                            is_airgun_list(Icount)=airgun_index(Ichoose(Ifile));
                            Icount=Icount+1;
                        end
                    end
                  
                end %Idd
            end %Ifold
        end %Iday
    end %Isite
end %Iyear
Icount=Icount-1;
%%%Save airgun information
if Icount<Nsamples
    file_name_list=file_name_list(1:Icount,:);
    is_airgun_list=is_airgun_list(1:Icount);

end


%%%Check results
fnames_final=dir([database_folder '/Unsupervised_database.dir/*mat']);
fprintf('There are %i files in database\n',length(fnames_final));
if length(is_airgun_list)~=length(fnames_final)
    keyboard
end

save([database_folder filesep 'Unsupervised_database.dir' filesep 'airgun_index.mat'],'file_name_list','is_airgun_list');


letters=zeros(size(fnames_final,1),1);
for I=1:size(fnames_final,1)
     letters(I)=str2num(fnames_final(I).name(end-4));
end
auto_frac=sum(letters==0)/length(letters);
fprintf('Manual fraction is %6.4f\n',1-auto_frac);

