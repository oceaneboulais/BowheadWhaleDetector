%%%%%master_create_supervised_dataset.m%%%
%
%
% Key points to remember when making spectrograms:
%       (1) They must be equalized;
%       (2) If not equalized they must be calibrated by the DASAR
%       calibation curve.
%       (3) They must be saved in uint16 to save as much space as possible.

close all
clear
!rm diary_output.txt
diary diary_output.txt
DASAR_strings='ABCDEFG';
GSI_file_dir='/Volumes/Shared/Data/';
WAV_file_dir='/Volumes/Bowhead4/';
data_file_type='GSI'; %'GSI' or 'WAV'
Manual_record_files_dir='../Shell_Manual_Results';
output_dir='../Spectrogram_Image_Database.dir';


%param.spec.debug_plot=false;

%debug.sec_to_load=6*60*60+1;
debug.Iday_start=1;
debug.Idasar_start=1;
debug.sec_to_load=Inf;
write_files=true;

year_want={'08','09','10','11','12','13','14'};
Site={'2','3','4','5'};

year_want={'10'};
Site={'5'};


sound_type='whale'; %whale, seal

%%%% Seconds of data to convert to spectrogram to conserve RAM memory.
%%%%   Duration should be short enough that background noise not expected
%%%%   to change...

chunk_sample=6*60*60;  %seconds
file_len_sec=10; %length of final file clip (includes noise estimate)
spectrogram_len_sec=5; %length of final spectrogram clip. (data used for noise removed)


%%%Parameters for event detection
param.event.dB_threshold = 5; % threshold above mean for detection
param.event.image_scale_factor = 5;  % factor to multiply SNR by for saving as unit8 image
param.event.fmin = 25;
param.event.fmax = 475;

param.spec.Nfft=256;
param.spec.ovlap=0.9;
param.spec.image_scale_factor = param.event.image_scale_factor;
param.spec.fmin = 0;
param.spec.fmax = 500;
%param.spec.final_dims=8*[16 10];

%Set up energy detector.  Example for bowhead whale analysis that monitors between 25 and 350 Hz,
%  using a set of detectors with 37 Hz bandwidth.
K=1;
param.energy.Nfft=256;
param.energy.Fs =1000;
param.energy.ovlap = 0.75;
param.energy.flo_det=param.event.fmin;
param.energy.fhi_det=param.event.fmax;
param.energy.burn_in_time=0.25;  %Time in minutes
param.energy.eq_time=5;   param.energy_desc{K}='Equalization time (s): should be roughly twice the duration of signal of interest';K=K+1;
param.energy.bandwidth=37;     param.energy_desc{K}='Bandwidth of sub-detector in kHz';K=K+1;
param.energy.threshold=5;  param.energy_desc{K}='Threshold in dB to accept a detection';K=K+1;
param.energy.TolTime=0.5;  param.energy_desc{K}='Minimum time in seconds that must elapse for two detections to be listed as separate';K=K+1;
param.energy.MinTime=0;     param.energy_desc{K}='Minimum time in seconds a required for a detection to be logged';K=K+1;
param.energy.MaxTime=5;     param.energy_desc{K}= 'Maximum time in seconds a detection is permitted to have';K=K+1;
param.energy.debug=0;       param.energy_desc{K}= '0: do not write out debug information. 1:  SEL output.  2:  equalized background noise. 3: SNR.';K=K+1;

nu=1.7;

if strcmpi(data_file_type,'gsi')
    if exist(GSI_file_dir,'dir')==0
        error('GSI_file_dir not present')
    end
elseif exist(WAV_file_dir,'dir')==0
    error('WAV_file_dir not present')

elseif exist(WAV_file_dir,'dir')==7
    head_info=load('GSI_header_table.mat');
end

for Iyear=1:length(year_want)
    for Isite=1:length(Site)
        for I=debug.Idasar_start:length(DASAR_strings)
            % DASAR_list{I}=['S314' strr(I) '0'];
            DASAR_list{I}=sprintf('S%s%s%s0',Site{Isite},year_want{Iyear},DASAR_strings(I));
        end
        ctmin=0;
        ctmax=Inf;
        fname=sprintf('%s%s20%s%sAllSite%s_20%s_manual_archive.txt', ...
            Manual_record_files_dir,filesep,year_want{Iyear},filesep,Site{Isite},year_want{Iyear});

        fname_mat=sprintf('%s%s20%s%sAllSite%s_20%s_manual_archive.mat', ...
            Manual_record_files_dir,filesep,year_want{Iyear},filesep,Site{Isite},year_want{Iyear});

        if ~exist(fname_mat,'file')
            [ind,localized]=read_tsv_archive(fname,ctmin,ctmax,DASAR_list);
            save(fname_mat,'ind','localized');
            clear ind localized
        end
        manual=load(fname_mat);


        %%%%%Restrict database to certain types of calls...%%%%%
        if strcmpi(sound_type,'whale')
            Itype=find(manual.localized.wctype<=7);  %bowhead whale calls only
        elseif strcmpi(sound_type,'seal')
            Itype=find(manual.localized.wctype==8 | manual.localized.wctype==9);  %seal and walrus only

        end

        if isempty(Itype)
            fprintf('No %s here\n',sound_type);
            continue
        end


        fieldnamess=fieldnames(manual.ind);
        %manual.ind.duration=manual.ind.duration(Itype,:);
        for JJ=1:length(fieldnamess)
            manual.ind.(fieldnamess{JJ})=manual.ind.(fieldnamess{JJ})(Itype,:);
        end
        call_type_all=manual.localized.wctype;
        call_type_all=call_type_all(Itype);

        %%%Loop through dates and create a selection file for each DASAR and day
        create_folder_flag=true;  %Flag to check for output directory structure when a new year-site combo is started
        
        for Id=debug.Idasar_start:size(manual.ind.wgt,2)  %For each DASAR

            %keyboard
            fprintf('DASAR %s\n',DASAR_list{Id});


            tabs_DASAR=datenum(1970,1,1,-8,0,manual.ind.ctime(:,Id)); %-8 converts from UTC time (archive) to local time (GSI WAV)

            Iexist=find(~isnan(tabs_DASAR));
            tabs_DASAR=tabs_DASAR(Iexist);
            SIG_all=manual.ind.sigdb(Iexist,Id);
            SNR_all=manual.ind.stndb(Iexist,Id);
            fmin_all=manual.ind.flo(Iexist,Id);
            fmax_all=manual.ind.fhi(Iexist,Id);
            call_type=call_type_all(Iexist);

            temp=datevec(tabs_DASAR);
            temp(:,4:6)=0;
            tabs_start=datenum(temp);
            tabs_start_unique=unique(tabs_start);

           
            if strcmpi(data_file_type,'gsi')
                dir_want=sprintf('%s/Shell20%s_GSI_Data/S%s%sgsif/S%s%s%s0', ...
                    GSI_file_dir,year_want{Iyear},Site{Isite},year_want{Iyear}, ...
                    Site{Isite},year_want{Iyear},DASAR_strings(Id));
                %fs=head.Fs*(1+head.tdrift/86400);
                if exist(dir_want,'dir')~=7
                    dir_want(end)='1';
                    if exist(dir_want,'dir')~=7
                        disp('Data do not exist')
                        continue
                    end
                end
                file_names=dir([dir_want '/*gsi']);
                head=readgsif_header([dir_want filesep file_names(1).name]);
            else %WAV file
                 dir_want=sprintf('%s/Shell20%s_GSI_Data/S%s%sgsif/S%s%s%s0_WAV', ...
                    WAV_file_dir,year_want{Iyear},Site{Isite},year_want{Iyear}, ...
                    Site{Isite},year_want{Iyear},DASAR_strings(Id));
                %fs=head.Fs*(1+head.tdrift/86400);
                if exist(dir_want,'dir')~=7
                    dir_want(end-4)='1';
                    if exist(dir_want,'dir')~=7
                        disp('Data do not exist')
                        continue
                    end
                end
                file_names=dir([dir_want '/*WAV']);

                %%%Get appropirate clock drift
                head=get_GSI_head_info(head_info,year_want{Iyear},Site{Isite},DASAR_strings(Id));
               
            end
           
            %%%%Get list of qualified files/days for this Year/Site/DASAR
            %%%%combo
            Icountt=1;
            clear file_array
            for JJ=1:length(file_names)
                if contains(file_names(JJ).name(1),'.')
                    continue
                end
                file_array{Icountt}=file_names(JJ).name;
                Icountt=Icountt+1;
            end

            for Iday=debug.Iday_start:length(tabs_start_unique)
                disp(datestr(tabs_start_unique(Iday)));

                Ithis_day=find(tabs_start==tabs_start_unique(Iday));
                fprintf('On this day there are %i manual detections.\n',length(Ithis_day));
                manual.tabs=tabs_DASAR(Ithis_day);
                manual.tsec=(tabs_DASAR(Ithis_day)-tabs_start_unique(Iday))*24*3600;
                manual.tsec=manual.tsec*(1+head.tdrift/86400);
                manual.duration=manual.ind.duration(Iexist(Ithis_day),Id);
                manual.tmid=manual.tsec+0.5*manual.duration;
                manual.tend=manual.tsec+manual.duration;
                manual.SNR=SNR_all(Ithis_day);
                manual.sig=SIG_all(Ithis_day);
                manual.call_type=call_type(Ithis_day);
                manual.fmin=fmin_all(Ithis_day);
                manual.fmax=fmax_all(Ithis_day);


                %%%Create directory structure%%%%%%%
                if create_folder_flag
                    mydir=pwd;
                    cd(output_dir)
                    %eval(sprintf('!mkdir 20%s', year_want{Iyear}));
                    %cd(sprintf('20%s',year_want{Iyear}));
                    %eval(sprintf('!mkdir Site%s',Site{Isite}));
                    %cd(sprintf('Site%s',Site{Isite}));
                    !mkdir Bowhead_calls.dir
                    !mkdir Other_sounds.dir
                    cd(mydir)
                    create_folder_flag=false;
                end
               
                Ifile_want=find(contains(file_array, datestr(tabs_start_unique(Iday),30)));
                if isempty(Ifile_want)
                    fprintf('%s is not present in %s\n',datestr(tabs_start_unique(Iday),30),dir_want);
                    continue
                end

                %%%%%Import data%%%%%%%%
                fprintf('Reading %s\n',file_array{Ifile_want});
                tic
                if strcmpi(data_file_type,'gsi')
                    [x,~,head]=readgsi_omni_only([dir_want filesep file_array{Ifile_want}],0,debug.sec_to_load);
                    x=x-2^15;

                else 
                    head.tabs_start=datenum(file_array{Ifile_want}(8:22),'yyyymmddTHHMMSS'); %'dd-mmm-yyyy HH:MM:SS'
                    head.tabs_end=head.tabs_start+datenum(0,0,1,0,0,0);
                    [x,Fs]=audioread([dir_want filesep file_array{Ifile_want}],[1/1000 debug.sec_to_load]*head.Fs,'native');
                end
                toc
                %x=int16(x-2^15);
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%Process and save all manual detections%%%%%%%%%%%%%%

                disp('Starting manual spectrograms')

                %Igood_org=Ipass(Igood);  %Ensure that we skipp the NaN..
                %Itemp is associated with Igood, which is associated with tabs.
                %Igood_org associated with original manual.ind.* fields

                for I=1:length(Ithis_day)
                    tmid=manual.tmid(I);

                    if length(x)/head.Fs-tmid<=0
                        continue
                    end

                    titstr{1}=sprintf('Manual detection: Filename: %s, middle time %6.2f seconds, %i of %i',file_array{Ifile_want},tmid,I,length(Ithis_day));
                    titstr{2}=sprintf('Final SNR image, SNR: %6.2f, abs start: %s Call_type %i',manual.SNR(I),datestr(manual.tabs(I)),manual.call_type(I));

                    param.spec.plot_fmin=manual.fmin(I);
                    param.spec.plot_fmax=manual.fmax(I);
                    param.spec.duration=manual.duration(I);
                    [SNR_gram,FF,TT]=create_spectrogram_sample(x,head.Fs,tmid,file_len_sec,spectrogram_len_sec,param.spec,titstr);
                    %if isempty(SNR_gram)
                   %     keyboard
                   % end
                    if write_files
                        output_name=file_array{Ifile_want}(1:(end-4));
                        tabs_mid=tabs_DASAR(Ithis_day(I))+datenum(0,0,0,0,0,tmid-manual.tsec(I));
                        temp=datestr(tabs_mid,30);
                        output_name(17:end)=temp(10:end);
                        output_name=sprintf('%s_Type%i',output_name,manual.call_type(I));
                       % output_name=[output_dir filesep '20' year_want{Iyear} filesep 'Site' Site{Isite} filesep 'Bowhead_calls.dir' filesep output_name '.mat'];
                        output_name=[output_dir filesep  'Bowhead_calls.dir' filesep output_name '.mat'];

                        save(output_name,'SNR_gram','FF','TT');
                    end
                end %I in Igood

                disp('Finished manual spectrograms')

                %%%%%%%%%%%%%%%%%%%%%%%%%%%%Energy Detector.m%%%%%%%
                %%% Now generate false detections by a simple event
                %%% detector and check that they aren't whale calls.
                %
                %First, to save memory we will load data into chunks.
                %Chunks must be short enough that the background spectrum
                %does not evolve substantially over the interval
                %chunk_sample=60;


                Nchunks=floor(length(x)/(chunk_sample*head.Fs));
                for Ichunk=1:Nchunks
                    fprintf('Chunk %i of %i in DASAR %s in day %s\n',Ichunk,Nchunks,DASAR_list{Id},datestr(tabs_start_unique(Iday)));
                    Iss=1+(Ichunk-1)*chunk_sample*head.Fs;
                    x_chunk=x(Iss:(Iss-1+chunk_sample*head.Fs));


                    param.energy.debug=false;
                    [detect,debugg]=MultipleBandEnergyDetector(double(x_chunk),head.tabs_start+datenum(0,0,0,0,0,(Ichunk-1)*chunk_sample),param.energy);
                    detect.tstart=detect.tstart+(Ichunk-1)*chunk_sample;
                    detect.tend=detect.tend+(Ichunk-1)*chunk_sample;

                    fprintf('There are %i automated detections in chunk %i which covers %i minutes.\n',length(detect.tend),Ichunk,chunk_sample/60 )
                    %detect.tmid_abs=0.5*(detect.tstart_abs+detect.tend_abs);

                    %%%Determine whether any overlap exists between
                    %%%manual detections and these detections.
                    %%%make comparisons in terms of absolute times.

                    param.compare.ovlap=0.5;
                    [Score{Ichunk},Manual_index]=evaluate_overlap_between_manual_automated(manual.tsec,manual.tend,detect.tstart,detect.tend,param.compare.ovlap);

                    Idet_match=find(Score{Ichunk}(:,1)>0);
                    Manual_index_match=Manual_index(:);
                    Manual_index_match=unique(Manual_index_match(~isnan(Manual_index_match)));  %unique may not be needed

                    %%%%%Determine manual annotations that were missed by
                    %%%%%  automated detector.
                    Imiss=setdiff(min(Manual_index_match):max(Manual_index_match),Manual_index_match);
                    fprintf('%i out of %i (%6.2f percent) manual detections in this chunk missing from automated detections\n',length(Imiss),max(Manual_index_match),100*length(Imiss)/max(Manual_index_match))

                    sub_plot_debug_manual_automated_comparison;

                    %%%Sometimes multiple manual annotations overlap in
                    %%%time and are sharing a single automated detection.
                   
                    Idet_notWhale=find(isnan(Score{Ichunk}(:,1)));
                    fprintf('%i Automated detections, %i manual calls in this chunk.\n \t%i match with manual whale annotations, %i are thus not whale calls, and %i manual annotations are missed\n', ...
                            length(detect.tstart),max(Manual_index_match), ...
                            length(Idet_match),length(Idet_notWhale),length(Imiss))

                     %%%Created automated detections that are not whale calls%%%%%%%
                     param.spec.debug_plot=false;
                     for Idet=1:length(Idet_notWhale)

                         II=Idet_notWhale(Idet);
                         tmid=0.5*(detect.tstart(II)+detect.tend(II));
                         titstr{1}=sprintf('Non-whale detection Filename: %s, middle time %6.2f seconds, %i of %i',file_array{Ifile_want},tmid,Idet,length(Idet_notWhale));
                         titstr{2}=sprintf('Final SNR image, SNR: %6.2f, abs start: %s, score overlap: %6.4f', ...
                             detect.dB_RMS(II),datestr(detect.tstart_abs(II)),Score{Ichunk}(II));

                         param.spec.plot_fmin=detect.fmin(II);
                         param.spec.plot_fmax=detect.fmax(II);
                         param.spec.duration=detect.duration(II);

                         [SNR_gram,FF,TT]=create_spectrogram_sample(x,head.Fs,tmid,file_len_sec,spectrogram_len_sec,param.spec,titstr);

                         if write_files
                             output_name=file_array{Ifile_want}(1:(end-4));
                             tabs_mid=detect.tstart_abs(II)+datenum(0,0,0,0,0,tmid-detect.tstart(II));
                             temp=datestr(tabs_mid,30);
                             output_name(17:end)=temp(10:end);
                             output_name=sprintf('%s_Type%i',output_name,0);
                             %output_name=[output_dir filesep '20' year_want{Iyear} filesep 'Site' Site{Isite} filesep 'Other_sounds.dir' filesep output_name '.mat'];
                             output_name=[output_dir filesep  'Other_sounds.dir' filesep output_name '.mat'];

                             save(output_name,'SNR_gram','FF','TT');
                         end
                     end %Idet
                   
                fprintf('Finished Chunk %i of %i in DASAR %s in day %s\n',Ichunk,Nchunks,DASAR_list{Id},datestr(tabs_start_unique(Iday)));
                   
                end %Ichunk
                
                pause(1);
                %debug_plot=false;

            end %Iday
        end %Id
        fprintf('Finished exporting this site and year.... \n\n\n')
    end %Isite
end %Iyear
