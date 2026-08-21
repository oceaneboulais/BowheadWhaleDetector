%%%%%master_create_datasets_v2.m%%%
%%%   RUN THIS SCRIPT FIRST WHEN CREATING A DATABASE FROM SCRATCH
%
%  This script imports raw acoustic data and generates normalized
%  spectrograms (units of SNR dB) that it saves as small *.mat files in uint8 format to save
%  space (usually with a multiplicative factor to allow a resolution of 0.1
%  dB
%
%  It will also estimate the bearing, NTV, and KE of the signal if the 'GSI' file type
%  is chosen.
%
%  The script first uploads manual annotation logs and then uses a simple CFAR energy
%  detector (with narrow bands) to flag additional samples that don't
%  overlap the manual identifications.  It uses the manual logs to identify
%  whale calls
%
%  After this script run master_index_database.m, and then
%  master_assemble_unsupervised_database.

close all
clear
addpath .
warning off

%%%Computer specific information
[~,hostname]=system('hostname');
if contains(hostname,'ishmael')
    GSI_file_dir='/Volumes/Shared-2/Data/';
    code_dir='/Users/thode/Desktop/DeepLearningBowhead/Software_repo/matlab';
    WAV_file_dir='/Volumes/Bowhead4/';
    Manual_record_files_dir='../../Shell_Manual_Results';

else
    %GSI_file_dir='/Volumes/Bowhead4/Shell_AllChannel_Demo/';
    GSI_file_dir='/Volumes/Shared/Data/';
    code_dir='/Users/thode/Projects/Greeneridge_bowhead_detection/DeepLearningNPRB_Project/Software/matlab';
    WAV_file_dir='/Volumes/Bowhead4/';
    Manual_record_files_dir='../../Shell_Manual_Results';

end
cd(code_dir)

!rm diary_output.txt
diary diary_output.txt


data_file_type='GSI'; %'GSI' or 'WAV' for raw audio data source
output_dir='../../Spectrogram_Image_Database.dir';  %%Where to save the database
eval(sprintf('!mkdir %s',output_dir));

%%%%%Fundamental parameters to change
param.spec.compute_azimuth=true;  %If true, compute the bearing of signals.  Slows down processing by a factor of 10.
file_len_sec=5; %length of final file clip (includes noise estimate)
spectrogram_len_sec=3; %length of final spectrogram clip. (data used for noise removed)
param.event.fmin = 25; %Hz
param.event.fmax = 500; %Hz
sound_type='whale'; %whale or seal or 'all_biologics': for filtering manual results
DASAR_strings='ADG';  %%What DASARs to sample data from.  Can be non-contiguous order: 'ACG';
%DASAR_strings='G';  %%What DASARs to sample data from.  Can be non-contiguous order: 'ACG';

%Event detector fundamental parameters
param.event.dB_threshold = 5; % dB threshold above mean for detection.  Higher value means fewer events selected.
param.energy.eq_time = 23.8;  %%Original choice

year_want={'08','09','10','11','12','13','14'};  %What years to process
Site={'2','3','4','5'};  %What sites to process

year_want={'08','10','12','14'};
Site={'5'};

%year_want={'08'};
%Site={'5'};

%%%%%%%%%Other parameters useful for debugging%%%%%%%%%%%%

max_files_per_directory=25000;  %%Maximum files allowed in an individual folder.  Lower numbers allow for easier manipulation using UNIX commands.
param.spec.debug_plot=false;  %Plot spectrograms as we go along
write_files=true;      %If true write the database

debug.Iday_start=1; %Set to one to process all days
debug.Idasar_start=1;  %Set to one to process all DASARs
debug.sec_to_load=Inf;  %Inf to load entire file at once
chunk_sample=6*60*60-1;  %%%% Seconds of data to process to spectrogram to conserve RAM memory.

%%%Parameters that rarely need to be changed at this point.

% scaling factors when converting images to uint8
param.event.image_scale_factor = 5;  % factor to multiply SNR by for saving as unit8 image
param.spec.NTV=100;
param.spec.KE_offset=20;
param.spec.KE_scale=6;
param.spec.Polar=1;
param.spec.image_scale_factor = param.event.image_scale_factor;

param.spec.Nfft=256;
param.spec.ovlap=0.9;
param.spec.fmin = param.event.fmin;
param.spec.fmax = param.event.fmax;
%param.spec.final_dims=8*[16 10];

%%%Set up energy event detector.  Example for bowhead whale analysis that monitors between 25 and 350 Hz,
%  using a set of detectors with 37 Hz bandwidth.
K=1;
%param.energy.eq_time=3;   param.energy_desc{K}='Equalization time (s): should be roughly twice the duration of signal of interest';K=K+1;
param.energy.threshold=param.event.dB_threshold;  param.energy_desc{K}='Threshold in dB to accept a detection';K=K+1;
param.energy.TolTime=0.05;  param.energy_desc{K}='Minimum time in seconds that must elapse for two detections to be listed as separate';K=K+1;
param.energy.MinTime=0.1;     param.energy_desc{K}='Minimum time in seconds a required for a detection to be logged';K=K+1;
param.energy.MaxTime=10;     param.energy_desc{K}= 'Maximum time in seconds a detection is permitted to have';K=K+1;

param.energy.Nfft=256;
param.energy.Fs =1000;
param.energy.ovlap = 0.75;
param.energy.flo_det=param.event.fmin;
param.energy.fhi_det=param.event.fmax;
param.energy.burn_in_time=0.25;  %Time in minutes
param.energy.bandwidth=37;     param.energy_desc{K}='Bandwidth of sub-detector in kHz';K=K+1;
param.energy.debug=0;       param.energy_desc{K}= '0: do not write out debug information. 1:  SEL output.  2:  equalized background noise. 3: SNR.';K=K+1;

if param.spec.compute_azimuth & contains(data_file_type,'WAV')
    error('Cannot compute bearings using single-channel WAV files.');
end

if strcmpi(data_file_type,'gsi')
    if exist(GSI_file_dir,'dir')==0
        error('GSI_file_dir not present')
    end
elseif exist(WAV_file_dir,'dir')==0
    error('WAV_file_dir not present')

elseif exist(WAV_file_dir,'dir')==7
    head_info=load('GSI_header_table.mat');
end

sample_count.manual=0;
mydir=pwd;


%%%Cycle though years and stites
for Iyear=1:length(year_want)
    for Isite=1:length(Site)
        cd(mydir)
        for I=debug.Idasar_start:length(DASAR_strings)
            % DASAR_list{I}=['S314' strr(I) '0'];
            DASAR_list{I}=sprintf('S%s%s%s0',Site{Isite},year_want{Iyear},DASAR_strings(I));
        end

        %%%%%%%Import manual analyst archive, and if needed, repackage as
        %%%%%%%convenient MAT file for future access.
        ctmin=0;
        ctmax=Inf;
        fname=sprintf('%s%s20%s%sAllSite%s_20%s_manual_archive.txt', ...
            Manual_record_files_dir,filesep,year_want{Iyear},filesep,Site{Isite},year_want{Iyear});

        fname_mat=sprintf('%s%s20%s%sAllSite%s_20%s_%s_manual_archive.mat', ...
            Manual_record_files_dir,filesep,year_want{Iyear},filesep,Site{Isite},year_want{Iyear},DASAR_strings);

        if ~exist(fname_mat,'file')  %%%Note we will use this only if DASAR_strings is the same as well
            disp('Reprocessing manual archive...')
            [ind,localized]=read_tsv_archive(fname,ctmin,ctmax,DASAR_list);  %%Note that this only downloads the DASARs requested.
            save(fname_mat,'ind','localized');
            clear ind localized
        end
        manual=load(fname_mat);

        %%%%%Restrict manual database to certain types of calls...%%%%%
        if strcmpi(sound_type,'whale')
            Itype=find(manual.localized.wctype<=7);  %bowhead whale calls only
        elseif strcmpi(sound_type,'seal')
            Itype=find(manual.localized.wctype==8 | manual.localized.wctype==9);  %seal and walrus only
        elseif strcmpi(sound_type,'all_biologics')
            Itype=find(manual.localized.wctype>=1 & manual.localized.wctype<=9);  %seal and walrus only

        end

        if isempty(Itype)
            fprintf('No %s here\n',sound_type);
            continue
        end

        fieldnamess=fieldnames(manual.ind);
        %manual.ind.duration=manual.ind.duration(Itype,:);
        %%%Sometimes the localized object is shorter than the ind
        %%%object.  So this is a safety check
        Itype=Itype(Itype<=size(manual.ind.wgt,1));

        for JJ=1:length(fieldnamess)
            manual.ind.(fieldnamess{JJ})=manual.ind.(fieldnamess{JJ})(Itype,:);
        end
        call_type_all=manual.localized.wctype;
        call_type_all=call_type_all(Itype);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%Start spectrogram creation loop 
        %%%Loop through dates and create a selection file for each DASAR and day
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        for Id=debug.Idasar_start:length(DASAR_list)  %For each DASAR desired
            % Note that length(DASAR_list) and size(ind.ctime) should be
            % the same, so Id is the correct index to access the manual
            % data.
            cd(mydir)
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
            tabs_start_unique=unique(tabs_start);  %%%The individual days present in the manual data for this DASAR/Site/year.

            %%%%%%%Download raw acoustic data%%%%%%%%%%%%%%
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

                %%%Get appropirate clock drift. Requires
                %%%master_create_tdrift_table.m be run first.
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

            %%%For each unique day for this DASAR/Site/year, create
            %%%database folder hierarchy.
            %%%    Cycle through day
            for Iday=debug.Iday_start:length(tabs_start_unique)
                disp(datestr(tabs_start_unique(Iday)));
                cd(mydir)

                Ithis_day=find(tabs_start==tabs_start_unique(Iday));
                fprintf('On this day there are %i manual detections.\n',length(Ithis_day));
               
                if length(Ithis_day)<3
                    disp('Not going to process this day as no manual detections \n');
                    continue
                end
                %%%Create directory structures and count files in current
                %%%directory
                %if Iday==debug.Iday_start && Id==debug.Idasar_start
                %mydir=pwd;
                cd(output_dir)
                eval(sprintf('!mkdir 20%s', year_want{Iyear}));
                cd(sprintf('20%s',year_want{Iyear}));
                eval(sprintf('!mkdir Site%s',Site{Isite}));
                cd(sprintf('Site%s',Site{Isite}));
                eval(sprintf('!mkdir Day_%s',datestr(tabs_start_unique(Iday),30)));
                cd(sprintf('Day_%s',datestr(tabs_start_unique(Iday),30)));


                !mkdir Manually_selected_bowhead_calls.dir
                !mkdir Manually_selected_bowhead_calls.dir/D1.dir

                !mkdir Event_sounds.dir
                !mkdir Event_sounds.dir/D1.dir


                cd('Event_sounds.dir/D1.dir')
                current_save_dir{1}=pwd;
                current_file_count(1)=length(dir('*mat'));
                cd ../..

                cd('Manually_selected_bowhead_calls.dir/D1.dir')
                current_save_dir{2}=pwd;
                current_file_count(2)=length(dir('*mat'));

                %%%Create a 'manual' variable that stored info about
                %%%detections for this specific day
                manual.tsec=(tabs_DASAR(Ithis_day)-tabs_start_unique(Iday))*24*3600;
                Ithis_day=Ithis_day(manual.tsec<debug.sec_to_load); %%In case only loaded part of file
                manual.tsec=(tabs_DASAR(Ithis_day)-tabs_start_unique(Iday))*24*3600;
                manual.tabs=tabs_DASAR(Ithis_day);
                manual.tsec=manual.tsec*(1+head.tdrift/86400);
               
                manual.duration=manual.ind.duration(Iexist(Ithis_day),Id);
                manual.tmid=manual.tsec+0.5*manual.duration;
                manual.tend=manual.tsec+manual.duration;
                manual.SNR=SNR_all(Ithis_day);
                manual.sig=SIG_all(Ithis_day);
                manual.call_type=call_type(Ithis_day);
                manual.fmin=fmin_all(Ithis_day);
                manual.fmax=fmax_all(Ithis_day);

                %%%%%Import data from specific file for this specific day. %%%%%%%%
                Ifile_want=find(contains(file_array, datestr(tabs_start_unique(Iday),30)));
                if isempty(Ifile_want)
                    fprintf('%s is not present in %s\n',datestr(tabs_start_unique(Iday),30),dir_want);
                    continue
                end
                fprintf('Reading %s\n',file_array{Ifile_want});
                tic
                if strcmpi(data_file_type,'gsi')
                    if param.spec.compute_azimuth  %Read all channels
                        [x,~,head]=readgsi([dir_want filesep file_array{Ifile_want}],0,debug.sec_to_load,'native');
                        x=x';
                        param.spec.brefa=head.brefa;
                    else
                        [x,~,head]=readgsi_omni_only([dir_want filesep file_array{Ifile_want}],0,debug.sec_to_load);
                    end
                    x=x-2^15;

                else
                    head.tabs_start=datenum(file_array{Ifile_want}(8:22),'yyyymmddTHHMMSS'); %'dd-mmm-yyyy HH:MM:SS'
                    head.tabs_end=head.tabs_start+datenum(0,0,1,0,0,0);
                    [x,Fs]=audioread([dir_want filesep file_array{Ifile_want}],[1/1000 debug.sec_to_load]*head.Fs,'native');
                end
                toc
                %x=int16(x-2^15);


                 %%%%%%%%%%%%%%%%%Energy Detector.m%%%%%%%%%%%%%%%%%%%
                %%% Now generate false detections by a simple event
                %%% detector and check that they aren't whale calls.
                %
                %First, to save memory we will load data as chunks.


                Nchunks=floor(max(size(x))/(chunk_sample*head.Fs));
                for Ichunk=1:Nchunks
                    fprintf('Chunk %i of %i in DASAR %s in day %s\n',Ichunk,Nchunks,DASAR_list{Id},datestr(tabs_start_unique(Iday)));
                    Iss=1+(Ichunk-1)*chunk_sample*head.Fs;
                    x_chunk=x(Iss:(Iss-1+chunk_sample*head.Fs),1);

                    param.energy.debug=false;
                    %Run energy event detector
                    [detect,debugg]=MultipleBandEnergyDetector(double(x_chunk),head.tabs_start+datenum(0,0,0,0,0,(Ichunk-1)*chunk_sample),param.energy);
                    detect.tstart=detect.tstart+(Ichunk-1)*chunk_sample;
                    detect.tend=detect.tend+(Ichunk-1)*chunk_sample;
                    detect.tpeak=detect.tpeak+(Ichunk-1)*chunk_sample;

                    fprintf('There are %i automated detections in chunk %i which covers %i minutes.\n',length(detect.tend),Ichunk,chunk_sample/60 )
                    %detect.tmid_abs=0.5*(detect.tstart_abs+detect.tend_abs);

                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    %%%%%%Determine whether any overlap exists between
                    %%%manual detections and these detections.

                    param.compare.ovlap=0.5; %Fraction of time overlap required to count as 'hit'
                    [Score{Ichunk},Manual_index]=evaluate_overlap_between_manual_automated(manual.tsec,manual.tend,detect.tstart,detect.tend,param.compare.ovlap);

                    %%%Remove detections that overlap manual detections (to
                    %%%   prevent duplicate spectrograms)
                    Iwhale_match=find(Score{Ichunk}(:,1)>0);
                    Manual_index_match=Manual_index(:);
                    Manual_index_match=unique(Manual_index_match(~isnan(Manual_index_match)));  %unique may not be needed

                    %%%%%Determine manual annotations that were missed by
                    %%%%%  automated detector.
                    Imiss=setdiff(min(Manual_index_match):max(Manual_index_match),Manual_index_match);
                    fprintf('%i out of %i (%6.2f percent) manual detections in this chunk missing from automated detections\n',length(Imiss),max(Manual_index_match),100*length(Imiss)/max(Manual_index_match))

                    %%%If debug option chosen, plot spectrograms of missed
                    sub_plot_debug_manual_automated_comparison;

                    %%%Sometimes multiple manual annotations overlap in
                    %%%time and are sharing a single automated detection.

                   Idet_notWhale=find(isnan(Score{Ichunk}(:,1)));
                    fprintf('%i Automated detections, %i manual calls in this chunk.\n \t%i match with manual whale annotations, %i are thus not whale calls, and %i manual annotations are missed\n', ...
                        length(detect.tstart),max(Manual_index_match), ...
                        length(Iwhale_match),length(Idet_notWhale),length(Imiss))

                    %%%%%%%Create spectrograms of all sounds%%%%%%%
                    %param.spec.debug_plot=false;
                    for Idet=1:length(detect.tstart)
                        if rem(Idet,500)==0, fprintf('%3.2f percent done\n',100*Idet/length(detect.tstart));end
                        %II=Idet_notWhale(Idet);
                        II=(Idet);
                        %tmid=0.5*(detect.tstart(II)+detect.tend(II));
                        tmid=detect.tpeak(II);
                        titstr{1}=sprintf('Detection Filename: %s, middle time %6.2f seconds, %i of %i',file_array{Ifile_want},tmid,Idet,length(detect.tstart));
                        titstr{2}=sprintf('Final SNR image, SNR: %6.2f, abs start: %s, score overlap: %6.4f', ...
                            detect.dB_RMS(II),datestr(detect.tstart_abs(II)),Score{Ichunk}(II));

                        param.spec.plot_fmin=detect.fmin(II);
                        param.spec.plot_fmax=detect.fmax(II);
                        param.spec.duration=detect.duration(II);
                        param.spec.debug_max_tmid=1*60;
                       % if Idet<param.spec.debug_count
                        %    param.spec.debug_plot=true;
                        %else
                        %    param.spec.debug_plot=false;
                        % end

                        %%%Identify if a whale call
                        if(~isnan(Score{Ichunk}(II,1)))
                            mytype=manual.call_type(Manual_index(II,1));
                            Idir=2;
                            param.spec.debug_plot=false;
                        else
                            Idir=1;
                            mytype=0;
                            param.spec.debug_plot=false;
                        end

                        [SNR_gram,VS_metrics,FF,TT,bearing]=create_spectrogram_sample(x,head.Fs,tmid,file_len_sec,spectrogram_len_sec,param.spec,titstr);
                        
                        %%%Sometimes file length not right
                        if isempty(SNR_gram)
                            continue
                        end

                        %%%Extract peak frequency and time%%%
                        [temp]=extract_features_from_SNRgram(TT(2)-TT(1),FF(2)-FF(1),SNR_gram);
                        PeakFrequency=temp.Fmax;
                        PeakTime=temp.Tmax;

                        
                        if  param.spec.debug_plot & mytype>0
                            Iman=Manual_index(II,1);
                            fprintf('Call %i of %i, Type %i, Score %6.2f\n',Iman,length(manual.tend),mytype,Score{Ichunk}(II,1));

                            subplot(2,1,1)
                            rectangle('Position',[file_len_sec/2-(tmid-manual.tsec(Iman)) manual.fmin(Iman) manual.tend(Iman)-manual.tsec(Iman) manual.fmax(Iman)-manual.fmin(Iman)],'edgecolor','w','Linewidth',1)

                            subplot(2,1,2)
                            rectangle('Position',[spectrogram_len_sec/2-(tmid-manual.tsec(Iman)) manual.fmin(Iman) manual.tend(Iman)-manual.tsec(Iman) manual.fmax(Iman)-manual.fmin(Iman)],'edgecolor','w','Linewidth',1)

                            pause;
                            close
                        end
                        %pause;  %Uncomment when viewing all detections,not just whales
                        %
                        % close

                        if isempty(SNR_gram)
                            continue
                        end
                        if write_files
                            output_name=file_array{Ifile_want}(1:(end-4));
                            tabs_tstartt=detect.tstart_abs(II)+datenum(0,0,0,0,0,tmid-detect.tstart(II));
                            tabs_tmid=detect.tstart_abs(II)+datenum(0,0,0,0,0,tmid);
                            temp=datestr(tabs_tstartt,30);
                            output_name(17:end)=temp(10:end);

                            
                            output_name=sprintf('%s_Type%i',output_name,mytype);

                            
                            %%%Save spectrograms to current subdirectory and
                            %%%start a new subdirectory if too many files in
                            %%%current one.
                 
                            
                            while current_file_count(Idir)>=max_files_per_directory
                                Idump=str2double(current_save_dir{Idir}(end-4));
                                current_save_dir{Idir}(end-4)=int2str(Idump+1);
                                if ~exist(current_save_dir{Idir},'dir')
                                    eval(sprintf('!mkdir %s',current_save_dir{Idir}))
                                end
                                cd(current_save_dir{Idir});
                                fprintf('Changing to %s\n',current_save_dir{Idir});
                                current_file_count(Idir)=length(dir('*.mat'));

                            end
                            current_file_count(Idir)=current_file_count(Idir)+1;
                            dF=FF(2)-FF(1);dT=TT(2)-TT(1);
                            NTV_gram=VS_metrics{2};KEtoPE_gram=VS_metrics{3};Polar_gram=VS_metrics{4};
                            save([current_save_dir{Idir} filesep output_name],'SNR_gram','NTV_gram','KEtoPE_gram','Polar_gram', ...
                                'dF','dT','bearing','tabs_tstartt','PeakFrequency','PeakTime');

                        end
                    end %Idet
                    fprintf('Finished Chunk %i of %i in DASAR %s in day %s\n',Ichunk,Nchunks,DASAR_list{Id},datestr(tabs_start_unique(Iday)));
                end %Ichunk
                pause(1);
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%Process and save all manual detections for this day%%%%%%%%%%%%%%

%                 disp('Starting manual spectrograms')
%                 %Igood_org=Ipass(Igood);  %Ensure that we skipp the NaN..
%                 %Itemp is associated with Igood, which is associated with tabs.
%                 %Igood_org associated with original manual.ind.* fields
% 
%                 for I=1:length(Ithis_day)
%                     tmid=manual.tmid(I);
% 
%                     if max(size(x))/head.Fs-tmid<=0
%                         continue
%                     end
% 
%                     titstr{1}=sprintf('Manual detection: Filename: %s, middle time %6.2f seconds, %i of %i',file_array{Ifile_want},tmid,I,length(Ithis_day));
%                     titstr{2}=sprintf('Final SNR image, SNR: %6.2f, abs start: %s Call_type %i',manual.SNR(I),datestr(manual.tabs(I)),manual.call_type(I));
% 
%                     param.spec.plot_fmin=manual.fmin(I);
%                     param.spec.plot_fmax=manual.fmax(I);
%                     param.spec.duration=manual.duration(I);
% 
% %                     if I<5
% %                         param.spec.debug_plot=true;
% %                     else
% %                         param.spec.debug_plot=false;
% %                     end
%                     [SNR_gram,VS_metrics,FF,TT,bearing]=create_spectrogram_sample(x,head.Fs,tmid,file_len_sec,spectrogram_len_sec,param.spec,titstr);
%                     if isempty(SNR_gram)
%                         disp('SNR_gram is empty')
%                         continue
%                     end
% 
%                     if write_files
%                         output_name=file_array{Ifile_want}(1:(end-4));
%                         tabs_tstartt=tabs_DASAR(Ithis_day(I))+datenum(0,0,0,0,0,tmid-manual.tsec(I));
%                         temp=datestr(tabs_tstartt,30);
%                         output_name(17:end)=temp(10:end);
%                         output_name=sprintf('%s_Type%i',output_name,manual.call_type(I));
% 
%                         %%%Save spectrograms to current subdirectory and
%                         %%%start a new subdirectory if too many files in
%                         %%%current one.
%                         while current_file_count>=max_files_per_directory
%                             Idump=str2double(current_save_dir(end-4));
%                             current_save_dir(end-4)=int2str(Idump+1);
%                             if ~exist(current_save_dir,'dir')
%                                 eval(sprintf('!mkdir %s',current_save_dir))
%                             end
%                             cd(current_save_dir);
%                             fprintf('Changing to %s\n',current_save_dir);
% 
%                             current_file_count=length(dir('*.mat'));
% 
%                         end
%                         if exist([output_name '.mat'],"file")>0  %if file already exists
%                             continue
%                         end
%                         current_file_count=current_file_count+1;
%                         dF=FF(2)-FF(1);dT=TT(2)-TT(1);
%                         NTV_gram=VS_metrics{2};KEtoPE_gram=VS_metrics{3};Polar_gram=VS_metrics{4};
%                         save(output_name,'SNR_gram','NTV_gram','KEtoPE_gram','Polar_gram','dF','dT','bearing','tabs_tstartt','param');
% 
%                         %database{Iyear,Isite,Id}.tabs=[]
% 
%                     end
%                     if current_file_count~=length(dir('*.mat'))
%                         keyboard
%                     end
%                 end %I in Igood
% 
%                 disp('Finished  spectrograms')

                %%%Update current file count in current subdirectory
               % if write_files
                %    cd ../../Event_sounds.dir/D1.dir
                %    current_save_dir=pwd;
                %    current_file_count=length(dir('*mat'));
                %end

               
            end %Iday
        end %Id
        fprintf('Finished exporting this site and year.... \n\n\n')
    end %Isite
end %Iyear
cd(mydir)