%%%%%master_compute_manual_miss_rate.m%%%
%
%  Computes the single aggregate statistic referenced by the TODO in
%  paper/BowheadAI_v0.tex (Sec. "Training and Evaluation Datasets"):
%
%     the percentage of manual annotations that did NOT overlap any
%     automated (CFAR) transient detection, across every date/DASAR
%     combination listed in Table I of that paper.
%
%  This reruns only the detection + overlap-matching portion of
%  master_create_datasets_v2.m (MultipleBandEnergyDetector followed by
%  evaluate_overlap_between_manual_automated) -- it does NOT regenerate any
%  spectrogram images -- and aggregates the missed-annotation count over all
%  dates instead of reporting it per-chunk as the original script does.
%
%  IMPORTANT: this requires the RAW acoustic data (GSI_file_dir /
%  WAV_file_dir below), not just the manual archive or the already-built
%  Spectrogram_Image_Database. The saved spectrogram .mat files only record
%  automated detections that were actually processed into an image; a
%  manual annotation with zero overlapping automated detections never
%  produces a file anywhere, so its "miss" can only be recovered by rerunning
%  the CFAR detector against the original audio.
%
%  Usage: set the dataset-specific paths below to match your mounted
%  volumes, then run. Results (per date/DASAR counts and the overall
%  percentage) are printed and saved to manual_miss_rate_results.mat.

close all
clear
addpath .
warning off

%%%Computer specific information (mirrors master_create_datasets_v2.m)
[~,hostname]=system('hostname');
if contains(hostname,'ishmael')
    GSI_file_dir='/Volumes/Shared-1/Data/'; %mounted via Jonah3; has Shell20{08,10,12,14}_GSI_Data
    WAV_file_dir='/Volumes/Bowhead4/';
    Manual_record_files_dir='/Volumes/Thode_AI_Working_Disk/Bowhead_DL_Project/Shell_Manual_Results';
else
    GSI_file_dir='/Volumes/Shared/Data/';
    WAV_file_dir='/Volumes/Bowhead4/';
    Manual_record_files_dir='../../Shell_Manual_Results';
end

data_file_type='GSI'; %'GSI' or 'WAV'
sound_type='whale';  %filter manual results to bowhead whale call types only (wctype<=7)
DASAR_strings='ADG'; %DASARs used for the Table I training dataset

if strcmpi(data_file_type,'gsi')
    raw_data_dir=GSI_file_dir;
else
    raw_data_dir=WAV_file_dir;
end
if exist(raw_data_dir,'dir')~=7
    error(['Raw acoustic data volume not mounted: %s\nThis script needs the raw audio ' ...
        '(not the manual archive or the pre-built spectrogram database) to rerun the CFAR ' ...
        'detector and recompute the missed-annotation percentage. Mount that volume and rerun.'], ...
        raw_data_dir);
end
if exist(Manual_record_files_dir,'dir')~=7
    error('Manual_record_files_dir not present: %s',Manual_record_files_dir);
end

%%%Energy event detector parameters (identical to master_create_datasets_v2.m)
param.event.fmin=25; param.event.fmax=500; param.event.dB_threshold=5;
param.energy.eq_time=23.8;
param.energy.threshold=param.event.dB_threshold;
param.energy.TolTime=0.05;
param.energy.MinTime=0.1;
param.energy.MaxTime=10;
param.energy.Nfft=256;
param.energy.Fs=1000;
param.energy.ovlap=0.75;
param.energy.flo_det=param.event.fmin;
param.energy.fhi_det=param.event.fmax;
param.energy.burn_in_time=0.25;
param.energy.bandwidth=37;
param.energy.debug=0;
param.compare.ovlap=0.5; %Fraction of time overlap required to count as a 'hit'

chunk_sample=6*60*60-1; %seconds of data processed per chunk, to conserve RAM

%%%Table I of BowheadAI_v0.tex: year/site/date combinations used for the
%%%100,000-sample training dataset. Dates are 'mm/dd' in the recording year.
table1(1)=struct('year','08','site','3','dates',{{'08/28','09/06','09/13','09/21','09/29'}});
table1(2)=struct('year','08','site','5','dates',{{'08/21','08/28','09/06','09/13','09/21','09/29'}});
table1(3)=struct('year','10','site','3','dates',{{'08/15','08/21','08/29','09/05','09/13','09/27'}});
table1(4)=struct('year','10','site','5','dates',{{'08/15','08/21','08/29','09/05','09/13'}});
table1(5)=struct('year','12','site','3','dates',{{'08/25','09/01','09/07','09/13','09/18','09/23','09/29','10/05'}});
table1(6)=struct('year','12','site','5','dates',{{'08/25','09/01','09/13','09/18','09/23','09/29'}});
table1(7)=struct('year','14','site','3','dates',{{'08/18','08/28','09/01','09/17','09/27'}});
table1(8)=struct('year','14','site','5','dates',{{'08/18','08/28','09/01','09/17','09/27'}});

%%%Running totals, aggregated across every date/site/year/DASAR combination
total_manual_matched=0;
total_manual_missed=0;
per_daterow=struct([]);

for Irow=1:length(table1)
    year_want=table1(Irow).year;
    Site=table1(Irow).site;

    for I=1:length(DASAR_strings)
        DASAR_list{I}=sprintf('S%s%s%s0',Site,year_want,DASAR_strings(I)); %#ok<AGROW>
    end

    %%%Load (or build a cached copy of) the manual archive, exactly as
    %%%master_create_datasets_v2.m does.
    fname=sprintf('%s%s20%s%sAllSite%s_20%s_manual_archive.txt', ...
        Manual_record_files_dir,filesep,year_want,filesep,Site,year_want);
    fname_mat=sprintf('%s%s20%s%sAllSite%s_20%s_%s_manual_archive.mat', ...
        Manual_record_files_dir,filesep,year_want,filesep,Site,year_want,DASAR_strings);
    if ~exist(fname_mat,'file')
        fprintf('Reprocessing manual archive for Site %s, 20%s...\n',Site,year_want);
        [ind,localized]=read_tsv_archive(fname,0,Inf,DASAR_list);
        save(fname_mat,'ind','localized');
        clear ind localized
    end
    manual_archive=load(fname_mat);

    if strcmpi(sound_type,'whale')
        Itype=find(manual_archive.localized.wctype<=7);
    else
        error('Only sound_type=''whale'' is wired up in this script.');
    end
    Itype=Itype(Itype<=size(manual_archive.ind.wgt,1));
    fieldnamess=fieldnames(manual_archive.ind);
    for JJ=1:length(fieldnamess)
        manual_archive.ind.(fieldnamess{JJ})=manual_archive.ind.(fieldnamess{JJ})(Itype,:);
    end
    call_type_all=manual_archive.localized.wctype(Itype);

    for Id=1:length(DASAR_list)
        tabs_DASAR=datenum(1970,1,1,-8,0,manual_archive.ind.ctime(:,Id)); %UTC->local, as in original pipeline
        Iexist=find(~isnan(tabs_DASAR));
        tabs_DASAR=tabs_DASAR(Iexist);
        call_type=call_type_all(Iexist);
        duration_all=manual_archive.ind.duration(Iexist,Id);

        temp=datevec(tabs_DASAR);
        temp(:,4:6)=0;
        tabs_start=datenum(temp);

        %%%Find the raw data directory/files for this DASAR
        if strcmpi(data_file_type,'gsi')
            dir_want=sprintf('%s/Shell20%s_GSI_Data/S%s%sgsif/S%s%s%s0', ...
                GSI_file_dir,year_want,Site,year_want,Site,year_want,DASAR_strings(Id));
            if exist(dir_want,'dir')~=7
                dir_want(end)='1';
            end
        else
            dir_want=sprintf('%s/Shell20%s_GSI_Data/S%s%sgsif/S%s%s%s0_WAV', ...
                WAV_file_dir,year_want,Site,year_want,Site,year_want,DASAR_strings(Id));
            if exist(dir_want,'dir')~=7
                dir_want(end-4)='1';
            end
        end
        if exist(dir_want,'dir')~=7
            fprintf('Skipping DASAR %s: raw data directory not found (%s)\n',DASAR_list{Id},dir_want);
            continue
        end

        if strcmpi(data_file_type,'gsi')
            file_names=dir([dir_want '/*gsi']);
        else
            file_names=dir([dir_want '/*WAV']);
        end
        file_array={};
        for JJ=1:length(file_names)
            if contains(file_names(JJ).name(1),'.')
                continue
            end
            file_array{end+1}=file_names(JJ).name; %#ok<AGROW>
        end

        %%%Restrict to the Table I dates for this year/site row only
        for Idate=1:length(table1(Irow).dates)
            date_str=table1(Irow).dates{Idate}; %'mm/dd'
            day_datenum=datenum(sprintf('20%s/%s',year_want,date_str),'yyyy/mm/dd');

            Ithis_day=find(tabs_start==day_datenum);
            if length(Ithis_day)<3
                fprintf('%s Site %s DASAR %s: fewer than 3 manual detections, skipping (matches original pipeline behavior).\n', ...
                    datestr(day_datenum,'mm/dd/yyyy'),Site,DASAR_strings(Id));
                continue
            end

            Ifile_want=find(contains(file_array,datestr(day_datenum,30)));
            if isempty(Ifile_want)
                fprintf('%s Site %s DASAR %s: raw data file not found, skipping.\n', ...
                    datestr(day_datenum,'mm/dd/yyyy'),Site,DASAR_strings(Id));
                continue
            end

            fprintf('Processing %s Site %s DASAR %s (%s)...\n', ...
                datestr(day_datenum,'mm/dd/yyyy'),Site,DASAR_strings(Id),file_array{Ifile_want});

            manual=struct();
            manual.tsec=(tabs_DASAR(Ithis_day)-day_datenum)*24*3600;
            manual.duration=duration_all(Ithis_day);
            manual.tend=manual.tsec+manual.duration;

            if strcmpi(data_file_type,'gsi')
                [x,~,head]=readgsi_omni_only([dir_want filesep file_array{Ifile_want}],0,Inf);
                x=x-2^15;
            else
                head.tabs_start=datenum(file_array{Ifile_want}(8:22),'yyyymmddTHHMMSS');
                [x,~]=audioread([dir_want filesep file_array{Ifile_want}],'native');
            end
            manual.tsec=manual.tsec*(1+head.tdrift/86400);
            manual.tend=manual.tend*(1+head.tdrift/86400);

            day_manual_matched=0;
            day_manual_missed=0;

            Nchunks=max(1,floor(max(size(x))/(chunk_sample*head.Fs)));
            for Ichunk=1:Nchunks
                Iss=1+(Ichunk-1)*chunk_sample*head.Fs;
                x_chunk=x(Iss:min(end,Iss-1+chunk_sample*head.Fs),1);

                [detect,~]=MultipleBandEnergyDetector(double(x_chunk), ...
                    head.tabs_start+datenum(0,0,0,0,0,(Ichunk-1)*chunk_sample),param.energy);
                detect.tstart=detect.tstart+(Ichunk-1)*chunk_sample;
                detect.tend=detect.tend+(Ichunk-1)*chunk_sample;

                %%%Manual detections whose midpoint falls in this chunk's time window
                chunk_tmin=(Ichunk-1)*chunk_sample;
                chunk_tmax=Ichunk*chunk_sample;
                Ichunk_manual=find(manual.tsec>=chunk_tmin & manual.tsec<chunk_tmax);
                if isempty(Ichunk_manual) || isempty(detect.tstart)
                    continue
                end

                [~,Manual_index]=evaluate_overlap_between_manual_automated( ...
                    manual.tsec(Ichunk_manual),manual.tend(Ichunk_manual), ...
                    detect.tstart,detect.tend,param.compare.ovlap);

                Manual_index_match=unique(Manual_index(~isnan(Manual_index)));
                n_matched=length(Manual_index_match);
                n_missed=length(Ichunk_manual)-n_matched;

                day_manual_matched=day_manual_matched+n_matched;
                day_manual_missed=day_manual_missed+n_missed;
            end

            fprintf('  -> %i matched, %i missed (of %i manual detections)\n', ...
                day_manual_matched,day_manual_missed,day_manual_matched+day_manual_missed);

            total_manual_matched=total_manual_matched+day_manual_matched;
            total_manual_missed=total_manual_missed+day_manual_missed;

            per_daterow(end+1)=struct('year',year_want,'site',Site,'dasar',DASAR_strings(Id), ...
                'date',date_str,'matched',day_manual_matched,'missed',day_manual_missed); %#ok<SAGROW>
        end %Idate
    end %Id
end %Irow

total_manual=total_manual_matched+total_manual_missed;
pct_missed=100*total_manual_missed/total_manual;

fprintf('\n=====================================================\n');
fprintf('TOTAL across all Table I dates/sites/DASARs: %i manual annotations\n',total_manual);
fprintf('  %i matched by an automated (CFAR) detection\n',total_manual_matched);
fprintf('  %i missed by the automated detector (%6.2f%%)\n',total_manual_missed,pct_missed);
fprintf('=====================================================\n');

save('manual_miss_rate_results.mat','per_daterow','total_manual_matched','total_manual_missed','total_manual','pct_missed');
