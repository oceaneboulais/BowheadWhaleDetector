
%function plot_manual_detection_allDASARs(manual_logs.manual_data{Isite,Iyear}.ind,time_zone_offset,head_info,Ibest)
function plot_manual_detection_allDASARs(fname,GSI_file_dir,ind,time_zone_offset,head_info,Ibest)

file_len_sec=5; %length of final file clip (includes noise estimate)
spectrogram_len_sec=3; %length of final spectrogram clip. (data used for noise removed)

param.spec.compute_azimuth=false;  %If true, compute the bearing of signals.  Slows down processing by a factor of 10.
param.spec.fmin = 25; %Hz
param.spec.fmax = 500; %Hz

param.spec.Nfft=256;
param.spec.ovlap=0.9;
param.spec.image_scale_factor = 5;  % factor to multiply SNR by for saving as unit8 image
param.spec.debug_plot=false;


spectrogram_window_length=file_len_sec; %sec
% [~,hostname]=system('hostname');
% if contains(hostname,'ishmael')
%     GSI_file_dir='~/mnt/jonah3/Shared/Data';
% else
%     GSI_file_dir='/Volumes/Shared/Data/';
% end


Isite=(fname(2));
Iyear=(fname(3:4));
Iletter=(fname(5));

head_table = get_GSI_head_info(head_info, Iyear, Isite, Iletter);

Nel=size(ind.wgt,2);


file_want=fname(1:22);
file_want(17:end)='0';
file_want=[file_want '.gsi'];



strr='ABCDEFG';

figure
for I=1:7

    ctime_start=ind.ctime(Ibest,I)+time_zone_offset+ind.duration(Ibest,I)/2-file_len_sec/2;
    if isnan(ctime_start)
        continue
    end

    dir_want=sprintf('%s/Shell20%s_GSI_Data/S%s%sgsif/S%s%s%s0', ...
        GSI_file_dir,Iyear,Isite,Iyear, ...
        Isite,Iyear,strr(I));

    file_want(5)=strr(I);
    [x,~,head]=readgsi([dir_want filesep file_want],ctime_start,file_len_sec,'native');
    head_table = get_GSI_head_info(head_info, Iyear, Isite, strr(I));

    titstr{1}=sprintf('%s, tdrift file: %6.4f, tdrift_logged: %6.4f',file_want, (head.tdrift),head_table.tdrift);
    [SNR_gram,~,TT,FF]=create_spectrogram_sample(x(1,:)',head.Fs,file_len_sec/2+1/head.Fs,file_len_sec,spectrogram_len_sec,param.spec,titstr);

    subplot(4,2,I);
    imagesc(FF,TT,double(SNR_gram));colormap jet;axis xy;title(titstr{1});

    hold on
    XX=0.5*spectrogram_len_sec-0.5*ind.duration(Ibest,I);
    rectangle('Position',[XX(1) ind.flo(Ibest,I) ind.duration(Ibest,I) ind.fhi(Ibest,I)-ind.flo(Ibest,I)],'edgecolor','y','Linewidth',1)
    hold off


end