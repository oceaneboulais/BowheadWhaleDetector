%%%%%sub_plot_debug_manual_automated_comparison
%%%
%%%   Plot misses and matches between manual and automated results from
%%%   'master_convert_manual_archive_to_spectrogram'   


%%%%%%Examine missed manual detections
%param.spec.debug_plot=true;
%Itemp=Imiss;

param.spec.debug_missed_manual=false;
if param.spec.debug_missed_manual
    fprintf('Displaying %i missed manual detections\n',length(Imiss));

    for I=1:length(Imiss)
        II=Imiss(I);
        tmid=manual.tmid(II);

        titstr{1}=sprintf('Missed manual detection: Filename: %s, middle time %6.2f seconds, %i of %i',file_array{Ifile_want},tmid,I,length(Imiss));
        titstr{2}=sprintf('Final SNR image, SNR: %6.2f, abs start: %s',manual.SNR(Imiss(I)),datestr(manual.tabs(Imiss(I))));

        
        param.spec.plot_fmin=manual.fmin(II);
        param.spec.plot_fmax=manual.fmax(II);
        param.spec.duration=manual.duration(II);

        [SNR_gram,FF,TT]=create_spectrogram_sample(x,head.Fs,tmid,file_len_sec,spectrogram_len_sec,param.spec,titstr);

    end %I in Imiss
end %debug_plot

%%%Display automated detections that match%%%%%%%
param.spec.debug_plot_matches=false;
if param.spec.debug_plot_matches
    for Idet=1:length(Idet_match)

        II=Idet_match(Idet);
        tmid=0.5*(detect.tstart(II)+detect.tend(II));
        titstr{1}=sprintf('Matching Auto detect Filename: %s, middle time %6.2f seconds, %i of %i',file_array{Ifile_want},tmid,Idet,length(Idet_match));
        titstr{2}=sprintf('Final SNR image, SNR: %6.2f, abs start: %s, score overlap: %6.4f', ...
            detect.dB_RMS(II),datestr(detect.tstart_abs(II)),Score{Ichunk}(II));
        
        param.spec.plot_fmin=detect.fmin(II);
        param.spec.plot_fmax=detect.fmax(II);
        param.spec.duration=detect.duration(II);

        [SNR_gram,FF,TT]=create_spectrogram_sample(x,head.Fs,tmid,file_len_sec,spectrogram_len_sec,param.spec,titstr);

    end %Idet
end %debug plot

