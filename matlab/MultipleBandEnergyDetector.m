%%%%MultipleBandEnergyDetector.m
%  Aaron Thode
%  Revised 10 May 2026 to allow timing of peak power across frequency to be
%   flagged.
% function [detect,debug]=MultipleBandEnergyDetector(x,tabs_start,params)
%
%%%  Given a time series x and a series of band parameters,
%%%  detect transients over multiple overlapping frequency bands
%%% x: time series.  Can be either a row or column vector (but haven't
%%%             checked)
%%% tabs_start:  datenumber representing start of x
%   params:  structure array with the following fields...
%
%   Nfft, Fs:
%   EqFormat:  'dB' or 'linear' for equalization function
%   ovlap: 0.75 is 75 percent overlap between FFTs.
%   burn_in_time: Time in minutes to build up equalization model.
%
%   flo_det, fhi_det:  The absolute minimum and maximum frequencies
%   	to monitor for signals of interest,
%
%   eq: if exists, use this as the initial equalization function
%   eq_time: an equalization time in seconds.  If eq_time is Inf, the equalization is not updated after the first 20 samples.
%           The longer the equalization time, the slower the changes in the
%           threshold over time.
%
%   bandwidth:  The bandwidth of a subdetector--must be less than fhi_det-flo_det
%
%   threshold:  dB threshold (SNR) needed to exceed to start detection.
%
%   bufferTime: How much time to stuff before and after detection start and stop when writing time series output.
%
%   TolTime: How much time in seconds must pass before new detection started?
%
%   MinTime: Minimum time in seconds a signal needs to exceed threshold SNR to register a detection.
%   MaxTime: 
%   debug: If 0, no subdetector output.  If 1, write subdetector SEL.  If 2, write suddetector equalization value.
%   	If 3, write ratio of current SEL to equalization SEL (SNR estimate).
%
%  Example for bowhead whale analysis that monitors between 25 and 350 Hz,
%  using a set of detectors with 37 Hz bandwidth.
%             param.Nfft=256;
%             param.Fs =1000;
%             param.ovlap = 0.75;
%             param.flo_det=25;
%             param.fhi_det=350;
%             param.burn_in_time=1;  %Time in minutes
%             param.eq_time=10;   param_desc{K}='Equalization time (s): should be roughly twice the duration of signal of interest';K=K+1;
%             param.bandwidth=37;     param_desc{K}='Bandwidth of sub-detector in kHz';K=K+1;
%             param.threshold=10;  param_desc{K}='Threshold in dB to accept a detection';K=K+1;
%             param.snips_chc=1;  param_desc{K}='0 for no snips file, 1 for snips file of one channel, 2 for snips file of all channels';K=K+1;
%             param.bufferTime=0.5; param_desc{K}='Buffer Time in seconds to store before and after each detection snip, -1 suppress snips file';K=K+1;
%             param.TolTime=1e-4;  param_desc{K}='Minimum time in seconds that must elapse for two detections to be listed as separate';K=K+1;
%             param.MinTime=0;     param_desc{K}='Minimum time in seconds a required for a detection to be logged';K=K+1;
%             param.MaxTime=3;     param_desc{K}= 'Maximum time in seconds a detection is permitted to have';K=K+1;
%             param.debug=0;       param_desc{K}= '0: do not write out debug information. 1:  SEL output.  2:  equalized background noise. 3: SNR.';K=K+1;
%
%
%%%%% Output
%     detect.tstart(count)=tstart_total;
%        detect.tend(count)=tend_total;
%        detect.fmin(count)=min(flo(tend~=0));
%        detect.fmax(count)=max(fhi(tend~=0));
%        detect.dB_RMS(count): dB re 1uPa amplitude RMS of signal
%                    (10*log10(sum(10.^(temp(1:2:end)/10),1)))
%        detect.duration=(detect.tend-detect.tstart);
%        detect.tstart_abs=tabs_start+datenum(0,0,0,0,0,detect.tstart);
%        detect.tend_abs=tabs_start+datenum(0,0,0,0,0,detect.tend);
%        debug.detector:  sum across spectrogram PSD bandwidth
       
function [detect,debug]=MultipleBandEnergyDetector(x,tabs_start,params)


%Define constants for easier understanding
STATE.OFF=-2;
STATE.POSSIBLE_ON=1;
STATE.ON=2;
STATE.POSSIBLE_OFF=-1;

isdB=true;
if isfield(params,'EqFormat')
  if ~contains(params.EqFormat,'dB')
      isdB=false;  %linear
  end
end

%%%Set up bands for subdetectors

flo=params.flo_det:(0.5*params.bandwidth):params.fhi_det;
fhi=flo+params.bandwidth;
Iflo=ceil(flo*params.Nfft/params.Fs);
%Iband=ceil(params.bandwidth*params.Nfft/params.Fs);
Ifhi=(fhi*params.Nfft/params.Fs);

Igood=find(Ifhi<=params.fhi_det*params.Nfft/params.Fs);
Ifhi=ceil(Ifhi(Igood));Iflo=Iflo(Igood);fhi=fhi(Igood);flo=flo(Igood);
Ndet=length(Ifhi);

if Ndet==0
    disp('Number of detectors is zero: bandwidth too large');
    detect=[];debug=[];
    return
end

%%%%%%%%%Create spectrogram %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[~,FF,TT,B] = spectrogram(x,hanning(params.Nfft),round(params.ovlap*params.Nfft),params.Nfft,params.Fs);
dF=FF(2)-FF(1);
Ispan=Iflo(1):Ifhi(end);
B=B(Ispan,:);
eq=median(B,2);

%%%%%%Quick test of power law detector%%%%%%%


if isdB  %JINGLONG
    B=10*log10(B); %Convert to dB
    threshold=params.threshold;
else
    threshold=10.^log10(threshold); %convert to linear
end
Ncol=size(B,2);


%Create equalization function

if ~isfield(params,'eq')
    [~,Iburn]=min(abs(params.burn_in_time*60-TT));
    eq=median(B(:,1:Iburn),2);
elseif contains(params.eq,'median')
    eq=median(B,2);
    Iburn=1;
end

%%%Adjust frequency indicies to account for filtering
Ifhi=Ifhi-Iflo(1)+1;
Iflo=Iflo-Iflo(1)+1;

%dT=TT(2)-TT(1);
dT=(1-params.ovlap)*params.Nfft/params.Fs;
Imin_time=round(params.MinTime/dT);
Itol_time=round(params.TolTime/dT);
Imax_time=round(params.MaxTime/dT);

%%% Create equalization and detection functions
if isdB  %JINGLONG
    debug.detect=zeros(length(Iflo),size(B,2));
    for Icol=1:length(Iflo)
        band.eq(Icol)=10*log10(dF*sum(10.^(eq(Iflo(Icol):Ifhi(Icol))/10)));
        debug.detect(Icol,:)=10*log10(dF*sum(10.^(B(Iflo(Icol):Ifhi(Icol),:)/10),1));  %Equal to dB RMS of transient
    end
    debug.peak_power=10*log10(dF*sum(10.^(B(Iflo(1):Ifhi(end),:)/10),1));
else
    for Icol=1:length(Iflo)
        band.eq(Icol)=dF*sum(eq(Iflo(Icol):Ifhi(Icol)));
        debug.detect(Icol,:)=dF*sum(B(Iflo(Icol):Ifhi(Icol),:),1);
    end
    
end
%%%Define equalization parameter
dn = (1-params.ovlap)*params.Nfft;
alpha = 0.01^(dn/(params.eq_time*params.Fs));  %20 dB depression
if isinf(params.eq_time)
    alpha=1;
end

%%Initialize detector.
count_estimate=round(10*length(x)/params.Fs);  %Assume around 10 detections/second
fieldnames={'tstart','tend','duration','dB_RMS','fmin','fmax','magnitude','tpeak'};
for Icol=1:length(fieldnames)
    detect.(fieldnames{Icol})=zeros(count_estimate,1);
end

count=0;

%%%detection_status reports status of each frequency band.  It is important
% to have more states than just 'on' or 'off'.  No enum type in MATLAB



detection_status=STATE.OFF*ones(Ndet,1);  %-2: off; -1: possible off 1: possible on  2: on
active_detectors=0;
tend=zeros(Ndet,1);
tstart=zeros(Ndet,1);
magnitude=tstart;

tstart_total=0;tend_total=0;magnitude_all=0;peak_index=0;
write_flag=false;

if size(band.eq,2)>1
    band.eq=band.eq';
end
debug.eq_history(Ndet,1:Iburn)=0;
debug.eq_history(:,1:Iburn)=repmat(band.eq,1,Iburn);
debug.detection_status(:,1:Iburn)=STATE.OFF*ones(Ndet,Iburn);


%%Cycle through detector.
%%% Written as for loop to make it easy
for Icol=((1+Iburn):Ncol)  %For every column of spectrogram (or incoming FFT)
    global_reset=false;
    val=debug.detect(:,Icol);  %Slice of frequency-summed spectrogram (dB rms)
    %sum_val=sum(val);
    for Jdetect=1:Ndet  %For each detector
        if write_flag||global_reset %Detection has offically ended, cycle thorugh rest of detectors quickly
            continue
        end
        
        if isdB  %JINGLONG
            criteria=band.eq(Jdetect)+threshold;
        else
            criteria=band.eq(Jdetect).*threshold;
        end
        if val(Jdetect)>=criteria  %threshold exceeded
            % STATE.OFF=-2;
            % STATE.POSSIBLE_ON=1;
            % STATE.ON=2;
            % STATE.POSSIBLE_OFF=-1;
            switch detection_status(Jdetect)
                case STATE.OFF %OFF
                    detection_status(Jdetect)=STATE.POSSIBLE_ON; %POSSIBLE ON
                    tstart(Jdetect)=Icol;
                    tend(Jdetect)=Icol;
                case STATE.POSSIBLE_ON %POSSIBLEON
                    if (Icol-tstart(Jdetect))>=Imin_time  %Is the detection long enough to track
                        detection_status(Jdetect)=STATE.ON;
                        active_detectors=active_detectors+1;
                        magnitude(Jdetect)=val(Jdetect);  %dB rms
                        if  debug.peak_power(Icol)>magnitude_all
                            peak_index=Icol;
                            magnitude_all=debug.peak_power(Icol);
                            
                        end
                        if active_detectors==1  %%If detection officially starts, include possible on
                            %tstart_total=Icol;  %Changed this 10 Mar 2026
                            tstart_total=tstart(Jdetect);
                        end
                    end
                    
                case STATE.ON %ON
                    magnitude(Jdetect)=max([magnitude(Jdetect) val(Jdetect)]);

                    if  debug.peak_power(Icol)>magnitude_all
                        peak_index=Icol;
                        magnitude_all=debug.peak_power(Icol);
                        
                    end

                    %%%Force reset if detection going on too long.
                    if Icol-tstart_total>=Imax_time
                        disp('too long')
                        active_detectors=0;
                        tend(Jdetect)=Icol;
                        tend_total=tend(Jdetect);
                        write_flag=true;
                        if ~global_reset
                            for KK=1:Ndet
                                detection_status(KK)=STATE.OFF;
                                if isdB  %JINGLONG
                                    band.eq(KK)=(1-0.25)*val(KK)+0.25*band.eq(KK);
                                else
                                    band.eq(KK)=(band.eq(Jdetect).^0.25).*(val(Jdetect).^.75); %Update background estimate
                                    
                                end
                                debug.eq_history(Jdetect,Icol)=band.eq(Jdetect);
                                tend(KK)=Icol;
                                %writeMe(KK)=true;
                            end
                            global_reset=true;
                        end
                        
                        
                    end % if Imax_time
                    
                case STATE.POSSIBLE_OFF %POSSIBLE OFF: we have just dipped below threshold for less than EndTolerance time, take back
                    detection_status(Jdetect)=STATE.ON;
                    tend(Jdetect)=Icol;
                    if  debug.peak_power(Icol)>magnitude_all
                        peak_index=Icol;
                        magnitude_all=debug.peak_power(Icol);
                        
                    end
            end
        else  %threshold not obtained
            
            switch detection_status(Jdetect)
                case STATE.OFF %OFF
                    if isdB
                        band.eq(Jdetect)=(alpha).*band.eq(Jdetect)+(1-alpha)*val(Jdetect); %Update background estimate
                    else
                        band.eq(Jdetect)=(band.eq(Jdetect).^alpha).*(val(Jdetect).^(1-alpha)); %Update background estimate
                        
                    end
                case STATE.ON %ON
                    tend(Jdetect)=Icol;  %%Mark end of detection
                    detection_status(Jdetect)=STATE.POSSIBLE_OFF; %Possible OFF
                    if  debug.peak_power(Icol)>magnitude_all
                        peak_index=Icol;
                        magnitude_all=debug.peak_power(Icol);
                        
                    end
                case STATE.POSSIBLE_ON %POSSIBLE ON:  We have briefly crossed threshold but have dipped back below
                    detection_status(Jdetect)=STATE.OFF;
                    tstart(Jdetect)=0;
                    tend(Jdetect)=0;
                    %magnitude_all=0;
                    %peak_index=0;
                case STATE.POSSIBLE_OFF %POSSIBLE OFF
                  
                    if tend(Jdetect)+Itol_time<=Icol&&(tend(Jdetect)~=0)  %%%if enough time has passed since last detection
                        detection_status(Jdetect)=STATE.OFF;
                        active_detectors=active_detectors-1;
                        %writeMe(J)=true;
                        if active_detectors==0
                            if Imin_time+Itol_time<=(tend(Jdetect)-tstart_total)
                                tend_total=tend(Jdetect);
                                write_flag=true;
                                global_reset=true;
                            else
                                disp('Too short')
                                reset_detect;  %detection is too short
                            end
                        end
                    end
                    
            end  %detection_status
            
        end %if threshold
        
    end %Jdetect loop through detectors.


    debug.eq_history(:,Icol)=band.eq;
    debug.detection_status(:,Icol)=detection_status;
    %%%We've now worked through all bands.  Create or close
    %%%  an official detection.
    
    if write_flag % detection completed
        %%%Is it too short?
        count=count+1;
        detect.tstart(count)=tstart_total;
        detect.tend(count)=tend_total;
        detect.fmin(count)=min(flo(tend~=0));
        detect.fmax(count)=max(fhi(tend~=0));
        detect.tpeak(count)=peak_index;
        
        if isdB  %JINGLONG
            temp=magnitude(tend~=0);
            detect.dB_RMS(count)=10*log10(sum(10.^(temp(1:2:end)/10),1));  %Equal to dB RMS of transient
            
        else
        end
        
        write_flag=false;
        reset_detect;
        
    end %write flag
    
end  %Icol-loop through time

%%%Convert times into seconds
%detect.tstart=0.5*params.Nfft/params.Fs-dT+dT*detect.tstart;
%detect.tend=0.5*params.Nfft/params.Fs-dT+dT*detect.tend;

detect.tstart=TT(detect.tstart(detect.tstart>0))';
detect.tend=TT(detect.tend(detect.tend>0))';
detect.tpeak=TT(detect.tpeak(detect.tpeak>0))';

detect.duration=detect.tend-detect.tstart;
debug.TT=TT;

for Icol=1:length(fieldnames)
    detect.(fieldnames{Icol})=detect.(fieldnames{Icol})(1:count);
end


if params.debug
    %close all
    figure(100+round(50*rand(1)));hold off
    subplot(4,1,1);hold off
    mineq=unique(debug.eq_history);
    imagesc(TT,0.5*(flo+fhi),(debug.eq_history));axis('xy'); colorbar('westoutside')
   % caxis([0 20])
    title('Equalization');
    
    %caxis([min(min(debug.detect)) max(max(debug.detect))]);
    subplot(4,1,2);hold off
    imagesc(TT,0.5*(flo+fhi),(debug.detect));axis('xy'); colorbar('westoutside')
    clim([min(min(debug.detect)) max(max(debug.detect))]);
    title('Detection function');
    
    subplot(4,1,3);hold off
    imagesc(TT,0.5*(flo+fhi),(debug.detect-debug.eq_history));axis('xy'); colorbar('westoutside')
    clim([params.threshold + [0 30]]);
    title('Detection Excess');colormap(jet)
    
    subplot(4,1,4);hold off
    imagesc(TT,0.5*(flo+fhi),debug.detection_status);axis('xy'); colorbar('westoutside')
    title('Detection Status');
    xlabel('seconds');
    
    hold on
    ylimm=ylim;
    for Icol=1:count
        line([detect.tstart(Icol) detect.tend(Icol)],detect.fmin(Icol)*[1 1],'color','g','linewidth',2);
        line([detect.tstart(Icol) detect.tend(Icol)],detect.fmax(Icol)*[1 1],'color','g','linewidth',2);
        
        line(detect.tpeak(Icol)*[1 1],ylimm,'color','g','linewidth',2);
    end
    linkaxes
    
end %if params.debug

%%%Convert times to absolute times
detect.tstart_abs=tabs_start+datenum(0,0,0,0,0,detect.tstart);
detect.tend_abs=tabs_start+datenum(0,0,0,0,0,detect.tend);
detect.tpeak_abs=tabs_start+datenum(0,0,0,0,0,detect.tpeak);
debug.flo=flo;
debug.fhi=fhi;


    function reset_detect
        active_detectors=0;
        tend_total=0;
        tstart_total=0;
        write_flag=false;
        magnitude_all=0;   
        peak_index=0;

        for KKK=1:Ndet
            magnitude(KKK)=0;
            %peak_index(KKK)=0;
            tstart(KKK)=0;
            tend(KKK)=0;
            detection_status(KKK)=STATE.OFF;
            
            
        end
    end
end



