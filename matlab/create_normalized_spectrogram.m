%function [SNR,output_array_fin,F,T,median_raw]=create_normalized_spectrogram(y,Fs,spectrogram_len_sec,param)
%  Load raw data, create spectrogram and normalize it using noise from
%  beginning and end.  Output type is uint8 for all output_array_fin
%
%  Include option for running compute_directional_metrics if multichannel
%  vector sensor data.
%
%  Output:
%       SNR: Signal to noise ratio of spectrogram in dB units, mulitplied
%           by param.image_scale_factor;
function [SNR,output_array_fin,F,T,median_raw]=create_normalized_spectrogram(y,Fs,spectrogram_len_sec,param)


param.debug_plot=false;
param.num_blobs=3;
param.blob_size=5;

image_key={'Azimuth','ItoERatio','KEtoPERatio','Polarization','Degree of Polarization'};
%param.NTV=100;
%param.KE_offset=60;
%param.KE_scale=3;
%param.Polar=1;


%t1=tic;
median_raw=[];

if size(y,2)>1

    param.sec_avg=0;
    param.instrument= {'DASAR_DASAR-omnisensor'  'DASAR_DASAR-Xsensor'  'DASAR_DASAR-Ysensor'};
    [T,F,output_array,B]=compute_directional_metrics(y',image_key(1:4), ...
        Fs,param.Nfft, param.ovlap, param);

    

    Ifreq=(F>=param.fmin & F<=param.fmax);
    B=B(Ifreq,:);

    dT=T(2)-T(1);
    T_noise=(T(end)-spectrogram_len_sec)/2;

    NN=round(T_noise/dT);
    median_noise=median([B(:,1:NN) B(:,(length(T)-NN):length(T))],2);

    Indexx=(NN+1):(length(T)-NN-1);
    SNR_org=(B(:,Indexx)-median_noise);

    %%%Package other metrics using appropirate scaling and conversion to int

    output_array_fin{1}=(output_array{1}(Ifreq,Indexx)); %azigram

    Wght=SNR_org.*output_array{2}(Ifreq,Indexx).^2;  %%Weight by NTV
    output_array_fin{2}=(param.NTV*output_array{2}(Ifreq,Indexx)); %NTV
    output_array_fin{2}=uint8(output_array_fin{2});
    SNR=uint8(param.image_scale_factor*SNR_org);

    output_array_fin{3}=10*log10(output_array{3}(Ifreq,Indexx)); %KE
    output_array_fin{3}(output_array_fin{3}<-param.KE_offset)=-param.KE_offset;
    output_array_fin{3}(output_array_fin{3}>param.KE_offset)=param.KE_offset;
    output_array_fin{3}=uint8(param.KE_scale*(param.KE_offset+output_array_fin{3}));  %3 is scaling factor

    %output_array_fin{4}=(param.Polar*output_array{4}(Ifreq,Indexx,1,4));  %Degree of polarization
    %output_array_fin{4}=uint8(output_array_fin{4});

    output_array_fin{4}=(imag(output_array{4}(Ifreq,Indexx,1,3)));  %%Stokes parameter 3 (Im(Vxy))
    output_array_fin{4}=uint8(output_array_fin{4});


    T=T(Indexx);
    T=T-T(1);
    F=F(Ifreq);
    %Igood=find(Wght(:)>5)


    %%%%Compute the estimated azimuth of the signal
    Igood = bwpropfilt(Wght>param.blob_size,'Area',param.num_blobs);
    angs=output_array_fin{1}(Igood);%azigram

    output_array_fin{1}=uint8(output_array_fin{1});
    median_raw=circ_median(angs(:)*pi/180)*180/pi;
    median_raw(median_raw<0)=median_raw(median_raw<0)+360;

    azi_math=90 - angs;   %%%Angles now defined c-clockwise relative to x axis
    Ifix=(azi_math<-180);
    azi_math(Ifix)=azi_math(Ifix)+360;

    % compute median and std
    median_math=180*circ_median(azi_math*pi/180)/pi;  %Original existing line
    iqr_out=180*circ_std(azi_math*pi/180)/pi;

    %Convert back into compass convention.
    median_compass=90-median_math;
    median_compass(median_compass<0)=median_compass(median_compass<0)+360;

    %param.debug_plot=true;
    if param.debug_plot
       % figure(1);set(gcf,'Position',[ 146     1   560   420])
       % imagesc(T,F,B);title('Spectrogram');axis xy;colormap(jet);clim([0 40]); colorbar

       %%%Plot raw outputs
        figure(2);set(gcf,'Position',[ 129    22   560   420])
        subplot(2,3,1)
        imagesc(T,F,SNR_org);title('SNR Spectrogram');axis xy;colormap(jet);clim([0 40]); colorbar

        for III=1:(length(output_array))
            subplot(2,3,III+1)
        
            switch III
                case 3
                    imagesc(T,F,10*log10(output_array{3}));title(image_key{III});axis xy;colormap(jet);colorbar
                case 4
                    imagesc(T,F,imag(output_array{4}(:,:,1,3)));title(image_key{III});axis xy;colormap(jet);colorbar

               % case 5
                %    imagesc(T,F,output_array{4}(:,:,1,4));title(image_key{III});axis xy;colormap(jet);colorbar
                otherwise
                    imagesc(T,F,output_array{III});title(image_key{III});axis xy;colormap(jet);colorbar
            end
        end

        %%%Plot outputs scaled to uint8 outputs
        figure(3);set(gcf,'Position',[ 146   519   560   420])
        subplot(2,3,1)
        imagesc(T,F,SNR);title('SNR');axis xy;colormap(jet);clim(param.image_scale_factor*[0 40]); colorbar

        for III=1:length(output_array_fin)
            subplot(2,3,III+1)
            imagesc(T,F,output_array_fin{III});title(image_key{III});axis xy;colormap(jet);colorbar
        end
    end
    %%%%


    if param.debug_plot
        figure(5);set(gcf,'Position',[ 663          41        1122         893])

        subplot(3,2,1)
        imagesc(T,F,SNR_org);title('SNR');axis xy;colormap(jet);colorbar;clim([5 30])
        subplot(3,2,2)
        imagesc(T,F,output_array{2}(Ifreq,Indexx));title('NTV');axis xy;clim([0 1]);colorbar
        subplot(3,2,3)
        imagesc(T,F,Wght);title('NTV*SNR^2');axis xy;clim([5 30]);colorbar
        subplot(3,2,4)
        imagesc(T,F,Igood);title(sprintf('top %i blobs',param.num_blobs));axis xy;colorbar
        subplot(3,2,5)

        azigram=output_array{1}(Ifreq,Indexx);
        imagesc(T,F,Igood.*azigram);title('filtered for top blobs');axis xy;colorbar;clim([0 360]);colorbar

        stats=regionprops(Wght>param.blob_size,'Area');Area=[stats.Area];
        subplot(6,2,10);plot(Area,'x');grid on;ylabel('Area(pixels)');%xlabel('Blob index')
        % figure;
        subplot(6,2,12);histogram(azigram(Igood));xlim([0 360])
        title(sprintf('Raw: %6.2f, Compass %6.2f',median_raw,median_compass),'Interpreter','none')
        %keyboard
        pause;
        close all
    end



    %   Output:
    %    TT,FF, output_array{Nmetric}:  TT vector of times and FF vector of Hz for
    %                 output_array grid, one for each value in metric_type.
    %                 If 'polarization' is a metric, output_array{}[freq times metric] will have
    %                 three dimensions, with third dimension being Stokes metric.



else  %Just a spectrogram please
    [B,F,T] = spectrogram(y(:,1),param.Nfft,round(param.ovlap*param.Nfft),param.Nfft,Fs);
    Ifreq=(F>=param.fmin & F<=param.fmax);
    B=10*log10(abs(B(Ifreq,:)));
    F=F(Ifreq);

    dT=T(2)-T(1);
    T_noise=(T(end)-spectrogram_len_sec)/2;

    NN=round(T_noise/dT);
    median_noise=median([B(:,1:NN) B(:,(length(T)-NN):length(T))],2);

    Indexx=(NN+1):(length(T)-NN-1);
    SNR=(param.image_scale_factor)*(B(:,Indexx)-median_noise);
    T=T(Indexx);
    T=T-T(1);
    if any(SNR(:)>255)
        fprintf('SNR greater than %6.2f\n',255/param.image_scale_factor);
        keyboard
    end
    SNR=uint8(SNR);
    output_array_fin=[];
end

%toc(t1)

end