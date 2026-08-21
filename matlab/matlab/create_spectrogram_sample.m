function [SNR_gram,output_array_fin,FF,TT,azi]=create_spectrogram_sample(x,Fs,tmid,file_len_sec,spectrogram_len_sec,param,titstr)
%function [SNR_gram,output_array_fin,FF,TT,azi]=create_spectrogram_sample(x,Fs,tmid,file_len_sec,spectrogram_len_sec,param,titstr)
% param.Nfft, param.ovlap, param.fmin, param.fmax
%   x[samples,channel]
if isempty(param)
    param.debug=false;
end
SNR_gram=[];TT=[];FF=[];azi=[];output_array_fin=[];
tsec_start=tmid-0.5*file_len_sec;
Ixx=round(Fs*(tsec_start+[0 file_len_sec]));

Ixx(1)=max([1 (Ixx(1))]);
Ixx(2)=min([length(x) Ixx(2)]);
y=x(Ixx(1):Ixx(2),:);  %%signal snippet including background noise

if length(y)>round(Fs*file_len_sec)
    y=y(1:round(Fs*file_len_sec));
end
if length(y)~=round(Fs*file_len_sec)  %if snippet from front or end of file.
    disp('File length not right')
    return
end

%[SNR,output_array_fin,F,T,median_raw]
[SNR_gram,output_array_fin,FF,TT,azi]=create_normalized_spectrogram(double(y),Fs,spectrogram_len_sec,param);

%%Ensure image dimensions are divisible by 8 to allow it to be close to
%%NUWC 128 by 144 image

target_dim=8*floor(size(SNR_gram)/8);

ncol_cut=floor((size(SNR_gram,2)-target_dim(2))/2);
nrow_cut=max([1 floor((size(SNR_gram,1)-target_dim(1))/2)]);

Itt=((ncol_cut+1):(length(TT)-ncol_cut));
if nrow_cut==1
    Iff=(nrow_cut+1):(length(FF));
else
    Iff=(nrow_cut+1):(length(FF)-nrow_cut);
end

SNR_gram=SNR_gram(Iff,Itt);
for III=1:length(output_array_fin)
    output_array_fin{III}=output_array_fin{III}(Iff,Itt);
end
TT=TT(Itt);FF=FF(Iff);

%if ~isfield(param,'debug_max_tmid')
%    param.debug_max_tmid=Inf;
%end

if param.debug_plot% & param.debug_max_tmid>tmid
    figure
    subplot(2,1,1)
    spectrogram(double(y(:,1)),param.Nfft,param.Nfft/2,param.Nfft,Fs,'yaxis')
    clim([0 30]);colorbar
    %title(sprintf('Filename: %s, middle time %6.2f seconds, %i of %i',GSI_names(Ifile_want).name,tmid,I,length(Itemp)))
    title(titstr{1})

    XX=0.5*file_len_sec-0.5*param.duration;
    hold on
    %line(XX,250*[1 1],'color','w','LineWidth',10);
    rectangle('Position',[XX(1) param.plot_fmin param.duration param.plot_fmax-param.plot_fmin],'edgecolor','y','Linewidth',1)
    hold off
  
    %     rectangle('Position',pos) creates a rectangle in 2-D coordinates.
    %     Specify pos as a four-element vector of the form [x y w h] in data
    %     units. The x and y elements determine the location and the w and h
    %     elements determine the size. The function plots into the current axes
    %     without clearing existing content from the axes.

    subplot(2,1,2)
    imagesc(TT,FF,SNR_gram);colorbar;axis xy
    title(titstr{2})
    hold on
    XX=0.5*spectrogram_len_sec-0.5*param.duration;
    rectangle('Position',[XX(1) param.plot_fmin param.duration param.plot_fmax-param.plot_fmin],'edgecolor','y','Linewidth',1)
    hold off
    %title(sprintf('Final SNR image, SNR: %6.2f, abs start: %s',manual.SNR(I),datestr(manual.tabs(I))));
    %
    % pause
end


end