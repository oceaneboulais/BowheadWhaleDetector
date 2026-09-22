%function  [outputs]=extract_features_from_SNRgram(dT,dF,SNR_gram)

function  [outputs]=extract_features_from_SNRgram(dT,dF,SNR_gram)

image_scale_factor=5;
%mindB=5;
debug_plot=false;

FF=dF*(1:size(SNR_gram,1));
TT=dT*(1:size(SNR_gram,2));
SNR_gram=double(SNR_gram)/image_scale_factor;
tmp=max(SNR_gram,[],2);
[~,Imax]=max(tmp);
outputs.Fmax=FF(Imax);

tmp=max(SNR_gram,[],1);
tmp=tmp-min(tmp);
[~,Imax]=max(tmp);
outputs.Tmax=TT(Imax);


%w=10.^((SNR_gram-min(min(SNR_gram)))/10);
%w_int=trapz(w);
%mean_f=trapz(w.*FF')./w_int;
%MWLB=median(2*sqrt(trapz(w.*((FF'-mean_f).^2))./w_int));

%%%Trying to estimate duration of signal but not promising...
%[pks,locs,ww,pp]=findpeaks(tmp,TT,'MinPeakHeight',5, ...
%    'MinPeakProminence',max([0 Pmax-8]),'SortStr','descend','Annotate','extents');
%outputs.duration=ww(1);

if debug_plot
    figure(101);
    subplot(2,1,1)
    imagesc(TT,FF,SNR_gram);%colorbar;
    ylim([0 500]);
    axis xy
    set(gca,'fontweight','bold','fontsize',14)
    title(sprintf('Peak Frequency: %6.2f Peak Time: %6.2f',outputs.Fmax,outputs.Tmax));
    colorbar
    
    subplot(2,1,2)
    findpeaks(tmp-min(tmp),TT,'MinPeakHeight',5,'MinPeakProminence',max([0 Pmax-8]),'SortStr','descend','Annotate','extents')
    keyboard;
    close(101)
end
% pause