function  [outputs]=extract_features_from_SNRgram(dT,dF,SNR_gram)

debug_plot=false;

image_scale_factor=5;
mindB=5;
min_prominence=3;

FF=dF*(1:size(SNR_gram,1));
TT=dT*(1:size(SNR_gram,2));
SNR_gram=double(SNR_gram)/image_scale_factor;
tmp=max(SNR_gram,[],2);
[Pmax,Imax]=max(tmp);
outputs.Fmax=FF(Imax);
outputs.SNR=Pmax;

tmp=max(SNR_gram,[],1);
tmp=tmp-min(tmp);
[Pmax,Imax]=max(tmp);
outputs.Tmax=TT(Imax);



%w=10.^((SNR_gram-min(min(SNR_gram)))/10);
%w_int=trapz(w);
%mean_f=trapz(w.*FF')./w_int;
%MWLB=median(2*sqrt(trapz(w.*((FF'-mean_f).^2))./w_int));

%%%Trying to estimate duration of signal but not promising...
[pks,locs,ww,pp]=findpeaks(tmp,TT,'MinPeakHeight',5, ...
    'MinPeakProminence',min_prominence,'SortStr','descend','Annotate','extents');
if ~isempty(ww)
    outputs.duration=ww(1);
else
    outputs.duration=[];
end

if debug_plot
    figure(101);
    subplot(2,1,1)
    imagesc(TT,FF,SNR_gram);%colorbar;
    ylim([0 500]);
    axis xy
    set(gca,'fontweight','bold','fontsize',14)
    title(sprintf('Peak Frequency: %6.2f Peak Time: %6.2f',outputs.Fmax,outputs.Tmax));
    colorbar('Location','north')
    
    subplot(2,1,2)
    findpeaks(tmp,TT,'MinPeakHeight',5,'MinPeakProminence',min_prominence,'SortStr','descend','Annotate','extents')
    %[pks,locs,ww,pp]=findpeaks(tmp,TT,'MinPeakHeight',5,'MinPeakProminence',max([0 Pmax-min_prominence]),'SortStr','descend','Annotate','extents');
    pause;
    close(101)
end
% pause