close all;
clear all;
fnames = dir('../Spectrogram_Image_Database.dir/Unsupervised_images.dir/*G*_Type4.mat');
% load ../Spectrogram_Image_Database.dir/Unsupervised_images.dir/S510G0T20100815T000017_Type0.mat

for I = 1:length(fnames)
    load([fnames(I).folder '/' fnames(I).name  ])
    imagesc(TT,FF,SNR_gram/5)
    axis xy
    colorbar
    xlabel('Time in Seconds')
    ylabel('Frequency (Hz)')
    grid on;
    set(gca, "FontWeight", "bold", "FontSize", 14);
    title(fnames(I).name)
    pause
end
