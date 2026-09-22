%%%%%master_load_and_test_network.m%%%%%%
clear all
close all

network_chc='Original_Manual_NotCentered';

[net_autoencoder,net_decoder,net_dir]=load_trained_network(network_chc);

%input_data=load('S314D0T20140901T131043_Type2.mat');
data_filenames=dir('S*.mat');
Nsamples=length(data_filenames);
for I=1:Nsamples
    input_data=load(data_filenames(I).name);

    if I==1
        sizz=size(input_data.SNR_gram);
        input_images=zeros(sizz(1),sizz(2),1,length(data_filenames),'single');
        %output_images=input_image;
    end
    input_images(:,:,1,I)=single(input_data.SNR_gram)./single(max(max(input_data.SNR_gram)));
    

end

[output_images] = minibatchpredict(net_autoencoder,input_images);
[latent_vectors] = minibatchpredict(net_autoencoder,input_images,'Outputs','TopLevelModule:to_latent');


temp=reshape(latent_vectors',size(latent_vectors,2),1,1,Nsamples);
output_image_decoder=minibatchpredict(net_decoder,temp);

%output_image=predict(net_autoencoder,input_image);
%latent=predict(net_autoencoder,input_image,'Outputs','TopLevelModule:to_latent');
%output_image_decoder=predict(net_decoder,latent);

figure(100)
for I=1:Nsamples
    subplot(3,1,1)
    imagesc(input_images(:,:,1,I));axis xy

    subplot(3,1,2)
    imagesc(output_images(:,:,1,I));title('Autoencoder output');
    grid on;axis xy
    

    subplot(3,1,3)
    imagesc(output_image_decoder(:,:,1,I));title('Decoder output');
    grid on;axis xy

    pause
end








%%%IMPORTANT!
%  Remove network from path in case we want to run different network in
%%%future

rmpath(net_dir);