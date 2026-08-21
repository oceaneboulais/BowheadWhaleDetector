
%%%%%master_load_and_test_network.m%%%%%%
clear all
close all

network_chc='Evaluation_Detection_Centered';

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

temp=reshape(latent_vectors',size(latent_vectors,2),1,Nsamples);
try
    output_image_decoder=minibatchpredict(net_decoder,temp);
catch
    disp('decoder not working')
end


figure(100)
for I=1:Nsamples
    subplot(3,1,1)
    imagesc(input_images(:,:,1,I));axis xy

    subplot(3,1,2)
    imagesc(output_images(:,:,1,I));title('Autoencoder output');
    grid on;axis xy

    try
        subplot(3,1,3)
        imagesc(output_image_decoder(:,:,1,I));title('Decoder output');
        grid on;axis xy
    end
    pause
end







%%%IMPORTANT!
%  Remove network from path in case we want to run different network in
%%%future

rmpath(net_dir);

% S512A0T20120901T192443_Type0.mat
% %%  1.0e+03 *
% 
%   Columns 1 through 11
% 
%     0.2065    0.0759    1.0544   -0.5370   -0.1628    0.3008    0.1203    0.0371   -0.3317   -0.4997    0.0918
% 
%   Columns 12 through 22
% 
%     0.6852    0.5845    0.2417   -0.2871    0.4487   -0.3470   -0.0225   -0.0634    0.0737    0.2089    0.8682
% 
%   Columns 23 through 32
% 
%     0.2504    0.0429   -0.2720   -0.2148   -0.2021    0.0660   -0.6091    0.2786    0.1708    0.0155