%%%%%master_perturb_latent_space.m%%%%%%
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


%%%Load latent vector space and do PCA
x_latent=load([net_dir filesep 'MATLAB' filesep 'latent_embeddings.mat'],'latent_embeddings');
[coeff,x,latent,tsquared,explained] = pca(x_latent.latent_embeddings,'NumComponents',15);
%coeff: projection of original axes onto new orthogonal axes (5
%by 3 matrix)
%
% x:  translation of each data point into the new PCA
% coordinates
%  latent: variance of each column of score
%  explained: percent varience explained by PCA component.
%  Used to judge which components to keep
%x=score;

temp=reshape(latent_vectors',size(latent_vectors,2),1,1,Nsamples);
output_image_decoder=minibatchpredict(net_decoder,temp);

%output_image=predict(net_autoencoder,input_image);
%latent=predict(net_autoencoder,input_image,'Outputs','TopLevelModule:to_latent');
%output_image_decoder=predict(net_decoder,latent);

a_pert=linspace(-1,1,9);

%a_pert=0;

for Idim=1:size(coeff,2)  %PCA dimension

    a_std=std(x(:,Idim));  %%Variance of the PCA dimension

    figure(100)
    for Ipert=1:length(a_pert)
        for I=1:Nsamples

            latent_perturbed=latent_vectors(I,:);


            latent_perturbed=latent_perturbed+a_pert(Ipert)*a_std*(coeff(:,Idim)');
            output_image_decoder=predict(net_decoder,latent_perturbed);

            subplot(3,3,3*I-2)
            imagesc(input_images(:,:,1,I));axis xy

            subplot(3,3,3*I-1)
            imagesc(output_images(:,:,1,I));title(sprintf('Original output'));
            grid on;axis xy


            subplot(3,3,3*I)
            imagesc(output_image_decoder);
            title(sprintf('Perturbed Decoder output, component %i, %3.2f percent done',Idim,100*Ipert/length(a_pert)));
            grid on;axis xy


        end %Isample
        pause
    end

end




%%%IMPORTANT!
%  Remove network from path in case we want to run different network in
%%%future

rmpath(net_dir);