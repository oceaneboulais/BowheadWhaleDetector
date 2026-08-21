
%%%%%master_compute_autoencoder_latent_vectors.m%%%%%%

% Igood                          46462x1                   371696  double
% clusters                           1x100000              400000  int32
% dataset_label                      1x16                      32  char
% date_adjusted                 100000x1                  1600000  datetime
% features                           1x1                 11201888  struct
% latent_embeddings             100000x32                12800000  single
% optimal_k                          1x1                        8  int64
% original_filenames                 1x100000            18400000  cell
% perplexity                         1x1                        8  double
% reconstruction_filenames           1x100000            20200000  cell
% reviewer                      100000x1                  5200112  string
% tsne_embeddings               100000x2                   800000  single
% umap_embeddings_3d            100000x3                  2400000  double

clear all
close all
addpath ../Displaying_Latent_Spaces
%%%UMAP parameters
addpath ../../../umapAndEppFileExchange_v4_6/umap
UMAP_dim=3;   %Dimension of UMAP to compute
n_neighbors=15;
min_dist=0.1;
save_template=false;


batch_size=2000;
output_label='eval_1to1';
network_chc='Evaluation_Detection_Centered';
image_folder='/Volumes/Thode_AI_Working_Disk/Bowhead_DL_Project/BCB_Whale_Datasets/Unsupervised_database_Evaluation_200K_1Auto1Manual_ADG_Y08101214_centered_07Aug2026.dir/';


[net_autoencoder,net_decoder,net_dir,Nlatent]=load_trained_network(network_chc);
data_filenames=dir([image_folder filesep 'S*.mat']);
Nsamples=length(data_filenames);

Ibatch_start=unique([1:batch_size:Nsamples Nsamples]);


for Ibatch=1:(numel(Ibatch_start)-1)

    Index=Ibatch_start(Ibatch):Ibatch_start(Ibatch+1);
    fprintf('Batch processing %i starting at index %i...\n',Ibatch,Index(1));

    tic
    for I=1:length(Index)
        Itop_index=Index(I);
        original_filenames{Itop_index}=data_filenames(Itop_index).name;
        input_data=load([image_folder filesep data_filenames(Itop_index).name]);

        if I==1
            sizz=size(input_data.SNR_gram);
            input_images=zeros(sizz(1),sizz(2),1,length(Index),'single');
        end

         feature_names=fieldnames(input_data.features);
           
        if Ibatch==1
            for Ifeat=1:length(feature_names)
                features.(feature_names{Ifeat})=zeros(Nsamples,1);
            end
            latent_embeddings=zeros(Nsamples,Nlatent);
            features.type=zeros(Nsamples,1);
            features.type_org=zeros(Nsamples,1);
            features.ICI=-ones(Nsamples,1);
        end
        input_images(:,:,1,I)=single(input_data.SNR_gram)./single(max(max(input_data.SNR_gram)));

        for Ifeat=1:length(feature_names)
            try
                features.(feature_names{Ifeat})(Itop_index)=input_data.features.(feature_names{Ifeat});
            catch
                fprintf('   %s in file %s failed.\n',feature_names{Ifeat}, data_filenames(Itop_index).name);
            end
        end
        features.type(Itop_index)=str2double(data_filenames(Itop_index).name(end-4));
        features.type_org(Itop_index)=features.type(Itop_index);
    end %I

    fprintf('Batch processing finished in %3.2f seconds...\n',toc);
    %[output_images] = minibatchpredict(net_autoencoder,input_images);
    tic
    [latent_vectors] = minibatchpredict(net_autoencoder,input_images,'Outputs','TopLevelModule:to_latent');
     fprintf('Network processing finished in %3.2f seconds...\n\n',toc);
   
    latent_embeddings(Index,:)=latent_vectors;

end %Ibatch
features.iscall=(features.type>0);


rmpath(net_dir);


[umap_embeddings_3d, umap, clusterIds, extras]= ...
    run_umap(double(latent_embeddings),'n_components', UMAP_dim,'min_dist',min_dist,'n_neighbors',n_neighbors,'verbose','text');



save(sprintf('%s_latent_embeddings_3d_%s_MATLAB.mat',network_chc,output_label),'latent_embeddings','features','original_filenames','image_folder','net_dir','umap_embeddings_3d');

zlimm_want=[-5 5];  %%%Restrict zaxis when selecting samples
color_label='iscall';  %%How to label colors in 3D scattering plot.

x=umap_embeddings_3d;
x_norm=(x-mean(x))./std(x);
x_color=features.(color_label);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%Plot all detections with UI controls%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%Optional flip to try to get better view of data...
x_norm=-x_norm;
myfig=scatter3_GUI_rotate_transparency_filter(x_norm,features,[78 90],zlimm_want); colormap jet
