
clear
close all
addpath ../../../umapAndEppFileExchange_v4_6/umap
n_neighbors=15;

%%%Load original UMAP from python output...
original_python=load('../../../Bowhead_DL_Project/Autoencoder_v13_100E_32LD_32C_AutoManual_Combined_100K_Date20251228-124835.dir/UMAP/umap_embeddings_3d_auto.mat');

[reduction, umap, clusterIds, extras]= ...
    run_umap(double(original_python.latent_embeddings),'n_components', 3,'n_neighbors',n_neighbors,'verbose','text');


alpha_value=0.1;

subplot(1,2,1)
x=reduction;
ss=scatter3(x(:,1),x(:,2),x(:,3),3,'k','filled');
ss.MarkerEdgeAlpha=alpha_value;
ss.MarkerFaceAlpha=alpha_value;
title('MATLAB'); grid on

subplot(1,2,2)
x=original_python.umap_embeddings_3d;
ss=scatter3(x(:,1),x(:,2),x(:,3),3,'k','filled');
ss.MarkerEdgeAlpha=alpha_value;
ss.MarkerFaceAlpha=alpha_value;
title('PYTHON');grid on

umap_embeddings_3d=single(reduction);
save('umap_embeddings_3d_auto_MATLAB.mat','umap_embeddings_3d');
