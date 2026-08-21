%%%Save the latent space....


tmp{1}=data.features;tmp{2}=data.date_adjusted;tmp{3}=data.reviewer;
data.features=data.features_org;
data.date_adjusted=data.date_adjusted_org;
data.reviewer=data.reviewer_org;

data=rmfield(data,{'features_org','date_adjusted_org','reviewer_org'});
save(sprintf('latent_embeddings_%id_%s_MATLAB.mat',UMAP_dim,dataset_chc),"-struct","data");

data.reviewer_org=data.reviewer;
data.date_adjusted_org=data.date_adjusted;
data.reviewer_org=data.reviewer;

data.features=tmp{1};
data.date_adjusted=tmp{2};
data.reviewer=tmp{3};

clear tmp