%%%%%%%master_cluster_analysis.m%%%%
% Nov 25, 2025

close all
clear all

data_dir='../../Cluster_Analysis/Autoencoder_v06_100E_32LD_MostlyManual_50K_Date20251121-170008';
data_dir='../../Cluster_Analysis/Autoencoder_v07_100E_32LD_AutoWithAirguns_50K_Date20251123-001830';
param.perplexity=[10 30 50 10 30 50];
param.standardize=[false false false true true true];

param.perplexity=[ 30 ];
param.standardize=[true ];


Itype_str={'Upcall','Downcall','Constant','U-shaped','N-shaped','Other FM','Complex','Bearded Seal','Walrus'};

data=load([data_dir '.dir' filesep 'latent_embeddings.mat']);
SNR_data=load([data_dir '.dir' filesep 'airgun_index.mat']);

SNR=-1*ones(size(data.latent_embeddings,1),1);
%%Assign SNR label to latent_embedding
for Isnr=1:size(SNR_data.SNR,1)
    Iwant=find(contains(data.filenames,char(SNR_data.file_name_list(Isnr,:))));
    SNR(Iwant)=SNR_data.SNR(Isnr);
end

%I_snr=find(SNR>=param.min_SNR);

ID=zeros(1,length(data.filenames));
for I=1:length(data.filenames)
    ID(I)=str2double(data.filenames{I}(end-4));
end
ID_type=unique(ID);
%ID=ID(I_snr);

for J=1:length(param.perplexity)
    disp('Starting tSNE');
    tic
    tt{J}=tsne(data.latent_embeddings,'verbose',2,'NumPCAComponents',0,'Perplexity',param.perplexity(J),'Standardize',param.standardize(J));
    toc
    figure(J);set(gcf,'Position',[42         190        1830         761]);
    histogram2(tt{J}(:,1),tt{J}(:,2),'DisplayStyle','tile');colorbar;grid on
    title(sprintf('Perplexity: %i Standardize: %i',param.perplexity(J),param.standardize(J)));

    saveas(gcf,sprintf('%s_all_%i.fig',data_dir,J))


    figure(20+J);set(gcf,'Position',[42         190        1830         761]);
    for Itype=1:length(ID_type)
        subplot(2,4,Itype+1)
        Igood=find(ID==Itype);
        histogram2(tt{J}(Igood,1),tt{J}(Igood,2),'DisplayStyle','tile');colorbar;grid on
        xlimm(Itype,:)=xlim;
        ylimm(Itype,:)=ylim;
    end

    xlimm_all(1)=min(xlimm(:,1));
    xlimm_all(2)=max(xlimm(:,2));
    ylimm_all(1)=min(ylimm(:,1));
    ylimm_all(2)=max(ylimm(:,2));

    for Itype=1:length(ID_type)
        handd(Itype)=subplot(2,4,Itype+1);
        xlim(xlimm_all);ylim(ylimm_all);
        title(sprintf('%s',Itype_str{Itype}));
    end
    handd(8)=subplot(2,4,1);
    histogram2(tt{J}(:,1),tt{J}(:,2),'DisplayStyle','tile');colorbar;grid on
    xlim(xlimm_all);ylim(ylimm_all)
    title('All')
    title(sprintf('All data: Perplexity: %i Standardize: %i',param.perplexity(J),param.standardize(J)));
    saveas(gcf,sprintf('%s_%i.fig',data_dir,J))
    linkaxes(handd,'xy');

end

figure
histogram(ID);grid on

%SNR=SNR(I_snr);

save(sprintf('%s.mat',data_dir),'tt','param','Itype_str','ID','SNR');


