%%%%%%%master_cluster_analysis.m%%%%
% Nov 25, 2025

close all
clear all

same_grid_size=true;
cluster_dir='../../Cluster_Analysis';
run_str='Autoencoder_v06_100E_32LD_MostlyManual_50K_Date20251121-170008';
images_dir='../../Spectrogram_Image_Database.dir/Unsupervised_database_MostlyManual.dir';

run_str='Autoencoder_v07_100E_32LD_AutoWithAirguns_50K_Date20251123-001830';
images_dir='../../Spectrogram_Image_Database.dir/Unsupervised_database_AutoWithAirguns.dir';
%images_dir='/Volumes/Maui2025/DeepLearningBowhead/Spectrogram_Image_Database.dir/Unsupervised_database_AutoWithAirguns.dir';
data_dir=[cluster_dir filesep run_str];

%data_dir='Autoencoder_v07_100E_32LD_AutoWithAirguns_50K_Date20251123-001830';
%data=load([data_dir '.dir' filesep 'latent_embeddings.mat']);
%param.perplexity=[10 30 50 10 30 50];
%param.standardize=[false false false true true true];

%Itype_str={'Upcall','Downcall','Constant','U-shaped','N-shaped','Other FM','Complex','Bearded Seal','Walrus'};


param_here.min_SNR=5*1;

param_here.alpha=0.1;
x_grid=-80:5:80;
y_grid=x_grid;

SNR_data=load([data_dir '.dir' filesep 'airgun_index.mat']);
I_snr=find(SNR_data.SNR>=param_here.min_SNR);

Icount=0;

for II=1:1
    Icount=Icount+1;
    load(sprintf('%s%s%s_v%i.mat',cluster_dir,filesep,run_str,II));

    % ID=zeros(1,length(data.filenames));
    % for I=1:length(data.filenames)
    %     ID(I)=str2double(data.filenames{I}(end-4));
    % end
    ID=ID(I_snr);
    ID_type=unique(ID);

    ID_type(2:3)=[3 2];
    temp=Itype_str{2};
    Itype_str{2}=Itype_str{3};
    Itype_str{3}=temp;
    %for J=1:length(param.perplexity)
    
    for J=1:1
        

        tt{J}=tt{J}(I_snr,:);
        figure(10*II+J);set(gcf,'Position',[42         190        1830         761]);
        histogram2(tt{J}(:,1),tt{J}(:,2),x_grid,y_grid,'DisplayStyle','tile');colorbar;grid on
        title(sprintf('Perplexity: %i Standardize: %i',param.perplexity(J),param.standardize(J)));

        saveas(gcf,sprintf('%s_all_%i.fig',data_dir,J))


        figure;set(gcf,'Position',[42         190        1830         761]);
        for Itype=1:length(ID_type)
            subplot(2,4,Itype+1)
            Igood=find(ID==ID_type(Itype));
            if ~same_grid_size
                histogram2(tt{J}(Igood,1),tt{J}(Igood,2),'DisplayStyle','tile');colorbar;grid on
            else
                histogram2(tt{J}(Igood,1),tt{J}(Igood,2),x_grid,y_grid,'DisplayStyle','tile');colorbar;grid on
            end
            set(gca,'FontWeight','bold','FontSize',18);
       
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
            title(sprintf('%s',Itype_str{Itype}),'FontSize',18);
        end
        handd(8)=subplot(2,4,1);
        if ~same_grid_size
            histogram2(tt{J}(:,1),tt{J}(:,2),'DisplayStyle','tile');colorbar;grid on
        else
            histogram2(tt{J}(:,1),tt{J}(:,2),x_grid,y_grid,'DisplayStyle','tile');colorbar;grid on
        end
        set(gca,'FontWeight','bold','FontSize',18);
        xlabel('tSNE1');ylabel('tSNE2')
        xlim(xlimm_all);ylim(ylimm_all)
        title(sprintf('All data: Perplexity: %i Standardize: %i',param.perplexity(J),param.standardize(J)), ...
            'Fontsize',18);
        % saveas(gcf,sprintf('%s_%i.fig',data_dir,J))
        linkaxes(handd,'xy');

        %%%%Scatter plots
        strr='rgbkkkkkkkk';
        figure(100+J-1);set(gcf,'Position',[42         190        1830         761]);
        subplot(1,2,II)
        for Itype=1:length(ID_type)
            switch Itype
                case {4,5,6}
                    continue
            end
            %subplot(2,4,Itype+1)
            Igood=find(ID==ID_type(Itype));
            hscatter=scatter(tt{J}(Igood,1),tt{J}(Igood,2),5,strr(Itype),'MarkerEdgeColor',strr(Itype));hold on;grid on
            hscatter.MarkerEdgeAlpha=param_here.alpha;
            set(gca,'FontWeight','bold','FontSize',18);
            xlabel('tSNE1');ylabel('tSNE2');

        end
        hleg=legend(Itype_str{[1:3 7]});
        title(sprintf('All data: Perplexity: %i Standardize: %i',param.perplexity(J),param.standardize(J)), ...
            'Fontsize',18);
        xlim(xlimm_all);ylim(ylimm_all)

    end %J-param combination trial

    %figure
    %histogram(ID);grid on

    %save([data_dir '.mat'],'tt','param','Itype_str','ID');

end %II run number



%%%Optional review of clusters
yes=input('Type 1 if want to analyze clusters: ');
if ~isempty(yes)
    tmp=ginput(2);
    Icluster=find(tt{J}(:,1)>=min(tmp(:,1)) & tt{J}(:,1)<=max(tmp(:,1))  ...
              & tt{J}(:,2)>=min(tmp(:,2))  & tt{J}(:,2)<=max(tmp(:,2)) ...
               );
              % & ID'==7);
    figure;scatter(tt{J}(Icluster,1),tt{J}(Icluster,2))

    temp_fnames=load(sprintf('%s%s%s.dir/latent_embeddings.mat',cluster_dir,filesep,run_str),'filenames');
    temp_fnames=temp_fnames.filenames;
    temp_fnames=temp_fnames(Icluster);

    Ncalls=30;
    Iwant=(randperm(length(Icluster),Ncalls));
    figure;set(gcf,'Position',[ 11          60        1745         874  ]);
    for JJ=1:Ncalls
        subplot(3,10,JJ)
        disp(temp_fnames{Iwant(JJ)})
        data=load(sprintf('%s%s%s',images_dir,filesep,temp_fnames{Iwant(JJ)}));
        FF=data.dF*(0:size(data.SNR_gram,1));
        TT=data.dT*(0:size(data.SNR_gram,2));

        imagesc(TT,FF,data.SNR_gram);%colorbar;
        axis xy
        title(sprintf('%s,%s',temp_fnames{Iwant(JJ)}(1:22),temp_fnames{Iwant(JJ)}(end-4)),'FontSize',10);
        if rem(JJ,10)~=1
            set(gca,'ytick',[])
        end
        if JJ<20
            set(gca,'xtick',[])
        end
    end
end

%%%Distribution of call types in database
figure;set(gcf,'Position',[ 8         286        1195         650]);
histogram(ID)
xtick_label=get(gca,'XTickLabel');
for I=1:length(xtick_label)
    xtick_label{I}=Itype_str{I};
end
xtick_label{2}='Downcall';
xtick_label{3}='Constant';
set(gca,'XTickLabel',xtick_label)
ylabel('Samples');
set(gca,'FontWeight','bold','FontSize',14);
grid on

