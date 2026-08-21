tmp=ginput(2);
Icluster=find(tt{J}(:,1)>=min(tmp(:,1)) & tt{J}(:,1)<=max(tmp(:,1))  ...
    & tt{J}(:,2)>=min(tmp(:,2))  & tt{J}(:,2)<=max(tmp(:,2)) ...
    );
% & ID'==7);
%figure;scatter(tt{J}(Icluster,1),tt{J}(Icluster,2))

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
    ylim([0 300]);
    axis xy
    set(gca,'fontweight','bold','fontsize',14)
    title(sprintf('%s,%s',temp_fnames{Iwant(JJ)}(1:22),temp_fnames{Iwant(JJ)}(end-4)),'FontSize',10);
    if rem(JJ,10)~=1
        set(gca,'ytick',[]);
    else
        ylabel('Hz')
    end
    if JJ<21
        set(gca,'xtick',[]);
    else
        xlabel('Time (sec)')
    end
end