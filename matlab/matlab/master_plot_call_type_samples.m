%%%master_plot_call_type_samples.m%%%%%
close all
clear all

print_figures=false;
Ncalls=18;
data_dir='../../Spectrogram_Image_Database.dir/Unsupervised_database_MostlyManual.dir';

fnames=dir([data_dir filesep 'S*.mat']);

call_type=-1*ones(length(fnames),1);
for I=1:length(fnames)
    call_type(I)=str2double(fnames(I).name(end-4));
end

call_types=unique(call_type);

for Itype=1:length(call_types)

    Icall_samples=find(call_type==call_types(Itype));
    Iwant=Icall_samples(randperm(length(Icall_samples),Ncalls));
    figure(Itype)
    set(gcf,'Position',[30          36        1560         849])
    for J=1:Ncalls
        subplot(3,6,J)
        disp(fnames(Iwant(J)).name)
        data=load(sprintf('%s%s%s',data_dir,filesep,fnames(Iwant(J)).name));
        FF=25+data.dF*(0:size(data.SNR_gram,1));
        TT=data.dT*(0:size(data.SNR_gram,2));

        imagesc(TT,FF,data.SNR_gram/5);%colorbar;
        axis xy
        set(gca,'FontWeight','bold','FontSize',14)
        title(sprintf('%s %i',fnames(Iwant(J)).name(1:(end-10)),call_types(Itype)),'FontSize',10);
        if rem(J,6)~=1
            set(gca,'ytick',[])
        else
            ylabel('Hz')
        end
        if J<13
            set(gca,'xtick',[])
        else
            xlabel('Time (sec)')
            set(gca,'xtick',0:3);
        end
        colorbar;caxis([0 20])

  
    end %J
    %text(0.5,0.5,sprintf('Call type %i',call_types(Itype)));
    if print_figures
        print('-djpeg','-r300',sprintf('CallSampletypes_%i.jpg',Itype))
    end
end %Itype