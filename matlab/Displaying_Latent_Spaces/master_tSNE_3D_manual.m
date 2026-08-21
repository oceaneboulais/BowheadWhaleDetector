%master_tSNE_3D_manual.m
%

close all
clear all

dir_names={'LD32/Autoencoder_v14_100E_32LD_32C_Manual_100K_Date20260122-190106.dir', ...
    'LD16/Autoencoder_v14_100E_16LD_32C_Manual_100K_Date20260122-190056.dir'};

images_dir{1}='/Volumes/Bowhead_DL_Project/BCB_Whale_Datasets/Unsupervised_database_Manual_100K_Y08101214.dir';
images_dir{2}=images_dir{1};


for Idir=1:length(dir_names)
    disp(dir_names{Idir})
    mydir=pwd;
    cd([dir_names{Idir} filesep 'MATLAB'])
    data=load('latent_embeddings.mat');

    save_flag=false;

    if ~isfield(data,'x_tsne')
        disp('Recomputing tSNE...')
        data.x_tsne=tsne(data.latent_embeddings,'NumDimensions',3);
        save_flag=true;
        save('latent_embeddings.mat','-struct','data');

    end
    cd(mydir)
    figure(Idir)
    type=str2double(extract(data.original_filenames,28));

    for J=1:1 %%Split by call type
        switch J
            case 1
                Itype=find(type<4);
                titstr='Upsweeps, downsweeps, and constant tones';
                alpha_value=0.1;
            case 2
                Itype=find(type==7);
                titstr='Complex Calls';
                alpha_value=0.5;
        end
        h(Idir,J)=subplot(1,2,J);
        ss(Idir,J)=scatter3(data.x_tsne(Itype,1), data.x_tsne(Itype,2), data.x_tsne(Itype,3), 3,type(Itype),'filled');
        ss(Idir,J).MarkerEdgeAlpha=alpha_value;
        ss(Idir,J).MarkerFaceAlpha=alpha_value;
        grid on
        axis equal
        colorbar
        %if J==1
        colormap jet
        % else
        %    colormap gray
        % end

        xlabel('Dimension 1');
        ylabel('Dimension 2');
        zlabel('Dimension 3');
        title([dir_names{Idir}(1:4) ' ' titstr]);
        %linkaxes([h(Idir,1) h(Idir,2)]);

        hLink = linkprop(h(Idir,:), {'CameraPosition','CameraUpVector','CameraTarget'});

        figure(3)

        x_tsne=data.x_tsne./std(data.x_tsne);
        scatter3_limits_with_azel_edits(x_tsne(Itype,:),type(Itype));

        colormap jet
        disp('Select rotation check and rotate figure');
        drawnow;
        
        create_gif=input('Enter 1 to create a rotating GIF, hit return otherwise...\n');
        if ~isempty(create_gif)
            GIF_movie_demo(x_tsne(Itype,:),type(Itype),alpha_value,titstr);
        end

        figure(3)
        Xt=gcf().UserData.Xt;

        display_sample=input('Enter 1 to review samples using ginput...\n');
       
        notready=true;
        while display_sample && notready
            tmp=ginput(2);
            tmp(:,3)=str2num(gcf().UserData.edtZ.String)';
            Icluster=find(Xt(:,1)>min(tmp(:,1))&Xt(:,1)<max(tmp(:,1)) ...
                &Xt(:,2)>min(tmp(:,2)) &Xt(:,2)<max(tmp(:,2)) ...
                &Xt(:,3)>min(tmp(:,3)) &Xt(:,3)<max(tmp(:,3)));

            temp_fnames=data.original_filenames(Itype(Icluster));
            
            hh=gcf().UserData.ax;
            hh.Title.String=sprintf('%i Samples in range',length(Icluster));
            hh.Title.FontWeight="bold";
            hh.Title.FontSize=14;

            Ncalls=30;
            Iwant=(randperm(length(Icluster),Ncalls));
            figure;set(gcf,'Position',[ 11          60        1745         874  ]);
            for JJ=1:Ncalls
                subplot(3,10,JJ)
                disp(temp_fnames{Iwant(JJ)})
               % type(Itype(Icluster(Iwant(JJ))));  %Should be same type as
               % in file name
                imgdata=load(sprintf('%s%s%s',images_dir{Idir},filesep,temp_fnames{Iwant(JJ)}));
                FF=imgdata.dF*(0:size(imgdata.SNR_gram,1));
                TT=imgdata.dT*(0:size(imgdata.SNR_gram,2));

                imagesc(TT,FF,imgdata.SNR_gram);%colorbar;
                ylim([0 300]);
                axis xy
                set(gca,'fontweight','bold','fontsize',14)
                title(sprintf('%s,%s',temp_fnames{Iwant(JJ)}(1:22),temp_fnames{Iwant(JJ)}(end-4)),'FontSize',8);
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
            notready=input('Enter 1 to make another selection:');
        end
    end %J


    clear data
    cd(mydir)
end