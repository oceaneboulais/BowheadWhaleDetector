%master_tSNE_3D_manual.m


%close all
clear all
addpath ..
addpath .

dataset_chc='auto';
UMAP_dim=3;   %Dimension of UMAP to load
color_label='type';  %%How to label colors in 3D scattering.  'PeakFrequency' or 'type','PeakTime'
advanced_labels=true;

[Database_dir,procdata_basedir,gitpath] = setUpDatabasePaths;
switch dataset_chc

    case 'manual'
        dir_names={[Database_dir 'LD32/Autoencoder_v14_100E_32LD_32C_Manual_100K_Date20260122-190106.dir'], ...
            [Database_dir 'LD16/Autoencoder_v14_100E_16LD_32C_Manual_100K_Date20260122-190056.dir']};

        images_dir{1}=[Database_dir '/BCB_Whale_Datasets/Unsupervised_database_Manual_100K_Y08101214.dir'];
        %images_dir{2}=images_dir{1};

        Ntypes=length(images_dir);
    case 'auto'
       % dir_names={[Database_dir '/LD16/Autoencoder_v13_100E_16LD_32C_AutoManual_Combined_100K_Date20260119-222955.dir']};
       %dir_names={[Database_dir '/LD32/Autoencoder_v13_100E_32LD_32C_AutoManual_Combined_100K_Date20251228-124835.dir']};
       dir_names={[Database_dir '/LD32/Autoencoder_v100E_32LD_32C_100kCombined_Centered_Date20260323-105320.dir/']};
       dir_names={pwd};

       %Original result
       images_dir{1,1}=[Database_dir '/BCB_Whale_Datasets/Unsupervised_database_AutoWithAirguns_100K_Y08101214.dir'];
       images_dir{1,2}=[Database_dir '/BCB_Whale_Datasets/Unsupervised_database_Manual_100K_Y08101214.dir'];

       %Centered result
       %images_dir{1,1}=[Database_dir '/BCB_Whale_Datasets/Unsupervised_database_AutoWithAirguns_100K_Y08101214_centered.dir.dir'];
       %images_dir{1,2}=[Database_dir '/BCB_Whale_Datasets/Unsupervised_database_Manual_100K_Y08101214_centered.dir'];


        Ntypes=1;
end

for Idir=1:length(dir_names)
    disp(dir_names{Idir})
    mydir=pwd;
    cd([dir_names{Idir} filesep 'UMAP'])
    %file_want=sprintf('umap_embeddings_%id_%s_MATLAB.mat',UMAP_dim,dataset_chc);
    file_want=sprintf('umap_embeddings_%id_%s.mat',UMAP_dim,dataset_chc);
    fpath = fullfile(pwd, file_want);    % current folder + filename

    if isfile(fpath)                  % or: exist(fpath,'file')==2
        data = load(fpath);
    else
        error('File "%s" not found in current folder: %s', file_want, pwd);
    end

    %data=load(file_want);
    field_want=sprintf('umap_embeddings_%id',UMAP_dim);
    x=data.(field_want);

    %%If frequency information not available, load from SNR_gram
    if advanced_labels & ~isfield(data,'PeakFrequency')
        Npp=size(x,1);
        data.PeakFrequency=ones(Npp,1);
        data.PeakTime=ones(Npp,1);
        for II=1:Npp
            if rem(II,100)==0,fprintf('%6.2f percent done\n', 100*II/Npp);end
            fname=data.original_filenames{II};
            if strcmp(dataset_chc,'manual')
                imgdata=load(sprintf('%s%s%s',images_dir{Idir},filesep,fname));
            else
                if strcmp(fname(end-4),'0')
                    imgdata=load(sprintf('%s%s%s',images_dir{1},filesep,fname));
                else
                    imgdata=load(sprintf('%s%s%s',images_dir{2},filesep,fname));
                end
            end

            [outputs]=extract_features_from_SNRgram(imgdata.dT,imgdata.dF,imgdata.SNR_gram);
            data.PeakFrequency(II)=outputs.Fmax;
            data.PeakTime(II)=outputs.Tmax;


        end %%II
        %save(sprintf('umap_embeddings_%id.mat',UMAP_dim),'PeakTime','PeakFrequency',"-append");
        save(sprintf('umap_embeddings_%id_%s.mat',UMAP_dim,dataset_chc),"-struct","data")
    end  %Advanced labels
    if UMAP_dim==5
        [coeff,score,latent,tsquared,explained] = pca(x,'NumComponents',3);
        %coeff: projection of original axes onto new orthogonal axes (5
        %by 3 matrix)
        %
        % score:  translation of each data point into the new PCA
        % coordinates
        %  latent: variance of each column of score
        %  explained: percent varience explained by PCA component.
        %  Used to judge which components to keep
        x=score;
    end
    %%update UMAP if data doesn't exist
    % save_flag=false;
    % if ~isfield(data,'x_tsne')
    %     disp('Recomputing tSNE...')
    %     data.x_tsne=tsne(data.latent_embeddings,'NumDimensions',3);siz
    %     save_flag=true;
    %     save('latent_embeddings.mat','-struct','data');
    % 
    % end
    cd(mydir)

    type=str2double(extract(data.original_filenames,28));

    figure(Idir)

    for J=1:Ntypes %%Split by call type

        switch dataset_chc
            case 'manual'
                initial_azi=45;
                initial_el=-25;

                switch J
                    case 1
                        Itype=find(type<4);
                        Itype=find(type<4 | type==7);
                        
                        titstr='Upsweeps, downsweeps, and constant tones';
                        alpha_value=0.3;
                    case 2
                        Itype=find(type==7);
                        titstr='Complex Calls';
                        alpha_value=0.5;
                end
            case 'auto'
                type(type>0)=1;
                Itype=1:length(type);
                %Itype=find(type==0);
                alpha_value=0.3;
                titstr='All detections';
                initial_azi=0;
                initial_el=-5;
        end
       % h(Idir,J)=subplot(1,2,J);
        x_norm=(x-mean(x))./std(x);
       
        
        switch color_label
            case 'PeakFrequency'
                x_color=data.PeakFrequency;
            case 'PeakTime'
                x_color=data.PeakTime;
            case 'type'
                x_color=type;
        end
        for K=1:2
            h(Idir,K)=subplot(1,2,K);
            if K==1
                ss(Idir,K)=scatter3(x(Itype,1), x(Itype,2), x(Itype,3), 3,x_color(Itype),'filled');
            else
                ss(Idir,K)=scatter3(x_norm(Itype,1), x_norm(Itype,2), x_norm(Itype,3), 3,x_color(Itype),'filled');

            end
            ss(Idir,K).MarkerEdgeAlpha=alpha_value;
            ss(Idir,K).MarkerFaceAlpha=alpha_value;
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
        end

        %hLink = linkprop(h(Idir,:), {'CameraPosition','CameraUpVector','CameraTarget'});


        scatter3_limits_with_azel_edits(x_norm(Itype,:),x_color(Itype));
        colormap jet
      
        myfig=gcf;

        colormap jet
        disp('Select rotation check and rotate figure');
        drawnow;

        create_gif=input('Enter 1 to create a rotating GIF, hit return otherwise...\n');
        if ~isempty(create_gif)
            titstr=sprintf('%s_%s_UMAP%idim.gif',dataset_chc,color_label,UMAP_dim);
            GIF_movie_demo(x(Itype,:),x_color(Itype),alpha_value,titstr,initial_azi,initial_el);
        end

        
        figure(myfig)
        display_sample= input('Switch to transform view, rotate and press 1 when ready...');
        Xt=gcf().UserData.Xt;

        %1display_sample=input('Enter 1 to review samples using ginput...\n');

        notready=true;
        while display_sample & notready
            Xt=gcf().UserData.Xt;

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

            Ncalls=min([30 length(Icluster)]);
            Iwant=(randperm(length(Icluster),Ncalls));
            figure;set(gcf,'Position',[ 11          60        1745         874  ]);
            for JJ=1:Ncalls
                subplot(3,10,JJ)
                disp(temp_fnames{Iwant(JJ)})
                % type(Itype(Icluster(Iwant(JJ))));  %Should be same type as
                % in file name

                if strcmp(dataset_chc,'manual')
                    imgdata=load(sprintf('%s%s%s',images_dir{Idir},filesep,temp_fnames{Iwant(JJ)}));
                else
                    if strcmp(temp_fnames{Iwant(JJ)}(end-4),'0')
                        imgdata=load(sprintf('%s%s%s',images_dir{1},filesep,temp_fnames{Iwant(JJ)}));
                    else
                        imgdata=load(sprintf('%s%s%s',images_dir{2},filesep,temp_fnames{Iwant(JJ)}));
                    end
                end
                FF=imgdata.dF*(0:size(imgdata.SNR_gram,1));
                TT=imgdata.dT*(0:size(imgdata.SNR_gram,2));

                imagesc(TT,FF,imgdata.SNR_gram);%colorbar;
                ylim([0 500]);
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