%%%%master_evaluate_autoencoder_performance.m

close all
clear all

Neighbors=40;  %Found no effect in using 10 or 40.
min_duration=0.5;


%%%Load data set used to train autoencoder
train_dir='/Users/thode/Projects/Greeneridge_bowhead_detection/DeepLearningNPRB_Project/Bowhead_DL_Project/LD32/Autoencoder_v13_100E_32LD_32C_AutoManual_Combined_100K_Date20260416-180022.dir/MATLAB';
train_fname=[train_dir filesep 'latent_embeddings_3d_train_MATLAB_5Aug2026_Angel.mat'];
train=load(train_fname);


%%%Load evaluation data set latent vectors

eval_dir=train_dir;
eval_fname=[eval_dir filesep 'latent_embeddings_3d_eval_8to1_MATLAB.mat'];
eval=load(eval_fname);

disp('Computing nearest neighbor....')

Idx_all=knnsearch(train.latent_embeddings, ...
    eval.latent_embeddings,'K',Neighbors,'IncludeTies',true,'Distance','euclidean');
Nsamples=length(Idx_all);

threshold=unique([ 0 0:1/Neighbors:1 1]);
strrr='kr';strr2='bg';
figure;
for I=1:2  %1 is all samples, 2 is restricted duration
    train.iscall=train.features.type_org>0 & train.features.type_org<12;
    eval.iscall=eval.features.type_org>0 & eval.features.type_org<12;
    switch I
        case 1
            Igood=1:Nsamples;
            Idx=Idx_all;

        case 2
            Igood=find(eval.features.duration1>=min_duration);
            Idx=Idx_all(Igood);
            titstr=sprintf('Neighbors: %i, Samples > %3.2f sec',Neighbors,min_duration);
    end
    titstr=sprintf('Neighbors: %i',Neighbors);
    score{I}=zeros(length(eval.iscall(Igood)),1);  %Score is size of test dataset
    idx_max=0;
    for Iscore=1:length(Idx)
        clusster=train.iscall(Idx{Iscore});  %ID (call or not) of nearest neighbors in training set
        score{I}(Iscore)=sum(clusster)/Neighbors;  %%%Fraction of neigbors that are calls.
        idx_max=max([idx_max Idx{Iscore}]);
    end
    idx_max
    Icall=eval.iscall(Igood)>0;
    Ino_call=(eval.iscall(Igood)==0);

    total_positives_in_test=sum(Icall);

    recall=zeros(1,length(threshold));
    precision=recall;
    for Ithresh=1:length(threshold)
        detected_positives=sum(score{I}(Icall)>=threshold(Ithresh));
        false_positives=sum(score{I}(Ino_call)>=threshold(Ithresh));
        recall(Ithresh)=detected_positives./total_positives_in_test;
        precision(Ithresh)=detected_positives./(detected_positives+false_positives);

    end

    subplot(1,2,1);hold on
    plot(recall,precision,[strrr(I) '-o']);grid on;hold on
    xlabel('Recall');ylabel('Precision');
    xlim([0 1]);ylim([0 1]);
    title(titstr)
    if I==2
        legend('All samples',sprintf('Samples greater than %3.2f seconds',min_duration),'location',' southwest')
    end
       text(0.1,0.95,'a)','fontweight','bold','FontSize',14)
  
    set(gca,'fontweight','bold','FontSize',14);


    subplot(1,2,2);hold on
    plot(1-recall,1-precision,[strrr(I) '-o']);grid on;
    xlabel('Miss fraction');ylabel('False discovery rate (Fraction of calls that are not calls)');
    xlim([0 1]);ylim([0 1]);
    title(titstr)
    if I==2
        legend('All samples',sprintf('Samples greater than %3.2f seconds',min_duration),'location','northeast')
    end
    text(0.05,0.95,'b)','fontweight','bold','FontSize',14)
    set(gca,'fontweight','bold','FontSize',14);


    %%%Estimate precision for realistic data set
    % R1=sum(Ino_call)/sum(Icall);
    % R2=7.8;  %Ratio of false hits to manual count in full dataset
    % precision_estimated=precision.*R1./(R2.*(1-precision)+R1.*precision);
    % subplot(1,2,1)
    % plot(recall,precision_estimated,[strr2(I) '-o']);
    % subplot(1,2,2);
    % plot(1-recall,1-precision_estimated,[strr2(I) '-o']);grid on;hold on

    fprintf('Dataset length: %i samples (%i calls, %i false)\n',length(eval.iscall(Igood)),sum(Icall),sum(Ino_call))

end %I
%title(sprintf('%i Neighbors',Neighbors))

orient landscape
print(sprintf('AE_performance_%i_Neighbors.jpg',Neighbors),'-djpeg','-r300')