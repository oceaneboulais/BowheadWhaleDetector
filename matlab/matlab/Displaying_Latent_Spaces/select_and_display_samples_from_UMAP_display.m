%%%select_and_display_samples_from_UMAP_display.m

figure(myfig);
ud=myfig.UserData;
Xt=ud.Xt;  %%Figure might have been rotated after selection, so need to update...
Igood=ud.Igood;  %%%Points visible on screen (survived filtering)

%%%Check that length of Igood matches points on screen
if (length(Igood)~=length(ud.h.CData)) | (length(Igood) ~= length(ud.h.XData))
    keyboard
end

tmp=ginput(2);
tmp(:,3)=str2num(ud.edtZ.String)';
Icluster=find(Xt(Igood,1)>min(tmp(:,1))&Xt(Igood,1)<max(tmp(:,1)) ...
    &Xt(Igood,2)>min(tmp(:,2)) &Xt(Igood,2)<max(tmp(:,2)) ...
    &Xt(Igood,3)>min(tmp(:,3)) &Xt(Igood,3)<max(tmp(:,3)));

Icluster=Igood(Icluster);  %Index now relates fo full data set

%%%Don''t count recent edits...
N_all_subsamples=length(Icluster);
try
    ignore_edits_after_this_date=datetime(ud.edt_time.String);
catch
    disp('Fixing edit string')
    ud.edt_time.String= datestr(datetime('now'),'yyyy-mm-dd HH:MM:SS');
    ignore_edits_after_this_date=datetime(ud.edt_time.String);
end
I_old_enough=find(data.date_adjusted(Icluster)<ignore_edits_after_this_date);  %exclude editing previously edited

Icluster=Icluster(I_old_enough);  %Index now relates fo full data set

if isempty(Icluster)
    disp('No points available because either none are there, or already edited.');

else

    temp_fnames=data.original_filenames(Icluster);
    temp_type=data.features.type(Icluster);  %%Note that edited calls will be used here...

    temp_Imanual=find(temp_type>0);
    temp_Iauto=find(temp_type<=0);

    N_manual=length(temp_Imanual);
    N_unmarked=length(temp_Iauto);
    fprintf('Out of %i original detections %i are editable... \n\t Of those remaining there are %i manual calls and %i unmarked signals in this sample \n', ...
        N_all_subsamples,length(Icluster),N_manual,N_unmarked);
    pause(0.5)
    %Display manual examples
    hh=ud.ax;
    hh.Title.String=sprintf('%i originals, %i Samples editable, %i annotated, %i unmarked', ...
        N_all_subsamples,length(Icluster),N_manual,N_unmarked);
    hh.Title.FontWeight="bold";
    hh.Title.FontSize=14;
    drawnow
    %Ncalls=min([30 length(Icluster)]);
    %Iwant=(randperm(length(Icluster),Ncalls));

    %Ichanged_manual=[];
    if display_manual& ~isempty(temp_Imanual)
        [data.features.type]=make_tile_spectrograms("Manual",temp_fnames(temp_Imanual), ...
            data.features.type,Icluster(temp_Imanual),dataset_chc,images_dir,manual_file,gsi_dir,display_NTV);
    end

    %Ichanged_auto=[];
    if display_auto & ~isempty(temp_Iauto)
        [data.features.type]=make_tile_spectrograms("Auto",temp_fnames(temp_Iauto), ...
            data.features.type,Icluster(temp_Iauto),dataset_chc,images_dir,manual_file,gsi_dir,display_NTV);

    end
    % close(2:length(get(0).Children))

    figgs = findall(0,'Type','figure');   % includes hidden handles
    for f = figgs.'
        if ~contains(f.Name,'Scatter3')
            close(f);
        end
    end
    clear figgs f

    %%%Ireviewed are indicies that have been reviewed,
   
    N_manual=length(temp_Imanual);
    N_unmarked=length(temp_Iauto);

    if N_manual+N_unmarked==0
        Ireviewed=[];

    elseif N_manual==0
        Ireviewed=Icluster(temp_Iauto);

    elseif N_unmarked==0
        Ireviewed=Icluster(temp_Imanual);
    else

        try
            Ireviewed=unique([Icluster(temp_Imanual); Icluster(temp_Iauto)]);

        catch
            Ireviewed=unique([Icluster(temp_Imanual) Icluster(temp_Iauto)]);
        end
    end
    %Ichanged=unique([Ichanged_manual Ichanged_auto]);
    data.date_adjusted(Ireviewed)=datetime('now');
    data.reviewer(Ireviewed,:)=reviewer_initials;

    %%%Change type, iscall, and ischanged features
    ud.features.type=data.features.type;
    data.features.iscall=double(data.features.type>0);
    ud.features.iscall=data.features.iscall;

    data.features.ischanged=(data.features.type~=data.features.type_org);
    ud.features.ischanged=data.features.ischanged;


    figure(myfig);
    set(myfig,'UserData',ud);

    %  if strcmpi(ud.selectedFeatureField,'iscall')
    %     Igood=find(ud.features.iscall>0);
    %     ud.Igood=Igood;
    %     data.Igood=Igood;
    %     set(myfig,'UserData',ud);

    %%%Update the internal 'Igood' variable stored in UserData.

    fhandle=myfig.UserData.edtFeature1.Callback;
    fhandle(myfig.UserData.edtFeature1);

    %%%Update info on edited samples
    former_manual=(data.features.type_org>0 & ~data.features.iscall);
    former_auto=(data.features.type_org==0 & data.features.iscall);
    fprintf('Call samples removed: %i, Call samples added: %i, total changes: %i\n\n', ...
        sum(former_manual),sum(former_auto),sum(data.features.type~=data.features.type_org));

end