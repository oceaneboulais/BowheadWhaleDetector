Npp=size(x,1);
for II=1:Npp

    if rem(II,100)==0,fprintf('%6.2f percent done\n', 100*II/Npp);end
    fname=data.original_filenames{II};

    try
        if strcmp(dataset_chc,'manual')
            imgdata=load(sprintf('%s%s%s',images_dir{Idir},filesep,fname),'features');
        else
            if strcmp(fname(end-4),'0')
                imgdata=load(sprintf('%s%s%s',images_dir{1},filesep,fname),'features');
            else
                imgdata=load(sprintf('%s%s%s',images_dir{2},filesep,fname),'features');
            end
        end
    catch
        fprintf('Could not load %s...\n',fname);
    end
    if II==1
        feature_names=fieldnames(imgdata.features);
        for Ifeature=1:length(feature_names)
            data.features.(feature_names{Ifeature})=ones(Npp,1);
        end
        data.features.type=ones(Npp,1);
    end

    for Ifeature=1:length(feature_names)
        try
            data.features.(feature_names{Ifeature})(II)=imgdata.features.(feature_names{Ifeature});
        catch
            data.features.(feature_names{Ifeature})(II)=-1;
        end
        data.features.type(II)=str2double(extract(fname,28));
    end
end %%II
data.features.type_org=data.features.type;
data.date_adjusted=repmat(datetime('now'),length(data.features.type),1);
data.features.score=zeros(size(data.features.type));  %Score is a feature!
data.reviewer=repmat("XX",length(data.features.type),1);

data.features=orderfields(data.features);

%%%Put iscall last so it is default color label in plot...
fieldname=fieldnames(data.features);
J_iscall=find(contains(fieldname,'iscall'));
data.features=orderfields(data.features,[1:(J_iscall-1) (J_iscall+1):length(fieldname) J_iscall]);