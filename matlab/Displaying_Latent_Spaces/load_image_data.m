function [imgdata, status]=load_image_data(dataset_chc,images_dir,fname)

status=true;
try
    if contains(lower(dataset_chc),'eval')
        imgdata=load(sprintf('%s%s%s',images_dir{1},filesep,fname));
    elseif contains(lower(dataset_chc),'train')
        if strcmp(fname(end-4),'0')
            imgdata=load(sprintf('%s%s%s',images_dir{1},filesep,fname));
        else
            imgdata=load(sprintf('%s%s%s',images_dir{2},filesep,fname));
        end
    else
        fprintf('%s is not a valid keyword for dataset_chc ...\n',dataset_chc);
        
    end
catch
    fprintf('Could not load %s...\n',fname);
    status=false;
    imgdata=[];
end

