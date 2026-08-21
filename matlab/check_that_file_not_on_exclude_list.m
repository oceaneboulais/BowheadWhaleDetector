% Find rows of fnames that are NOT present in original_filenames (cell array of strings)
% Assume fnames is a cell array of strings (rows) and original_filenames is a cell array of strings
function  Ipass=check_that_file_not_on_exclude_list(fnames,original_filenames);

fnames_str=string(char(fnames));
[~, ia] = ismember(fnames_str, original_filenames);
Ipass = find(ia == 0);
fprintf('%i out of %i files pass...\n',length(Ipass),length(fnames_str));
