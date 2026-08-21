close all
clear all

file_name1='~/Downloads/latent_embeddings_3d_auto_MATLAB.mat';
file_name2='~/Downloads/latent_embeddings_3d_auto_MATLAB2.mat';
file_name_out='~/Downloads/latent_embeddings_3d_auto_MATLAB_merged.mat';
reviewer='AH';

data1=load(file_name1);
data2=load(file_name2);


I_reviewed1=contains(data1.reviewer,reviewer);
I_reviewed2=contains(data2.reviewer,reviewer);
I_both=I_reviewed1|I_reviewed2;

N1=sum(I_reviewed1);
N2=sum(I_reviewed2);
Nboth=sum(I_reviewed1|I_reviewed2);
fprintf('File 1 had %i edits, File 2 had %i edits, combined files have %i total edits with %i overlaps.\n', ...
    N1,N2,Nboth,(N1+N2)-Nboth);

data3=data1;

data3.reviewer(I_reviewed2)=reviewer;
data3.date_adjusted(I_reviewed2)=data2.date_adjusted(I_reviewed2);
data3.features.type(I_reviewed2)=data2.features.type(I_reviewed2);

I_reviewed3=contains(data3.reviewer,reviewer);
fprintf('Merged file has %i edits\n',sum(I_reviewed3));
save(file_name_out,'-struct','data3');

