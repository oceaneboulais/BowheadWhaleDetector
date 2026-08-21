%  [score,manual_index]=evaluate_overlap_between_manual_automated(t1_s,t1_e,t2_s,t2_e,ovlap)
%  t1 are manual detections, t2 are automated
%  score and manual_index have dimensions of the automated detections
%      score(Idet) is fraction overlap the automated detection has with any
%      of the manual detections
%
%       manual_index(Idet,1) is the index of the manual detection that
%       matches Idet

%[Score{Ichunk},Manual_index]=evaluate_overlap_between_manual_automated(manual.tsec,manual.tend,detect.tstart,detect.tend,param.compare.ovlap);

function [score,manual_index]=evaluate_overlap_between_manual_automated(t1_s,t1_e,t2_s,t2_e,ovlap)
%score is percentage overlap between signal in t1_s
t1_s=t1_s(:);
t1_e=t1_e(:);
t2_s=t2_s(:);
t2_e=t2_e(:);
score=NaN(length(t2_s),3);
manual_index=NaN(length(t2_s),3);
for I=1:length(t1_s)  %%For each manual detection

    tmin=max([repmat(t1_s(I),length(t2_s),1) t2_s],[],2);
    tmax=min([repmat(t1_e(I),length(t2_e),1) t2_e],[],2);

    duration_1=t1_e-t1_s;
    duration_2=t2_e-t2_s;

    min_duration=min([repmat(duration_1(I),length(duration_2),1) duration_2],[],2);
    [frac_ovlap,Imax]=max((tmax-tmin)./min_duration);
    %score(Imax)=frac_ovlap>=ovlap;
    if frac_ovlap<0
        continue
    end
   
    if isnan(manual_index(Imax,1))
        score(Imax,1)=frac_ovlap;
        manual_index(Imax,1)=I;
    elseif isnan(manual_index(Imax,2))
        manual_index(Imax,2)=I;
        score(Imax,2)=frac_ovlap;
    elseif isnan(manual_index(Imax,3))
        manual_index(Imax,3)=I;
        score(Imax,3)=frac_ovlap;
       % keyboard

       % JJ=manual_index(Imax,:)
       % round([t1_s(JJ) t1_e(JJ)])
    else
        disp('Too many manual detections match this automated detection')
        %keyboard
    end


end
%keyboard
