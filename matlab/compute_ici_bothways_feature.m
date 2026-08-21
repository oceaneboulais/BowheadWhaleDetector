%%%%%%%%%%%compute_ici_bothways_feature.m%%%%%%%%%%%%%%%%%
%function [ICI]=compute_ici_bothways_feature(tabs,ici_range,feature_array,feature_names, ...
%       num_clicks,num_misses,tol_feature,tol_time,Idebug),
%
%%% Given a vector of datenumbers, search for regular occurrances and
%%% estimate inter-click interval (ICI) and multipath
%    tabs: [1 x Ntimes] vector of datenumbers
%       to enable the first detected sound to be assigned an ICI
%    ici_range=[0.1 2; 5 10]; %Range of ICIs to test for in seconds.  If
%       multiple rows, test each range
%    feature_array: [Nfeature x Ntimes] array of features
%    feature_names: cell array of feature names, corresponds with rows of
%       feature array.
%    num_clicks=3;  %number of adjacent detections to examine for ICI
%    num_misses=1;  %How many "gaps" can exist in the ICI trace
%    tol_feature: [Nfeature x 1] array of tolerances (absolute) for feature
%       matching
%    tol_time=[0.01; 0.5];  %Tolerance for accepting a detection if close to ICI,(sec)
%       The number of rows much match the number of rows in ici_range.
%    Idebug:
%       .fname: raw GSI file from which data extracted.
%       .names:  cell array (Nfeature cells) with descriptive string of
%           features in feature_array.
%       .index:  index of element of tabs to begin plotting debug output.
%
%
%   Aaron Thode
%   Dec 14, 2008
%   simulation script embedded in code

function ICI=compute_ici_bothways_feature(tabs,ici_range,feature_array,feature_names, ...
    num_clicks,num_misses,tol_feature,tol_time,Idebug)

ICI=[];
if exist('Idebug')&&~isempty(Idebug)
    debug_flag=1;
    if isfield(Idebug,'time')
        [junk,debug_index]=min(abs(tabs-Idebug.time));
    else
        debug_index=Idebug.index;
    end
else
    debug_flag=0;
end

% if ~exist('max_duration_val')
%     max_duration_val=Inf;
% end

if isempty(tabs)
    ICI=[];
    return
end
if nargin<2,
    ici_range=[0.1 2]; %Range of ICIs to test for
    num_clicks=3;  %Number of clicks that needd to have same ICI
    tol_time=0.01;  %Tolerance for accepting ICI in sec
end
if num_clicks<3,
    disp('num_clicks must be greater than 2');
    return;
end
if num_clicks<4&&num_misses>0,
    disp('Too few num_clicks to permit any nonzero num_misses');
    num_misses=0;
end


max_future_clicks=200;

ctime=86400*(tabs-datenum(1970,1,1,0,0,0));
trel=ctime-ctime(1);

if size(trel,2)>1,
    trel=trel';
end

ICI=zeros(size(trel));
Istart=find(trel(1:min(500,length(trel)))<max(ici_range(:)));
Istart=Istart(1);

%%Process each event detection with index I
for I=Istart:length(trel)
    %if rem(I,1000)==0,disp(sprintf('%6.2f percent done',round(100*I/length(trel))));plot(trel,ICI,'x');pause(1);end


    ICI(I)=-1;
    current_time=trel(I);

    %%find how far in the future and past max_future_clicks (events) occur
    min_index=max([1 I-max_future_clicks]);
    max_index=min([length(trel) I+max_future_clicks]);

    %%Identify possible ICI candidates. Look both forward and backward in
    %%time by num_clicks*20*maximum interval desired.  Thus search range is the smaller
    % of max_future_clicks or the time span num_clicks*20*maximum

    %Icand=find((trel((I+1):max_index)-current_time)<num_clicks*max(ici_range));
    Icand_t=(min_index-1)+find(abs(trel(min_index:max_index)-current_time)<(1+num_clicks)*20*max(ici_range(:)));

    %Filter by feature
    Icand=feature_filter(Icand_t);

    %t_cand=trel(I+Icand);
    t_cand=trel(Icand);

    ici_cand=sort(abs(t_cand-current_time));  %Possible ICI times
    %ici_cand=ici_cand(find(ici_cand>min(ici_range)&ici_cand<max(ici_range)));
    
    
    %Restrict ICI possibilities to those bracketed by input variable
    %   ici_range
    Ipass=[];
    for Irow=1:size(ici_range,1)
        Ipass=union(Ipass, find(ici_cand>(ici_range(Irow,1))&ici_cand<(ici_range(Irow,2))));
    end
    ici_cand=ici_cand(Ipass);
    
    %If the difference between two ICI times is small, take mean value...
    %Idiff=find(diff(ici_cand)<tol_time);
    %ici_cand(Idiff)=0.5*(ici_cand(Idiff)+ici_cand(Idiff+1));
    %ici_cand(Idiff+1)=[];
    
    %%Add rounded times
    %ici_cand=sort([ici_cand ;10 ;20 ;25]);

    if isempty(Icand)
        continue
    end
    if debug_flag&&I>=debug_index
        
        plot_intermediate_debug_info;
    
    end


    %num_misses_org=num_misses;
    max_hits=0;  %Records the maximum number of hits for best candidate so far
    for JJ=1:length(ici_cand) % Start from shortest ICI, working to largest...
        hits=0;
        %num_misses=num_misses_org;
        %for K=1:(num_clicks-1),
        Iclicks=floor(num_clicks/2);
        Icount=0;
        
        %%Adjust tolerance time based on candidate interval. generally smaller intervals have smaller
        %%tolerances
        for Irow=1:size(ici_range,1)
            if (ici_range(Irow,1)<=ici_cand(JJ))&&(ici_cand(JJ)<=ici_range(Irow,2))
                tol_time_cand=tol_time(Irow);
            end
        end
        for K=1:Iclicks
            threshold=tol_time_cand*sqrt(K);
            pred_time=current_time+[-1 1]*K*ici_cand(JJ);
            
            %If count is dropping off edge of +/- max_num_pulses count, skip
            if (max(pred_time)<=trel(min_index))||(max(pred_time)>=trel(max_index))
                continue;
            end

            for L=1:2
                [best_guess(L),Ibest(L)]=min(abs(pred_time(L)-t_cand));
            end

            if all(best_guess<threshold)
                hits=hits+2;
                %ici_cand(JJ)=((hits)/(hits+1))*ici_cand(JJ)+abs(closest_click-current_time)/((hits+1)*abs(K));  %Average ICI of selection
                %ici_cand(JJ)=0.5*ici_cand(JJ)+0.5*diff(pred_time)/(2*K);
                ici_cand(JJ)=abs(diff(t_cand(Ibest)))/(2*K);
               % keyboard
            elseif any(best_guess<threshold)
                hits=hits+1;
            end
            Icount=Icount+1;

%             if debug_flag&&(I>=debug_index)
%                 disp(sprintf('ICI being tested: %6.2f, candidate %i of %i, %i good so far',ici_cand(JJ),Icount,num_clicks,hits));
%                 tpred=pred_time+ctime(1)-t0;
%                 disp(sprintf('ICI %6.2f adjusted tol_time=%6.2f',ici_cand(JJ),threshold));
%                % subplot(2,1,1);
%                 if all(best_guess<threshold)
%                     hh=plot(tpred,350,'ro','markerfacecolor',[1 1 1],'markersize',18);
%                 elseif any(best_guess<threshold)
%                     hh=plot(tpred,350,'go','markerfacecolor',[1 1 1],'markersize',15);
%                     disp(sprintf('mismatch %10.6f seconds\n',best_guess'));
%                 else
%                     hh=plot(tpred,350,'k^','markerfacecolor',[1 1 1],'markersize',12);
%                     disp(sprintf('mismatch %10.6f seconds\n',best_guess));
%                 end
%                 pause;
% 
%                 set(hh,'Vis','off')
% 
%             end

        end

        %If we find a pulse at all predicted ICIs within tolerance that has
        %best score so far
            
        if hits>=max([max_hits+1 num_clicks-num_misses]), %ICI(I)=ici_cand(JJ)/hits;
            if debug_flag&&(I>=debug_index)
                disp(sprintf('for candidate %i, %s, ICI value %6.2f had %i hits logged out of %i trials, need %i', ...
                    I, datestr(tabs(I)),ici_cand(JJ),hits,Icount,num_clicks-num_misses));
                disp(sprintf('current ICI value %6.2f final ICI value %6.2f', ...
                    ICI(I),ici_cand(JJ)));
               pause;
                
            end

            ICI(I)=ici_cand(JJ);
            if ICI(I)<min(ici_range)
                %disp('ICI does not fit inside range');
                ICI(I)=-1;
            end
            
            %If have a hit every time, don't bother to go further.
            if hits==num_clicks
                break
            end
            max_hits=hits;
            %%Option to break out if criteria satisfied...
            break
%         elseif debug_flag&&(I>=debug_index)
%             disp(sprintf('no ICI value fit for candidate %i.',I));
%             
       
        end


    end  %Have cycled through all candidates--didn't find one..
    if debug_flag&&(I>=debug_index)
        disp(sprintf('Final ICI value %6.2f ', ICI(I)));
        close all;
    end
end

% plot(tabs,feature_array(2,:)/1000,'gx')
% datetick('x',14)
% ylim([0 10])
% hold on
% plot(tabs,med/1000,'rx')
% keyboard;

    function [Icand]=feature_filter(Jcand)
        for J=1:length(tol_feature)

%             if strcmp(feature_names{J},'peak_duration')
%                 median_val=median(feature_array(J,Jcand));
%                 mean_val=mean(feature_array(J,Jcand));
%                 if median_val>=max_duration_val
%                     continue
%                 end
% 
%             end
            fitt=feature_array(J,Jcand)-feature_array(J,I);
            Igood=find(abs(fitt)<tol_feature(J));
            Jcand=Jcand(Igood);
            if debug_flag&&(I>=debug_index)
                subplot(length(tol_feature),1,J);
                [N,X]=hist(fitt,100);
                bar(X,N);
                hold on
                line([1 1]*(tol_feature(J)),[0 max(N)]);
                line(-[1 1]*tol_feature(J),[0 max(N)]);
                hold off;
                grid on;
                title(sprintf('mismatch between target and candidate %s...',Idebug.names{J}));
                %keyboard;
            end
        end

        Icand=Jcand;
    end

    function plot_intermediate_debug_info
        
        twindow=120;
        tback=min([trel(I) twindow/2]);
        t0=ctime(I)-tback;
        [y,t,head]=readfile(Idebug.fname,t0,twindow,1,'ctime','no');
        [S,T,F,PP] = spectrogram(y,256,2*(64+32),256,head.Fs);
        figure;
        set(gcf,'pos',[555         469        1342         545]);
        subplot(2,1,1)
        imagesc(F,T,10*log10(PP));axis('xy');caxis([0 40]);
        set(gca,'fontweight','bold','fontsize',14);
        xlabel('Time (sec)');
        ylabel('Frequency (Hz)');
        title(sprintf('interval test for detection at %s',ctime2str(ctime(I))));
        
        %Mark candidates
        t_true=(ctime(1)+trel-t0);
        t_cand_plot=ctime(1)+t_cand-t0;
        hold on;
        
        %plot features
        IIplot=[];
        for JJ=1:length(feature_names),
            if ~isempty(findstr(feature_names{JJ},'freq'))
                hh=plot(t_true,feature_array(JJ,:),'rs','markerfacecolor',((JJ-1)/2)*[1 1 1]);
                IIplot=[IIplot JJ];
            elseif ~isempty(findstr(feature_names{JJ},'duration'))
                line([t_true'; t_true'+feature_array(JJ,:)],[1 1]'*(feature_array([1 ],:)),'Color',[1 1 1]);
            elseif ~isempty(findstr(feature_names{JJ},'bearing'))
                subplot(2,1,2)
                plot(tabs(Icand_t),feature_array(JJ,Icand_t),'x');hold on;
                plot(tabs(I),feature_array(JJ,I),'ro');
                plot(tabs(Icand),feature_array(JJ,Icand),'rs');
                xlim([tabs(I)+[-1 1]*datenum(0,0,0,0,0,twindow/2)]);
                datetick('x',14,'keeplimits');grid on
                
                subplot(2,1,1)
                hh=plot(t_true,feature_array(JJ,:),'rh','markerfacecolor',((JJ-1)/2)*[1 1 1],'markersize',7);
                
            end
        end
        plot(tback,200,'wd','markerfacecolor',[0 0 0],'markersize',14);
        %hh=plot(t_true+feature_array(2,:)/head.Fs,feature_array(1,:),'rx');
        
        %Plot durations
        plot(t_cand_plot,250,'go','markerfacecolor',[1 0 1],'markersize',10);
        
        if isempty(ici_cand),
            disp(sprintf('Index %i has no ici candidates',I));
        end
        pause;
    end
end
