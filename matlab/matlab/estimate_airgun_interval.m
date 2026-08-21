function ICI=estimate_airgun_interval(tabs,bearing,param)
%function ICI=estimate_airgun_interval(tabs,bearing,param)

%%Compute ICI (regular intervals) from raw data, using selected features
%%for assistance.


try
    ICI=compute_ici_bothways_feature(tabs,param.ICI_range, ...
        bearing, {'bearing'},...
        param.Ndet,param.Nmiss,...
        param.tol_feature,param.ICItol);


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
    %     debug: if true, plot results
    %    titstr:  if debug is true, this is title to put on graph.
    %
catch
    disp('Check your MATLAB path: ICI_detectors may be missing');
    keyboard
end

if isfield(param,'debug') & param.debug
    %figure(2+10*Iday+Id)
    figure
    subplot(2,2,1)
    plot(tabs,ICI,'x');
    xtickangle(90);grid on
    datetick('x',30)
    title(param.titstr,'interp','none')

    subplot(2,2,3)
    plot(bearing,ICI,'x');grid on;xlabel('Azimuth (deg)');ylabel('ICI (sec)');

end
%%Added April 5, 2010:  Sometimes walrus sequences and heavy bowhead whale
%%sequences can have an ICI, but the ICI is inconsistent between calls.
%  Thus here we march through each ICI detection and check whether detections
%  nearby share the same ICI.

Iguns=find(ICI>0);

tsec=(tabs-tabs(1))*24*3600;

ICI_score=ones(size(ICI));
for I=1:length(Iguns)
    current_time=tsec((Iguns(I)));
    current_ICI=ICI(Iguns(I));
    Itest=find(abs(current_time-tsec((Iguns)))<=0.5*param.Ndet*current_ICI);
    Ipass=0;
    for J=1:2  %Harmonic loop:  checks for possibility that a 10 s ICI may have been assigned a 20 s ICI.
        Ipass=Ipass+length(find(abs(ICI(Iguns(Itest))/J-current_ICI)<=param.ICI_std));
    end

    %%Are there enough matching ICIs close to the value of the current ICI?
    if (Ipass-1)<param.Nstd
        ICI_score(Iguns(I))=0;  %%%Not an airgun
    else
        %disp('good');
    end

end

ICI=ICI.*ICI_score;

if isfield(param,'debug') & param.debug
    %figure(2)
    subplot(2,2,2)
    plot(tabs((ICI>0)),ICI(ICI>0),'x');
    xtickangle(90);grid on
    datetick('x',30)
    title(param.titstr,'interp','none')

    subplot(2,2,4)
    plot(bearing,ICI,'x');grid on;xlabel('Azimuth (deg)');ylabel('ICI (sec)');
    pause
end
