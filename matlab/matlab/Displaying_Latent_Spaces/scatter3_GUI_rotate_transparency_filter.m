


function [fig]=scatter3_GUI_rotate_transparency_filter(X0,features,default_view,zlimm_want)
% Scatter3 with independent X,Y,Z sliders + edit boxes for axis limits
% and edit boxes for Azimuth and Elevation
% Alpha (transparency) slider + edit



feature_names=fieldnames(features);
type=features.(feature_names{end});
Igood=1:size(X0,1);

if ~exist("default_view","var")
    default_view=[];
end
if ~exist("zlimm_want","var")
    zlimm_want=[];
end
rng(0)
%X0 = randn(500,3);
alpha_value=0.2;  %Initial value for transparency.
sizze=8;
sizze=10;

% Figure and axes
fig = figure('Name','Scatter3 Limits & Az/El Edits','Units','normalized','Position',[0.05 0.9 0.9 0.9]);
ax = axes('Parent',fig,'Position',[0.05 0.15 0.75 0.8]);
h = scatter3(ax,X0(Igood,1),X0(Igood,2),X0(Igood,3),sizze,type,"o","filled");
h.MarkerEdgeAlpha=alpha_value;
h.MarkerFaceAlpha=alpha_value;
% %I would like
%to change shape of points that have changed their type, but this would
%require a separate scatterplot.  Consider using a unique color...


xlabel('x');ylabel('y');zlabel('z');
axis(ax,'equal')
colorbar('location','south')
%view(ax,[0 0]);

% UI positions
x0 = 1300; y = 420+150; w = 320; htxt = 20; gap = 50; editW = 80;
[~,user_name]=system('whoami');
if contains(user_name,'angel')
    disp('Adjusting controls on Angels computer...')
    x0=x0-300;
elseif contains(user_name,'raquel')
    disp('Adjusting controls on Raquel computer...')
    x0=x0-500;
end

% Add an edit box displaying current date and time
y = y + gap;
dtstr = datestr(datetime('now'),'yyyy-mm-dd HH:MM:SS');
uicontrol(fig,'Style','text','Position',[x0 y w htxt],'String','Freeze edits after','HorizontalAlignment','left');
edt_time=uicontrol(fig,'Style','edit','Position',[x0+110 y 190 24],...
    'String',dtstr,'Enable','on','BackgroundColor',[1 1 1]);

% X limit label, slider, edit
y = y - gap;
uicontrol(fig,'Style','text','Position',[x0 y w htxt],'String','X limit [min max]','HorizontalAlignment','left');
sldX = uicontrol(fig,'Style','slider','Position',[x0 y-20 w-140 15],'Min',-8,'Max',8,'Value',3,'Callback',@onControl);
edtX = uicontrol(fig,'Style','edit','Position',[x0+w-130 y-22 editW 22],'String',sprintf('%.2f %.2f',-3,3),'Callback',@onEdit);

% Y limit label, slider, edit
y = y - gap;
uicontrol(fig,'Style','text','Position',[x0 y w htxt],'String','Y limit [min max]','HorizontalAlignment','left');
sldY = uicontrol(fig,'Style','slider','Position',[x0 y-20 w-140 15],'Min',-8,'Max',8,'Value',3,'Callback',@onControl);
edtY = uicontrol(fig,'Style','edit','Position',[x0+w-130 y-22 editW 22],'String',sprintf('%.2f %.2f',-3,3),'Callback',@onEdit);

% Z limit label, slider, edit
y = y - gap;
uicontrol(fig,'Style','text','Position',[x0 y w htxt],'String','Z limit [min max]','HorizontalAlignment','left');
sldZ = uicontrol(fig,'Style','slider','Position',[x0 y-20 w-140 15],'Min',-8,'Max',8,'Value',3,'Callback',@onControl);
edtZ = uicontrol(fig,'Style','edit','Position',[x0+w-130 y-22 editW 22],'String',sprintf('%.2f %.2f',-3,3),'Callback',@onEdit);

% Azimuth slider + edit
y = y - gap;
uicontrol(fig,'Style','text','Position',[x0 y w htxt],'String','Azimuth (deg)','HorizontalAlignment','left');
sldAz = uicontrol(fig,'Style','slider','Position',[x0 y-20 w-100 15],'Min',0,'Max',360,'Value',45,'Callback',@onControl);
edtAz = uicontrol(fig,'Style','edit','Position',[x0+w-90 y-22 editW 22],'String',sprintf('%.1f',45),'Callback',@onAzElPEdit);

% Elevation slider + edit
y = y - gap;
uicontrol(fig,'Style','text','Position',[x0 y w htxt],'String','Elevation (deg)','HorizontalAlignment','left');
sldEl = uicontrol(fig,'Style','slider','Position',[x0 y-20 w-100 15],'Min',-90,'Max',90,'Value',30,'Callback',@onControl);
edtEl = uicontrol(fig,'Style','edit','Position',[x0+w-90 y-22 editW 22],'String',sprintf('%.1f',30),'Callback',@onAzElPEdit);

% Pitch slider + edit
y = y - gap;
uicontrol(fig,'Style','text','Position',[x0 y w htxt],'String','Pitch (deg)','HorizontalAlignment','left');
sldP = uicontrol(fig,'Style','slider','Position',[x0 y-20 w-100 15],'Min',0,'Max',360,'Value',30,'Callback',@onControl);
edtP = uicontrol(fig,'Style','edit','Position',[x0+w-90 y-22 editW 22],'String',sprintf('%.1f',30),'Callback',@onAzElPEdit);


%Transparency slider
y = y - gap; % position above other controls
uicontrol(fig,'Style','text','Position',[x0 y w htxt],'String','Marker Alpha (0-1)','HorizontalAlignment','left');
sldAlpha = uicontrol(fig,'Style','slider','Position',[x0 y-20 w-100 15],'Min',0,'Max',1,'Value',alpha_value,'Callback',@onAlphaControl);
edtAlpha = uicontrol(fig,'Style','edit','Position',[x0+w-90 y-22 editW 22],'String',sprintf('%.2f',alpha_value),'Callback',@onAlphaEdit);

% Dropdown: feature field names from 'feature' struct (or variable)
% Determine field names safely
y = y - gap; % position above other controls
if exist('features','var') && isstruct(features)
    fnames = fieldnames(features);
else
    fnames = {};
end

Idefault=find(contains(fnames,'iscall'));

uicontrol(fig,'Style','text','Position',[x0 y w htxt],'String','Feature Color','HorizontalAlignment','left');
if ~isempty(fnames)
    ddFeature = uicontrol(fig,'Style','popupmenu','Position',[x0 y-20 200 22],...
        'String',fnames,'Value',Idefault,'Callback',@onFeatureSelect);
else
    ddFeature = uicontrol(fig,'Style','popupmenu','Position',[x0 y-20 200 22],...
        'String',{'<no feature struct>'},'Value',1,'Enable','off');
end

% Dropdown: filter field names from 'feature' struct (or variable)

y = y - gap; % position above other controls

ddFilter_text=uicontrol(fig,'Style','text','Position',[x0 y w htxt],'String','Filter Feature','HorizontalAlignment','left');
if ~isempty(fnames)
    ddFilter = uicontrol(fig,'Style','popupmenu','Position',[x0 y-20 200 22],...
        'String',fnames,'Value',Idefault,'Callback',@onFilterSelect);
else
    ddFilter = uicontrol(fig,'Style','popupmenu','Position',[x0 y-20 200 22],...
        'String',{'<no feature struct>'},'Value',1,'Enable','off');
end
% store handle
sel = fnames{Idefault};
% store the selected field name for other callbacks to use

% Two edit boxes to the right of ddFeature with same callback
edtFeature1 = uicontrol(fig,'Style','edit','Position',[x0+210 y-20 80 22],...
    'String',num2str(min(features.(sel)),'%3.1f'),'Callback',@onFilterFeatureEdit);
edtFeature2 = uicontrol(fig,'Style','edit','Position',[x0+300 y-20 80 22],...
    'String',num2str(max(features.(sel)),'%3.1f'),'Callback',@onFilterFeatureEdit);


% Checkbox: transform or change view
chk = uicontrol(fig,'Style','checkbox','Position',[x0 20 320 20],'String', ...
    'Rotate coordinates (transform points)','Value',1,'Callback',@onControl);

% Store handles/data
% store handles
ud = fig.UserData;
ud.ddFeature = ddFeature;
ud.selectedFeatureField = sel;
ud.selectedFilterField = sel;

ud.ddFilter = ddFilter;
ud.ddFilter_text=ddFilter_text;
ud.minVal=str2double(edtFeature1.String); 
ud.maxVal=str2double(edtFeature2.String); 
ud.edtFeature1 = edtFeature1;
ud.edtFeature2 = edtFeature2;

ud.X0 = X0; ud.h = h; ud.ax = ax;
ud.Igood=Igood;  %%%Points that meet the filter criteria...
ud.sldX = sldX; ud.sldY = sldY; ud.sldZ = sldZ;
ud.edtX = edtX; ud.edtY = edtY; ud.edtZ = edtZ;
ud.edt_time=edt_time;
ud.sldAz = sldAz; ud.sldEl = sldEl; ud.edtAz = edtAz; ud.edtEl = edtEl;
ud.sldP = sldP;  ud.edtP= edtP; 
ud.chk = chk;
ud.sldAlpha = sldAlpha;
ud.edtAlpha = edtAlpha;
ud.features=features;
ud.CData=h.CData;
fig.UserData = ud;


% Initialize symmetric limits [-3 3] and update display
setInitialLimits([-5 5], [-5 5], [-5 5]);
if ~isempty(default_view)

    set(ud.edtAz,'String',sprintf('%.1f', default_view(1)));
    set(ud.edtEl,'String',sprintf('%.1f', default_view(2)));
    set(ud.sldAz,'Value',default_view(1));
    set(ud.sldEl,'Value',default_view(2));
    % updateEditsFromAxes();
    %applyRotationAndView();
end

%%%Slice the cake to a chosen layer...
if ~isempty(zlimm_want)
    zlim(ud.ax,zlimm_want);
    set(ud.sldZ,'Value',max(abs(zlimm_want)));
    set(ud.edtZ,'String',sprintf('%4.3f %4.2f',zlimm_want(1),zlimm_want(2)));
end
updateDisplay();

% --- Callbacks -------------------------------------------------------

function onFeatureSelect(src,~)
        udtmp = fig.UserData;
        idx = get(src,'Value');
        items = get(src,'String');
        sel = items{idx};
        % store the selected field name for other callbacks to use
        udtmp.selectedFeatureField = sel;
        udtmp.CData=udtmp.features.(sel);
        %udtmp.h.CData=double(udtmp.CData(udtmp.Igood));  %%Change color...
        set(udtmp.h,'CData',double(udtmp.CData(udtmp.Igood)));  %Update color
        minval=min(udtmp.h.CData);
        maxval=max(udtmp.h.CData);
        hh = findobj(fig,'type','colorbar');
        set(hh.Label,"String",sel, "FontSize",12,"FontWeight","bold")

        clim(udtmp.ax,[minval maxval]);

        fig.UserData = udtmp;
        
end

% Callback shared by both feature edit boxes
    function onFilterFeatureEdit(src,~)
        udtmp = fig.UserData;

        val = get(src,'String');
        % store last entered string in UserData for later use
        if src == udtmp.edtFeature1
            udtmp.minVal = str2double(val);
        elseif src == udtmp.edtFeature2
            udtmp.maxVal = str2double(val);
        end

        sel=udtmp.selectedFilterField;
        udtmp.Igood=find(udtmp.features.(sel)>=udtmp.minVal & ...
            udtmp.features.(sel)<=udtmp.maxVal);

        warning off
        doTransform = get(udtmp.chk,'Value');
        if doTransform
            set(udtmp.h,'XData',udtmp.Xt(udtmp.Igood,1),'YData',...
                udtmp.Xt(udtmp.Igood,2),'ZData',udtmp.Xt(udtmp.Igood,3), ...
                'CData',double(udtmp.CData(udtmp.Igood)));
        else
            set(udtmp.h,'XData',udtmp.X0(udtmp.Igood,1),'YData',udtmp.X0(udtmp.Igood,2), ...
                'ZData',udtmp.X0(udtmp.Igood,3), ...
                'CData',double(udtmp.CData(udtmp.Igood)));
        end
        minval=min(udtmp.h.CData);maxval=max(udtmp.h.CData);
        if minval==maxval
            maxval=maxval+0.1;
        end
        
        clim(udtmp.ax,[minval maxval]);

       
        %changed_from_manual_to_auto=length()
        udtmp.ddFilter_text.String=sprintf('Filter Feature %i samples',length(udtmp.Igood));
        warning on
        fig.UserData = udtmp;
    end

    function onFilterSelect(src,~)
        udtmp = fig.UserData;
        idx = get(src,'Value');
        items = get(src,'String');
        sel = items{idx};
        % store the selected field name for other callbacks to use
        udtmp.selectedFilterField = sel;
        disp(sel)

        %%%Update edit boxes
        udtmp.minVal = min(udtmp.features.(sel));
        udtmp.maxVal = max(udtmp.features.(sel));
       
        udtmp.edtFeature1.String=num2str(udtmp.minVal,'%3.1f');
        udtmp.edtFeature2.String=num2str(udtmp.maxVal,'%3.1f');

        fig.UserData = udtmp;
        onFilterFeatureEdit(udtmp.edtFeature1);
    end

    

    function onAlphaControl(src,~)
        % slider changed -> update edit and scatter alpha
        udtmp = fig.UserData;
        a = get(src,'Value');
        set(udtmp.edtAlpha,'String',sprintf('%.2f',a));
        if isfield(udtmp,'h') && isvalid(udtmp.h)
            udtmp.h.MarkerEdgeAlpha = a;
            udtmp.h.MarkerFaceAlpha = a;
        end
        fig.UserData = udtmp;
    end

    function onAlphaEdit(src,~)
        % edit changed -> validate and update slider and scatter alpha
        udtmp = fig.UserData;
        a = str2double(get(src,'String'));
        if isnan(a) || a<0 || a>1
            % revert
            if isfield(udtmp,'sldAlpha')
                set(src,'String',sprintf('%.2f',get(udtmp.sldAlpha,'Value')));
            else
                set(src,'String',sprintf('%.2f',alpha_value));
            end
            return
        end
        set(udtmp.sldAlpha,'Value',a);
        if isfield(udtmp,'h') && isvalid(udtmp.h)
            udtmp.h.MarkerEdgeAlpha = a;
            udtmp.h.MarkerFaceAlpha = a;
        end
        fig.UserData = udtmp;
    end


    function onControl(src,~)
        % Generic callback for sliders/checkbox: update edits/view
        ud = fig.UserData;
        % Read symmetric magnitudes from sliders (use abs to ensure non-neg)
        Lx = get(ud.sldX,'Value');
        Ly = get(ud.sldY,'Value');
        Lz = get(ud.sldZ,'Value');
        xlim(ud.ax,[-abs(Lx) abs(Lx)]);
        ylim(ud.ax,[-abs(Ly) abs(Ly)]);
        zlim(ud.ax,[-abs(Lz) abs(Lz)]);
        % If az/el sliders changed, sync edit boxes
        set(ud.edtAz,'String',sprintf('%.1f', get(ud.sldAz,'Value')));
        set(ud.edtEl,'String',sprintf('%.1f', get(ud.sldEl,'Value')));
        set(ud.edtP,'String',sprintf('%.1f', get(ud.sldP,'Value')));
        updateEditsFromAxes();
        applyRotationAndView();
    end

    function onEdit(src,~)
        % Parse edited text like "min max" and set axes limits; update sliders
        ud = fig.UserData;
        str = strtrim(get(src,'String'));
        vals = str2num(str);
        if numel(vals)==2 
            if src==ud.edtX
                xlim(ud.ax,[vals(1) vals(2)]);
                set(ud.sldX,'Value',max(abs(vals)));
            elseif src==ud.edtY
                ylim(ud.ax,[vals(1) vals(2)]);
                set(ud.sldY,'Value',max(abs(vals)));
            else % edtZ
                zlim(ud.ax,[vals(1) vals(2)]);
                set(ud.sldZ,'Value',max(abs(vals)));
            end
            applyRotationAndView();
        else
            % Invalid input: revert edit box to current axes limits
            updateEditsFromAxes();
        end
    end

    function onAzElPEdit(src,~)
        % Parse single numeric az or el, update slider and view/transform
        ud = fig.UserData;
        v = str2double(get(src,'String'));
        if isnan(v)
            % revert
            set(src,'String',sprintf('%.1f', get(src==ud.edtAz && ud.sldAz || ud.sldEl,'Value')));
            return
        end
        if src==ud.edtAz 
            v = mod(v,360); % keep 0-360
            set(ud.sldAz,'Value',v);
            set(ud.edtAz,'String',sprintf('%.1f',v));
        elseif src== ud.edtP
            v = mod(v,360); % keep 0-360
            set(ud.sldP,'Value',v);
            set(ud.edtP,'String',sprintf('%.1f',v));
        else
            v = max(min(v,90),-90);
            set(ud.sldEl,'Value',v);
            set(ud.edtEl,'String',sprintf('%.1f',v));
        end
        applyRotationAndView();
    end

% --- Helpers ----------------------------------------------------------
    function setInitialLimits(xlim0, ylim0, zlim0)
        xlim(ax,xlim0); ylim(ax,ylim0); zlim(ax,zlim0);
        set(sldX,'Value',max(abs(xlim0)));
        set(sldY,'Value',max(abs(ylim0)));
        set(sldZ,'Value',max(abs(zlim0)));
        updateEditsFromAxes();
    end

    function updateEditsFromAxes()
        xl = xlim(ax); yl = ylim(ax); zl = zlim(ax);
        set(edtX,'String',sprintf('%.3g %.3g',xl(1),xl(2)));
        set(edtY,'String',sprintf('%.3g %.3g',yl(1),yl(2)));
        set(edtZ,'String',sprintf('%.3g %.3g',zl(1),zl(2)));
    end

    function applyRotationAndView()
        ud = fig.UserData;
        az = get(ud.sldAz,'Value');
        el = get(ud.sldEl,'Value');
        p = get(ud.sldP,'Value');
        doTransform = get(ud.chk,'Value');
        if doTransform
            R = rotationMatrix(az,el,p);
            Xt = (R * ud.X0')';
            set(ud.h,'XData',Xt(ud.Igood,1),'YData',Xt(ud.Igood,2),'ZData',Xt(ud.Igood,3));
            % keep camera fixed in transformed-data mode
            view(ud.ax,[0 90]);
            fig.UserData.Xt=Xt;
        else
            set(ud.h,'XData',ud.X0(ud.Igood,1),'YData',ud.X0(ud.Igood,2),'ZData',ud.X0(ud.Igood,3));
            view(ud.ax,[az el]);
        end
        %drawnow
    end

    function updateDisplay()
        updateEditsFromAxes();
        set(edtAz,'String',sprintf('%.1f', get(sldAz,'Value')));
        set(edtEl,'String',sprintf('%.1f', get(sldEl,'Value')));
        applyRotationAndView();
    end
end

function R = rotationMatrix(az,el,p)
az = deg2rad(az); el = deg2rad(el);p = deg2rad(p);
Rz = [ cos(az) -sin(az) 0; sin(az) cos(az) 0; 0 0 1];
Ry = [ cos(el) 0 sin(el); 0 1 0; -sin(el) 0 cos(el)];
Rx=[1 0 0; 0 cos(p) -sin(p); 0 sin(p) cos(p)];
R = Rx* Ry * Rz;
end
