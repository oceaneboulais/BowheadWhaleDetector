
function simple_scatter3_classic(X0,type)
% Scatter3 with independent X,Y,Z sliders + edit boxes showing current limits
rng(0)
%X0 = randn(500,3);

% Figure and axes
fig = figure('Name','Scatter3 Limits & Edits','Position',[200 200 880 520]);
ax = axes('Parent',fig,'Position',[0.05 0.15 0.62 0.8]);
h = scatter3(ax,X0(:,1),X0(:,2),X0(:,3),3,type,'filled');
axis(ax,'equal')

% UI positions
x0 = 540; y = 400; w = 300; htxt = 20; gap = 50; editW = 120;

% X limit label, slider, edit
uicontrol(fig,'Style','text','Position',[x0 y w htxt],'String','X limit [min max]','HorizontalAlignment','left');
sldX = uicontrol(fig,'Style','slider','Position',[x0 y-20 w-140 15],'Min',-8,'Max',8,'Value',3,'Callback',@onControl);
edtX = uicontrol(fig,'Style','edit','Position',[x0+w-130 y-22 editW 22],...
    'String',sprintf('%.2f %.2f',-3,3),'Callback',@onEdit);

% Y limit label, slider, edit
y = y - gap;
uicontrol(fig,'Style','text','Position',[x0 y w htxt],'String','Y limit [min max]','HorizontalAlignment','left');
sldY = uicontrol(fig,'Style','slider','Position',[x0 y-20 w-140 15],'Min',-8,'Max',8,'Value',3,'Callback',@onControl);
edtY = uicontrol(fig,'Style','edit','Position',[x0+w-130 y-22 editW 22],...
    'String',sprintf('%.2f %.2f',-3,3),'Callback',@onEdit);

% Z limit label, slider, edit
y = y - gap;
uicontrol(fig,'Style','text','Position',[x0 y w htxt],'String','Z limit [min max]','HorizontalAlignment','left');
sldZ = uicontrol(fig,'Style','slider','Position',[x0 y-20 w-140 15],'Min',-8,'Max',8,'Value',3,'Callback',@onControl);
edtZ = uicontrol(fig,'Style','edit','Position',[x0+w-130 y-22 editW 22],...
    'String',sprintf('%.2f %.2f',-3,3),'Callback',@onEdit);

% Az/El controls (sliders)
y = y - gap;
uicontrol(fig,'Style','text','Position',[x0 y w htxt],'String','Azimuth (deg)','HorizontalAlignment','left');
sldAz = uicontrol(fig,'Style','slider','Position',[x0 y-20 w 15],'Min',0,'Max',360,'Value',45,'Callback',@onControl);

y = y - gap;
uicontrol(fig,'Style','text','Position',[x0 y w htxt],'String','Elevation (deg)','HorizontalAlignment','left');
sldEl = uicontrol(fig,'Style','slider','Position',[x0 y-20 w 15],'Min',-90,'Max',90,'Value',30,'Callback',@onControl);

% Checkbox: transform or change view
chk = uicontrol(fig,'Style','checkbox','Position',[x0 30 320 20],...
    'String','Rotate coordinates (transform points)','Value',0,'Callback',@onControl);

% Store handles/data
ud.X0 = X0; ud.h = h; ud.ax = ax;
ud.sldX = sldX; ud.sldY = sldY; ud.sldZ = sldZ;
ud.edtX = edtX; ud.edtY = edtY; ud.edtZ = edtZ;
ud.sldAz = sldAz; ud.sldEl = sldEl; ud.chk = chk;
fig.UserData = ud;

% Initialize limits to symmetric [-3 3]
setInitialLimits([-3 3], [-3 3], [-3 3]);
updateDisplay();

% --- Nested callbacks -------------------------------------------------
    function onControl(src,~)
        % Generic callback for sliders/checkbox: update limits/view
        ud = fig.UserData;
        % Read current limits from axes (they may be changed earlier)
        % But prefer sliders for symmetric magnitude: slider value is positive magnitude.
        Lx = get(ud.sldX,'Value');
        Ly = get(ud.sldY,'Value');
        Lz = get(ud.sldZ,'Value');
        % Keep symmetric around zero when using sliders
        xlim(ud.ax,[-abs(Lx) abs(Lx)]);
        ylim(ud.ax,[-abs(Ly) abs(Ly)]);
        zlim(ud.ax,[-abs(Lz) abs(Lz)]);
        % Update edit boxes to reflect current limits
        updateEditsFromAxes();
        % Apply rotation or view
        applyRotationAndView();
    end

    function onEdit(src,~)
        % Parse edited text like "min max" and set axes limits; update sliders
        ud = fig.UserData;
        str = strtrim(get(src,'String'));
        vals = sscanf(str,'%f %f');
        if numel(vals)==2 && vals(1)if src==ud.edtX
                xlim(ud.ax,[vals(1) vals(2)]);
                % set slider to symmetric magnitude (max absolute)
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
        doTransform = get(ud.chk,'Value');
        if doTransform
            R = rotationMatrix(az,el);
            Xt = (R * ud.X0')';
            set(ud.h,'XData',Xt(:,1),'YData',Xt(:,2),'ZData',Xt(:,3));
            % keep camera fixed for transformed-data mode
            view(ud.ax,[45 30]);
        else
            set(ud.h,'XData',ud.X0(:,1),'YData',ud.X0(:,2),'ZData',ud.X0(:,3));
            view(ud.ax,[az el]);
        end
        drawnow
    end

    function updateDisplay()
        % refresh edits and plot to be consistent with current state
        updateEditsFromAxes();
        applyRotationAndView();
    end
end

function R = rotationMatrix(az,el)
az = deg2rad(az); el = deg2rad(el);
Rz = [ cos(az) -sin(az) 0; sin(az) cos(az) 0; 0 0 1];
Ry = [ cos(el) 0 sin(el); 0 1 0; -sin(el) 0 cos(el)];
R = Ry * Rz;
end