
function Scatterplot_GUI
% Simple GUI: adjust symmetric axis limits and rotate coordinates

% Sample data
rng(0)
N = 500;
X0 = randn(N,3);

% Create UI
fig = uifigure('Name','Scatter3 Limits & Rotation','Position',[200 200 800 500]);

ax = uiaxes(fig,'Position',[25 80 550 390]);
ax.Box = 'on';
ax.View = [45 30];
ax.Toolbar.Visible = 'off';
hold(ax,'on')

% Scatter plot (store original data in appdata)
h = scatter3(ax,X0(:,1),X0(:,2),X0(:,3),36,'filled');
axis(ax,'equal')

% Default limits
Lx = 3; Ly = 3; Lz = 3;
xlim(ax,[-Lx Lx]); ylim(ax,[-Ly Ly]); zlim(ax,[-Lz Lz]);

% UI controls positions
x0 = 600; w = 160; hctrl = 20; gap = 40;
ypos = 400;

% X limit slider (symmetric ±L)
uuilabel(fig,Position=[x0 ypos+10 160 20],Text='X limit ±L');
sldX = uislider(fig,Position=[x0 ypos  w hctrl],Limits=[0.1 10],Value=Lx,ValueChangingFcn=@update);

ypos = ypos - gap;
uuilabel(fig,Position=[x0 ypos+10 160 20],Text='Y limit ±L');
sldY = uislider(fig,Position=[x0 ypos  w hctrl],Limits=[0.1 10],Value=Ly,ValueChangingFcn=@update);

ypos = ypos - gap;
uuilabel(fig,Position=[x0 ypos+10 160 20],Text='Z limit ±L');
sldZ = uislider(fig,Position=[x0 ypos  w hctrl],Limits=[0.1 10],Value=Lz,ValueChangingFcn=@update);

ypos = ypos - gap;
uuilabel(fig,Position=[x0 ypos+10 160 20],Text='Azimuth (deg)');
sldAz = uislider(fig,Position=[x0 ypos  w hctrl],Limits=[0 360],Value=45,ValueChangingFcn=@update);

ypos = ypos - gap;
uuilabel(fig,Position=[x0 ypos+10 160 20],Text='Elevation (deg)');
sldEl = uislider(fig,Position=[x0 ypos  w hctrl],Limits=[-90 90],Value=30,ValueChangingFcn=@update);

% Checkbox to rotate coordinates (vs. only changing view)
chk = uicheckbox(fig,Position=[x0 20 200 20],Text='Rotate coordinates (transform points)',Value=false,ValueChangedFcn=@update);

% Store data in figure for callbacks
fig.UserData.X0 = X0;
fig.UserData.h = h;
fig.UserData.ax = ax;
fig.UserData.sldX = sldX;
fig.UserData.sldY = sldY;
fig.UserData.sldZ = sldZ;
fig.UserData.sldAz = sldAz;
fig.UserData.sldEl = sldEl;
fig.UserData.chk = chk;

% Initial update
update();

% Callback
    function update(~,evt)
        % Read UI values (ValueChangingFcn supplies evt with Value)
        if nargin>1 && isfield(evt,'Value')
            % ValueChangingFcn passed only the changed value; but we always
            % read all slider values to keep consistent behavior.
        end
        Lx = fig.UserData.sldX.Value;
        Ly = fig.UserData.sldY.Value;
        Lz = fig.UserData.sldZ.Value;
        az = fig.UserData.sldAz.Value;
        el = fig.UserData.sldEl.Value;
        doTransform = fig.UserData.chk.Value;

        ax = fig.UserData.ax;
        % Apply symmetric limits
        xlim(ax,[-Lx Lx]);
        ylim(ax,[-Ly Ly]);
        zlim(ax,[-Lz Lz]);

        % Either transform coordinates or just set view
        X0 = fig.UserData.X0;
        if doTransform
            R = rotationMatrix(az,el);    % rotates points by az/el (degrees)
            Xt = (R * X0')';
            set(fig.UserData.h,'XData',Xt(:,1),'YData',Xt(:,2),'ZData',Xt(:,3));
            % keep a default view for inspect
            view(ax, [45 30]);
        else
            set(fig.UserData.h,'XData',X0(:,1),'YData',X0(:,2),'ZData',X0(:,3));
            view(ax, [az el]);
        end
        drawnow
    end

end

% Rotation: first rotate about z by az, then about x (or y) for elevation.
function R = rotationMatrix(az,el)
az = deg2rad(az);
el = deg2rad(el);
Rz = [ cos(az) -sin(az) 0;
       sin(az)  cos(az) 0;
       0        0       1];
Ry = [ cos(el) 0 sin(el);
       0       1 0;
      -sin(el) 0 cos(el)];
R = Ry * Rz;
end