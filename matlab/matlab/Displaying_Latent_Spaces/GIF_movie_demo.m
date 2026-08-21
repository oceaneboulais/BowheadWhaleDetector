%%%sample rotate image
function GIF_movie_demo(x,type,alpha_value,titstr,initial_azi,initial_el)
% Create sample 3D scatter data
%rng(0);
%n = 500;
%X = randn(n,1);
%Y = randn(n,1);
%Z = randn(n,1) + 0.5*X; % correlated for nicer structure
%c = sqrt(X.^2+Y.^2+Z.^2);

% Create figure and initial scatter3
scale=1;  %Size of plot.  The bigger the scale the smaller the file.
fig = figure('Color','w','Position',[200 200 1000/scale 1000/scale]);
ax = axes('Parent',fig);
% ss(Idir,J)=scatter3(data.x_tsne(Itype,1), data.x_tsne(Itype,2), data.x_tsne(Itype,3), 3,type(Itype),'filled');

%%%Plot different symbols for different types
% str={'o','square','diamond'};
% types=unique(type);
% for I=1:length(types)
%     Itype=find(type==types(I));
%     s{I} = scatter3(ax, x(Itype,1), x(Itype,2),x(Itype,3), 3, type(Itype),'filled');
%     s{I}.MarkerEdgeAlpha=alpha_value;
%     s{I}.MarkerFaceAlpha=alpha_value;
%     s{I}.Marker=str{I};
%     hold on
% end


s = scatter3(ax, x(:,1), x(:,2),x(:,3), 3, type,'filled');
s.MarkerEdgeAlpha=alpha_value;
s.MarkerFaceAlpha=alpha_value;
%s.Marker=str{I};
hold on

colormap(jet)
colorbar
axis equal
xlabel('UMAP Dimension 1');
ylabel('UMAP Dimension 2');
zlabel('UMAP Dimension 3');
%title(titstr);

% Lighting and view
%view(45,25)
view(initial_azi,initial_el);
grid on

drawnow

% Parameters for rotation and output
nFrames = 120/2;         % number of frames in full rotation
az0 = initial_azi;              % starting azimuth
el0 =initial_el;               % elevation (fixed)
outputGIF = true;      % set false to skip GIF creation
gifName = titstr;
delayTime = 8*0.03;      % delay between frames in seconds
maxlimm=4;
% Preallocate capture
frames(nFrames) = struct('cdata',[],'colormap',[]);

% Rotate by changing azimuth angle

axis vis3d
view([0 90]);
for k = 1:nFrames
    az = bnorm(az0 + 180*(k-1)/nFrames);
    az=az0;
    el = bnorm(el0 + 360*(k-1)/nFrames);

    R = rotationMatrix(az,el);
    Xt = (R * x')';
    set(s,'XData',Xt(:,1),'YData',Xt(:,2),'ZData',Xt(:,3));
    xlim(maxlimm*[-1 1]);ylim(maxlimm*[-1 1]);zlim(maxlimm*[-1 1])
    %view(ax, az, el)

    %camorbit(360/nFrames,0)
    title(sprintf('%s az: %6.2f el:%6.2f',titstr,el,az),'Interpreter','none');
    % drawnow
    frames(k) = getframe(fig);
    if outputGIF
        img = frame2im(frames(k));
        [A,map] = rgb2ind(img,256);
        if k == 1
            imwrite(A,map,gifName,'gif','LoopCount',Inf,'DelayTime',delayTime);
        else
            imwrite(A,map,gifName,'gif','WriteMode','append','DelayTime',delayTime);
        end
    end
end

% If not writing GIF, play movie once
if ~outputGIF
    movie(fig, frames, 1, 1/delayTime)
end
end
% Close or leave figure open as desired (comment/uncomment)
% close(fig)

function R = rotationMatrix(az,el)
az = deg2rad(az); el = deg2rad(el);
Rz = [ cos(az) -sin(az) 0; sin(az) cos(az) 0; 0 0 1];
Ry = [ cos(el) 0 sin(el); 0 1 0; -sin(el) 0 cos(el)];
R = Ry * Rz;
end