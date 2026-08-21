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
scale=2;  %Size of plot.  The bigger the scale the smaller the file.
fig = figure('Color','w','Position',[200 200 700/scale 600/scale]);
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
xlabel('Dimension 1');
ylabel('Dimension 2');
zlabel('Dimension 3');
%title(titstr);

% Lighting and view
%view(45,25)
view(initial_azi,initial_el);
grid on
%xlim([-2 2]);ylim([-2 2]);zlim([-2 2]);

drawnow

% Parameters for rotation and output
nFrames = 120/2;         % number of frames in full rotation
az0 = initial_azi;              % starting azimuth
el =initial_el;               % elevation (fixed)
outputGIF = true;      % set false to skip GIF creation
gifName = titstr;
delayTime = 8*0.03;      % delay between frames in seconds

% Preallocate capture
frames(nFrames) = struct('cdata',[],'colormap',[]);

% Rotate by changing azimuth angle

axis vis3d
for k = 1:nFrames
    az = bnorm(az0 + 180*(k-1)/nFrames);
    %view(ax, az, el)

    camorbit(360/nFrames,0)
    title(sprintf('%s az: %6.2f el:%6.2f',titstr,az,el),'Interpreter','none');
    drawnow
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

% Close or leave figure open as desired (comment/uncomment)
% close(fig)