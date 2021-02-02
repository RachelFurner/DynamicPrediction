%% -----------------------------------------------------------------------------

% Decide whether or not to plot uncoloured contours.
schematic = 0;

% Extract the bolus velocity and streamfunctions from the file.
[ data.iso_slope data.bmoc ] = load_bolus( name, grd, ndx, kt );

% Integrate the meridional bolus velocity to get overturning streamfunction.
addlayer = 1;

% on w points.
data.emoc = calc_moc( data.w, grd.cmask, grd.rac, addlayer );
data.rmoc = data.emoc + data.bmoc;

%% -----------------------------------------------------------------------------
% Plot the Eulerian overturning.

figure;

bstmax = 15.0;
bstint = 2*bstmax/40;
bstlev = ( -bstmax:bstint:bstmax );

contourf( [ -60., grd.latc( 1:end-grd.ky ), 60. ], -grd.zgpsi, ...
    [ zeros( grd.nz+1, 1 ), data.emoc( :, 1:end-grd.ky ), zeros( grd.nz+1, 1 ) ]/1E6, bstlev );
hold on;

caxis( [ -1 1 ]*bstmax );
colorbar( 'v' );
colormap( bluewhitered( 40 ) );
axis( [ -60 60 -4000 0 ] );

set( gca, ...
    'dataaspectratio', [ 35 2500 1 ], 'plotboxaspectratio', [ 35 2500 1 ], ...
    'fontsize', 25, 'xtick', ( -60.:20.:60. ), 'ytick', ( -4000.:1000.:0 ) );

%tit = title( 'Eulerian Overturning on w grid (Sv)' );
%set( tit, 'fontsize', 30, 'fontweight', 'bold' );

xl = xlabel( 'Latitude' );
yl = ylabel( 'Depth' );
set( [ xl yl ], 'fontsize', 30, 'fontweight', 'bold' );

set( gcf, 'color', 'w' );

%% -----------------------------------------------------------------------------

figure;

contourf( [ -60., grd.latc( 1:end-grd.ky ), 60. ], -grd.zgpsi, ...
    [ zeros( grd.nz+1, 1 ), data.bmoc( :, 1:end-grd.ky ), zeros( grd.nz+1, 1 ) ]/1E6, bstlev );
hold on;

caxis( [ -1 1 ]*bstmax );
colorbar( 'v' );
colormap( bluewhitered( 40 ) );
axis( [ -60 60 -4000 0 ] );

set( gca, ...
    'dataaspectratio', [ 35 2500 1 ], 'plotboxaspectratio', [ 35 2500 1 ], ...
    'fontsize', 25, 'xtick', ( -60.:20.:60. ), 'ytick', ( -4000.:1000.:0 ) );

%tit = title( 'Bolus Overturning on w grid (Sv)' );
%set( tit, 'fontsize', 30, 'fontweight', 'bold' );

xl = xlabel( 'Latitude' );
yl = ylabel( 'Depth' );
set( [ xl yl ], 'fontsize', 30, 'fontweight', 'bold' );

set( gcf, 'color', 'w' );

%% -----------------------------------------------------------------------------

figure;

contourf( [ -60., grd.latc( 1:end-grd.ky ), 60. ], -grd.zgpsi, ...
    [ zeros( grd.nz+1, 1 ), data.rmoc( :, 1:end-grd.ky ), zeros( grd.nz+1, 1 ) ]/1E6, bstlev );
hold on;

caxis( [ -1 1 ]*bstmax );
cb = colorbar( 'v' );
colormap( bluewhitered( 40 ) );
axis( [ -60 60 -4000 0 ] );

set( gca, 'color', 'none', 'dataaspectratio', [ 35 2500 1 ], ...
    'fontname', 'arial', 'fontsize', 36, 'fontweight', 'bold', ...
    'linewidth', 5, ...
    'plotboxaspectratio', [ 35 2500 1 ], ...
    'xtick', ( -60.:20.:60. ), 'ytick', ( -4000.:1000.:0 ) );

set( cb, 'fontname', 'arial', 'fontsize', 36, 'fontweight', 'bold', 'linewidth', 5 );

%l = line( [ -60 -40 -40 ], [ -2000. -2000. 0. ] );
%set( l, 'color', [ 0 0 0 ], 'linestyle', '--', 'linewidth', 5 );
 
%tit = title( 'Residual Overturning on w grid (Sv)' );
%set( tit, 'fontsize', 30, 'fontweight', 'bold' );
   
xl = xlabel( 'Latitude' );
yl = ylabel( 'Depth' );
set( [ xl yl ], 'fontsize', 48, 'fontweight', 'bold' );

set( gcf, 'color', 'w' );

%% -----------------------------------------------------------------------------

if schematic;
    figure;

    bstlev = ( -15:0.5:15 );

    contour( [ -60., grd.latc( 1:end-grd.ky ), 60. ], -grd.zgpsi, ...
        [ zeros( grd.nz+1, 1 ), data.bmoc ]/1E6, bstlev, '-k' );
    hold on;

    caxis( [ -1 1 ]*15.0 );
    colorbar( 'v' );
    colormap( bluewhitered( 40 ) );
    axis( [ -60 60 -4000 0 ] );

    set( gca, ...
        'dataaspectratio', [ 35 2500 1 ], 'plotboxaspectratio', [ 35 2500 1 ], ...
        'fontsize', 25, 'xtick', ( -60.:20.:60. ), 'ytick', ( -4000.:1000.:0 ) );

    l = line( [ -60 -40 -40 ], [ -2500. -2500. 0. ] );
    set( l, 'color', [ 0 0 0 ], 'linestyle', '--', 'linewidth', 5 );
    
    xl = xlabel( 'Latitude' );
    yl = ylabel( 'Depth' );
    set( [ xl yl ], 'fontsize', 30, 'fontweight', 'bold' );

    set( gcf, 'color', 'w' );
end;

%close( 1:4 );

%% -----------------------------------------------------------------------------
% Flush the memory.

clear bstlev;
clear tit xl yl;

%% -----------------------------------------------------------------------------
