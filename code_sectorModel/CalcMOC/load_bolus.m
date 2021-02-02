function [ iso_slope bolus ] = load_bolus( name, grid, order, time_level )
% This function loads in the diagnostic file containing the bolus velocities. It
% extracts the required elements of the GM tensor (32/wy), and calculates
% isoneutral slope from these two diagnostics. The isoneutral slopes
% are then used to calculate the Y components of the bolus streamfunction, which
% is then used to find the bolus overturning using Stuart's method of simply
% integrating Fy in the y-direction. This overturning is averaged to the
% velocity grid points, which is one of the outputs.
%% -----------------------------------------------------------------------------

% Pre-allocate the array for speed.
kwy = cell( name.no_proc, 1 );

for j = 1:1:grid.npy;
    for i = 1:1:grid.npx;
        % Set a dummy variable for the tile number.
        k = i + ( j-1 )*grid.npx;
        % Set another dummy variable that points at the correct mnc_test_????
        % dir for the current tile.
        l = order( k );
        
        % Open the netcdf file.
        ncid = netcdf.open( fullfile( name.dname, name.run{ l }, name.su_file{ l } ), 'NOWRITE' );

         % Load the 32/wy element of the GM tensor.
        kwy{ k } = permute( netcdf.getVar( ncid, netcdf.inqVarID( ncid, 'GM_Kwy' ), ...
            [ 0 0 0 time_level-1 ], [ grid.nx/grid.npx grid.ny/grid.npy grid.nz 1 ] ), [ 3 2 1 ] );

        % Close the netcdf file.
        netcdf.close( ncid );
    end;
end;

% Concatenate the tiles together.
kwy = cat3Dgrid( grid.nx, grid.ny, grid.npx, grid.npy, kwy{ : } );

%% -----------------------------------------------------------------------------

% Convert to the Y component of the bolus streamfunction.
iso_slope = kwy/2.0;

% Integrate in the x-direction.
bolus = sum( change( iso_slope.* ...
    permute( repmat( grid.dxf, [ 1 1 grid.nz ] ), [ 3 1 2 ] ).*grid.cmask, ...
    '==', NaN, 0 ), 3 );

% Add an extra level at the bottom with zero streamfunction.
bolus( grid.nz+1, : ) = 0.0;

%% -----------------------------------------------------------------------------
