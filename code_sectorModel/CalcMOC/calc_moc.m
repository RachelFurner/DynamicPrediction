function psi = calc_moc( w, wmask, rac, addlayer )
% This function is going to attempt to calculate the overturning streamfunction
% from the vertical velocity, instead of the v velocity.
%% -----------------------------------------------------------------------------

% Get the grid dimenions.
[ nz ny nx ] = size( wmask );

% Calculate the flux through the top of a w cell.
flux = w .* permute( repmat( rac, [ 1 1 nz ] ), [ 3 1 2 ] ) .* wmask;

% Sum up over the x direction.
flux_ave = nansum( flux, 3 );

% Integrate/cumulatively sum in the y direction.
psi = cumsum( flux_ave, 2 );

if addlayer;
    psi( end+1, : ) = 0.0;
end;

%% -----------------------------------------------------------------------------



