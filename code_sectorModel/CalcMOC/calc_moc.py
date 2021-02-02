# Generated with SMOP  0.41
import numpy as np

def calc_moc(w=None,wmask=None,rac=None,addlayer=None):

    # This function is going to attempt to calculate the overturning streamfunction
    # from the vertical velocity, instead of the v velocity.
    ## -----------------------------------------------------------------------------
    
    # Calculate the flux through the top of a w cell.
    flux = np.multiply( np.multiply( w, np.broadcast_to(rac,w.shape) ), np.broadcast_to(wmask,w.shape) )
    print(flux.shape)

    # Sum up over the x direction.
    flux_ave=np.sum(flux,3)
    print(flux_ave.shape)

    # Integrate/cumulatively sum in the y direction.
    psi=np.cumsum(flux_ave,2)
    print(psi.shape)
    if addlayer:
        zeros = np.zeros((psi.shape[0],1,psi.shape[2]))
        psi = np.append(psi, zeros, axis=1)
    print(psi.shape)
 
    ### -----------------------------------------------------------------------------
    #
    #varargin = calc_moc.varargin
    #nargin = calc_moc.nargin

    ## This function is going to attempt to calculate the overturning streamfunction
    ## from the vertical velocity, instead of the v velocity.
    ### -----------------------------------------------------------------------------
    #
    ## Get the grid dimenions.
    #nz,ny,nx=size(wmask,nargout=3)
    ## Calculate the flux through the top of a w cell.
    #flux=multiply(multiply(w,permute(repmat(rac,concat([1,1,nz])),concat([3,1,2]))),wmask)
    ## Sum up over the x direction.
    #flux_ave=nansum(flux,3)
    ## Integrate/cumulatively sum in the y direction.
    #psi=cumsum(flux_ave,2)
    #if addlayer:
    #    psi[end() + 1,arange()]=0.0
    #
    ### -----------------------------------------------------------------------------
    
    return psi
