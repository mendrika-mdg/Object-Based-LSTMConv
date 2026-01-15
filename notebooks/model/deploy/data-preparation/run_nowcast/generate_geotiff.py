
import numpy as np
#import u_interpolate as uinterp
import rasterio, os,sys 
from osgeo import gdal,osr
gdal.UseExceptions()
import shlex, subprocess
import pdb
import scipy.spatial.qhull as qhull
from scipy.interpolate import griddata
import pyproj


class UKCEH_PortalProduct:
    def __init__(self,user,product,root,lats,lons,crs='EPSG:4326',weightpath='.',missing = -999.999, units = "%", minval=None,maxval =None,bounds=None,irregular=True,dx =None):
        """

        IN: irregular (optional): Are the lat-long coordinates on an irregular grid (non-constant pixel size). If so, data will be reprojected
        IN: dx (optional) : If irregular, can specify fixed resolution to project only. If not specified, average pixel size from irregular grid will be used
        IN: lats    : 2-D numpy array of irregular latitude points (same size as image)
        IN: lons    : 2-D numpy array of irregular longitude points (same size as image)
        """
        
        self.user = user
        self.product = product
        self.units = units
        self.crs = crs
        self.root = root
        self.weightpath= weightpath
        self.missing = missing
        self.minval = minval
        self.maxval  = maxval
        self.bounds = bounds
        self.irregular = irregular
        self.dx = dx
        self.dx_set = dx
        self.lats= lats
        self.lons = lons


        print("Writer initialised for "+str(self.user)+'_'+str(self.product))



    def generate_portal_geotiff(self,image,originTime,leadTime,outFile =None):
        """
        resample image onto a fixed resolution array
        IN: image   : 2-D numpy array of image to be resampled onto a fixed grid
        IN: originTime: datetime object of origin validity time of image
        IN: leadTime:  integer indicating  lead time from origin in number of minutes 
        IN: outFile (optional) : Output path and filename of final geotiff File
        
        OUT: Geotiff file in self.root folder, under datestamped subfolder, unless location specified specifically by outFile
        """


        try:
            leadTime = str(leadTime)
            if outFile is None:
                # assume it will be the main portal Directory
                outDir = os.path.join(self.root,self.user+'_'+self.product,originTime.strftime("%Y%m%d"))
                os.makedirs(outDir,exist_ok=True)
                outBasename = 'nowcast_'+originTime.strftime("%Y%m%d%H%M_")+leadTime.zfill(4)+'.tif'
                outBasename_precrop = 'nowcast_'+originTime.strftime("%Y%m%d%H%M_")+leadTime.zfill(4)+'_precrop.tif'
                outFile = os.path.join(outDir,outBasename)
                outFile_precrop = os.path.join(outDir,outBasename_precrop)
            else:
                os.makedirs(os.path.dirname(outFile),exist_ok=True)



            lat_min, lat_max= np.nanmin(self.lats[self.lats !=self.missing]),np.nanmax(self.lats[self.lats !=self.missing])
            lon_min, lon_max= np.nanmin(self.lons[self.lons !=self.missing]),np.nanmax(self.lons[self.lons !=self.missing])
      

    
            if self.irregular:
                # need to resample
                print("Resampling on to regular grid")
                # Use dx for resolution if supplied, else use lat, lon mean of  (max-min)/number of cells
                if self.dx is None:
                    av_lat = (lat_max - lat_min)/self.lats.shape[1]
                    av_lon = (lon_max - lon_min)/self.lons.shape[0]
                    dx = (av_lat+av_lon)/2.0
                    self.dx_set = dx
                else:
                    dx = self.dx
                    self.dx_set = dx

                fixed_lats = np.arange(lat_min,lat_max ,dx)
                fixed_lons = np.arange(lon_min,lon_max ,dx)


               # sys.exit(0)
                grid_lon, grid_lat = np.meshgrid(fixed_lons,fixed_lats)
                # check for weights
                coordstr= str(round(lon_min,2)).replace('-','m')+'-'+str(round(lon_max,2)).replace('-','m')+'_'+str(round(lat_min,2)).replace('-','m')+'-'+str(round(lat_max,2)).replace('-','m')
                if os.path.exists(os.path.join(self.weightpath,self.user+'_'+self.product+'_'+coordstr+'_weights.npz')):
                    print("Reading nzp weights")
                    weightdata = np.load(os.path.join(self.weightpath,self.user+'_'+self.product+'_'+coordstr+'_weights.npz'))
                    inds = weightdata['inds']
                    weights = weightdata['weights']
                    new_shape=tuple(weightdata['new_shape'])
                else:
                    print("Calculating interpolation weights")
                    inds, weights, new_shape=interpolation_weights(self.lons[np.isfinite(self.lons)], self.lats[np.isfinite(self.lats)],grid_lon, grid_lat, irregular_1d=True)
                    np.savez(os.path.join(self.weightpath,self.user+'_'+self.product+'_'+coordstr+'_weights.npz'),inds=inds,weights=weights,new_shape=np.array(new_shape))

                data_interp=interpolate_data(image, inds, weights, new_shape)
                transform = rasterio.transform.from_origin(lon_min,lat_max,dx,dx)
    
            else:
                grid_lon, grid_lat = np.meshgrid(self.lons,self.lats)
                dx = (lat_max - lat_min)/self.lats.shape[1]
                data_interp = image
                transform = rasterio.transform.from_origin(lon_min,lat_max,dx,dx)

            # apply any minmax values
            if not self.minval is None:
                data_interp[data_interp < self.minval] = self.missing
            if not self.maxval is None:
                data_interp[data_interp > self.maxval] = self.missing
            print("Generating GeoTIFF")


            if self.crs.lower()!='epsg:3857':
                rasFile = os.path.splitext(outFile)[0]+'_tmp.tif'
            else:
                rasFile = outFile
            rasImage = rasterio.open(rasFile,'w',driver='GTiff',
                                        height=data_interp.shape[0],width=data_interp.shape[1],
                                        count=1, dtype= str(data_interp.dtype),
                                        crs = self.crs,
                                        nodata=self.missing,
                                        transform = transform
                                        )   
            rasImage.write(np.flipud(data_interp[:]),1)
            rasImage.close()   
    
            # now reproject if required and remove tmp
            if self.crs.lower()!='epsg:3857':
                ds = gdal.Warp(outFile, rasFile, srcSRS=self.crs, dstSRS='EPSG:3857', format='GTiff',creationOptions=["COMPRESS=DEFLATE", "TILED=YES"])
                ds = None 
                os.system('rm '+rasFile)

            # now crop if needed
            if not self.bounds is None:
                transformer = pyproj.Transformer.from_crs(self.crs, f"EPSG:3857", always_xy=True)
                x_min, y_min = transformer.transform(self.bounds[0], self.bounds[1])
                x_max, y_max = transformer.transform(self.bounds[2], self.bounds[3])
                os.system('mv '+outFile+' '+outFile_precrop)
                gdal.Warp(outFile,outFile_precrop,outputBounds=(x_min, y_min, x_max, y_max),cropToCutline=True)
                os.system('rm '+outFile_precrop)
    
            # Include some Metatdata
           # xmp_metadata = f"""xml:XMP=<x:xmpmeta xmlns:x='adobe:ns:meta/'><rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'><rdf:Description rdf:about='' xmlns:xmp='http://ns.adobe.com/xap/1.0/' xmlns:dc='http://purl.org/dc/elements/1.1/'><dc:title><rdf:Alt><rdf:li xml:lang='en'>{self.product}</rdf:li></rdf:Alt></dc:title><dc:creator><rdf:Seq><rdf:li>{self.user}</rdf:li></rdf:Seq></dc:creator><dc:description><rdf:Alt><rdf:li xml:lang='en'>Units: {self.units}</rdf:li></rdf:Alt></dc:description></rdf:Description></rdf:RDF></x:xmpmeta>"""
            
            xmp_metadata = f"""xml:XMP=<x:xmpmeta xmlns:x=\'adobe:ns:meta/\'>
            <rdf:RDF xmlns:rdf=\'http://www.w3.org/1999/02/22-rdf-syntax-ns#\'>
                <rdf:Description rdf:about=\'\' xmlns:xmp=\'http://ns.adobe.com/xap/1.0/\' xmlns:dc=\'http://purl.org/dc/elements/1.1/\'>
                <dc:title><rdf:Alt><rdf:li xml:lang=\'en\'>{self.product}</rdf:li></rdf:Alt></dc:title>
                <dc:creator><rdf:Seq><rdf:li>{self.user}</rdf:li></rdf:Seq></dc:creator>
                <dc:description><rdf:Alt><rdf:li xml:lang=\'en\'>Units: {self.units}</rdf:li></rdf:Alt></dc:description>
                </rdf:Description>
            </rdf:RDF>
            </x:xmpmeta>"""
            metadata_option = f"-mo {shlex.quote(xmp_metadata)}"
            subprocess.run(f"gdal_edit.py {outFile} {metadata_option}", shell=True, check=True)
            #os.system("gdal_edit.py -mo "+xmp_metadata+" "+outFile)
            print("File "+str(outFile)+" written")
        except Exception as e:
            print("Error generating GeoTIFF")
            print(e)



# FUNCTIONS HARVESTED FROM CONNI'S U_INTERPOLATE.PY 

def _interp_weights(xyz, uvw, d=None):

    """
    :param xyz: flattened coords of current grid
    :param uvw: flattened coords of target grid
    :param d: number of dimensions of new grid
    :return: triangulisation lookup table, point weights
    """
    tri = qhull.Delaunay(xyz)
    simplex = tri.find_simplex(uvw)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uvw - temp[:, d]
    bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

def _interpolate(values, vtx, wts, fill_value=np.nan):

    """
    :param values: flattened data values
    :param vtx: lookup table
    :param wts: point weights
    :param fill_value: fill value for extrapolated regions
    :return: interpolated data
    """
    ret = np.einsum('nj,nj->n', np.take(values, vtx), wts)
    ret[np.any(wts < 0, axis=1)] = fill_value
    return ret


#def interpolation_weights(x, y, new_x, new_y):

#    """
#    :param x: current x variables (1 or 2d, definitely 2d if irregular!)
#    :param y: current y variables (1 or 2d, definitely 2d if irregular!)
#    :param new_x: target x vars
#    :param new_y: target y vars
#    :return:  triangulisation lookup table, point weights, 2d shape - inputs for interpolation func
#    """

#    if x.ndim == 1:
#        grid_xs, grid_ys = np.meshgrid(x, y)
#    else:
#        grid_xs = x
 #       grid_ys = y

#    if new_x.ndim == 1:
#        new_xs, new_ys = np.meshgrid(new_x, new_y)
#    else:
#        new_xs = new_x
#        new_ys = new_y

#    points = np.array((grid_xs.flatten(), grid_ys.flatten())).T
 #   inter = np.array((np.ravel(new_xs), np.ravel(new_ys))).T

#    inds, weights = _interp_weights(points, inter, d=2)

#    return inds, weights, new_xs.shape


def interpolation_weights(x, y, new_x, new_y, irregular_1d=False):

    """
    :param x: current x variables (1 or 2d)
    :param y: current y variables (1 or 2d)
    :param new_x: target x vars
    :param new_y: target y vars
    :keyword irregular_1d = False , set to True if input is non-ordered points in 1d arrays
    :return:  triangulisation lookup table, point weights, 2d shape - inputs for interpolation func
    """

    if (x.ndim == 1) & ~irregular_1d:
        grid_xs, grid_ys = np.meshgrid(x, y)
    else:
        grid_xs = x
        grid_ys = y

    if new_x.ndim == 1:
        new_xs, new_ys = np.meshgrid(new_x, new_y)
    else:
        new_xs = new_x
        new_ys = new_y

    points = np.array((grid_xs.flatten(), grid_ys.flatten())).T
    inter = np.array((np.ravel(new_xs), np.ravel(new_ys))).T

    inds, weights = _interp_weights(points, inter, d=2)

    return inds, weights, new_xs.shape




def interpolation_weights_grid(lon, lat, grid):

    inter, points = u_grid.griddata_input(lon, lat, grid)
    inds, weights = _interp_weights(points, inter, d=2)

    return inds, weights, (grid.ny, grid.nx)

def interpolate_data(data, inds, weights, shape):

    """
    This routine interpolates only over the 2d plane i.e. spatial interpolation
    :param data: original data, 2d, 3d or 4d (e.g. incl. time steps and pressure levels).

    :param inds: lookup table from weights func
    :param weights: index weights from weights func
    :param shape: 2d shape of plane
    :return: interpolated data, same number of dimensions as input data
    """

    if (data.ndim < 2) | (data.ndim > 4):
        print('Error. Only data with 2 - 4 dimensions allowed.')
        return
    # interpolate 2d arrays
    coll = []
    if data.ndim > 2:
        for d in data:
            if d.ndim == 2:

                d2d = _interpolate(d.flatten(), inds, weights)
                d2d = d2d.reshape(shape)
                coll.append(d2d[None, ...])

            if d.ndim == 3:
                plevs = []

                for pl in d:
                    pdb.set_trace()
                    pl2d = _interpolate(pl.flatten(), inds, weights)
                    pl2d = pl2d.reshape(shape)
                    plevs.append(pl2d[None, ...])
                if len(plevs) > 1:
                    plevs = np.concatenate(plevs, axis=0)
                coll.append(plevs[None, ...])
        if len(coll) > 1:
            coll = np.concatenate(coll, axis=0)
    else:
        d2d = _interpolate(data.flatten(), inds, weights)
        d2d = d2d.reshape(shape)
        coll = d2d


    return coll


def regrid_irregular_quick(x, y, new_x, new_y, data):

    """
    Combines all steps of data interpolation, does not provide weights etc
    Useful for quick interpolation of single 3d to 4d arrays
    :param x: array, current x coordinates
    :param y: array, current y coordinates
    :param new_x: array, new x coordinates
    :param new_y: array, new y coordinates
    :param data: array, input data
    :return:
    """
    inds, weights, shape = interpolation_weights(x, y, new_x, new_y)
    # interpolate 2d arrays
    coll = interpolate_data(data, inds, weights, shape)

    return coll


