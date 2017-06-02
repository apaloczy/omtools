# Description: Objective mapping functions.
# Author/date: André Palóczy, November/2016
# E-mail:      paloczy@gmail.com

import numpy as np

# TODO: Implement zero-bias MSE (fix the code below following equation 16 in Dan's notes):
# if unbiased:
#     v = np.matrix(np.ones((n,1)))
#     eonum = np.array(1. - v.T*np.linalg.solve(Cqq, Cqm))**2
#     print(eonum)
#     print(eonum.shape)
#     print(Cmm.shape, v.T.shape, Cqq.shape, v.shape)
#     eoden = Cmm*np.float(v.T*np.linalg.solve(Cqq, v))
#     eo = 1. - np.diag(Cqm.T*a/Cmm) + np.diag(eonum/eoden)
# else:
#     eo = 1. - np.diag(Cqm.T*a/Cmm) # Bretherton et al. (1976), eq. (29).
# eo = np.squeeze(np.array(eo))

def om1(q, t, ti, Td=5., errvar=0.1, dcovmat=None, unbiased=True, verbose=True):
    """
    Calculates a 1D objective map for the scalar variable q(t)
    assuming Gaussian autocovariance with decorrelation timescale
    'Td' for the signal.

    REFERENCES
    ----------
    Bretherton et al. (1976)
    Rudnick SIO221B lecture notes.
    """
    q, t, ti = map(np.matrix, (q, t, ti)) # x = x(t).
    assert q.size==t.size, "Data and data grid shape mismatch."

    # Demean and detrend data.
    qbar = q.mean()
    bb, aa, _ = linear_trend(q, return_line=False)
    qlin = bb*t + aa
    q = q - qbar - qlin

    if verbose:
        print("Calculating squared distance matrices.")
    n, ni = t.size, ti.size
    t0 = np.tile(t, (n,1))
    dtt2 = np.array(t0.T - t0)**2     # Squared distance between all pairs of data points.
    t0m = np.tile(ti, (ni,1))
    dtiti2 = np.array(t0m.T - t0m)**2 # Squared distance between all pairs of grid points.
    tt = np.tile(t, (ni,1))
    tti = np.tile(ti.T, (1,n))
    dtti2 = np.array(tti - tt)**2 # Squared distance between data points and interpolation grid points.

    if verbose:
        print("Calculating covariance matrices.")
    q, t = q.T, t.T
    if dcovmat is not None:
        Cqq = np.matrix(dcovmat) # Data autocovariance matrix directly from averaged realizations of the data.
    else:
        Cqq = np.exp(-dtt2/Td**2)    # Assuming the data autocovariance function is Gaussian instead.
        Cqq = np.matrix(Cqq).T       # Model covariance matrix evaluated at data locations.
        Cqq = Cqq + errvar*np.eye(n) # Get the actual data covariance matrix by adding the sampling noise. Assuming sampling error to be random, normal and spatially uncorrelated.

    Cmm = np.exp(-dtiti2/Td**2)
    Cmm = np.matrix(Cmm).T        # Model autocovariance matrix.

    Cqm = np.exp(-dtti2/Td**2)
    Cqm = np.matrix(Cqm).T        # Data-model covariance matrix.

    if verbose:
        print("Calculating gain matrix.")
    a = gain(Cqq, Cqm, unbiased=unbiased)

    if verbose:
        print("Calculating objective map.")
    qi = a.T*q                     # Calculate objective map.

    # Add the mean and linear trend back to the objective map.
    qi = np.squeeze(np.array(qi))
    qlini = np.array(bb*ti + aa) # Linear trend evaluated at grid points.
    qi = qi + qbar + qlini
    qi = np.squeeze(qi)

    # Normalized mean-square error between objective map and signal normalized by signal variance, i.e.,
    # (qi - qtilde)**2/qtilde**2, it is a measure of the interpolation error squared.
    if verbose:
        print("Calculating normalized MSE matrix (interpolation error matrix).")

    eo = 1. - np.diag(Cqm.T*a/Cmm) # Bretherton et al. (1976), eq. (29).
    eo = np.squeeze(np.array(eo))
    if verbose:
        print("Done.")

    return qi, eo

def om2_iso(q, x, y, xi, yi, Lr=5., errvar=0.1, dcovmat=None, unbiased=True, geographic=True, verbose=True):
    """
    Calculates a 2D objective map for the scalar variable q(x,y)
    assuming a radially-symmetric Gaussian autocovariance with
    decorrelation length 'Lr' for the signal.

    REFERENCES
    ----------
    Bretherton et al. (1976)
    Rudnick SIO221B lecture notes.
    """
    x, y, q = map(np.asarray, (x, y, q))

    if geographic:
        x, y = lonlat2xy(x, y)
        xi, yi = lonlat2xy(xi, yi)

    origshp = xi.shape
    q, x, y, xi, yi = map(np.ravel, (q, x, y, xi, yi))
    q, x, y, xi, yi = map(np.matrix, (q, x, y, xi, yi)) # x = x(t).
    assert q.size==x.size==y.size, "Data and data grid size mismatch."
    assert xi.size==yi.size, "interpolation grid size mismatch."

    # Demean and detrend data (subtract a least-squares plane fit).
    qc, qsx, qsy = deplane(x, y, q, return_what='coeffs', verbose=False)
    qplane = qc + qsx*np.array(x) + qsy*np.array(y)
    q = q - qplane

    if verbose:
        print("Calculating squared distance matrices.")

    n, ni = x.size, xi.size
    x0 = np.tile(x, (n,1))
    y0 = np.tile(y, (n,1))
    dr2 = np.array(x0.T - x0)**2 + np.array(y0.T - y0)**2    # Squared radial distance between all pairs of data points.

    x0m = np.tile(xi, (ni,1))
    y0m = np.tile(yi, (ni,1))
    drm2 = np.array(x0m.T - x0m)**2 + np.array(y0m.T - y0m)**2 # Squared radial distance between all pairs of grid points.

    xx = np.tile(x, (ni,1))
    yy = np.tile(y, (ni,1))
    xxm = np.tile(xi.T, (1,n))
    yym = np.tile(yi.T, (1,n))
    drrm2 = np.array(xx - xxm)**2 + np.array(yy - yym)**2 # Squared radial distance between all pairs of data and grid points.

    if verbose:
        print("Calculating covariance matrices.")
    q, x, y = q.T, x.T, y.T
    if dcovmat is not None:
        Cqq = np.matrix(dcovmat) # Data autocovariance matrix directly from averaged realizations of the data.
    else:
        Cqq = np.exp(-dr2/Lr**2)    # Assuming the data autocovariance function is Gaussian instead.
        Cqq = np.matrix(Cqq).T       # Model covariance matrix evaluated at data locations.
        Cqq = Cqq + errvar*np.eye(n) # Get the actual data covariance matrix by adding the sampling noise. Assuming sampling error to be random, normal and spatially uncorrelated.

    Cmm = np.exp(-drm2/Lr**2)
    Cmm = np.matrix(Cmm).T        # Model autocovariance matrix.

    Cqm = np.exp(-drrm2/Lr**2)
    Cqm = np.matrix(Cqm).T        # Data-model covariance matrix.

    if verbose:
        print("Calculating gain matrix.")
    a = gain(Cqq, Cqm, unbiased=unbiased)

    if verbose:
        print("Calculating objective map.")
    qi = a.T*q                     # Calculate objective map.

    # Add the plane (mean and linear trends in x and y) back to the objective map.
    qi = np.squeeze(np.array(qi))
    qplanei = qc + qsx*np.array(xi) + qsy*np.array(yi) # Linear trend evaluated at grid points.
    qi = qi + qplanei
    qi = np.squeeze(qi)

    # Normalized mean-square error between objective map and signal normalized by signal variance, i.e.,
    # (qi - qtilde)**2/qtilde**2, it is a measure of the interpolation error squared.
    if verbose:
        print("Calculating normalized MSE matrix (interpolation error matrix).")

    ei = 1. - np.diag(Cqm.T*a/Cmm) # Bretherton et al. (1976), eq. (29).
    ei = np.squeeze(np.array(ei))
    if verbose:
        print("Done.")

    ## Reshape arrays.
    qi = np.reshape(qi, origshp)
    ei = np.reshape(ei, origshp)

    return qi, ei

def om2_aniso(q, x, y, xi, yi, Lx=5., Ly=20., theta=0., errvar=0.1, dcovmat=None, unbiased=True, geographic=True, verbose=True):
    """
    Calculates a 2D objective map for the scalar variable q(x,y)
    assuming an anisotropic Gaussian autocovariance rotated by 'theta'
    radians with decorrelation lengths 'Lx' and 'Ly' for the signal.

    REFERENCES
    ----------
    Bretherton et al. (1976)
    Rudnick SIO221B lecture notes.
    """
    x, y, q = map(np.asarray, (x, y, q))

    if geographic:
        x, y = lonlat2xy(x, y)
        xi, yi = lonlat2xy(xi, yi)

    origshp = xi.shape
    q, x, y, xi, yi = map(np.ravel, (q, x, y, xi, yi))
    q, x, y, xi, yi = map(np.matrix, (q, x, y, xi, yi)) # x = x(t).
    assert q.size==x.size==y.size, "Data and data grid size mismatch."
    assert xi.size==yi.size, "interpolation grid size mismatch."

    # Demean and detrend data (subtract a least-squares plane fit).
    qc, qsx, qsy = deplane(x, y, q, return_what='coeffs', verbose=False)
    qplane = qc + qsx*np.array(x) + qsy*np.array(y)
    q = q - qplane

    if verbose:
        print("Calculating squared distance matrices.")

    n, ni = x.size, xi.size
    x0 = np.tile(x, (n,1))
    y0 = np.tile(y, (n,1))
    dx = np.array(x0.T - x0)    # x-distance between all pairs of data points.
    dy = np.array(y0.T - y0)    # y-distance between all pairs of data points.

    x0m = np.tile(xi, (ni,1))
    y0m = np.tile(yi, (ni,1))
    dxm = np.array(x0m.T - x0m) # x-distance between all pairs of grid points.
    dym = np.array(y0m.T - y0m) # y-distance between all pairs of grid points.

    xx = np.tile(x, (ni,1))
    yy = np.tile(y, (ni,1))
    xxm = np.tile(xi.T, (1,n))
    yym = np.tile(yi.T, (1,n))
    dxxm = np.array(xx - xxm)   # x-distance between all pairs of data and grid points.
    dyym = np.array(yy - yym)   # y-distance between all pairs of data and grid points.

    if verbose:
        print("Calculating covariance matrices.")
    q, x, y = q.T, x.T, y.T
    if dcovmat is not None:
        Cqq = np.matrix(dcovmat) # Data autocovariance matrix directly from averaged realizations of the data.
    else:
        Cqq = gauss2d((dx,dy), 0., 0., 1., Lx, Ly, theta, 0.) # Assuming the data autocovariance function is an anisotropic and rotated Gaussian function.
        Cqq = np.matrix(Cqq).T       # Model covariance matrix evaluated at data locations.
        Cqq = Cqq + errvar*np.eye(n) # Get the actual data covariance matrix by adding the sampling noise. Assuming sampling error to be random, normal and spatially uncorrelated.

    Cmm = gauss2d((dxm,dym), 0., 0., 1., Lx, Ly, theta, 0.)
    Cmm = np.matrix(Cmm).T        # Model autocovariance matrix.

    Cqm = gauss2d((dxxm,dyym), 0., 0., 1., Lx, Ly, theta, 0.)
    Cqm = np.matrix(Cqm).T        # Data-model covariance matrix.

    if verbose:
        print("Calculating gain vector.")
    a = gain(Cqq, Cqm, unbiased=unbiased)

    if verbose:
        print("Calculating objective map.")
    qi = a.T*q                     # Calculate objective map.

    # Add the plane (mean and linear trends in x and y) back to the objective map.
    qi = np.squeeze(np.array(qi))
    qplanei = qc + qsx*np.array(xi) + qsy*np.array(yi) # Linear trend evaluated at grid points.
    qi = qi + qplanei
    qi = np.squeeze(qi)

    # Normalized mean-square error between objective map and signal normalized by signal variance, i.e.,
    # (qi - qtilde)**2/qtilde**2, it is a measure of the interpolation error squared.
    if verbose:
        print("Calculating normalized MSE matrix (interpolation error matrix).")

    ei = 1. - np.diag(Cqm.T*a/Cmm) # Bretherton et al. (1976), eq. (29).
    ei = np.squeeze(np.array(ei))
    if verbose:
        print("Done.")

    ## Reshape arrays.
    qi = np.reshape(qi, origshp)
    ei = np.reshape(ei, origshp)

    return qi, ei

def om3():
    """
    Dhat(x) = Dbar(x) + \sum_{i=1}^N b_i*(d_i - dbar) # Thomson & Emery p.319, eq. 4.4.
    """
    abar = a.mean()
    a -= abar # Remove the mean.
    a -= deplane(a, order=order) # Remove backgroud trend (linear or not).

    # Add the trend and mean back to the interpolated field.
    a = a + abar + apl

    return a

def gain(Cxx, Cxy, unbiased=True):
    """
    Calculates the gain matrix from the data autocovariance matrix Cxx
    and data-signal covariance matrix Cxy.
    """
    assert Cxx.shape[0]==Cxx.shape[0], "The data autocovariance matrix is not square."
    n = Cxx.shape[0]
    if unbiased: # Solve for gain constrained with zero-bias mean (Rudnick notes on objective mapping, eq. 15).
        v = np.matrix(np.ones((n,1)))
        num = 1. - v.T*np.linalg.solve(Cxx, Cxy)
        den = v.T*np.linalg.solve(Cxx, v)
        frac = num/den
        Cxy2 = Cxy + v*frac
        a = np.linalg.solve(Cxx, Cxy2)
    else:
        a = np.linalg.solve(Cxx, Cxy) # Solve a = Cqq.I*Cqm for the gain matrix a (for zero-mean data). Bretherton et al. (1976), eq. (7).

    return a

def gauss2d(x,y, x0, y0, A, Lx, Ly, th, c):
    xx = x - x0
    yy = y - y0
    ex = ((xx*np.cos(th) - yy*np.sin(th))/Lx)**2
    ey = ((xx*np.sin(th) + yy*np.cos(th))/Ly)**2
    exy = ex + ey
    gauss2 = A*np.exp(-exy) + c

    return gauss2

def nllstsq_gauss2d(x, y, data, init_guess):
    """
    Find the parameters for a rotated and anisotropic 2D Gaussian function
    that minimize the sum of squared errors between the data and the 2D Gaussian
    using scipy.optimize.curve_fit (via nonlinear least-squares).

    gauss2d((x,y), x0, y0, A, Lx, Ly, th, c)
    Parameters (in order): x0, y0, A, Lx, Ly, theta, offset.
    """
    if data.ndim>1:
        data = data.ravel() # Input data to curve_fit() must be 1D.

    popt, _ = curve_fit(gauss2d, (x,y), data, p0=initial_guess)
    model = gauss2d((x, y), *popt).reshape(x.shape)

    return popt, model

def binacorr_iso():
    """
    Compute the covariance matrix by block-averaging data points
    in radially-separated bins.
    """
    return 1

def binacorr_aniso():
    return 1

def desurf(order=1):
    """
    Returns a surface based on a least-squares polynomial fit with order
    specified by the variable 'order' (defaults to 1, i.e., a plane).
    """
    return 1

def lonlat2xy(lon, lat):
	"""
	USAGE
	-----
	x, y = lonlat2xy(lon, lat)

	Calculates zonal and meridional distance 'x' and 'y' (in meters)
	from the 'lon' and 'lat' 2D meshgrid-type arrays (in degrees), using centered
	(forward/backward) finite-differences for the interior (edge) points.

    Adapted from deg2m_dist (from ap_tools.dyn) module.
	"""
	lon, lat = map(np.asanyarray, (lon, lat))

	lon = lon - lon.min()                # [deg]
	lat = lat - lat.min()                # [deg]
	deg2m = 111120.0                     # [m/deg]

	# Account for divergence of meridians in zonal distance.
	x = lon*deg2m*np.cos(lat*np.pi/180.) # [m]
	y = lat*deg2m                        # [m]

	return x, y

def linear_trend(series, return_line=True):
    """
    USAGE
    -----
    line = linear_trend(series, return_line=True)

    OR

    b, a, x = linear_trend(series, return_line=False)

    Returns the linear fit (line = b*x + a) associated
    with the 'series' array.

    Adapted from pylab.detrend_linear.
    (From ap_tools.utils module)
    """
    series = np.asarray(series)
    x = np.arange(series.size, dtype=np.float_)

    C = np.cov(x, series, bias=1) # Covariance matrix.
    b = C[0, 1]/C[0, 0] # Angular coefficient.

    a = series.mean() - b*x.mean() # Linear coefficient.
    line = b*x + a

    if return_line:
        return line
    else:
        return b, a, x

def deplane(x, y, Q, return_what='deplaned_array', verbose=True):
    """
    Fit a plane z(x,y) = a*x + b*y + c via least-squares and subtract it
    from a scalar variable Q(x,y).
    """
    x, y, Q = map(np.asarray, (x, y, Q))
    q = np.copy(Q)
    oshp = x.shape

    # Remove NaNs in data.
    fgud = ~np.isnan(Q)
    Q = q[fgud].ravel()

    x = x[fgud].ravel()
    y = y[fgud].ravel()

    # Save indices of good values to rebuild matrix later.
    denan_idx = np.where(fgud)
    denanj, denani = denan_idx
    d = np.matrix(Q).T

    ## A plane is described by the equation a*x + b*y + c*z + d = 0
    ## If z = z(x,y):
    ## z(x,y) = A*x + B*y + C
    ## with A = -a/c, B = -b/c, C = -d/c
    n = x.size
    c = np.repeat(1., n)
    c, x, y = map(np.expand_dims, (c, x, y), (1, 1, 1))
    G = np.matrix(np.concatenate((c, x, y), axis=1))

    # Solve the overdetermined least-squares problem to get the plane coefficients.
    m = (G.T*G).I*G.T*d

    if verbose:
        print("")
        print("Model parameter matrix:")
        print(m)

    # Intercept, slope in x, slope in y. m/s, 1/s, 1/s
    Qc, Qsx, Qsy = np.array(m).flatten()
    z = np.array(G*m).flatten() # Values of the model plane fit at each grid point.

    # Rebuild original arrays with nans.
    zskel = np.nan*np.ones(oshp)
    zlst = z.tolist()
    for j,i in zip(denanj,denani):
        zskel[j,i] = zlst.pop(0)

    Q = q - zskel # Subtract the plane from the data.

    if return_what=='plane':
        return zskel
    elif return_what=='coeffs':
        return Qc, Qsx, Qsy
    elif return_what=='deplaned_array':
        return Q

def deplane_vec(x, y, u, v, return_what='deplaned_array', verbose=True):
    """
    Fit a plane z(x,y) = a*x + b*y + c via least-squares and subtract it
    from a vector variable U = xhat*u(x,y) + yhat*v(x,y).

    The outputs for deplane(x, y, u), deplane(x, y, v) and deplane_vec(x, y, u, v)
    should be identical within floating point precision. The difference is just that
    deplane_vec() sets up and solves the least-squares problem in one go.
    """
    x, y, u, v = map(np.asarray, (x, y, u, v))
    uu, vv = map(np.copy, (u, v))
    oshp = x.shape

    # Remove NaNs in data.
    fgud = ~np.isnan(uu)
    fgudv = ~np.isnan(vv)
    assert np.all(fgud==fgudv), "Different NaN locations in u(x,y) and v(x,y)."
    u = uu[fgud].ravel()
    v = vv[fgud].ravel()

    x = x[fgud].ravel()
    y = y[fgud].ravel()

    # Save indices of good values to rebuild matrix later.
    denan_idx = np.where(fgud)
    denanj, denani = denan_idx
    u, v = map(np.expand_dims, (u, v), (1, 1))
    d = np.matrix(np.concatenate((u, v), axis=0))

    ## A plane is described by the equation a*x + b*y + c*z + d = 0
    ## If z = z(x,y):
    ## z(x,y) = A*x + B*y + C
    ## with A = -a/c, B = -b/c, C = -d/c
    n = x.size
    c = np.repeat(1., n)
    c, x, y = map(np.expand_dims, (c, x, y), (1, 1, 1))
    g = np.matrix(np.concatenate((c, x, y), axis=1))
    g0 = np.zeros_like(g)
    G_upper = np.concatenate((g, g0), axis=1)
    G_lower = np.concatenate((g0, g), axis=1)
    G = np.concatenate((G_upper, G_lower), axis=0)

    # Solve the overdetermined least-squares problem to get the plane coefficients.
    m = (G.T*G).I*G.T*d

    if verbose:
        print("")
        print("Model parameter matrix:")
        print(m)

    # Intercept, slope in x, slope in y. m/s, 1/s, 1/s
    Uc, Usx, Usy = np.array(m[:3]).flatten()
    Vc, Vsx, Vsy = np.array(m[3:]).flatten()
    z = np.array(G*m).flatten() # Values of the model plane fit at each grid point.
    zu, zv = z[:n], z[n:]

    # Rebuild original arrays with nans.
    zuskel = np.nan*np.ones(oshp)
    zvskel = zuskel.copy()
    zulst = zu.tolist()
    zvlst = zv.tolist()
    for j,i in zip(denanj,denani):
        zuskel[j,i] = zulst.pop(0)
        zvskel[j,i] = zvlst.pop(0)

    # Subtract the plane from the data.
    zu, zv = map(np.copy, (zuskel, zvskel))
    u = uu - zuskel
    v = vv - zvskel

    if return_what=='plane':
        return zuskel, zvskel
    elif return_what=='coeffs':
        return Uc, Usx, Usy, Vc, Vsx, Vsy
    elif return_what=='deplaned_array':
        return u, v
