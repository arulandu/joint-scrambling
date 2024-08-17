import numpy as np
import numdifftools as nd
import scipy
from scipy import stats
from scipy.signal import convolve
from scipy.integrate import quad

deriv = lambda f,o=1: nd.Derivative(f, n=o)

def zkde(y, Zs, k, h=0.1):
    return (1/(Zs.shape[0]*h))*np.sum(k((y-Zs.ravel())/h))
    
def rrta(Y, T, n=100):
    return Y.rvs(size=n)+stats.uniform(loc=0, scale=T).rvs(size=n)

def rrtm(Y, T, n=100):
    return Y.rvs(size=n)*stats.uniform(loc=0, scale=T).rvs(size=n)

def rrtpdfa(Zs, T, k, h=0.1):
    return np.vectorize(lambda y: quad(lambda t: zkde(t, Zs, k, h), y, y+T)[0]/T)

def rrtpdfm(Zs, T, k, h=0.1, dt=0.1):
    return np.vectorize(lambda y: -y*(T**2)*(zkde(y*T+dt/2, Zs, k, h)-zkde(y*T-dt/2, Zs, k, h))/dt)

def integrate(f, a, b, m=4):
    return scipy.integrate.quad(f, a, b, epsabs=1e-1)[0]
    
def rrtcdfa(Zs, T, k, h=0.1):
    return np.vectorize(lambda y: integrate(
        lambda z: zkde(z, Zs, k, h), -np.inf, y+T
        #epsabs=1e-1
    )-integrate(lambda z: (z-y)*zkde(z, Zs, k, h), y, y+T)/T)

def js(Y, S, r=1, n=100):
  Ys = Y.rvs(size=n).reshape((n,1))
  Ss = S.rvs(size=r*n).reshape((n,r))
  return np.sort(np.hstack((Ys, Ss)), axis=1)

def jscdfest(Zs, S, r=1, n=100):
  est_cdfY = np.vectorize(lambda y : (1/n)*np.sum(Zs.ravel() < y) - r*S.cdf(y))
  return est_cdfY

def jspdfest(Zs, S, k, Sbnd=[-np.inf, np.inf], h=1, r=1, n=100):
  def est_pdfY(y):
      kde = (1/n)*np.sum(k((y-Zs.ravel())/h)/h)
      C = quad(lambda s: (k((y-s)/h)/h)*S.pdf(s), Sbnd[0], Sbnd[1])[0]
      # C = S.pdf(y)
      return kde-r*C
      
  return np.vectorize(est_pdfY)
