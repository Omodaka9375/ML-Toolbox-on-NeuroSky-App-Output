import numpy as np
from scipy import stats
from scipy.interpolate import interp1d
import statistics
from scipy.stats import kurtosis
from scipy.stats import skew

def pSpectrum(vector):
    '''get the power spectrum of a vector of raw EEG data'''
    A = np.fft.fft(vector)
    ps = np.abs(A)**2
    ps = ps[:len(ps)//2]
    return ps
  
def entropy(power_spectrum,q):
    '''get the entropy of a power spectrum'''
    q = float(q)
    
    power_spectrum = np.array(power_spectrum)
        
    if not q ==1:
        S = 1/(q-1)*(1-np.sum(power_spectrum**q))
    else:
        S = - np.sum(power_spectrum*np.log2(power_spectrum))
        
    return S

def binnedPowerSpectra (pspectra,nbin):
    '''compress an array of power spectra into vectors of length nbins'''
    l = len(pspectra)
    array = np.zeros([l,nbin])

    for i,ps in enumerate(pspectra):
        x = np.arange(1,len(ps)+1)
        f = interp1d(x,ps)#/np.sum(ps)
        array[i] = f(np.arange(1, nbin+1))

    index = np.argwhere(array[:,0]==-1)
    array = np.delete(array,index,0)
    return array

# get the power spectrum
def spectra (readings):
  "Parse + calculate the power spectrum for every reading in a list"
  return [pSpectrum(v) for v in readings]

def avgPowerSpectrum (arrayOfPowerSpectra, modifierFn):
    '''
    get the mean of an array of power spectra, and apply modifierFn to it
    example: 
    avgPowerSpectrum(binnedPowerSpectra(pspectra,100), np.log10)
    '''
    # ra = modifierFn(np.mean(arrayOfPowerSpectra, 0))
    # return  np.array_str(ra, max_line_width=np.inf)
    return modifierFn(np.mean(arrayOfPowerSpectra, 0))

def avgPercentileUp (arrayOfPowerSpectra, confidenceIntervalParameter):
    '''confidenceIntervalParameter of 1 is 1%-99%'''
    return np.percentile(spectra,100-confidenceIntervalParameter,axis=0)

def avgPercentileDown (arrayOfPowerSpectra, confidenceIntervalParameter):
    return np.percentile(spectra,confidenceIntervalParameter,axis=0)

def pinkNoiseCharacterize(pspectrum,normalize=True,plot=True):
    '''Compute main power spectrum characteristics'''
    if normalize:
        pspectrum = pspectrum/np.sum(pspectrum)
    
    S = entropy(pspectrum,1)
    
    x = np.arange(1,len(pspectrum)+1)
    lx = np.log10(x)
    ly = np.log10(pspectrum)
    
    c1 = (x > 0)*(x < 80)
    c2 = x >= 80
    
    fit1 = stats.linregress(lx[c1],ly[c1])
    fit2 = stats.linregress(lx[c2],ly[c2])
    
    #print fit1
    #print fit2
    
    if plot:
        plot(lx,ly)
        plot(lx[c1],lx[c1]*fit1[0]+fit1[1],'r-')
        plot(lx[c2],lx[c2]*fit2[0]+fit2[1],'r-')
        
    return {'S':S,'slope1':fit1[0],'slope2':fit2[0]}

# A function we apply to each group of power spectra
def makeFeatureVector (readings, bins): 
  '''
  Create 100, log10-spaced bins for each power spectrum.
  For more, see http://blog.cosmopol.is/eeg/2015/06/26/pre-processing-EEG-consumer-devices.html
  '''
  return avgPowerSpectrum(
    binnedPowerSpectra(spectra(readings), bins)
    , np.log10)

def extractFeatures(ll):
    ss = []
    max_value = max(ll)
    min_value = min(ll)
    standard_dev = statistics.stdev(ll)
    sk = skew(ll)
    ku = kurtosis(ll)
    ss.append(max_value)
    ss.append(min_value)
    ss.append(standard_dev)
    ss.append(sk)
    ss.append(ku)
    return ss

def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts] 
             for i in range(wanted_parts) ]

