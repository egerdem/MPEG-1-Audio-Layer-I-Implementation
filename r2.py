"""
Ege Erdem, MMI-726

Mpeg 1 Layer 1 Psychoacoustical Model Implementation

"""
import numpy as np
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
from geo import geo
from ftobark import ftobark
#from FrequencySpectrum import fspectrum

rate, audio0 = read("sin.wav")  # read pcm audio file
#audio0 = audio0[:,0]   # select only left channel, if the wav is mono, ignore this line

""" Fundamental definitions and frequency indices for plotting graph."""

b = 16  # number of bits per sample
N = 512 # fft length

f_ = np.arange(N/2)  # 0'dan 256'ya bir array
f_hz = np.arange(1, (N/2+1))*(rate/N) # hearing range divided to 256 bins each of them 86.13Hz apart
freq_bark = ftobark(f_hz) # bark scale conversion of frequency range

th = 3.64*((f_hz/1000)**-0.8)-6.5*np.exp(-0.6*((f_hz/1000-3.3)**2))+(10**-3)*((f_hz/1000)**4) #threshold of hearing formula
th[187:256] = 69.13 # manipulating the last part of the threshold curve

# center frequencies for 25 critical filter bank
cb_c = np.array([50,150,250,350,450,570,700,840,1000,1175,1370,1600,1850,
             2150,2500,2900,3400,4000,4800,5800,7000,8500,10500,13500,19500])  

cb_c_bark = ftobark(cb_c) # bark conversion of center frequencies    
cb_in = np.divide(np.multiply(cb_c,N//2),(rate/2))  # converting center freqs. to 0-256 range
cb_in = cb_in.astype(int) # making it integer

bnd = np.array([0,1,2,3,5,6,8,9,11,13,15,17,20,23,27,32,37,45,52,62,74,88,108,132,180,232]) #critical band boundaries
bark = np.arange(25)

bandwidth_hz = np.array([0,100,200,300,400,510,630,770,920,1080,1270,1480,1720,2000,2320,2700,3150,3700,4400,5300,6400,7700,9500,12000,15500,22050])

k_bar = np.zeros(25)

for i in range(0,25):
    k_bar[i] = geo(bandwidth_hz[i],bandwidth_hz[i+1])
    
k_bark = ftobark(k_bar)
k_bark256 = np.divide(np.multiply(k_bar,N//2),(rate/2))
k_bark256 = k_bark256.astype(int)


cb_c_bark = k_bark
cb_in = k_bark256


"""STEP 1 Step 1: Spectral Analysis and SPL Normalization."""

for i in range(0,512,512):

    audio = audio0[i:(i+N)] # getting new 512 sample in each loop
    audio = audio/(2**(b-1)) # normalization according to bits
    audio = audio / 512 # normalization according to FFT length
    
        
    h = np.hanning(M=512)*1.63 # 1.63 is the gain to compensate the average power of window 
        
    X = np.fft.fft(h*audio,512)/512 # FFT with h
    
    fft = abs(X)
    fft = fft[0:(N//2)]
        
    p = 20*np.log10(fft)
    delta = 96 - np.amax(p)  
    p += delta
    
    p_tm = []
    k_tm = []
        
    
    """ Step 2: Identification of Tonal and Noise Maskers """
    
    #finding tonal set "p_tm" from spectral peaks
    
    for k in np.arange(2,250):
        if (p[k-1] < p[k] and p[k] > p[k+1]):
            
            del_k = []
            
            if k > 2 and k < 63 :
                del_k = np.array([-2,+2])
            elif k >= 63 and k < 127 :
                del_k = np.array([-3,-2,2,3])
            elif k>=127 and k<=256 :
                del_k = np.array([-6, -5, -4, -3, -2, +2, +3, +4, +5, +6])
            else:
                del_k = 0
            #del_k = np.asarray(del_k)
            
            if all(p[k]>p[k+del_k]+7):
                
                p_tm.append(10*np.log10(10**(0.1*p[k-1])+10**(0.1*p[k])+10**(0.1*p[k+1])))
                k_tm.append(k)
                
               
                k_tm_f = np.multiply(k_tm,22050/256)
                k_tm_f_bark = ftobark(k_tm_f) 
            
                
    """
    for a in range(0,len(k_tm)):
        k = k_tm[a]
        p[k-1] = 0
        p[k] = 0
        p[k+1] = 0
    """
                       
    """ Calculating nontonal noise maskers in critical bands """ 
    
    # we need to exclude already found spectral lines
    
    
    idxx = [] # noise masker indices
    idxe = [] # indices that should be excluded
    p_nm = [] # noise masker array    
    
    for x in range(0,len(bnd)-1):            
    
        for idx in range(bnd[x], bnd[x+1]):
            del_k = []
        
            if idx > 2 and idx < 63 :
                del_k = np.array([-2,-1,0,1,2])
            elif idx >= 63 and idx < 127 :
                del_k = np.array([-3,-2,-1,0,1,2,3])
            elif idx>=127 and idx<=256 :
                del_k = np.array([-6, -5, -4, -3, -2,-1,0,1,2, +3, +4, +5, +6])
            
            for j in range(0,len(k_tm)):  
                for f in range(0,len(del_k)):
                              
                    if (idx != (k_tm[j]+del_k[f])): 
                                           
                        #print("include",idx)
                        c   = "no need"                                            
                        
                    elif (idx == (k_tm[j]+del_k[f])):     
                                          
                        #print("exclude",idx)  
                        idxe.append(idx)
                                    
    bnd_k = np.arange(0,232)    
    
    # indices without previously found spectral lines
    idxx = list(set(bnd_k)^set(idxe))   
            
    for x in range(0,len(bnd)-1):  
        total = 0
        for a in range(len(idxx)):
            
                if (bnd[x]<=idxx[a] and idxx[a]<bnd[x+1]):    
                    
                    total += 10**(0.1*p[idxx[a]])                                       
        
        p_nm.append(10*np.log10(total))
                        
        
    """Step 3: Decimation and Reorganization of Maskers."""
    
    p_tm_th = []
    p_nm_th = []
    k_nm_th = []
    k_tm_th = []
    
    for k in range(len(k_tm)):        
                   
        if (p_tm[k] >= th[k_tm[k]]):
            
            p_tm_th.append(p_tm[k])
            k_tm_th.append(k_tm[k])
                
    for l in range(len(cb_in)):        
           
        if (p_nm[l] >= th[cb_in[l]]):
    
             p_nm_th.append(p_nm[l])
             k_nm_th.append(cb_in[l])
        else:
            dummy = l
    """0.5 Bark Window."""
        
    k_tm_bark = ftobark(np.multiply(k_tm_th,22050/256))    
    k_nm_bark = ftobark(np.multiply(k_nm_th,22050/256))
    
    ptm2 = []
    ktm2 = []
    pnm2 = []
    knm2 = []
    ptm3 = []
    ktm3 = []
    pnm3 = []
    knm3 = []
    
    for k in range(len(k_tm_bark)-1): 
        if np.absolute(k_tm_bark[k+1]-k_tm_bark[k]) <= 0.5 :
            if p_tm_th[k+1] > p_tm_th[k]:
                ptm2.append(p_tm_th[k+1])
                ktm2.append(k_tm_bark[k+1])
            else:
                ptm2.append(p_tm_th[k])
                ktm2.append(k_tm_bark[k])
        else:
            ptm2.append(p_tm_th[k])
            ktm2.append(k_tm_bark[k])
    ptm2.append(p_tm_th[len(k_tm_bark)-1])
    ktm2.append(k_tm_bark[len(k_tm_bark)-1])
            
    for k in range(len(k_nm_bark)-1): 
        if np.absolute(k_nm_bark[k+1]-k_nm_bark[k]) <= 0.5 :
            if p_nm_th[k+1] > p_nm_th[k]:
                pnm2.append(p_nm_th[k+1])
                knm2.append(k_nm_bark[k+1])
            else:
                pnm2.append(p_nm_th[k])
                knm2.append(k_nm_bark[k])            
        else:
            pnm2.append(p_nm_th[k])
            knm2.append(k_nm_bark[k])
    
    pnm2.append(p_nm_th[len(k_nm_bark)-1])
    knm2.append(k_nm_bark[len(k_nm_bark)-1])
      
    
    for k in range(len(ktm2)):
        alone = True
        for j in range(len(knm2)):
            if np.absolute(ktm2[k] - knm2[j]) <= 0.5:
                alone = False
                
                if ptm2[k] < pnm2[j]:
                    pnm3.append(pnm2[j])
                    knm3.append(knm2[j])
                if ptm2[k] >= pnm2[j]:
                    ptm3.append(ptm2[k])
                    ktm3.append(ktm2[k])
        if alone:
            ptm3.append(ptm2[k])
            ktm3.append(ktm2[k])
    
    for k in range(len(knm2)):
        alone = True
        for j in range(len(ktm3)):
            if np.absolute(ktm3[j] - knm2[k]) <= 0.5:
                alone = False
                
        if alone:
            pnm3.append(pnm2[k])
            knm3.append(knm2[k])
    
    """Step 4: Calculation of Individual Masking Thresholds."""
    
    TTMny = []
    TTMty = []
    TTMnx = []
    TTMtx = []
    
    # finding spread of masking functions
    
    def SF(i,j,p):
        
        dz = i-j
        sf = 0
        if -3<=dz<-1:
            sf = 17*dz - 0.4*p+11
        elif -1<=dz<0:
            sf = (0.4*p+6)*dz
        elif 0<=dz<1:
            sf = -17*dz
        elif 1<=dz<8:
            sf = (0.15*p-17)*dz-0.15*p
            
        return sf
        
    for k in range(len(ktm3)):
        j = ktm3[k]
        Ttmy=[]
        Ttmx=[]
        for i in np.arange(j-3,j+8,0.1):
            
            """
            Appending individual tonal masker thresholds.
            """
            Ttmy.append( ptm3[k] - 0.275*j -6.025 + SF(i,j,ptm3[k]) ) 
            Ttmx.append( i )         
            
        TTMty.append(Ttmy)
        TTMtx.append(Ttmx)
        
    for k in range(len(knm3)):
        j = knm3[k]
        Ttmy=[]
        Ttmx=[]
        for i in np.arange(j-3,j+8,0.1):
            
            """
            Appending individual noise masker thresholds.
            """
            Ttmy.append( pnm3[k] - 0.175*j -2.025 + SF(i,j,pnm3[k]) ) 
            Ttmx.append( i ) 
                    
        TTMny.append(Ttmy)
        TTMnx.append(Ttmx)
    
    
    """Step 5: Calculation of Global Masking Thresholds."""
    
    g_mth_y = []
    g_mth_x = []
    for i in np.arange(0, 25, 0.1):
        total = 0
        allvals = []
        for j in range(len(freq_bark)):
            if i <= freq_bark[j] < i+0.1:
                
                allvals.append(th[j])
       
        if len(allvals) > 0:
            total += 10 ** (0.1 * np.mean(allvals))
         
        allvals = []
        for n in range(len(TTMnx)):
            
            for k in range(len(TTMnx[n])):
                
                if i <= TTMnx[n][k] < i+0.1:
                    allvals.append(TTMny[n][k])
                    if len(allvals) > 0:
                        total += np.sum(np.power(10, np.multiply(0.1, allvals)))
        allvals = []
        for t in range(len(TTMtx)):
            for k in range(len(TTMtx[t])):
                if i <= TTMtx[t][k] < i+0.1:
                    allvals.append(TTMty[t][k])
                    if len(allvals) > 0:
                        total += np.sum(np.power(10, np.multiply(0.1, allvals)))
        
        if total:
            g_mth_y.append(10 * np.log10(total))
            g_mth_x.append(i)
        
       
    """Bark Plots."""
    
    
    plt.figure(1)
    
    plt.plot(freq_bark,p)
    
    plt.plot(k_tm_f_bark,p_tm,'bo')
    
    plt.plot(cb_c_bark,p_nm,'rx')
    
    plt.plot(freq_bark, th) 
    
    plt.xticks(np.arange(1, max(freq_bark)+1, 1))
    
    
    """ Fr. Index Plots."""  
    
    plt.figure(2)
    
    plt.plot(f_,p)
    
    plt.plot(k_tm,p_tm,'bo')
    
    plt.plot(cb_in,p_nm,'rx')
    
    plt.plot(k_nm_th,p_nm_th,'gx')
    
    plt.plot(k_tm_th,p_tm_th,'go')
    
    plt.plot(f_,th)
         
    
    plt.figure(3)
    
    plt.plot(freq_bark,p)
    
    plt.plot(k_tm_f_bark,p_tm,'ro')
    plt.plot(ktm3,ptm3,'bo')
    
    plt.plot(cb_c_bark,p_nm,'rx')
    plt.plot(knm3,pnm3,'bx')
    
    plt.plot(freq_bark, th) 
    
    plt.xticks(np.arange(1, max(freq_bark)+1, 1))
    
    for i in range(len(TTMny)):
        plt.plot(TTMnx[i], TTMny[i])
    for i in range(len(TTMty)):
        plt.plot(TTMtx[i], TTMty[i])
        
    plt.plot(g_mth_x, g_mth_y)
    
    plt.figure(4)
    
    plt.plot(g_mth_x, g_mth_y)
    plt.plot(freq_bark,p)
    plt.show()
    
    
    
