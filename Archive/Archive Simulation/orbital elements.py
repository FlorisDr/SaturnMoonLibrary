import numpy as np

G=6.674*10**-11 #m^3 kg^-1 s^-2
M=5.6834*10**26 #kg

def orbital_elements(moon_index):
    """
    takes a moon index and spits out the important orbital parameters:
    numbers,vectors: (inclination, longitude of ascending node, argument of periapsis, eccentricity, semimajor axis, Period)
    , (angular momentum vector (without mass), ascending node (without mass), ecc vector)
    angular momentum vector, 
    """
    h=np.cross(x[:,moon_index]-x[:,0],v[:,moon_index]-v[:,0],axis=1) #angular momentum
    habs=np.sqrt(h[:,0]**2+h[:,1]**2+h[:,2]**2)
    n=np.cross(np.array([0,0,1]),h)
    nabs=np.sqrt(n[:,0]**2+n[:,1]**2+n[:,1]**2)
    e_vec=np.cross(v[:,moon_index],h)/G/M-np.transpose(np.transpose(x[:,moon_index,:])/np.sqrt(x[:,moon_index,0]**2+x[:,moon_index,1]**2+x[:,moon_index,2]**2)) #eccentricity vector
    e=np.sqrt(e_vec[:,0]**2+e_vec[:,1]**2+e_vec[:,2]**2) #eccentricity scalar
    a=habs**2/G/M #semimajor axis
    i=np.arccos(h[:,2]/habs) #inclination
    Omega_temp=np.arccos(n[:,0]/nabs) #longitude of ascending node
    if np.all(n[:,1]>=0):
        Omega=Omega_temp
    elif np.all(n[:,1]<0):
        Omega=2*np.pi-Omega_temp
    else:
        Omega=2*np.pi*(n[:,1]<0)+(1-2*(n[:,1]<0))*Omega_temp # This case should only happen in the rare event that n is very dependent on time
    # calculate argument of periapsis
    omega_temp=np.arccos(np.einsum("ij,ij->i",n,e_vec)/nabs/e) #einsum works, but most importantly best i can find, quite fast as well can do 1e4 computations in .4s
    if np.all(e_vec[:,2]>=0):
        omega=omega_temp
    elif np.all(e_vec[:,2]<0):
        omega=2*np.pi-omega_temp
    else:
        omega=2*np.pi*(e_vec[:,2]<0)+(1-2*(e_vec[:,2]<0))*omega_temp # This case should only happen in the rare event that n is very dependent on time
    T=np.pi*2*np.sqrt(a**3/G/M)
    return (np.degrees(i),np.degrees(Omega),np.degrees(omega),e,a,T),(h,n,e_vec)