import math
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.integrate

r_vec=np.array([594.193479,-5881.90168,-4579.2909])# km
v_vec=np.array([5.97301650,2.50988687,-2.448802691]) # km/s
mu=3.986004418*math.pow(10,5)# km^3/s^2
oe=[6798,.1,math.pi/4,math.pi/3,math.pi/4,math.pi]# a(in km),e,i,omega,OMEGA,nu

# Converts r and v to orbital elements, source of equations:
# https://orbital-mec1hanics.space/classical-orbital-elements/orbital-elements-and-the-state-vector.html
def covert_to_oe(r,v,mu):
    oe=[0,0,0,0,0,0]

    h=np.cross(r,v) # angular momentum vector
    k=np.array([0,0,1]) # z unit vector
    n=np.cross(k,h) # line of nodes
    rmag=np.linalg.norm(r)
    vmag=np.linalg.norm(v)
    hmag=np.linalg.norm(h)
    nmag=np.linalg.norm(n)

    #OMEGA
    OMEGA=np.arccos(n[0]/nmag) # right ascension of the ascending node
    if OMEGA<0:
        OMEGA=2*math.pi-OMEGA # plane check for right ascension of the ascending node

    #Eccentricity
    e_vec=np.cross(v,h)/mu-(r/rmag) # eccentricity vector
    e=np.linalg.norm(e_vec) # eccentricity normalized
    i=np.arccos(h[2]/hmag) # inclination

    #omega, w
    omega=np.arccos(np.dot(e_vec,n)/(nmag*e)) # argument of periapsis
    if omega<0:
        omega=2*math.pi-omega# plane check for argument of periapsis

    #nu
    nu=np.arccos(np.dot(e_vec,r)/(e*rmag)) # true anomaly
    if nu<0:
        nu=2*math.pi-nu # plane check for true anomaly

    #semi-major axis
    a=1/(2/rmag-(vmag**2)/mu) # semi-major axis
    
    i=math.acos(h[2]/hmag)
    OMEGA=math.acos(n[0]/nmag)
    omega=math.acos(np.dot(n,e_vec)/(nmag*e))
    nu=math.acos(np.dot(e_vec,r)/(e*rmag))
    oe=[a,e,i,omega,OMEGA,nu]
    return oe

# Converts orbital elements to r and v:
# https://space.stackexchange.com/questions/19322/converting-orbital-elements-to-cartesian-state-vectors
def convert_to_rv(oe,mu):
    #intaializations
    r=[0,0,0]
    v=[0,0,0]
    
    a=oe[0]
    e=oe[1]
    i=oe[2]
    omega=oe[3]
    OMEGA=oe[4]
    nu=oe[5]
    p=a*(1-e**2)
    ri=p/(1+e*math.cos(nu))
    h=math.sqrt(mu*a*p)
    # equations
################
    #PQW frame
    r=np.array([[ri*math.cos((nu))],
                [ri*math.sin(nu)],
                [0]])
    v=math.sqrt(mu/p)*np.array([[-math.sin(nu)],
                                [(e+math.cos(nu))],
                                [0]])
    #313 rotation
    Rw=rotate(2,omega)
    Ri=rotate(0,i)
    RO=rotate(2,OMEGA)

    
    Rtot=np.transpose(Rw@Ri@RO)
    r_vec=Rtot@r
    v_vec=Rtot@v


    return r_vec,v_vec

def find_period(oe,mu):
    a=oe[0]
    return 2*math.pi*math.sqrt(a**3/mu)
def rotate(val,angle): # value 0 corresponds to first column or x and so on
    if val==0:
        R=np.array([
        [1, 0, 0],
        [0, np.cos(angle), np.sin(angle)],
        [0, -np.sin(angle), np.cos(angle)]
        ])
    elif(val==1):
        R=np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
        ])
    elif(val==2):
        R=np.array([
        [np.cos(angle), np.sin(angle), 0],
        [-np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
        ])
    return R

def fun_def(t,y): #function format passed to scipy
    #initializations
    mu=3.986004418*math.pow(10,5)
    r=y[0:3]
    v=y[3:6]
    r_norm=np.linalg.norm(r)
    mu=int(mu)
    a1=-mu*r[0]/r_norm**3
    a2=-mu*r[1]/r_norm**3
    a3=-mu*r[2]/r_norm**3
    a=([a1,a2,a3])
    A=np.concatenate([v,a])

    return A



if __name__ == "__main__":
    # prompt that checks if I want to convert to oe, rv, or input new oe, rv
    val=int(input("convert to oe(1), convert to rv(2), new oe(3), new rv(4): "))
    if val==1:
        oe=covert_to_oe(r_vec,v_vec,mu)
        print("Orbital Elements: ")
        print(oe)
    elif val==2:
        r,v=convert_to_rv(oe,mu)
        print("r and v: ")
        print(r)
        print(v)
    elif val==3:
        oe=[float(x) for x in input("Enter oe[a,e,i,omega,OMEGA,nu]: ").split()]
        r,v=convert_to_rv(oe,mu)
        oe=covert_to_oe(r,v,mu)
        print('Orbital Elements: ')
        print(oe)
    elif val==4:
        r=[float(x) for x in input("Enter r: ").split()]
        v=[float(x) for x in input("Enter v: ").split()]
        oe=covert_to_oe(r,v,mu)
        r,v=convert_to_rv(oe,mu)
        print('r and v: ')
        print(r)
        print(v)
    else:
        pass
    new_val=int(input("Close program(1), plot_solve_numerical(2), plot_oes (3): "))
    tspan=find_period(oe,mu)
    if new_val==1:
        exit()
    elif new_val==2:
        r=r_vec
        v=v_vec
    
        y0=np.concatenate([r,v])
        t=np.linspace(0,tspan,100)
        y=scipy.integrate.solve_ivp(fun_def,(0,tspan),y0,method='DOP853',t_eval=t)
        print(y)
    
        #Plotting
        fig=plt.figure()
        plt.plot(y.y[0],y.y[1])
        plt.xlabel('x-s')
        plt.ylabel('y-km')
        plt.show()

        fig=plt.figure()
        ax=plt.axes(projection='3d')
        ax.plot3D(y.y[0],y.y[1],y.y[2],'red')
        plt.xlabel('x-km')
        plt.ylabel('y-km')
        plt.clabel('z-km')
        plt.show()
    elif new_val==3:
        r=r_vec
        v=v_vec
    
        y0=np.concatenate([r,v])
        t=np.linspace(0,tspan,100)
        a=np.zeros(len(t))
        inc=np.zeros(len(t))
        e=np.zeros(len(t))
        omega=np.zeros(len(t))
        OMEGA=np.zeros(len(t))
        nu=np.zeros(len(t))

        y=scipy.integrate.solve_ivp(fun_def,(0,tspan),y0,method='DOP853',t_eval=t)
        t=np.linspace(0,tspan,100)
        xx=y.y[0]
        yy=y.y[1]
        zz=y.y[2]

        vx=y.y[3]
        vy=y.y[4]
        vz=y.y[5]


        for i in range(len(t)):
            r_i=[xx[i],yy[i],zz[i]]
            v_i=[vx[i],vy[i],vz[i]]
            oe=covert_to_oe(r_i,v_i,mu)

            a[i]=oe[0]
            e[i]=oe[1]
            inc[i]=oe[2]
            omega[i]=oe[3]
            OMEGA[i]=oe[4]
            nu[i]=oe[5]
        fig,ax=plt.subplots(6)
        ax[0].plot(t,a)
        ax[0].set_title('Semi-major axis')
        ax[1].plot(t,e)
        ax[1].set_title('Eccentricity')
        ax[2].plot(t,inc)
        ax[2].set_title('Inclination')
        ax[3].plot(t,omega)
        ax[3].set_title('Argument of periapsis')
        ax[4].plot(t,OMEGA)
        ax[4].set_title('Right ascension of the ascending node')
        ax[5].plot(t,nu)
        ax[5].set_title('True anomaly')
        plt.show()
        #Notice how all graphs are more or less constant expect nu which varies from [0,pi]

    else:
        pass

