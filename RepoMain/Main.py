#----------------------------------------------------------
#
#   This is a Firedrake implementation of the AVS-FE method 
#   for a convection-diffusion problem
#    -epsilon*laplacian(u) + b dot grad(u) = f  on UnitSquare,
#                                 u = 0   on boundary.
#   
#   Note that this is the 'mixed' formulation without 
#   any static condenzation of the error representation 
#   function, thus making the global system large and 
#   inefficient to solve:
#   H - B = -f
#   B^* = 0
#   Written by Eirik Valseth August 2021
#-----------------------------------------------------------
import os
import mshr
import numpy as np
import time
import dolfin
from fenics import *
from dolfin import *
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import math

############## JOSH FUNCTIONS###################
def my_cross(c):
    return (c[1],-c[0])

def my_outer(c):
    return as_vector(my_cross(c))

#Define Boundary at t=0 
def boundary_D(x,on_boundary):
    return on_boundary and near(x[0],0,tol)    

def genMesh(t0,tc,tf,n):
    mesh   = Mesh()
    editor = MeshEditor()

    A = t0
    B = tc
    C = tf
    N_BC = n # Number of cells in BC

    vertices = np.linspace(B, C, N_BC+1)[1:]

    hCap=(C-B)/N_BC
    tNew=B
    tO=B
    i=0
    while 2*tNew-tO>0: #Check to bisect the last 2 points
        tNew=tO-hCap*pow(1.5,i)
        vertices=np.insert(vertices,0,tNew)
        i+=1
        print(i)

    vertices=np.insert(vertices,0,[0 ,tNew/2])
    
    topological_dim = 1
    geometrical_dim = 1

    num_local_vertices = len(vertices)
    num_local_cells    = len(vertices)-1

    num_global_vertices = num_local_vertices
    num_global_cells    = num_local_cells

    editor.open(mesh, "interval" , topological_dim, geometrical_dim)
    editor.init_vertices_global(num_local_vertices, num_global_vertices)
    editor.init_cells_global(num_local_cells, num_global_cells)

    # Add vertices
    for i, vertex in enumerate(vertices):
        editor.add_vertex(i, np.array([vertex], dtype='float'))
    # Add cells
    for i in range(num_local_cells):
        editor.add_cell(i, np.array([i, i+1], dtype='uint'))

    # Close editor
    editor.close()

    fig, ax1 = plt.subplots()
    fig=plot(mesh)

    print(vertices)

    print(len(vertices))

    plt.show()


    return (mesh)

def genFunctionSpace(mesh,pdegree):
    #Generate Function Space
    Vx=VectorElement('CG',mesh.ufl_cell(),degree=pdegree,dim=2)
    Vq=VectorElement('CG',mesh.ufl_cell(),degree=pdegree,dim=2)
    Vex=VectorElement('DG',mesh.ufl_cell(),degree=pdegree+1,dim=2)
    Veq=VectorElement('DG',mesh.ufl_cell(),degree=pdegree+1,dim=2)
    Ve=MixedElement([Vx,Vq,Vex,Veq])
    V=FunctionSpace(mesh,Ve)
    return (V)

def genTestTrialBCFunctions(V):
    #Define Test and Trial Function\\s
    (delx,delq,er,Er)=TrialFunctions(V)    
    (pTest,rTest,gamma,xi)=TestFunctions(V)


    #Exact solutions
    del_x_guess=Expression(('0','0'),degree=20)#,a1=a1,a2=a2,x2o=x2o,tf=tf
    del_q_guess=Expression(('0','0'),degree=20)

    ex_guess=Expression(('0','0'),degree=20)
    eq_guess=Expression(('0','0'),degree=20)

    #Define BC's of delx
    bc_delx=DirichletBC(V.sub(0),del_x_guess,boundary_D)
    bc_delq=DirichletBC(V.sub(1),del_q_guess,boundary_D)
    bc_ex=DirichletBC(V.sub(2),ex_guess,boundary_D)
    bc_eq=DirichletBC(V.sub(3),eq_guess,boundary_D)

    bc=[bc_delx,bc_delq,bc_ex,bc_eq]

    return (delx,delq,er,Er,pTest,rTest,gamma,xi,bc)

#Should be used currently
def genXnQn(x0,y0,qx0,qy0,V,Vref):
    #Define initial values
    x_n=Expression(('qx0*x[0]','qy0*x[0]+y0'),degree=10,x0=x0,qx0=qx0,y0=y0,qy0=qy0)
    x_n=project(x_n,V.sub(0).collapse()) #not sure if I'm supposed to interp cos(x[0])
    x_nref=project(x_n,Vref.sub(0).collapse()) #not sure if I'm supposed to interp cos(x[0])

    q_n=Expression(('qx0','qy0'),degree=10,qx0=qx0,qy0=qy0)
    q_n=project(q_n,V.sub(1).collapse()) #not sure if I'm supposed to interp
    q_nref=project(q_n,Vref.sub(1).collapse()) #not sure if I'm supposed to interp

    return(x_n,q_n,x_nref,q_nref)

def genXcQc(V,Vref):
    #Define initial values
    x_n=Expression(('0','0'),degree=10,x0=x0,qx0=qx0,y0=y0,qy0=qy0)
    x_n=project(x_n,V.sub(0).collapse()) #not sure if I'm supposed to interp cos(x[0])

    q_n=Expression(('0','0'),degree=10,qx0=qx0,qy0=qy0)
    q_n=project(q_n,V.sub(1).collapse()) #not sure if I'm supposed to interp

    x_nref=Expression(('0','0'),degree=10,x0=x0,qx0=qx0,y0=y0,qy0=qy0)
    x_nref=project(x_nref,Vref.sub(0).collapse()) #not sure if I'm supposed to interp cos(x[0])

    q_nref=Expression(('0','0'),degree=10,qx0=qx0,qy0=qy0)
    q_nref=project(q_nref,Vref.sub(1).collapse()) #not sure if I'm supposed to interp

    return(x_n,q_n,x_nref,q_nref)


def genLHS(x_n,q_n,delx,delq,er,Er,pTest,rTest,gamma,xi,mesh):#,delx_E,delq_E
    #Inner Product
    h = Circumradius(mesh)
    #print

    hmax = mesh.hmax()

    #h1
    iprod=(inner(er,gamma)+inner(Er,xi)+h*h*inner(er.dx(0),gamma.dx(0))+h*h*inner(Er.dx(0),xi.dx(0)))

    #L2
    #iprod=inner(er,gamma)+inner(Er,xi)

    Lambda=(epsilon+x_n[1])/D
    dLambda=delx[1]/D

    #Define Left Hand Side######
    dJqx=inner(delq-delx.dx(0),xi)*wMult
    dJma=inner(m*delq.dx(0),gamma)
    dJsd=inner(3*pi*nu*D*(delq),gamma)
    dJsl=-inner(1.62*mu*D*D*sqrt(tau/nu)*my_outer(delq),gamma)
    dJvdw=-inner(H*dLambda/(6*D)*(1/pow(Lambda,2)-1/pow(Lambda,3)-1/pow(Lambda+1,2)-1/pow(Lambda+1,3))*en,gamma)
    dJel=-inner(2*Q/pow(D/2+x_n[1],3)*en*delx[1],gamma)

    dJ=dJqx+dJma+dJsd+dJsl+dJvdw+dJel

    #Define the test/Error Product###########
    edLambda=pTest[1]/D

    edJqx=inner(rTest-pTest.dx(0),Er)*wMult
    edJma=inner(m*rTest.dx(0),er)
    edJsd=inner(3*pi*nu*D*(rTest),er)
    edJsl=-inner(1.62*mu*D*D*sqrt(tau/nu)*my_outer(rTest),er)
    edJvdw=-inner(H*edLambda/(6*D)*(1/pow(Lambda,2)-1/pow(Lambda,3)-1/pow(Lambda+1,2)-1/pow(Lambda+1,3))*en,er)
    edJel=-inner(2*Q/pow(D/2+x_n[1],3)*en*pTest[1],er)
    
    edJ=edJqx+edJma+edJsd+edJsl+edJvdw+edJel

    Lhs=(iprod-dJ+edJ)*dx

    return (Lhs)

def genRHS(x_n,q_n,gamma,xi,capBool):
    u=Expression(('x_n[1]>blDelta?Vx0:Vx0*pow(x_n[1]/blDelta,2)','0'),degree=2,x_n=x_n,Vx0=Vx0,blDelta=blDelta) #Velocity of the fluid
    # u=Expression(('Vx0','0'),degree=2,x_n=x_n,Vx0=Vx0,blDelta=blDelta) #Velocity of the fluid

    #Define the normal right hand side
    rLambda=(epsilon+x_n[1])/D

    Jqx=dot(q_n-x_n.dx(0),xi)*wMult
    Jma=dot(m*q_n.dx(0),gamma)
    Jsd=-dot(3*pi*nu*D*(u-q_n),gamma)
    Jsl=-dot(1.62*mu*D*D*sqrt(tau/nu)*my_outer(u-q_n),gamma)
    Jvdw=-dot(H/(12*D)*(2/rLambda-1/pow(rLambda,2)-2/(rLambda+1)-1/pow(rLambda+1,2))*en,gamma)
    Jel=-dot(Q/pow(D/2+x_n[1],2)*en,gamma)

    if capBool==True:
        Fc=Expression(('0','x_n[1]<hc?-2*M_PI*sigma*D:0'),x_n=x_n,degree=2,hc=hc,D=D,sigma=sigma)
        print("here")
    else:
        Fc=Expression(('0','0'),degree=1)
    #Fc=Expression(('0','x_n[1]<hc?-2*M_PI*sigma*D:0'),x_n=x_n,degree=2,hc=hc,D=D,sigma=sigma)
    #Fc=Expression(('0','0'),degree=1)
    Jc=dot(Fc,gamma)
    Fg=Expression(('0','-m*grav'),m=m,grav=grav,degree=2)
    Jg=dot(Fg,gamma)

    J=Jqx+Jma+Jsd+Jsl+Jel+Jvdw 

    Rhs=(J-Jc-Jg)*dx    #-Jc due to it being a force
    return (Rhs)

#Rebuild x_n,q_n onto Vcurrent
def projFunc(func1,func2,V):
    func1=project(func1,V.sub(0).collapse())
    func2=project(func2,V.sub(1).collapse())
    return (func1,func2)

def sortMesh(M):
    npMesh=np.sort(np.array(M.coordinates()),axis=0)

    #print(npMesh)

    num_local_vertices = npMesh.size
    num_local_cells    = num_local_vertices-1

    mesh   = Mesh()
    editor = MeshEditor()

    topological_dim = 1
    geometrical_dim = 1

    editor.open(mesh, "interval" , topological_dim, geometrical_dim)
    editor.init_vertices_global(num_local_vertices, num_local_vertices)
    editor.init_cells_global(num_local_cells, num_local_cells)

    # Add vertices
    for i in range(npMesh.size):
        editor.add_vertex(i, np.array(npMesh[i], dtype='float'))
    # Add cells
    for i in range(num_local_cells):
        editor.add_cell(i, np.array([i, i+1], dtype='uint'))

    #print(mesh.coordinates())

    return mesh

def test(V,x_n,q_n,capBool,errorTarget):

    #Taylor Series stuff
    iter=0
    iter=1
    #maxiter=50
    errorCurr=1
    #errorTarget=1e-20


    #Generate x_n and q_n
    #(x_ans,q_ans,x_c,q_c)=genXcQc(V,V)#Vref)

    vtkX=File('Iterative Avs/Results'+str(n)+"_p"+str(pdegree)+'/'+testName+'x_mult'+str(wMult)+'.pvd')
    vtkQ=File('Iterative Avs/Results'+str(n)+"_p"+str(pdegree)+'/'+testName+'q_mult'+str(wMult)+'.pvd')

    vtkDX=File('Iterative Avs/Results'+str(n)+"_p"+str(pdegree)+'/'+testName+'dx_mult'+str(wMult)+'.pvd')
    vtkDQ=File('Iterative Avs/Results'+str(n)+"_p"+str(pdegree)+'/'+testName+'dq_mult'+str(wMult)+'.pvd')

    vtkEX=File('Iterative Avs/Results'+str(n)+"_p"+str(pdegree)+'/'+testName+'erx_mult'+str(wMult)+'.pvd')
    vtkEQ=File('Iterative Avs/Results'+str(n)+"_p"+str(pdegree)+'/'+testName+'erq_mult'+str(wMult)+'.pvd')

    tstart = time.time()

    exVect=Function(V)

    while errorCurr > errorTarget and iter<maxiter:
        iter=iter+1
        global M
        # Define the necessary function spaces: 
        (V)=genFunctionSpace(M,pdegree)
        #(x_n,q_n,x_nref,q_nref)=genXnQn(x0,y0,qx0,qy0,V,Vref)
        #(x_n,q_n)=projFunc(x_n,q_n,V)

        x_n=project(x_c,V.sub(0).collapse())
        q_n=project(q_c,V.sub(1).collapse())

        vtkX<<(x_n,iter)
        vtkQ<<(q_n,iter)

        x_c.assign(x_n)
        q_c.assign(q_n)

        #Gen test and trial functions and bcs
        # Define the test and trial functions on the total space, 
        # e.g. phi and w belong to the discontinuous vector valued space V1
        # and u and r belong to the space of continuous polynomials V4 etc.
        (delx,delq,er,Er,pTest,rTest,gamma,xi,bc)=genTestTrialBCFunctions(V)

        # To compute the inner product we need the element diameter 
        h = Circumradius(M)
        hmax = M.hmax()
        
        #print(M.hmin())
#x_n,q_n,pTest,rTest,delx,delq,gamma,xi,er,Er,mesh
        #GEN LHS RHS
        Lhs=genLHS(x_n,q_n,delx,delq,er,Er,pTest,rTest,gamma,xi,M)#,del_x_E,del_q_E
        Rhs=genRHS(x_n,q_n,gamma,xi,capBool)

        #Generate answer
        an=Function(V)

        # Call the solver
        solve(Lhs==Rhs, an, bcs=bc, solver_parameters = {'linear_solver' : 'mumps'}) #, bcs=[bc,bc1]
        # Split the solution vector
        (delx_nsol,delq_nsol,ex_sol,eq_sol)=an.split()

        # vtkfileE<<(delx_nsol,iter)
        # vtkfileEq<<(delq_nsol,iter)

        #Prep to print file and add to x_n in while loop if mesh refining is done
        delx_nsol=project(delx_nsol,V.sub(0).collapse())
        delq_nsol=project(delq_nsol,V.sub(1).collapse())

        vtkDX<<(delx_nsol,iter)
        vtkDQ<<(delq_nsol,iter)

        vtkEX<<(ex_sol,iter)
        vtkEQ<<(eq_sol,iter)

        x_c.vector()[:]=x_n.vector()+alpha*delx_nsol.vector()
        q_c.vector()[:]=q_n.vector()+alpha*delq_nsol.vector()

        #### Eirik Mesh refinement ####

        # compute error indicators from the DPG mixed variable e
        PC = FunctionSpace(M,"DG", 0)
        c  = TestFunction(PC)             # p.w constant fn

        ge = ( inner(eq_sol, eq_sol)*c+ inner(ex_sol, ex_sol)*c  + h**2 * inner(grad(ex_sol), grad(ex_sol))*c+ h**2 * inner(grad(eq_sol), grad(eq_sol))*c  )*dx
        g = assemble(ge)

        # Mark cells for refinement
        cell_markers = MeshFunction("bool", M, M.topology().dim())    
        Mold=M

        #Mesh Refinement area
        g0 = sorted(g, reverse=True)[int(len(g)*REFINE_RATIO)]
        for cellm in cells(M):
            cell_markers[cellm] = g[cellm.index()] > g0
            #print(cellm)
        M=refine(M,cell_markers)

        M=sortMesh(M)

        #print(M.coordinates())

        ##### END MESH REFINEMENT #####
        #print(iter, errorCurr,errorCurr/int(len(g)))   

    errorCurr=errornorm(q_n,q_c,'L2')#project(x_c,V.sub(0).collapse()))

    print(iter, errorCurr,errorCurr/int(len(g)))   

    tstopAVS = time.time()
    return(x_c,interpolate(x_c,V.sub(0).collapse()),q_c,Mold)
###################  END JOSH FUNCTIONS   ######################

########################################################################
#                  Stuff that actually gets adjusted
########################################################################

##################################
# Implementation
##################################

#Iteratitive stuff
maxiter=50 #Taylor Series Stuff
REFINE_RATIO = 0.0 #Adaptive refinment
alpha=1e-1 #Multiplier for delx  (x+alpha*delx)

#Weights for implementation
wMult=1e-3 #Weight Multiplier for mixed formulation

#Mesh Generation
n=50 #number of elements
pdegree=3 #poly degree

#Test Name
testName='Find_Zero'

##################################
# Problem info
##################################

#FLow
Vx0=-2 #Flow x Velocitiy
blDelta=1e-2 #boundarylayer thickness

#Initializing Particles
v0=1 #Particle velocities
theta=10 #Particle entry angle

y0=2e-2#-qy0 #meters
x0=0

thetarad=math.radians(theta)

qx0=cos(thetarad)*v0
qy0=-sin(thetarad)*v0

#Generate Mesh
t0=0
tf=-y0/qy0*1.1#3.45e-6#3.417945e-06

print(tf)

#hc=1e10 #To find out right timeframe
hc=1e-6

###############

################## JOSH PREP SHIT ####################
#End loop
tol=1E-14

#Define of Particle onstants
mu=Constant(1.488e-5) #m2/s
D=Constant(50e-6)#mConstant(1)#
m=Constant(1.3e-9)
tau=Constant(1)
nu=Constant(1.81e-5) #kg/(m*s)
H=Constant(1e-20) #J
Q=Constant(0) #zero if high humidity
sigma=Constant(72e-3) #N/m
grav=Constant(9.81)#m/s^2

en=Constant((0,1))
ep=Constant((1,0))

epsilon=Constant(5e-10)

#alpha=1e-1

########### END JOSH PREP ##########

File_name = 'Figures'
TrialTypeDir = os.path.join(File_name, 'P_' + str(pdegree))
if not os.path.isdir(TrialTypeDir): os.makedirs(TrialTypeDir)


#Generate some stuff
M=IntervalMesh(n,t0,tf)
V=genFunctionSpace(M,pdegree)
(x_n,q_n,x_c,q_c)=genXnQn(x0,y0,qx0,qy0,V,V)#Vref)

errorTarg=1e-8
maxiter=150

#Main Run
(xRes,projRes,qRes,M)=test(V,x_n,q_n,False,errorTarg) #Have V and M maybe be inputs

#Finding the time of impact
x_Ans_Array=np.array(xRes.vector().get_local())
x_Ans_Array=np.reshape(x_Ans_Array,(-1,2))
x_Ans_Sort=(x_Ans_Array[x_Ans_Array[:,1].argsort()])#have results array backwards so interpolating works
meshInt=np.linspace(tf,0,n*pdegree+1)

#interp Stuff
tf=np.interp(0,x_Ans_Sort[:,1],meshInt)
tCap=np.interp(hc*2,x_Ans_Sort[:,1],meshInt)

print(t0,tCap,tf)


############# SECOND RUN ##############
testName="FUCK"

n=100

M = genMesh(0,tCap,tf,n)
V=genFunctionSpace(M,pdegree)

xNew=project(xRes,V.sub(0).collapse())
qNew=project(qRes,V.sub(0).collapse())

vtkX=File('Iterative Avs/ResultsRun2.pvd')
vtkX<<(xNew,1)

(xRes,projRes,qRes,M)=test(V,xNew,qNew,True,errorTarg) #Have V and M maybe be inputs

# #Finding the time of impact
# x_Ans_Array=np.array(xRes.vector().get_local())
# x_Ans_Array=np.reshape(x_Ans_Array,(-1,2))
# x_Ans_Sort=(x_Ans_Array[x_Ans_Array[:,1].argsort()])#have results array backwards so interpolating works


# Minterp = genMesh(0,tCap,tf,pdegree*n,pdegree*n*10)
# tf=np.interp(-1e-9,x_Ans_Sort[:,1],Minterp.coordinates()[:,0])
# tCap=np.interp(hc*10,x_Ans_Sort[:,1],Minterp.coordinates()[:,0])

# print(t0,tCap,tf)

# ############# THIRD RUN ###############
# testName="REFUCK"

# M = genMesh(0,tCap,tf,n,n*10)
# V=genFunctionSpace(M,pdegree)

# xNew=project(xRes,V.sub(0).collapse())
# qNew=project(qRes,V.sub(0).collapse())

# vtkX=File('Iterative Avs/ResultsRun3.pvd')
# vtkX<<(xNew,1)


# (xRes,projRes,qRes,M)=test(V,xNew,qNew,True) #Have V and M maybe be inputs
# print(t0,tCap,tf)

# print(tf-tCap)