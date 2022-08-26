# from dolfin import *

# mesh = Mesh("mesh.xml")

# 

# from ast import Interactive
# from dolfin import *
# from fenics import *

# from fenics import *
# import numpy as np
# #from fenics import Mesh, MeshEditor, FunctionSpace
import matplotlib.pyplot as plt


# # using https://bitbucket.org/fenics-project/dolfin/src/b55804ecca7d010ff976967af869571b56364975/dolfin/generation/IntervalMesh.cpp#lines-76:98 as template
# N = 5 # we want to work with 5 vertices on the mesh
# gdim = 2 # geometric dimension
# tdim = 0
# vertices = np.random.rand(N, gdim)

# mesh = IntervalMesh(1,0,1) # empty mesh
# editor = MeshEditor()
# editor.open(mesh,type='interval',tdim=1,gdim=1)
# editor.init_vertices(3)
# editor.init_cells(2)
# editor.add_vertex(2,Point(0.95))
# editor.add_cell(2,np.array([0.95]),np.array([1]))
# editor.close()

# fig, ax1 = plt.subplots()
# fig=plot(mesh)

# plt.show()

# from dolfin import *
# import numpy as np
# editor = MeshEditor()
# mesh = Mesh()
# editor.open(mesh, type="point",tdim=2, gdim=2)  # top. and geom. dimension are both 2
# editor.init_vertices(4)  # number of vertices
# editor.init_cells(2)     # number of cells
# editor.add_vertex(0, np.array([0.0, 0.0]))
# editor.add_vertex(1, np.array([1.0, 0.0]))
# editor.add_vertex(2, np.array([0.0, 1.0]))
# editor.add_vertex(3, np.array([1.0, 1.0]))
# editor.add_cell(0, np.array([0, 1, 3], dtype=np.uintp))
# editor.add_cell(1, np.array([0, 2, 3], dtype=np.uintp))
# editor.close()
# fig, ax1 = plt.subplots()
# fig=plot(mesh)

# plt.show()

from dolfin import *
import numpy

mesh   = Mesh()
editor = MeshEditor()

A =0
B = 4.5
C = 10
N_AB = 10 # Number of cells in AB
N_BC = 10 # Number of cells in BC


vertices = numpy.linspace(B, C, N_BC+1)[1:]

hCap=(C-B)/N_BC
tNew=0
tO=B
i=0
while tO-2*tNew>0:
    tNew=tO-hCap*pow(1.5,i)
    vertices=numpy.insert(vertices,0,tNew)
    i=+1

vertices=numpy.insert(vertices,0,[0 ,tNew/2])

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
    editor.add_vertex(i, numpy.array([vertex], dtype='float'))
# Add cells
for i in range(num_local_cells):
    editor.add_cell(i, numpy.array([i, i+1], dtype='uint'))

print(mesh.coordinates())

# Close editor
editor.close()

fig, ax1 = plt.subplots()
fig=plot(mesh)

plt.show()

