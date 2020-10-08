import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import vg

fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
ax = fig.gca(projection='3d')


def rotate(x,y,z,axis_x, axis_y, axis_z, theta):
    axis = [axis_x, axis_y, axis_z]
    theta = theta
    for i in range(len(x)):
        x[i],y[i],z[i] = np.dot(rotation_matrix(axis, theta), [x[i],y[i],z[i]])
    return x,y,z


def getAngle(x1,y1,z1,x2,y2,z2,x3,y3,z3, x4=None,y4=None,z4=None,x5=None,y5=None,z5=None,x6=None,y6=None,z6=None, ex=None, ey=None, ez=None):  

    v1 = [x2-x1,y2-y1,z2-z1]
    v2 = [x3-x1,y3-y1,z3-z1]

    print(v1, v2)
    if (x4 != None):
        v3 = [x5-x4,y5-y4,z5-z4]
        v4 = [x6-x4,y6-y4,z6-z4]


    nv1 = np.cross(np.array(v1), np.array(v2))
    print(" nv1:", nv1)
    if (x4 != None):
        nv2 = np.cross(np.array(v3), np.array(v4))
    else:
        nv2 = [ex,ey,ez]
    
    d = ( nv1[0] * nv2[0] + nv1[1] * nv2[1] + nv1[2] * nv2[2] ) 
    e1 = np.sqrt( nv1[0] * nv1[0] + nv1[1] * nv1[1] + nv1[2] * nv1[2]) 
    e2 = np.sqrt( nv2[0] * nv2[0] + nv2[1] * nv2[1] + nv2[2] * nv2[2]) 
    d = d / (e1 * e2) 
    A = np.arccos(d)
    return A


def getXYZAngles(x1,y1,z1,x2,y2,z2,x3,y3,z3):  

    v1 = np.array([x2-x1,y2-y1,z2-z1])
    v2 = np.array([x3-x1,y3-y1,z3-z1])

    nv = np.cross(v1, v2)
    nn = nv/ np.linalg.norm(nv)
    angles = (np.arccos(nn))
    return angles


# z1 = [0 for i in np.linspace(0,3,10)]
# x1 = [i for i in np.linspace(0,3,10)]
# y1 = [0 for i in np.linspace(0,3,10)]
# ax.plot3D(x1, y1, z1, label='parametric curve')

# z2 = [i for i in np.linspace(1,3,10)]
# x2 = [i for i in np.linspace(1,3,10)]
# y2 = [i for i in np.linspace(1,3,10)]
# ax.plot3D(x2, y2, z2, label='parametric curve')





center_plane_x = np.array([[1,0],[1,0]])
center_plane_y = np.array([[0,1],[1,2]])
center_plane_z = np.array([[0,0],[0.5,0.5]])

ax.plot_surface(center_plane_x, center_plane_y, center_plane_z,alpha=0.5)



center_plane1_x = np.array([[0,0],[0,0]])
center_plane1_y = np.array([[0,1],[0,1]])
center_plane1_z = np.array([[0,0],[1,1]])

ax.plot_surface(center_plane1_x, center_plane1_y, center_plane1_z,alpha=0.5)


center_plane2_x = np.array([[0,1],[0,1]])
center_plane2_y = np.array([[0,0],[0,0]])
center_plane2_z = np.array([[0,0],[1,1]])

ax.plot_surface(center_plane2_x, center_plane2_y, center_plane2_z,alpha=0.5)


center_plane3_x = np.array([[0,0],[1,1]])
center_plane3_y = np.array([[0,2],[0,2]])
center_plane3_z = np.array([[1,1],[1,1]])

ax.plot_surface(center_plane3_x, center_plane3_y, center_plane3_z,alpha=0.5)




ang_x = getAngle(center_plane_x[0][0],center_plane_y[0][0],center_plane_z[0][0],center_plane_x[0][1],center_plane_y[0][1],center_plane_z[0][1],center_plane_x[1][0],center_plane_y[1][0],center_plane_z[1][0],  center_plane1_x[0][0],center_plane1_y[0][0],center_plane1_z[0][0],center_plane1_x[0][1],center_plane1_y[0][1],center_plane1_z[0][1],center_plane1_x[1][0],center_plane1_y[1][0],center_plane1_z[1][0])
print("ang_x:", ang_x*180/np.pi)
ang_y = getAngle(center_plane_x[0][0],center_plane_y[0][0],center_plane_z[0][0],center_plane_x[0][1],center_plane_y[0][1],center_plane_z[0][1],center_plane_x[1][0],center_plane_y[1][0],center_plane_z[1][0],  center_plane2_x[0][0],center_plane2_y[0][0],center_plane2_z[0][0],center_plane2_x[0][1],center_plane2_y[0][1],center_plane2_z[0][1],center_plane2_x[1][0],center_plane2_y[1][0],center_plane2_z[1][0])
print("ang_y:", ang_y*180/np.pi)
ang_z = getAngle(center_plane_x[0][0],center_plane_y[0][0],center_plane_z[0][0],center_plane_x[0][1],center_plane_y[0][1],center_plane_z[0][1],center_plane_x[1][0],center_plane_y[1][0],center_plane_z[1][0],  center_plane3_x[0][0],center_plane3_y[0][0],center_plane3_z[0][0],center_plane3_x[0][1],center_plane3_y[0][1],center_plane3_z[0][1],center_plane3_x[1][0],center_plane3_y[1][0],center_plane3_z[1][0])
print("ang_z:", ang_z*180/np.pi)




xyz_angles = getXYZAngles(center_plane_x[0][0],center_plane_y[0][0],center_plane_z[0][0],center_plane_x[0][1],center_plane_y[0][1],center_plane_z[0][1],center_plane_x[1][0],center_plane_y[1][0],center_plane_z[1][0])

print("xyz_angles:", np.degrees(xyz_angles))



ax.set(xlabel='x', ylabel='y', zlabel='z')

plt.show()