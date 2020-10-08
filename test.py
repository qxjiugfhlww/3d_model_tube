from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

import math

import time

fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
ax = fig.gca(projection='3d')

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])



def rotate(x,y,z,axis_x, axis_y, axis_z, theta):
    axis = [axis_x, axis_y, axis_z]
    theta = theta
    for i in range(len(x)):
        x[i],y[i],z[i] = np.dot(rotation_matrix(axis, theta), [x[i],y[i],z[i]])
    return x,y,z

def rotateX(y,z,cy,cz, radians):
    for i in range(len(y)):
        _y      = y[i] - cy
        _z      = z[i] - cz
        d      = math.hypot(_y, _z)
        theta  = math.atan2(_y, _z) + radians
        z[i] = cz + d * math.cos(theta)
        y[i] = cy + d * math.sin(theta)
    return y,z

def rotateY(x,z,cx,cz, radians):
    for i in range(len(x)):
        _x      = x[i] - cx
        _z      = z[i] - cz
        d      = math.hypot(_x, _z)
        theta  = math.atan2(_x, _z) + radians
        z[i] = cz + d * math.cos(theta)
        x[i] = cx + d * math.sin(theta)

def rotateZ(x,y, cx,cy, radians):   
    for i in range(len(x)):
        _x      = x[i] - cx
        _y      = y[i] - cy
        d      = math.hypot(_x, _y)
        theta  = math.atan2(_x, _y) + radians
        x[i] = cx + d * math.sin(theta)
        y[i] = cy + d * math.cos(theta)
    return x,y



x = [i for i in np.arange(0,1,0.05)]
y = [np.cos(i) for i in x]
z = [0 for i in x]
ax.scatter(x, y, z, c='r', marker='o')


x,y = rotateZ(x,y, x[0],y[0], np.pi/4)

#x,y,z = rotate(x,y,z,1, 0, 0, -np.pi/2)





x1 = [i for i in np.arange(0,1,0.05)]
y1 = [1 for i in x1]
z1 = [np.sin(i**(1/2)) for i in x1]

theta = 4
z_1 = [i*np.cos(theta)-j*np.sin(theta) for i,j in zip(y1,z1) ]
y_1 = [i*np.sin(theta)+j*np.cos(theta) for i,j in zip(y1,z1) ]


x2 = [(i+j)/2 for i, j in zip(x,x1)]
y2 = [(i+j)/2 for i, j in zip(y,y1)]
z2 = [(i+j)/2 for i, j in zip(z,z1)]


print(len(x2))


# cset = ax.contour(x,y,z, zdir='z', offset=-100, cmap=cm.coolwarm)
# cset = ax.contour(x,y,z, zdir='x', offset=-40, cmap=cm.coolwarm)
# cset = ax.contour(x,y,z, zdir='y', offset=40, cmap=cm.coolwarm)



# axes = np.array([   [2,0.0,0.0],
#                     [0.0,2,0.0],
#                     [0.0,0.0,2]])
# # rotate accordingly
# for i in range(len(axes)):
#     axes[i] = np.dot(axes[i], 2)


# # plot axes
# for p in axes:
#     X3 = np.linspace(-p[0], p[0], 100) + 0
#     Y3 = np.linspace(-p[1], p[1], 100) + 0
#     Z3 = np.linspace(-p[2], p[2], 100) + 0
#     ax.plot(X3, Y3, Z3, color='black')




def getAngleBetweenPointsXY(x1,y1,x2,y2):
    return np.arctan2(y2-y1, x2-x1)

def getAngleBetweenPointsXZ(x1,z1,x2,z2):
    return np.arctan2(z2-z1, x2-x1)

def getAngleBetweenPointsYZ(y1,z1,y2,z2):
    return np.arctan2(z2-z1, y2-y1)




ax.scatter(x, y, z, c='r', marker='o')
ax.scatter(x2, y2, z2, c='g', marker='o')
ax.scatter(x1, y1, z1, c='b', marker='o')
ax.scatter(x1, y_1, z_1, c='m', marker='o')



#ax.scatter(ellipse_x, ellipse_y, ellipse_z, c='g', marker='o')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.set_xlim(-1, 2); ax.set_ylim(-1, 2); ax.set_zlim(-1, 1);

plt.show()

