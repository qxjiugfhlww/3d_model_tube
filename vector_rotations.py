import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def rotateVector3D(v, theta, axis):
    """ Takes a three-dimensional vector v and rotates it by the angle theta around the specified axis.
    """
    return np.dot(rotationMatrix3D(theta, axis), v)


def rotationMatrix3D(theta, axis):
    """ Return the rotation matrix associated with counterclockwise rotation about
        the given axis by theta radians.
    """
    axis = np.asarray(axis) / np.sqrt(np.dot(axis, axis)) 
    a = np.cos(theta/2.0)
    b, c, d = -axis*np.sin(theta/2.0)
    aa, bb, cc, dd = a**2, b**2, c**2, d**2
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])


def drawObject(ax, pts, color="red"):
    """ Draws an object on a specified 3D axis with points and lines between consecutive points.
    """
    map(lambda pt: ax.scatter(*pt, s=10, color=color), pts)
    pts = list(pts)
    for k in range(len(pts)-1):
        x, y, z = zip(*pts[k:k+2])
        ax.plot(x, y, z, color=color, linewidth=1.0)
    x, y, z = zip(*[pts[0],pts[1]])
    ax.plot(x, y, z, color=color, linewidth=1.0)


def normalVector(obj):
    """ Takes a set of points, assumed to be flat, and returns a normal vector with unit length.
    """
    print("fhgfh", obj)
    n = np.cross(np.array(list(obj[1]))-np.array(list(obj[0])), np.array(obj[2])-np.array(obj[0]))
    return n/np.sqrt(np.dot(n,n))


# Set the original object (can be any set of points)
obj = [[2, 0, 2], [2, 0, 4], [4, 0, 4], [4, 0, 2]]
mObj = [3, 0, 3]
nVecObj = normalVector(obj)

# Given vector.
vec = [6, 6, 6]

# Find rotation axis and angle.
rotAxis = normalVector([(0,0,0), nVecObj, vec])
angle =  np.arccos(np.dot(nVecObj, vec) / (np.sqrt(np.dot(vec, vec)) * np.sqrt(np.dot(nVecObj, nVecObj))))
print("Rotation angle: {:.2f} degrees".format(angle/np.pi*180))


# Generate the rotated object.
rotObj = map(lambda pt: rotateVector3D(pt, angle, rotAxis), obj)
mRotObj = rotateVector3D(mObj, angle, rotAxis)
nVecRotObj = normalVector(list(rotObj))


# Set up Plot.
fig = plt.figure()
fig.set_size_inches(18,18)
ax = fig.add_subplot(111, projection='3d')

# Draw.
drawObject(ax, [[0,0,0], np.array(vec)/np.sqrt(np.dot(vec,vec))], color="gray")
drawObject(ax, [mObj, mObj+nVecObj], color="red")
drawObject(ax, obj, color="red")
drawObject(ax, [mRotObj, mRotObj + nVecRotObj], color="green")
drawObject(ax, rotObj, color="green")

# Plot cosmetics.
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

# Check if the given vector and the normal of the rotated object are parallel (cross product should be zero).
print(np.round(np.sum(np.cross(vec, nVecRotObj)**2), 5))