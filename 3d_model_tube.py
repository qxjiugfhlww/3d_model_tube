from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import math
import random
from pyellipsoid import drawing

import vg

from mpl_toolkits.mplot3d import proj3d

import mpl_toolkits.mplot3d as m3d

from numpy.polynomial.polynomial import polyfit

import pandas as pd

from scipy.interpolate import griddata

from itertools import chain 

class Coord(object):
    def __init__(self,x,y):
        self.x = x
        self.y = y

    def __sub__(self,other):
        # This allows you to substract vectors
        return Coord(self.x-other.x,self.y-other.y)

    def __repr__(self):
        # Used to get human readable coordinates when printing
        return "Coord(%f,%f)"%(self.x,self.y)

    def length(self):
        # Returns the length of the vector
        return math.sqrt(self.x**2 + self.y**2)

    def angle(self):
        # Returns the vector's angle
        return math.atan2(self.y,self.x)

def normalize(coord):
    return Coord(
        coord.x/coord.length(),
        coord.y/coord.length()
        )

def perpendicular(coord):
    # Shifts the angle by pi/2 and calculate the coordinates
    # using the original vector length
    return Coord(
        coord.length()*math.cos(coord.angle()+math.pi/2),
        coord.length()*math.sin(coord.angle()+math.pi/2)
        )

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

def getAngleBetweenPointsXY(x1,y1,x2,y2):
    return np.arctan2(y2-y1, x2-x1)

def getAngleBetweenPointsXZ(x1,z1,x2,z2):
    return np.arctan2(z2-z1, x2-x1)

def getAngleBetweenPointsYZ(y1,z1,y2,z2):
    return np.arctan2(z2-z1, y2-y1)

def getPerpendicular(x1,y1,x2,y2, length):
    dx = x1-x2
    dy = y1-y2
    dist = math.sqrt(dx*dx + dy*dy)
    dx /= dist
    dy /= dist
    x3 = x1 + (length)*dy
    y3 = y1 - (length)*dx
    x4 = x1 - (length)*dy
    y4 = y1 + (length)*dx
    return x3,y3,x4,y4

def getAngleBetweenLines(x1, y1, z1, x2, y2, z2, x3, y3, z3):                 
    # Find direction ratio of line AB 
    ABx = x1 - x2; 
    ABy = y1 - y2; 
    ABz = z1 - z2; 
  
    # Find direction ratio of line BC 
    BCx = x3 - x2; 
    BCy = y3 - y2; 
    BCz = z3 - z2; 
    # Find the dotProduct 
    # of lines AB & BC 
    dotProduct = (ABx * BCx + ABy * BCy + ABz * BCz); 
  
    # Find magnitude of 
    # line AB and BC 
    magnitudeAB = (ABx * ABx + ABy * ABy + ABz * ABz); 
    magnitudeBC = (BCx * BCx + BCy * BCy + BCz * BCz); 
    # Find the cosine of 
    # the angle formed 
    # by line AB and BC 
    angle = dotProduct; 
    angle /= math.sqrt(magnitudeAB * magnitudeBC); 
  
    # Find angle in radian 
    angle = (angle * 180) / 3.14; 
    # Print angle 
    return angle

def multiDimenDist(point1,point2):  
   #find the difference between the two points, its really the same as below
   deltaVals = [point2[dimension]-point1[dimension] for dimension in range(len(point1))]
   runningSquared = 0
   #because the pythagarom theorm works for any dimension we can just use that
   for coOrd in deltaVals:
       runningSquared += coOrd**2
   return runningSquared**(1/2)
def findVec(point1,point2,unitSphere = False):
  #setting unitSphere to True will make the vector scaled down to a sphere with a radius one, instead of it's orginal length
  finalVector = [0 for coOrd in point1]
  for dimension, coOrd in enumerate(point1):
      #finding total differnce for that co-ordinate(x,y,z...)
      deltaCoOrd = point2[dimension]-coOrd
      #adding total difference
      finalVector[dimension] = deltaCoOrd
  if unitSphere:
      totalDist = multiDimenDist(point1,point2)
      unitVector =[]
      for dimen in finalVector:
          unitVector.append( dimen/totalDist)
      return unitVector
  else:
      return finalVector

def getAnlgeBetweenVectors(x1, y1, z1, x2, y2, z2,x3,y3,z3,x4,y4,z4):

    vec1 = findVec([x1,y1,z1],[x2,y2,z2])
    vec2 = findVec([x3,y3,z3],[x4,y4,z4])   
    return np.arccos(np.dot(vec1 / np.linalg.norm(vec1), vec2 / np.linalg.norm(vec2)))
    #return np.arccos( (x1*x2 + y1*y2 + z1*z2) / np.sqrt( (x1*x1 + y1*y1 + z1*z1)*(x2*x2+y2*y2+z2*z2) ) )

def getPlaneNormal(x1,y1,z1,x2,y2,z2,x3,y3,z3):
    v1 = [x2-x1,y2-y1,z2-z1]
    v2 = [x3-x1,y3-y1,z3-z1]
    x = 1*v1[1]*v1[2]+1*v1[2]*v2[1]
    y = 1*v1[2]*v2[0]+1*v1[0]*v2[2]
    z = 1*v1[1]*v2[0]+1*v1[0]*v2[1]
    return [x,y,z]

def plot_arc3d(vector1, vector2, radius=0.2, fig=None, colour='black'):
    """ Plot arc between two given vectors in 3D space. """

    """ Confirm correct input arguments """
    assert len(vector1) == 3
    assert len(vector2) == 3

    """ Calculate vector between two vector end points, and the resulting spherical angles for various points along 
        this vector. From this, derive points that lie along the arc between vector1 and vector2 """
    v = [i-j for i, j in zip(vector1, vector2)]
    v_points_direct = [(vector2[0]+v[0]*l, vector2[1]+v[1]*l, vector2[2]+v[2]*l) for l in np.linspace(0, 1)]
    v_phis = [math.atan2(v_point[1], v_point[0]) for v_point in v_points_direct]
    v_thetas = [math.acos(v_point[2]/np.linalg.norm(v_point)) for v_point in v_points_direct]

    v_points_arc = [(radius*np.sin(theta)*np.cos(phi), radius*np.sin(theta)*np.sin(phi), radius*np.cos(theta))
                    for theta, phi in zip(v_thetas, v_phis)]
    v_points_arc.append((0, 0, 0))

    # if fig is None:
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    # else:
    #     ax = fig.gca()

    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    """ Plot polygon (face colour must be set afterwards, otherwise it over-rides the transparency)
        https://stackoverflow.com/questions/18897786/transparency-for-poly3dcollection-plot-in-matplotlib """
    points_collection = Poly3DCollection([v_points_arc], alpha=0.4)
    points_collection.set_facecolor(colour)
    ax.add_collection3d(points_collection)

    return fig

def getAngle(x1,y1,z1,x2,y2,z2,x3,y3,z3, x4=None,y4=None,z4=None,x5=None,y5=None,z5=None,x6=None,y6=None,z6=None, ex=None, ey=None, ez=None):  

    v1 = [x2-x1,y2-y1,z2-z1]
    v2 = [x3-x1,y3-y1,z3-z1]


    if (x4 != None):
        v3 = [x5-x4,y5-y4,z5-z4]
        v4 = [x6-x4,y6-y4,z6-z4]


    nv1 = np.cross(np.array(v1), np.array(v2))
    if (x4 != None):
        nv2 = np.cross(np.array(v3), np.array(v4))
    else:
        nv2 = [ex,ey,ez]
    
    d = ( nv1[0] * nv2[0] + nv1[1] * nv2[1] + nv1[2] * nv2[2] ) 
    e1 = math.sqrt( nv1[0] * nv1[0] + nv1[1] * nv1[1] + nv1[2] * nv1[2]) 
    e2 = math.sqrt( nv2[0] * nv2[0] + nv2[1] * nv2[1] + nv2[2] * nv2[2]) 
    d = d / (e1 * e2) 
    A = math.acos(d)
    return A

def getXYZAngles(x1,y1,z1,x2,y2,z2,x3,y3,z3):  

    v1 = np.array([x2-x1,y2-y1,z2-z1])
    v2 = np.array([x3-x1,y3-y1,z3-z1])
    nv = np.cross(v1, v2)
    nn = nv/ np.linalg.norm(nv)
    angles = (np.arcsin(nn))
    return angles

def vrange(stops):
    """Create concatenated ranges of integers for multiple [1]/stop

    Parameters:
        starts (1-D array_like): starts for each range
        stops (1-D array_like): stops for each range (same shape as starts)

    Returns:
        numpy.ndarray: concatenated ranges
    """
    starts = np.array([1] * len(stops))
    stops = np.asarray(stops) + 1
    L_p = stops - starts
    return np.array(np.split(np.repeat(stops - L_p.cumsum(), L_p) + np.arange(L_p.sum()), np.cumsum(stops - 1)[:-1]))

def get_points_v(x_v, y_v, z_v, d=None, n_p= None):
        "DESEMPACAR ARRAY DE WKT"
        x_v, y_v, z_v = np.asarray(x_v), np.asarray(y_v), np.asarray(z_v)

        "DISTANCIAS ENTRE X(n) - X(n-1), Y(n) - Y(n-1)"
        Lx, Ly = np.array(x_v[1:] - x_v[:-1]), np.array(y_v[1:] - y_v[:-1])

        "EXCLUIR LINEAS DE LONGITUD MENOR A LA DISTANCIA 'd'"
        if d and np.sum(np.asarray(((x_v[1:] - x_v[:-1]) ** 2 + (y_v[1:] - y_v[:-1]) ** 2) ** 0.5)) < d:
            print(np.sum(Lx), np.sum(Ly))
            pass
        else:
            "NUMERO DE PUNTOS ENTRE VERTICES"
            if n_p is None:
                nx, ny = np.array(np.around(np.abs(Lx / d), decimals=0)), np.array(np.around(np.abs(Ly / d), decimals=0))
                nx, ny = np.where(nx <= 0, 1, nx).astype(np.int), np.where(ny <= 0, 1, ny).astype(np.int)
                n_points = np.maximum(nx, ny)
            else:
                n_points = np.array([1] * len(Lx)) * np.array(n_p)

            "LONGUITUD DE SEGMENTOS ENTRE PUNTOS"
            x_space, y_space = Lx / (n_points + 1), Ly / (n_points + 1)

            "CREAR 2D ARRAY DE VALORES INICIALES"
            x_init, y_init = np.array(np.split(np.repeat(x_v[:-1], n_points), np.cumsum(n_points)[:-1])), np.array(np.split(np.repeat(y_v[:-1], n_points), np.cumsum(n_points)[:-1]))

            "CREAR RANGO DE NUMERO DE SEGMENTOS (n_points)"
            range_n = vrange(n_points)

            "CALCULO DE PUNTOS INTERMEDIOS ENTRE SEGMENTOS DE X_V y Y_v"
            if n_p is None:
                points_x, points_y = x_init + (range_n * x_space).T, y_init + (range_n * y_space).T
            else:
                points_x, points_y = x_init + (range_n * x_space[:, None]), y_init + (range_n * y_space[:,None])

            "GENERAR ARRAY DE VALORES z_v"
            points_z = np.split(np.repeat(np.array([z_v[0]] * len(points_x)), n_points), np.cumsum(n_points)[:-1])

            return points_x, points_y, points_z

def lin(z):
    x = (z - c_xz)/m_xz
    y = (z - c_yz)/m_yz
    return x,y

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
    return x,z

def rotateZ(x,y, cx,cy, radians):   
    for i in range(len(x)):
        _x      = x[i] - cx
        _y      = y[i] - cy
        d      = math.hypot(_x, _y)
        theta  = math.atan2(_x, _y) + radians
        #y_r[i] = cy + d * math.cos(theta)
        #x_r[i] = cx + d * math.sin(theta)
        x[i] = cy + d * math.sin(theta)
        y[i] = cx + d * math.cos(theta)
    return x,y

def createEllipse(x,y,z,rx,ry,rz, index, x_prev=None, y_prev=None, z_prev=None, type=None, color=None):
    if (index == len(x)-1):
        return
    index = int(index)
    size = len(x)
    ellipses_amount = size//int(size*0.1)
    u = v = None
    e_x = []
    e_y = [] 
    e_z = []
    e_x_ret = None
    e_y_ret = None
    e_z_ret = None
    if (type == 'skelet'):

        u=0     #x-position of the center
        v=0    #y-position of the center
        a=ry     #radius on the x-axis
        b=rz    #radius on the y-axis

        t = np.linspace(0, 2*np.pi, 100)
        e_x = [0 for i in np.linspace(0, 2*np.pi, 100)]
        e_y = u+a*np.cos(t)
        e_z = v+b*np.sin(t)

        # e_x_ret = e_x+x[index]
        # e_y_ret = e_y+y[index]
        # #e_y_ret = e_y
        # e_z_ret = e_z+z[index]

        e_x = np.array([[e_x[i], e_x[i+1]] for i in range(len(e_x)-1)])
        e_y = np.array([[e_y[i], e_y[i+1]] for i in range(len(e_y)-1)])
        e_z = np.array([[e_z[i], e_z[i+1]] for i in range(len(e_z)-1)])

        '''
        xyz = {'x': e_x, 'y': e_y, 'z': e_z}
        # put the data into a pandas DataFrame (this is what my data looks like)
        df = pd.DataFrame(xyz, index=range(len(xyz['x']))) 

        # re-create the 2D-arrays
        y1 = np.linspace(df['y'].min(), df['y'].max(), len(df['y'].unique()))
        z1 = np.linspace(df['z'].min(), df['z'].max(), len(df['z'].unique()))
        y2, z2 = np.meshgrid(y1, z1)
        x2 = griddata((df['x'], df['y']), df['z'], (y2, z2), method='cubic')
        '''
        
        u = np.linspace(0.0, 2 * np.pi, 10)
        v = np.linspace(0.0, np.pi, 10)
        ed_x = rx * np.outer(np.cos(u), np.sin(v))
        ed_y = ry * np.outer(np.sin(u), np.sin(v))
        ed_z = rz * np.outer(np.ones_like(u), np.cos(v))
        



        #e_x, e_y, e_z = rotate(e_x,e_y,e_z,0, 1, 0, np.pi/2)
        #ax.plot_surface(e_x_ret, e_y_ret, e_z_ret,alpha=0.3, color=color)

    elif (type == 'model'):
        # u = np.linspace(0.0, 2 * np.pi, 100)
        # v = np.linspace(0.0, np.pi, 100)
        # e_x = rx * np.outer(np.cos(u), np.sin(v))
        # e_y = ry * np.outer(np.sin(u), np.sin(v))
        # e_z = rz * np.outer(np.ones_like(u), np.cos(v))

        u=0     #x-position of the center
        v=0    #y-position of the center
        a=ry     #radius on the x-axis
        b=rz    #radius on the y-axis

        t = np.linspace(0, 2*np.pi, 100)
        e_x = [0 for i in np.linspace(0, 2*np.pi, 100)]
        e_y = u+a*np.cos(t)
        e_z = v+b*np.sin(t)

        e_x = np.array([[e_x[i], e_x[i+1]] for i in range(len(e_x)-1)])
        e_y = np.array([[e_y[i], e_y[i+1]] for i in range(len(e_y)-1)])
        e_z = np.array([[e_z[i], e_z[i+1]] for i in range(len(e_z)-1)])

    print(index)
    perp_x1, perp_y1, perp_x2, perp_y2 = getPerpendicular(x[index],y[index],x[index+1],y[index+1], 0.3)
    perp_x3, perp_z1, perp_x4, perp_z2 = getPerpendicular(x[index],z[index],x[index+1],z[index+1], 0.3)

    # gax = getAngle(perp_x3,y[index],perp_z1,x[index],y[index],z[index],perp_x1,perp_y1,z[index], ex=1, ey=0,ez=0)
    # print("getAngle x:", gax)
    # gay = getAngle(perp_x3,y[index],perp_z1,x[index],y[index],z[index],perp_x1,perp_y1,z[index], ex=0, ey=1,ez=0)
    # print("getAngle y:", gay)
    # gaz = getAngle(perp_x3,y[index],perp_z1,x[index],y[index],z[index],perp_x1,perp_y1,z[index], ex=0, ey=0,ez=1)
    # print("getAngle z:", gaz)

    '''
    gax = getAngle(perp_x3,y[index],perp_z1,x[index],y[index],z[index],perp_x1,perp_y1,z[index],    center_plane1_x[0][0],center_plane1_y[0][0],center_plane1_z[0][0],center_plane1_x[0][1],center_plane1_y[0][1],center_plane1_z[0][1],center_plane1_x[1][0],center_plane1_y[1][0],center_plane1_z[1][0])
    #print("getAngle x:", gax)
    gay = getAngle(perp_x3,y[index],perp_z1,x[index],y[index],z[index],perp_x1,perp_y1,z[index],    center_plane2_x[0][0],center_plane2_y[0][0],center_plane2_z[0][0],center_plane2_x[0][1],center_plane2_y[0][1],center_plane2_z[0][1],center_plane2_x[1][0],center_plane2_y[1][0],center_plane2_z[1][0])
    #print("getAngle y:", gay)
    gaz = getAngle(perp_x3,y[index],perp_z1,x[index],y[index],z[index],perp_x1,perp_y1,z[index],    center_plane3_x[0][0],center_plane3_y[0][0],center_plane3_z[0][0],center_plane3_x[0][1],center_plane3_y[0][1],center_plane3_z[0][1],center_plane3_x[1][0],center_plane3_y[1][0],center_plane3_z[1][0])
    #print("getAngle z:", gaz)
    '''
    
    if (type == 'skelet'):
        ax.scatter([perp_x3,perp_x1,perp_x2,perp_x4], [y[index],perp_y1,perp_y2,y[index]], [perp_z1,z[index],z[index],perp_z2],  c='cyan', marker='o')
    vg_angle_x = vg.angle(np.array([perp_x3,y[index], perp_z1]), np.array([x[index],y[index],z[index]]), look=vg.basis.x)/180*np.pi
    vg_angle_y = vg.angle(np.array([perp_x3,y[index], perp_z1]), np.array([x[index],y[index],z[index]]), look=vg.basis.y)/180*np.pi
    vg_angle_z = vg.angle(np.array([perp_x3,y[index], perp_z1]), np.array([x[index],y[index],z[index]]), look=vg.basis.z)/180*np.pi


    if (type == 'skelet'):
        pass
        #ax2.plot_surface(e_x_ret, e_y_ret, e_z_ret,alpha=0.3, color=color)

    xyz_angles = getXYZAngles(perp_x3,y[index],perp_z1,x[index],y[index],z[index],perp_x1,perp_y1,z[index])
    #print("xyz_angles:", np.degrees(xyz_angles))

    # vg_angle_x = vg.angle(np.array([perp_x1,perp_y1, z[index]]), np.array([x[index],y[index],z[index]]), look=vg.basis.x)/180*np.pi
    # vg_angle_y = vg.angle(np.array([perp_x1,perp_y1, z[index]]), np.array([x[index],y[index],z[index]]), look=vg.basis.y)/180*np.pi
    # vg_angle_z = vg.angle(np.array([perp_x1,perp_y1, z[index]]), np.array([x[index],y[index],z[index]]), look=vg.basis.z)/180*np.pi
    #plot_arc3d([perp_x1,perp_y1, z[index]], [x[index],y[index],z[index]], radius=0.2, fig=ax, colour='C0')

    #print("vg.angle x", vg_angle_x, vg_angle_x*180/np.pi)
    #print("vg.angle y", vg_angle_y, vg_angle_y*180/np.pi)
    #print("vg.angle z", vg_angle_z, vg_angle_z*180/np.pi)

    x_ = np.array([[perp_x3,perp_x1 ], [perp_x2, perp_x4]])
    y_ = np.array([[y[index], perp_y1], [perp_y2, y[index]]])
    z_ = np.array([[perp_z1, z[index]], [z[index], perp_z2]])

    #ax.plot3D([perp_x1,perp_x2,perp_x3,perp_x4], [perp_y1,perp_y2, y[index], y[index]], [z[index],z[index],perp_z1,perp_z2], 'blue')

    if (type == 'skelet'):
        ax.plot_surface(x_, y_, z_,alpha=0.5, color='cyan')

    angle1 = getAngleBetweenLines(perp_x3, y[index], perp_z1, x[index], y[index], z[index],x[index+1], y[index+1],z[index+1])

    if (type == 'skelet'):
        ax.plot3D([perp_x3, x[index],x[index+1]], [y[index],y[index],y[index+1]], [perp_z1,z[index],z[index+1]], 'mo-')
    #print("aaaangle1",angle1)

    angleBetweenPointsYZ = getAngleBetweenPointsYZ(y[index],z[index],y[index+1],z[index+1])
    theta = -xyz_angles[0] #gax #np.radians(17) #vg_angle_x #angle1+np.pi/2#angleBetweenPointsYZ #+1.5708
    #print("x:", theta, theta*180/np.pi)

    e_x, e_y, e_z = rotate(e_x,e_y,e_z,1, 0, 0, theta)
    if (type == 'skelet'):
        ed_x, ed_y, ed_z = rotate(ed_x,ed_y,ed_z,1, 0, 0, theta)

    if (type == 'skelet'):
        ax.plot3D([perp_x1,perp_x2,perp_x3,perp_x4], [perp_y1,perp_y2, y[index], y[index]], [z[index],z[index],perp_z1,perp_z2], 'blue')

    #ax.plot_surface(np.array([perp_x1,perp_x2,perp_x3,perp_x4]), np.array([perp_y1,perp_y2, y[index], y[index]]), np.array([z[index],z[index],perp_z1,perp_z2]), color='green')

    if (type == 'skelet'):
        ax.scatter([perp_x3,x[index]], [y[index],y[index]], [perp_z1,z[index]],  c='cyan', marker='o')
    #angle1 = getAnlgeBetweenVectors(perp_x3, y[index], perp_z1, x[index], y[index], z[index])
    
    angleBetweenPointsXZ = getAngleBetweenPointsXZ(x[index],z[index],x[index],z[index+1])
    theta = -xyz_angles[2] #gay #np.radians(19) #vg_angle_y #angleBetweenPointsXZ # + 90deg
    #print("y:", theta, theta*180/np.pi)
   
    e_x, e_y, e_z = rotate(e_x,e_y,e_z,0, 1, 0, theta)
    if (type == 'skelet'):
        ed_x, ed_y, ed_z = rotate(ed_x,ed_y,ed_z,0, 1, 0, theta)

    #print("angle1:", angle1, angle1*180/np.pi)
    angleBetweenPointsXY = getAngleBetweenPointsXY(x[index],0,0,0)
    theta = xyz_angles[1]#gaz #np.radians(55) #vg_angle_z#angle1#angleBetweenPointsXY
    #print("z:", theta, theta*180/np.pi)
   
    e_x, e_y, e_z = rotate(e_x,e_y,e_z,0, 0, 1, theta)
    if (type == 'skelet'):
        ed_x, ed_y, ed_z = rotate(ed_x,ed_y,ed_z,0, 0, 1, theta)

    #ax.plot_wireframe(e_x+x[index], e_y+y[index], e_z+z[index],  rstride=2, cstride=2, color='black', alpha=0.1)

    if (type == 'skelet'):
        ax.plot_surface(ed_x+x[index], ed_y+y[index], ed_z+z[index],alpha=0.3, color=color)    
    elif (type == 'model'):
        ax1.plot_wireframe(e_x+x[index], e_y+y[index], e_z+z[index],  rstride=20, cstride=20, color=color, alpha=0.8)
    #print("ret", ret)
    return e_x+x[index], e_y+y[index], e_z+z[index]


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')


# u=0     #x-position of the center
# v=0    #y-position of the center
# a=1     #radius on the x-axis
# b=1    #radius on the y-axis

# t = np.linspace(0, 2*np.pi, 100)
# x = [i for i in np.linspace(0, 2*np.pi, 100)]

# y = u+a*np.cos(t)
# z = v+b*np.sin(t)

# y2 = np.outer(y, 1)
# z2 = np.outer(z, 1)



# R = np.linspace(0, 5, 5)
# h = 0
# u = np.linspace(0,  2*np.pi, 20)

# y2 = np.outer(R, np.cos(u))
# z2 = np.outer(R, np.sin(u))

# xyz = {'x': x, 'y': y, 'z': z}

# # put the data into a pandas DataFrame (this is what my data looks like)
# df = pd.DataFrame(xyz, index=range(len(xyz['x']))) 

# # re-create the 2D-arrays
# y1 = np.linspace(df['y'].min(), df['y'].max(), len(df['y'].unique()))
# z1 = np.linspace(df['z'].min(), df['z'].max(), len(df['z'].unique()))
# y2, z2 = np.meshgrid(y1, z1)


# X = np.arange(-2, 2, 1)
# Y = np.arange(-2, 2, 1)
# print("X", X)
# print("Y", Y)
# R = np.sqrt(X**2 + Y**2)
# Z = np.sin(R)
# print("X", len(x2[0]))

# print("Y", len(y2[0]))
# print("Z", len(z2[0]))
# ax.plot_surface(0, y2, z2, color='tab:red')
# plt.show()
# input()


fig = plt.figure()
fig1 = plt.figure()
fig2 = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax1 = fig1.add_subplot(111, projection='3d')
ax2 = fig2.add_subplot(111, projection='3d')
#ax = fig.gca(projection='3d') #,autoscale_on=False

from copy import copy


'''
x = [i for i in np.linspace(0,2,50)]
y = [np.cos(i) for i in x]
z = [0 for i in x]
x,y = rotateZ(x,y, x[0],y[0], np.pi/4)
x1 = [i for i in np.linspace(1,3,50)]
y1 = [1 for i in x1]
z1 = [np.sin(i**(1/2)) for i in x1]
x2 = np.array([(i+j)/2 for i, j in zip(x,x1)])
y2 = np.array([(i+j)/2 for i, j in zip(y,y1)])
z2 = np.array([(i+j)/2 for i, j in zip(z,z1)])

x = copy(x2)
y = copy(y2)
z = copy(z2)
'''


x = np.array([i for i in np.linspace(1,3,50)])
y = np.array([i**2 for i in x])
z = np.array([np.sin(i) for i in x])



ax.plot3D(x, y, z, c='tab:orange', marker='_')
ax1.plot3D(x, y, z, c='tab:orange', marker='_')


print(x)


# Fit with polyfit
a, b = polyfit(x, y, 1)
c, d = polyfit(y, z, 1)
line_x = np.array([i for i in np.linspace(1,3,50)])
line_y = a + b * line_x
line_z = c+d*line_y



# data = np.array([[i,j,k] for i,j,k in zip(x,y,z)])
# print(data)

# datamean = data.mean(axis=0)
# uu, dd, vv = np.linalg.svd(data - datamean)
# print("np.mgrid[-3:3:2j][:, np.newaxis]", np.mgrid[-3:3:2j][:, np.newaxis])
# linepts = vv[0] * np.mgrid[-1:1:2j][:, np.newaxis]
# linepts += datamean
# print(linepts)

# line_x, line_y, line_z = get_points_v([linepts[0][0], linepts[1][0]], [linepts[0][1], linepts[1][1]], [linepts[0][2], linepts[1][2]], n_p= 50)
# ax.plot3D(line_x[0], line_y[0], line_z[0], c='tab:red', marker="_")


# # this will find the slope and x-intercept of a plane
# # parallel to the y-axis that best fits the data
# A_xz = np.vstack((x, np.ones(len(x)))).T
# m_xz, c_xz = np.linalg.lstsq(A_xz, z)[0]

# # again for a plane parallel to the x-axis
# A_yz = np.vstack((y, np.ones(len(y)))).T
# m_yz, c_yz = np.linalg.lstsq(A_yz, z)[0]

# # the intersection of those two planes and
# # the function for the line would be:
# # z = m_yz * y + c_yz
# # z = m_xz * x + c_xz
# # or:


# line_z = np.linspace(0,5)
# line_x,line_y = lin(line_z)

'''
min_x = np.min(x)
max_x = np.max(x)
new_line_x = []

for j in range(len(line_x)):
    if (line_x[j] > min_x and line_x[j] < max_x):
        new_line_x.append(line_x[j])

min_y = np.min(y)
max_y = np.max(y)
new_line_y = []

for j in range(len(line_y)):
    if (line_y[j] > min_y and line_y[j] < max_y):
        new_line_y.append(line_y[j])
'''


'''
min_z = np.min(z)
max_z = np.max(z)

min_x = np.min(x)
max_x = np.max(x)

min_y = np.min(y)
max_y = np.max(y)

new_line_z = []
new_line_x = []
new_line_y = []

for j in range(len(line_z)):
    if (line_z[j] > min_z and line_z[j] < max_z):
        new_line_z.append(line_z[j])
        new_line_x.append(line_x[j])
        new_line_y.append(line_y[j])

new_line_z1 = []
new_line_x1 = []
new_line_y1 = []

for j in range(len(new_line_y)):
    if (new_line_y[j] > min_y and new_line_y[j] < max_y):
        new_line_z1.append(new_line_z[j])
        new_line_x1.append(new_line_x[j])
        new_line_y1.append(new_line_y[j])

new_line_z2 = []
new_line_x2 = []
new_line_y2 = []

for j in range(len(new_line_x1)):
    if (new_line_x1[j] > min_x and new_line_x1[j] < max_x):
        new_line_z2.append(new_line_z1[j])
        new_line_x2.append(new_line_x1[j])
        new_line_y2.append(new_line_y1[j])




#ax.scatter3D(*data.T)
print("line_x", line_x)
print("line_x", line_y)
print("line_x", line_z)


line_x = line_x[0]
line_y = line_y[0]
line_z = line_z[0]
line_z = [np.sin(i) for i in x]

line_x = new_line_x2
line_y = new_line_y2
line_z = new_line_z2
'''



def getAngle_BetweenLineAxisXYZ(x,y,z):
    deltaY = line_y[0] - line_y[len(line_y)-1]
    deltaX = line_x[0] - line_x[len(line_x)-1]
    deltaZ = line_z[0] - line_z[len(line_z)-1]
    angleX = math.atan(deltaY / deltaZ)
    angleY = math.atan(deltaX / deltaZ)
    angleZ = math.atan(deltaX / deltaY)

    print("angleX", angleX, np.degrees(angleX))
    print("angleY", angleY, np.degrees(angleY))
    print("angleZ", angleZ, np.degrees(angleZ))
    print()
    return angleX, angleY, angleZ




angleX, angleY, angleZ = getAngle_BetweenLineAxisXYZ(line_x,line_y,line_z)

line_y_0 = line_y[0]
line_z_0 = line_z[0]
if (angleX < 0): 
    line_y, line_z = rotateX(line_y, line_z, line_y[0], line_z[0], (-np.pi/2-angleX))
    y, z = rotateX(y, z, line_y_0, line_z_0, (-np.pi/2-angleX))
else:
    line_y, line_z = rotateX(line_y, line_z, line_y[0], line_z[0], np.pi/2-angleX)   
    y, z = rotateX(y, z, line_y_0, line_z_0, (-np.pi/2-angleX))

angleX, angleY, angleZ = getAngle_BetweenLineAxisXYZ(line_x,line_y,line_z)

line_x_0 = line_x[0] 
line_z_0 = line_z[0]
if (angleY < 0): 
    line_x, line_z = rotateY(line_x, line_z, line_x_0, line_z_0, (-np.pi/2-angleY))
    x, z = rotateY(x, z, line_x_0, line_z_0, (-np.pi/2-angleY))
else:

    line_x, line_z = rotateY(line_x, line_z, line_x_0, line_z_0, np.pi/2-angleY)
    x, z = rotateY(x, z, line_x_0, line_z_0, np.pi/2-angleY)


angleX, angleY, angleZ = getAngle_BetweenLineAxisXYZ(line_x,line_y,line_z)




line_x_0 = line_x[0]
line_y_0 = line_y[0]
if (angleZ < 0):
    line_x, line_y = rotateZ(line_x, line_y, line_x_0, line_y_0, (-np.pi/2-angleZ))
    x, y = rotateZ(x, y, line_x_0, line_y_0, (-np.pi/2-angleZ))
else:
    line_x, line_y = rotateZ(line_x, line_y, line_x_0, line_y_0, np.pi/2-angleZ)
    x, y = rotateZ(x, y, line_x_0, line_y_0, np.pi/2-angleZ)


print(x)
#line_x, line_y, line_z = rotate(line_x, line_y, line_z,0, 0, 1, np.pi/2-angleZ)
#x, y, z = rotate(x, y, z,0, 0, 1, np.pi/6)

angleX, angleY, angleZ = getAngle_BetweenLineAxisXYZ(line_x,line_y,line_z)

# line_y_0 = line_y[0]
# line_z_0 = line_z[0]
# if (angleX < 0): 
#     line_y, line_z = rotateX(line_y, line_z, line_y[0], line_z[0], (-np.pi/2-angleX))
#     y, z = rotateX(y, z, line_y_0, line_z_0, (-np.pi/2-angleX))
# else:
#     line_y, line_z = rotateX(line_y, line_z, line_y[0], line_z[0], np.pi/2-angleX)   
#     y, z = rotateX(y, z, line_y_0, line_z_0, (-np.pi/2-angleX))

# angleX, angleY, angleZ = getAngle_BetweenLineAxisXYZ(line_x,line_y,line_z)

print()

#line_y, line_z = rotateX(line_y, line_z, line_y[0], line_z[0], np.pi/2)
# line_x, line_z = rotateY(line_x, line_z,line_x[0], line_z[0], -angleY)
# line_x, line_y = rotateZ(line_x, line_y,line_x[0], line_y[0], np.pi/2-angleZ)



# line_x, line_y, line_z = rotate(line_x, line_y, line_z,1, 0, 0, straight_line_angles[0])
# line_x, line_y, line_z = rotate(line_x, line_y, line_z,0, 1, 0, straight_line_angles[2])
# line_x, line_y, line_z = rotate(line_x, line_y, line_z,0, 0, 1, -straight_line_angles[1])






ax.plot3D(line_x, line_y, line_z, c='tab:blue', marker="_")
#ax.plot3D(new_line_x1, new_line_y1, new_line_z1, c='tab:green', marker="_")
ax1.plot3D(line_x, line_y, line_z, c='tab:blue', marker='_')



'''
center_plane1_x = np.array([[0,0],[0,0]])
center_plane1_y = np.array([[0,0.5],[0,0.5]])
center_plane1_z = np.array([[0,0],[0.5,0.5]])

ax.plot_surface(center_plane1_x, center_plane1_y, center_plane1_z,alpha=0.5)


center_plane2_x = np.array([[0,0.1],[0,0.1]])
center_plane2_y = np.array([[0,0],[0,0]])
center_plane2_z = np.array([[0,0],[0.1,0.1]])

ax.plot_surface(center_plane2_x, center_plane2_y, center_plane2_z,alpha=0.5)


center_plane3_x = np.array([[0,0],[0.1,0.1]])
center_plane3_y = np.array([[0,0.1],[0,0.1]])
center_plane3_z = np.array([[0.1,0.1],[0.1,0.1]])

ax.plot_surface(center_plane3_x, center_plane3_y, center_plane3_z,alpha=0.5)
'''


curve_e_x =  []
curve_e_y= []
curve_e_z = []
straight_e_x = [] 
straight_e_y = []
straight_e_z = []
straight_line_angles= []

ellipses_amount = 0
for i in np.linspace(1,len(x),5):
    c_e = createEllipse(x,y,z,0,random.uniform(0.4,0.6),random.uniform(0.4,0.6), int(i-1), type='skelet', color='tab:orange')
    if (c_e != None):
        curve_e_x.append(c_e[0]); curve_e_y.append(c_e[1]); curve_e_z.append(c_e[2])
        ellipses_amount += 1
    s_e = createEllipse(line_x,line_y,line_z,0,0.5,0.5, i-1, type='skelet', color='tab:blue')
    if (s_e != None):
        #straight_line_angles = s_e[3]
        straight_e_x.append(s_e[0]); straight_e_y.append(s_e[1]); straight_e_z.append(s_e[2])
    

for i in range(1,len(line_x)): 
    createEllipse(x,y,z,0,0.5,0.5, i-1, type='model', color='tab:orange')
    createEllipse(line_x,line_y,line_z,0,0.5,0.5, i-1, type='model', color='tab:blue')








# ax.plot3D(line_x, line_y, line_z, c='tab:red', marker=".")




# offset = 10
# z = [0 for i in range(-offset,offset)]
# x = [i for i in range(-offset,offset)]
# y = [0 for i in range(-offset,offset)]
# ax.plot3D(x, y, z, 'gray',alpha=0.2)
# ax1.plot3D(x, y, z, 'gray',alpha=0.2)

# z = [i for i in range(-offset,offset)]
# x = [0 for i in range(-offset,offset)]
# y = [0 for i in range(-offset,offset)]
# ax.plot3D(x, y, z, 'gray',alpha=0.2)
# ax1.plot3D(x, y, z, 'gray',alpha=0.2)

# z = [0 for i in range(-offset,offset)]
# x = [0 for i in range(-offset,offset)]
# y = [i for i in range(-offset,offset)]
# ax.plot3D(x, y, z, 'gray',alpha=0.2)
# ax1.plot3D(x, y, z, 'gray',alpha=0.2)


ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')


ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')

ax.set_xlim(-7, 13); ax.set_ylim(-10, 10); ax.set_zlim(-10, 10)
ax1.set_xlim(-7, 13); ax1.set_ylim(-10, 10); ax1.set_zlim(-10, 10)
#ax2.set_zlim(1, 2)








from shapely.geometry.point import Point
from shapely.geometry import Polygon as sPolygon
from shapely import affinity
from matplotlib.patches import Polygon as mPolygon


def create_ellipse(center, lengths, angle=0):
    """
    create a shapely ellipse. adapted from
    https://gis.stackexchange.com/a/243462
    """
    circ = Point(center).buffer(1)
    ell = affinity.scale(circ, int(lengths[0]), int(lengths[1]))
    ellr = affinity.rotate(ell, angle)
    return ellr
 
# ellipse1 = create_ellipse((0,0),(2,4),10)






fig, axs = plt.subplots(ellipses_amount//2, ellipses_amount - (ellipses_amount//2), sharex='col', sharey='row',
                        gridspec_kw={'hspace': 0, 'wspace': 0})


#(ax1, ax2), (ax3, ax4) = axs
fig.suptitle('Intersections')
# for i in range(len(curve_e_x)):
#     curve_e_x[i] = list(chain.from_iterable(curve_e_x[i]))
#     curve_e_y[i] = list(chain.from_iterable(curve_e_y[i]))
#     curve_e_z[i] = list(chain.from_iterable(curve_e_z[i]))

#     straight_e_x[i] = list(chain.from_iterable(straight_e_x[i]))
#     straight_e_y[i] = list(chain.from_iterable(straight_e_y[i]))
#     straight_e_z[i] = list(chain.from_iterable(straight_e_z[i]))

   
    #curve_e_x[i], curve_e_y[i], curve_e_z[i] = rotate(curve_e_x[i], curve_e_y[i], curve_e_z[i],0,1,0, np.pi/2)

# for i in range(len(curve_e_x)):
#     straight_e_x[i], straight_e_y[i], straight_e_z[i] = rotate(straight_e_x[i], straight_e_y[i], straight_e_z[i],0,1,0, np.pi/2)

# for i in range(len(curve_e_x)):     
#     curve_e_x[i], curve_e_y[i], curve_e_z[i] = rotate(curve_e_x[i], curve_e_y[i], curve_e_z[i],1,0,0, np.pi/2)


# for i in range(len(curve_e_x)):     
#     curve_e_x[i], curve_e_y[i], curve_e_z[i] = rotate(curve_e_x[i], curve_e_y[i], curve_e_z[i],0,0,1, np.pi/2)



# curve_e_x_tmp = []
# curve_e_z_tmp = []
# curve_e_x_tmp = copy(curve_e_x) #y
# curve_e_z_tmp = copy(curve_e_z) #x

# curve_e_x[0] = list(chain.from_iterable(curve_e_x[0]))
# curve_e_y[0] = list(chain.from_iterable(curve_e_y[0]))
# curve_e_z[0] = list(chain.from_iterable(curve_e_z[0]))


curve_e_x = copy(curve_e_y)
straight_e_x = copy(straight_e_y)


axs_counter = 0
for i in range(len(axs)):
    for j in range(len(axs[i])):
        axs[i][j].set_xlim(-2, 2)
        axs[i][j].set_ylim(-2, 2)
        coords_x_1 = []
        coords_y_1 = []

        coords_x_2 = []
        coords_y_2 = []

        for k in range(len(curve_e_x[axs_counter])):
            for l in range(len(curve_e_x[axs_counter][k])):
                coords_y_1.append(curve_e_x[axs_counter][k][l])
                coords_x_1.append(curve_e_z[axs_counter][k][l])  
        

        for k in range(len(straight_e_x[axs_counter])):
            for l in range(len(straight_e_x[axs_counter][k])):
                coords_y_2.append(straight_e_x[axs_counter][k][l])
                coords_x_2.append(straight_e_z[axs_counter][k][l])
        '''
        max_x_1 = np.max(coords_y_1)
        min_x_1 = np.min(coords_y_1)

        max_x_2 = np.max(coords_y_2)
        min_x_2 = np.min(coords_y_2)

        max_x = max(max_x_1, max_x_2)
        min_x = min(min_x_1, min_x_2)

        if (np.abs(max_x) > np.abs(min_x)):
            coords_y_1 = coords_y_1 - max_x
            coords_y_2 = coords_y_2 - max_x
        else:
            coords_y_1 = coords_y_1 - min_x
            coords_y_2 = coords_y_2 - min_x

        max_x_1 = np.max(coords_x_1)
        min_x_1 = np.min(coords_x_1)

        max_x_2 = np.max(coords_x_2)
        min_x_2 = np.min(coords_x_2)

        max_x = max(max_x_1, max_x_2)
        min_x = min(min_x_1, min_x_2)

        if (np.abs(max_x) > np.abs(min_x)):
            coords_x_1 = coords_x_1 - max_x
            coords_x_2 = coords_x_2 - max_x
        else:
            coords_x_1 = coords_x_1 - min_x
            coords_x_2 = coords_x_2 - min_x

        '''
        coords_1 = np.array([coords_x_1,coords_y_1])
        coords_2 = np.array([coords_x_2,coords_y_2])

        patch4 = mPolygon(coords_1.T, color = 'tab:orange', alpha = 0.5)
        axs[i][j].add_patch(patch4)

        patch5 = mPolygon(coords_2.T, color = 'tab:blue', alpha = 0.5)
        axs[i][j].add_patch(patch5)


        coords_xy_1 = []
        for k in range(len(curve_e_x[axs_counter])):
            for l in range(len(curve_e_x[axs_counter][k])):
                coords_xy_1.append((curve_e_z[axs_counter][k][l], curve_e_x[axs_counter][k][l]))

        coords_xy_2 = []
        for k in range(len(straight_e_x[axs_counter])):
            for l in range(len(straight_e_x[axs_counter][k])):
                coords_xy_2.append((straight_e_z[axs_counter][k][l], straight_e_x[axs_counter][k][l]))


        try:
            coords_xy_1 = sPolygon(coords_xy_1)
            coords_xy_2 = sPolygon(coords_xy_2)
            ##the intersect will be outlined in black
            intersect = coords_xy_1.intersection(coords_xy_2)


            #print("intersect", intersect, intersect == "POLYGON EMPTY", intersect.length)
            if (intersect.length == 0.0):
                print(axs_counter, "no intersection")
                axs_counter += 1
                continue
            else: 
                print(axs_counter)
            
            verts3 = np.array(intersect.exterior.coords.xy)
            patch3 = mPolygon(verts3.T, facecolor = 'none', edgecolor = 'black')
            axs[i][j].add_patch(patch3)

            ##compute areas and ratios 
            print('area of ellipse 1:',coords_xy_1.area)
            print('area of ellipse 2:',coords_xy_2.area)
            print('area of intersect:',intersect.area)
            print('intersect/ellipse1:', intersect.area/coords_xy_1.area)
            print('intersect/ellipse2:', intersect.area/coords_xy_2.area)
        except:
            print("self-intersection")

        axs_counter += 1







for ax_i in axs.flat:
    ax_i.label_outer()




plt.show()

