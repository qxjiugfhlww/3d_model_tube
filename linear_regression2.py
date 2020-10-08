import numpy as np

pts = np.add.accumulate(np.random.random((10,3)))
x,y,z = pts.T

# this will find the slope and x-intercept of a plane
# parallel to the y-axis that best fits the data
A_xz = np.vstack((x, np.ones(len(x)))).T
m_xz, c_xz = np.linalg.lstsq(A_xz, z)[0]

# again for a plane parallel to the x-axis
A_yz = np.vstack((y, np.ones(len(y)))).T
m_yz, c_yz = np.linalg.lstsq(A_yz, z)[0]

# the intersection of those two planes and
# the function for the line would be:
# z = m_yz * y + c_yz
# z = m_xz * x + c_xz
# or:
def lin(z):
    x = (z - c_xz)/m_xz
    y = (z - c_yz)/m_yz
    return x,y

#verifying:
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure()
ax = Axes3D(fig)
zz = np.linspace(0,5)
xx,yy = lin(zz)
ax.scatter(x, y, z)
ax.plot(xx,yy,zz)
plt.show()