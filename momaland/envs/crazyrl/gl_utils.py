"""Graphical representation of the UAV in 3D space. the reference is based on the Crazyflie position reference: https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/functional-areas/lighthouse/terminology_definitions/ ."""
import numpy as np
from OpenGL.GL import (
    GL_LINES,
    GL_QUADS,
    glBegin,
    glColor3f,
    glColor4f,
    glEnd,
    glTranslatef,
    glVertex3f,
    glVertex3fv,
)
from OpenGL.GLU import gluNewQuadric
from OpenGL.raw.GLU import gluSphere


def axes():
    """Draw axes on the opengl simulation view."""
    glBegin(GL_LINES)

    glColor3f(0, 0, 1.0)
    glVertex3fv((0, 0, -2))
    glVertex3fv((0, 0, -1))

    glColor3f(0, 1.0, 0)
    glVertex3fv((0, 0, -1.98))
    glVertex3fv((-1, 0, -1.98))

    glColor3f(1.0, 0, 0)
    glVertex3fv((0, 0, -1.98))
    glVertex3fv((0, 1, -1.98))

    glEnd()


def field(size):
    """Draw the field on the opengl simulation view.

    Args:
        size: int the size of the side field
    """
    glBegin(GL_QUADS)
    glVertex3f(-size, -size, -2)
    glVertex3f(size, -size, -2)
    glVertex3f(size, size, -2)
    glVertex3f(-size, size, -2)
    glEnd()

    glColor3f(1.0, 1.0, 1.0)
    glBegin(GL_LINES)
    for i in np.arange(-size, size, 1):
        glVertex3f(-size, i, -1.99)
        glVertex3f(size, i, -1.99)

        glVertex3f(i, size, -1.99)
        glVertex3f(i, -size, -1.99)

    glEnd()


def point(point):
    """Draw the drone as a little red dot with a stick to visualize better the projection on the grid.

    Args:
        point: tuple x,y,z position
    """
    sphere = gluNewQuadric()
    glTranslatef(-point[1], point[0], point[2] - 2)
    glColor4f(0.5, 0.2, 0.2, 1)
    gluSphere(sphere, 0.1, 32, 16)

    glBegin(GL_LINES)
    # glColor4f(0.5, 0.2, 0.2, 0.3)
    glVertex3f(0, 0, 0)
    glVertex3f(0, 0, -2 - point[2])
    glEnd()


def target_point(point):
    """Draw the target point as a bigger yellow dot with a stick to visualize better the projection on the grid.

    Args:
        point: tuple x,y,z position
    """
    sphere = gluNewQuadric()
    glTranslatef(-point[1], point[0], point[2] - 2)
    glColor4f(0.6, 0.6, 0, 0.7)
    gluSphere(sphere, 0.2, 32, 16)

    glBegin(GL_LINES)
    glColor4f(0.7, 0.7, 0, 0.3)
    glVertex3f(0, 0, 0)
    glVertex3f(0, 0, -2 - point[2])
    glEnd()
