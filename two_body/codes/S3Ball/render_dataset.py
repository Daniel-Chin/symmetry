import time
import random
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import cv2
from physics import *

WIN_W = 320
WIN_H = 320
IMG_W = 32
IMG_H = 32
DT = 0.2
TRAJ_LEN = 20
IMG_FOLDER_PATH = 'Ball3DImg'
IMG_SAVE_PATH = f'{IMG_FOLDER_PATH}/{IMG_H}_{IMG_W}_{DT}_{TRAJ_LEN}_3_init_points_colorful_continue_evalset'
IMG_NAME_WITH_POSITION = True
DRAW_GIRD = True

MODE_MAKE_IMG = 'make_img'
MODE_LOCATE = 'locate'
MODE_OBV_ONLY = 'obv_only'

RUNNING_MODE = MODE_LOCATE

VIEW = np.array([-0.8, 0.8, -0.8, 0.8, 1.0, 15.0])  # 视景体的left/right/bottom/top/near/far六个面
SCALE_K = np.array([1.0, 1.0, 1.0])  # 模型缩放比例
EYE = np.array([0.0, 4.0, 2.0])  # 眼睛的位置（默认z轴的正方向）
LOOK_AT = np.array([0.0, 0.0, -5.0])  # 瞄准方向的参考点（默认在坐标原点）
EYE_UP = np.array([0.0, 1.0, 0.0])  # 定义对观察者而言的上方（默认y轴的正方向）
BALL_RADIUS = 1

class BallViewer:
    def __init__(self) -> None:
        self.bodies = None
    
    def drawBall(self):
        ...

def makeBall(x, y, z, color3f=(1, 1, 1)):
    glPushMatrix()
    glColor3f(* color3f)
    glTranslatef(x, y, -z)  # Move to the place
    quad = gluNewQuadric()
    gluSphere(quad, BALL_RADIUS, 90, 90)
    gluDeleteQuadric(quad)
    glPopMatrix()

def locate_with_ball():
    makeBall(-3, 1, 1)
    makeBall(0, 1, -1)
    makeBall(3, 1, 1)

    makeBall(-5, 1, 4)
    makeBall(0, 1, 4)
    makeBall(4, 1, 4)

    makeBall(-4, 1, 8)
    makeBall(2, 1, 8)
    makeBall(8, 2, 8)

def drawGird():
    glBegin(GL_LINES)
    glColor4f(0.0, 0.0, 0.0, 1)  # 设置当前颜色为黑色不透明
    for i in range(101):
        glVertex3f(-100.0 + 2 * i, - BALL_RADIUS, -100)
        glVertex3f(-100.0 + 2 * i, - BALL_RADIUS, 100)
        glVertex3f(-100.0, - BALL_RADIUS, -100 + 2 * i)
        glVertex3f(100.0, - BALL_RADIUS, -100 + 2 * i)
    glEnd()
    glLineWidth(3)

def make_dirs_if_need():
    if RUNNING_MODE == MODE_MAKE_IMG:
        if not os.path.isdir(IMG_FOLDER_PATH):
            os.mkdir(IMG_FOLDER_PATH)
        if not os.path.isdir(IMG_SAVE_PATH):
            os.mkdir(IMG_SAVE_PATH)

def init():
    glutInit()
    displayMode = GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH
    glutInitDisplayMode(displayMode)

    glutInitWindowSize(WIN_W, WIN_H)
    glutInitWindowPosition(300, 50)
    glutCreateWindow('Ball Throwing Simulation')

    # 初始化画布
    glClearColor(0.4, 0.4, 0.4, 1.0)  # 设置画布背景色。注意：这里必须是4个参数
    glEnable(GL_DEPTH_TEST)  # 开启深度测试，实现遮挡关系
    glDepthFunc(GL_LEQUAL)  # 设置深度测试函数（GL_LEQUAL只是选项之一）-
    glEnable(GL_LIGHT0)  # 启用0号光源
    glLightfv(GL_LIGHT0, GL_POSITION, GLfloat_4(0, 1, 4, 0))  # 设置光源的位置
    glLightfv(GL_LIGHT0, GL_SPOT_DIRECTION, GLfloat_3(0, 0, -1))  # 设置光源的照射方向
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)  # 设置材质颜色
    glEnable(GL_COLOR_MATERIAL)
    
    glutDisplayFunc(draw)  # 注册回调函数draw()
    glutIdleFunc(draw)
    glutReshapeFunc(reshape)  # 注册响应窗口改变的函数reshape()
    # glutMouseFunc(mouseclick)  # 注册响应鼠标点击的函数mouseclick()
    # glutMotionFunc(mousemotion)  # 注册响应鼠标拖拽的函数mousemotion()
    # glutKeyboardFunc(keydown)  # 注册键盘输入的函数keydown()

def draw():
    initRender()
    glEnable(GL_LIGHTING)  # 启动光照
    if DRAW_GIRD:
        drawGird()
    ballViewer.drawBall()
    glDisable(GL_LIGHTING)  # 每次渲染后复位光照状态

    # 把数据刷新到显存上
    glFlush()
    glutSwapBuffers()  # 切换缓冲区，以显示绘制内容

def initRender():
    # 清除屏幕及深度缓存
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # 设置投影（透视投影）
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()

    glFrustum(
        VIEW[0], VIEW[1], VIEW[2] * WIN_H / WIN_W,
        VIEW[3] * WIN_H / WIN_W, VIEW[4], VIEW[5], 
    )

    # 设置模型视图
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    # 几何变换
    glScale(* SCALE_K)

    # 设置视点
    gluLookAt(
        * EYE,
        * LOOK_AT, 
        * EYE_UP, 
    )

    # 设置视口
    glViewport(0, 0, WIN_W, WIN_H)

def screenShot(w, h, sub_folder_dir, imgName):
    glReadBuffer(GL_FRONT)
    # 从缓冲区中的读出的数据是字节数组
    data = glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE)
    arr = np.zeros((h * w * 3), dtype=np.uint8)
    for i in range(0, len(data), 3):
        # 由于opencv中使用的是BGR而opengl使用的是RGB所以arr[i] = data[i+2]，而不是arr[i] = data[i]
        arr[i] = data[i + 2]
        arr[i + 1] = data[i + 1]
        arr[i + 2] = data[i]
    arr = np.reshape(arr, (h, w, 3))
    # 因为opengl和OpenCV在Y轴上是颠倒的，所以要进行垂直翻转，可以查看cv2.flip函数
    cv2.flip(arr, 0, arr)
    resized = cv2.resize(arr, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    cv2.imshow('scene', resized)
    cv2.imwrite(f'{sub_folder_dir}/{imgName}.png', resized)  # 写入图片
    cv2.waitKey(1)

def reshape(width, height):
    global WIN_W, WIN_H
    WIN_W, WIN_H = width, height
    glutPostRedisplay()

if __name__ == "__main__":
    make_dirs_if_need()
    init()
    ballViewer = BallViewer()
    glutMainLoop()  # 进入glut主循环
