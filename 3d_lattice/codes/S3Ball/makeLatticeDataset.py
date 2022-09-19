from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np
import time
import random
from ball_img_maker_initials import *

from typing import List

from ball_throwing_physical_model import ballNextState
import cv2
from codes.common_utils import random_in_range
from shared import *

WIN_W = 320
WIN_H = 320
DT = 0.2
TRAJ_LEN = DATASET_SIZE
BALL_INIT_POSITION = [0.0, 1.0, 0.0]
IMG_FOLDER_PATH = 'Ball3DImg'
IMG_SAVE_PATH = f'{IMG_FOLDER_PATH}/{IMG_H}_{IMG_W}_{DT}_{TRAJ_LEN}_3_init_points_colorful_continue_evalset'
IMG_NAME_WITH_POSITION = False
DRAW_GIRD = True

MODE_MAKE_IMG = 'make_img'
MODE_LOCATE = 'locate'
MODE_OBV_ONLY = 'obv_only'

RUNNING_MODE = MODE_MAKE_IMG


"""
 7 | 8 | 9
——— ——— ———
 4 | 5 | 6
——— ——— ———
 1 | 2 | 3
"""

BALL_INITIAL_STATE = NORMAL_DATA
BALL_INITIAL_COLOR = COLOR_INIT_ALWAYS_GREEN

state_range = List[tuple]


def init_random_states(s_range: state_range) -> List:
    return [random_in_range(r) for r in s_range]


def init_ball_state():
    sample_list = list(filter(lambda point: point[ENABLE], BALL_INITIAL_STATE))
    position = random.sample(sample_list, 1)[0]
    init_s = init_random_states(position[S0])
    init_v = init_random_states(position[V0])
    return init_s, init_v


def init_ball_color():
    sample_list = list(filter(lambda point: point[ENABLE], BALL_INITIAL_COLOR))
    color_range = random.sample(sample_list, 1)[0][COLOR3F]
    color = [random_in_range(r) for r in color_range]
    print('color', color)
    return color

def make_dirs_if_need():
    if RUNNING_MODE == MODE_MAKE_IMG:
        if not os.path.isdir(IMG_FOLDER_PATH):
            os.mkdir(IMG_FOLDER_PATH)
        if not os.path.isdir(IMG_SAVE_PATH):
            os.mkdir(IMG_SAVE_PATH)


class BallViewer:
    def __init__(self):
        init_s, init_v = init_ball_state()
        # 小球位置信息
        self.sX = init_s[0]
        self.sY = init_s[1]
        self.sZ = init_s[2]

        # 小球速度信息
        self.vX, self.vY, self.vZ = init_v

        # 小球半径
        self.ballRadius = 0.5

        # opengl视角信息
        self.IS_PERSPECTIVE = True  # 透视投影
        self.VIEW = np.array([-0.8, 0.8, -0.8, 0.8, 1.0, 15.0])  # 视景体的left/right/bottom/top/near/far六个面
        self.SCALE_K = np.array([1.0, 1.0, 1.0])  # 模型缩放比例
        self.EYE = np.array([0.0, 4.0, 2.0])  # 眼睛的位置（默认z轴的正方向）
        self.LOOK_AT = np.array([0.0, 0.0, -5.0])  # 瞄准方向的参考点（默认在坐标原点）
        self.EYE_UP = np.array([0.0, 1.0, 0.0])  # 定义对观察者而言的上方（默认y轴的正方向）
        self.DIST, self.PHI, self.THETA = self.getposture()  # 眼睛与观察目标之间的距离、仰角、方位角
        self.WIN_W, self.WIN_H = WIN_W, WIN_H  # 保存窗口宽度和高度的变量

        # 鼠标操作信息
        self.LEFT_IS_DOWNED = False  # 鼠标左键被按下
        self.MOUSE_X, self.MOUSE_Y = 0, 0  # 考察鼠标位移量时保存的起始位置

        self.openGLInit()
        self.currTime = time.time()
        self.lastScreenShot = time.time()  # 上次截图时间d
        self.timeInOneTest = 0

        self.sub_folder_dir = ''
        self.curr_idx = 0
        self.first_render = True
        self.last_position = []
        self.color = init_ball_color()

        self.running_modes = {
            MODE_MAKE_IMG: self.fastDrawBall,
            MODE_OBV_ONLY: self.drawBall,
            MODE_LOCATE: self.locate_with_ball
        }

        self.make_not_screenshot = True
        os.makedirs(DATASET_PATH, exist_ok=True)
        self.sub_folder_dir = DATASET_PATH
        self.preheat = 5    # openGL shit

    def getposture(self):
        dist = np.sqrt(np.power((self.EYE - self.LOOK_AT), 2).sum())
        if dist > 0:
            phi = np.arcsin((self.EYE[1] - self.LOOK_AT[1]) / dist)
            theta = np.arcsin((self.EYE[0] - self.LOOK_AT[0]) / (dist * np.cos(phi)))
        else:
            phi = 0.0
            theta = 0.0
        return dist, phi, theta

    def init(self):
        glClearColor(0.4, 0.4, 0.4, 1.0)  # 设置画布背景色。注意：这里必须是4个参数
        glEnable(GL_DEPTH_TEST)  # 开启深度测试，实现遮挡关系
        glDepthFunc(GL_LEQUAL)  # 设置深度测试函数（GL_LEQUAL只是选项之一）-
        glEnable(GL_LIGHT0)  # 启用0号光源
        glLightfv(GL_LIGHT0, GL_POSITION, GLfloat_4(0, 1, 4, 0))  # 设置光源的位置
        glLightfv(GL_LIGHT0, GL_SPOT_DIRECTION, GLfloat_3(0, 0, -1))  # 设置光源的照射方向
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)  # 设置材质颜色
        glEnable(GL_COLOR_MATERIAL)

    def initRender(self):
        # 清除屏幕及深度缓存
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # 设置投影（透视投影）
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        if self.WIN_W > self.WIN_H:
            if self.IS_PERSPECTIVE:
                glFrustum(self.VIEW[0] * self.WIN_W / self.WIN_H, self.VIEW[1] * self.WIN_W / self.WIN_H, self.VIEW[2],
                          self.VIEW[3], self.VIEW[4], self.VIEW[5])
            else:
                glOrtho(self.VIEW[0] * self.WIN_W / self.WIN_H, self.VIEW[1] * self.WIN_W / self.WIN_H, self.VIEW[2],
                        self.VIEW[3], self.VIEW[4], self.VIEW[5])
        else:
            if self.IS_PERSPECTIVE:
                glFrustum(self.VIEW[0], self.VIEW[1], self.VIEW[2] * self.WIN_H / self.WIN_W,
                          self.VIEW[3] * self.WIN_H / self.WIN_W, self.VIEW[4], self.VIEW[5])
            else:
                glOrtho(self.VIEW[0], self.VIEW[1], self.VIEW[2] * self.WIN_H / self.WIN_W,
                        self.VIEW[3] * self.WIN_H / self.WIN_W, self.VIEW[4], self.VIEW[5])

        # 设置模型视图
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # 几何变换
        glScale(self.SCALE_K[0], self.SCALE_K[1], self.SCALE_K[2])

        # 设置视点
        gluLookAt(
            self.EYE[0], self.EYE[1], self.EYE[2],
            self.LOOK_AT[0], self.LOOK_AT[1], self.LOOK_AT[2],
            self.EYE_UP[0], self.EYE_UP[1], self.EYE_UP[2]
        )

        # 设置视口
        glViewport(0, 0, self.WIN_W, self.WIN_H)

    def drawGird(self):
        glBegin(GL_LINES)
        glColor4f(0.0, 0.0, 0.0, 1)  # 设置当前颜色为黑色不透明
        for i in range(101):
            glVertex3f(-100.0 + 2 * i, -self.ballRadius, -100)
            glVertex3f(-100.0 + 2 * i, -self.ballRadius, 100)
            glVertex3f(-100.0, -self.ballRadius, -100 + 2 * i)
            glVertex3f(100.0, -self.ballRadius, -100 + 2 * i)
        glEnd()
        glLineWidth(3)

    def makeBall(self, x, y, z, color3f=(1, 1, 1)):
        glPushMatrix()
        glColor3f(color3f[0], color3f[1], color3f[2])
        glTranslatef(x, y, -z)  # Move to the place
        quad = gluNewQuadric()
        gluSphere(quad, self.ballRadius, 90, 90)
        gluDeleteQuadric(quad)
        glPopMatrix()

    def locate_with_ball(self):
        self.makeBall(-3, 1, 1)
        self.makeBall(0, 1, -1)
        self.makeBall(3, 1, 1)

        self.makeBall(-5, 1, 4)
        self.makeBall(0, 1, 4)
        self.makeBall(4, 1, 4)

        self.makeBall(-4, 1, 8)
        self.makeBall(2, 1, 8)
        self.makeBall(8, 2, 8)

    def drawBall(self):
        dt = time.time() - self.currTime
        self.currTime += dt
        self.timeInOneTest += dt
        [self.sX, self.sY, self.sZ], [self.vX, self.vY, self.vZ] = ballNextState([self.sX, self.sY, self.sZ],
                                                                                 [self.vX, self.vY, self.vZ], dt)
        if self.timeInOneTest > 4:
            self.color = init_ball_color()
            init_s, init_v = init_ball_state()
            self.timeInOneTest = 0
            self.resetBallPosition(init_s)
            self.vX, self.vY, self.vZ = init_v
        self.makeBall(self.sX, self.sY, self.sZ, self.color)

    def fastDrawBall(self):
        if self.preheat > 0:
            self.preheat -= 1
            return
        idx = self.curr_idx
        x_i = idx %  N_CURVE_VERTICES
        idx = idx // N_CURVE_VERTICES
        y_i = idx %  N_CURVE_VERTICES
        idx = idx // N_CURVE_VERTICES
        z_i = idx %  N_CURVE_VERTICES
        if self.make_not_screenshot:
            # make ball
            self.makeBall(
                X_LATTICE[x_i], 
                Y_LATTICE[y_i], 
                Z_LATTICE[z_i], 
                self.color, 
            )
        else:
            # screenshot
            self.screenShot(
                self.WIN_W, self.WIN_H, 
                f'{x_i}_{y_i}_{z_i}', 
            )
            self.curr_idx += 1
            if self.curr_idx % 16 == 0:
                print(format(
                    self.curr_idx / DATASET_SIZE, '.1%', 
                ))
            if self.curr_idx == TRAJ_LEN:
                print('DONE')
                raise Exception('Program finished normally.')
        self.make_not_screenshot = not self.make_not_screenshot

    def resetBallPosition(self, position=None):
        if position is None:
            position = [0, 1, 0]
        self.sX = position[0]
        self.sY = position[1]
        self.sZ = position[2]

    def draw(self):
        self.initRender()
        glEnable(GL_LIGHTING)  # 启动光照
        if DRAW_GIRD:
            self.drawGird()
        self.running_modes[RUNNING_MODE]()
        glDisable(GL_LIGHTING)  # 每次渲染后复位光照状态

        # 把数据刷新到显存上
        glFlush()
        glutSwapBuffers()  # 切换缓冲区，以显示绘制内容

    def reshape(self, width, height):
        self.WIN_W, self.WIN_H = width, height
        glutPostRedisplay()

    def mouseclick(self, button, state, x, y):
        self.MOUSE_X, self.MOUSE_Y = x, y
        if button == GLUT_LEFT_BUTTON:
            self.LEFT_IS_DOWNED = state == GLUT_DOWN
        elif button == 3:
            self.SCALE_K *= 1.05
            glutPostRedisplay()
        elif button == 4:
            self.SCALE_K *= 0.95
            glutPostRedisplay()

    def mousemotion(self, x, y):
        if self.LEFT_IS_DOWNED:
            dx = self.MOUSE_X - x
            dy = y - self.MOUSE_Y
            self.MOUSE_X, self.MOUSE_Y = x, y

            self.PHI += 2 * np.pi * dy / self.WIN_H
            self.PHI %= 2 * np.pi
            self.THETA += 2 * np.pi * dx / self.WIN_W
            self.THETA %= 2 * np.pi
            r = self.DIST * np.cos(self.PHI)

            self.EYE[1] = self.DIST * np.sin(self.PHI)
            self.EYE[0] = r * np.sin(self.THETA)
            self.EYE[2] = r * np.cos(self.THETA)

            if 0.5 * np.pi < self.PHI < 1.5 * np.pi:
                self.EYE_UP[1] = -1.0
            else:
                self.EYE_UP[1] = 1.0

            glutPostRedisplay()

    def keydown(self, key, x, y):
        if key in [b'x', b'X', b'y', b'Y', b'z', b'Z']:
            if key == b'x':  # 瞄准参考点 x 减小
                self.LOOK_AT[0] -= 0.01
            elif key == b'X':  # 瞄准参考 x 增大
                self.LOOK_AT[0] += 0.01
            elif key == b'y':  # 瞄准参考点 y 减小
                self.LOOK_AT[1] -= 0.01
            elif key == b'Y':  # 瞄准参考点 y 增大
                self.LOOK_AT[1] += 0.01
            elif key == b'z':  # 瞄准参考点 z 减小
                self.LOOK_AT[2] -= 0.01
            elif key == b'Z':  # 瞄准参考点 z 增大
                self.LOOK_AT[2] += 0.01

            self.DIST, self.PHI, self.THETA = self.getposture()
            glutPostRedisplay()
        elif key == b'\r':  # 回车键，视点前进
            self.EYE = self.LOOK_AT + (self.EYE - self.LOOK_AT) * 0.9
            self.DIST, self.PHI, self.THETA = self.getposture()
            glutPostRedisplay()
        elif key == b'\x08':  # 退格键，视点后退
            self.EYE = self.LOOK_AT + (self.EYE - self.LOOK_AT) * 1.1
            self.DIST, self.PHI, self.THETA = self.getposture()
            glutPostRedisplay()
        elif key == b' ':  # 空格键，切换投影模式
            self.IS_PERSPECTIVE = not self.IS_PERSPECTIVE
            glutPostRedisplay()

    def openGLInit(self):
        glutInit()
        displayMode = GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH
        glutInitDisplayMode(displayMode)

        glutInitWindowSize(self.WIN_W, self.WIN_H)
        glutInitWindowPosition(300, 50)
        glutCreateWindow('Ball Throwing Simulation')

        self.init()  # 初始化画布
        glutDisplayFunc(self.draw)  # 注册回调函数draw()
        glutIdleFunc(self.draw)
        glutReshapeFunc(self.reshape)  # 注册响应窗口改变的函数reshape()
        glutMouseFunc(self.mouseclick)  # 注册响应鼠标点击的函数mouseclick()
        glutMotionFunc(self.mousemotion)  # 注册响应鼠标拖拽的函数mousemotion()
        glutKeyboardFunc(self.keydown)  # 注册键盘输入的函数keydown()

    def mainLoop(self):
        glutMainLoop()  # 进入glut主循环

    def screenShot(self, w, h, imgName):
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
        cv2.imwrite(f'{self.sub_folder_dir}/{imgName}.png', resized)  # 写入图片
        cv2.waitKey(1)


if __name__ == "__main__":
    make_dirs_if_need()
    ballView = BallViewer()
    ballView.mainLoop()
