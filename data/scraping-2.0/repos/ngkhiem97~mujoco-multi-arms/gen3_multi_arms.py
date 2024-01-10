import mujoco_py
from mujoco_py import load_model_from_xml, MjSim, MjViewer, MjRenderContextOffscreen
from mujoco_py import cymj
from mujoco_py.utils import remove_empty_lines
from mujoco_py.builder import build_callback_fn
import numpy as np
from collections import deque
import socket
import threading
import json
import xml.etree.ElementTree as ET
import os
import cv2
from scipy import stats
from pyhull.convex_hull import ConvexHull
from scipy.spatial import ConvexHull as ConvexHull2D
from stl.mesh import Mesh
from VelocityController import *
import time
import argparse

GAP = 15

class SimulatorVelCtrl: #a communication wrapper for MuJoCo
                   
    def init(self, modelFile, nv, action="pap"):
        
        # create thread lock for performance improvement
        self.lock = threading.Lock()
        self.lock1 = threading.Lock()
        self.lockcv2 = threading.Lock()

        # create velocity controller
        self.velocityCtrl = VelocityController()

        # create simulation world in MuJoCo
        parser = ET.XMLParser(encoding="utf-8")
        tree = ET.parse(modelFile, parser=parser)
        root = tree.getroot()
        for child in root:
            if child.tag == "mujoco":
                worldbody = child.find('worldbody')
                folder = '/tmp/mujoco_objects/'
                if not os.path.exists(folder):
                    os.makedirs(folder) 
                obj1 = ET.SubElement(worldbody, "body")
                pos_table = np.array([0.475, 0.0, 0.8])
                size_table = np.array([0.6,0.3,0.015])            
                self.generate_obj_table(obj1,pos_table,size_table) 
                obj3 = ET.SubElement(worldbody, "body")
                self.generate_obj_box(obj3, pos_table, 0.1, "ConSurf1")
                obj4 = ET.SubElement(worldbody, "body")
                self.generate_obj_box(obj4, pos_table, -0.1, "ConSurf2")
                self.modelStr = ET.tostring(child, encoding='utf8', method='xml').decode("utf-8") 
                with open(os.path.join(folder + 'model' + '.xml'), 'w') as f:
                    f.write(self.modelStr)
            elif child.tag == "learning":
                for node in child:
                    if node.tag == "init": 
                        statesStr = node.get('qpos')                       
                        self.initQpos = [float(n) for n in statesStr.split(",")]
                        gpCtrlStr = node.get('gpCtrl')
                        self.initGripper = float(gpCtrlStr)
                    else:
                        continue
        self.model = mujoco_py.load_model_from_xml(self.modelStr)
        self.sim = MjSim(self.model)
        self.viewer = MjViewer(self.sim)
        self.offscreen = MjRenderContextOffscreen(self.sim, 0, quiet = True)
        # self.goal = self.sim.data.get_site_xpos('target0')
        self.viewer.render()

        # buffers
        self.nv = nv
        self.v_tgt = np.zeros(self.nv)
        self.queue = deque(maxlen=10)
        self.queue_img = deque(maxlen=10)
        self.action = action

        # CV2 interface viewing
        self.thViewer = threading.Thread(target=self.cv2_viewer, args=())
        self.thViewer.start()

        
    def generate_obj_box(self, obj, pos_table, prefix, name):
        pos_box = np.array([0.45, 0.0, pos_table[2]+0.015+0.125])
        obj.set('name', name)                
        obj.set('pos', '{} {} {}'.format(pos_box[0], pos_box[1]+prefix, pos_box[2]))# 0.035))
        geom = ET.SubElement(obj, "geom")
        geom.set('name', name) 
        geom.set('type','box')
        geom.set('size','0.035 0.015 0.025')
        geom.set('rgba','0.999 0.999 0.999 1')    
        geom.set('friction','0.45 0.005 0.0001')
        geom.set('solimp','0.95 0.95 0.01')
        geom.set('solref','0.01 1')
        geom.set('condim','3')
        geom.set('priority','2')
        geom.set('margin','0.0')
        
        joint_x = ET.SubElement(obj, "joint")
        joint_x.set('name','object0:joint_x'+name)
        joint_x.set('pos','0 0 0')
        joint_x.set('type','slide')        
        joint_x.set('axis','1 0 0')
        joint_x.set('damping','0.01')
        
        joint_y = ET.SubElement(obj, "joint")
        joint_y.set('name','object0:joint_y'+name)
        joint_y.set('pos','0 0 0')
        joint_y.set('type','slide')        
        joint_y.set('axis','0 1 0')
        joint_y.set('damping','0.01')
        
        joint_z = ET.SubElement(obj, "joint")
        joint_z.set('name','object0:joint_z'+name)
        joint_z.set('pos','0 0 0')
        joint_z.set('type','slide')        
        joint_z.set('axis','0 0 1')
        joint_z.set('damping','0.01')
        
        joint_r = ET.SubElement(obj, "joint")
        joint_r.set('name','object0:joint_r'+name)
        joint_r.set('pos','0 0 0')
        joint_r.set('type','hinge')        
        joint_r.set('axis','1 0 0')
        joint_r.set('damping','0.01')
        
        joint_p = ET.SubElement(obj, "joint")
        joint_p.set('name','object0:joint_p'+name)
        joint_p.set('pos','0 0 0')
        joint_p.set('type','hinge')        
        joint_p.set('axis','0 1 0')
        joint_p.set('damping','0.01')
        
        joint_yw = ET.SubElement(obj, "joint")
        joint_yw.set('name','object0:joint_yw'+name)
        joint_yw.set('pos','0 0 0')
        joint_yw.set('type','hinge')        
        joint_yw.set('axis','0 0 1')
        joint_yw.set('damping','0.01')
        
        site = ET.SubElement(obj, "site")
        site.set('name', 'object0'+name)
        site.set('size', '0.0001 0.0001')
        site.set('rgba', '0 1 0 1')
        site.set('type', 'cylinder')
        site.set('pos', '{} {} {}'.format(0,0,-0.025))
        
        return pos_box
    
    def generate_obj_table(self, obj, pos_table, size_table):
        #table top
        obj.set('name', 'Table')                
        
        obj.set('pos', '{} {} {}'.format(pos_table[0],pos_table[1],pos_table[2]))
        
        geom = ET.SubElement(obj, "geom")
        geom.set('size','{} {} {}'.format(size_table[0],size_table[1],size_table[2]))
        geom.set('type','box')
        geom.set('rgba','0.8 0.2 0.1 1')
        geom.set('friction','0.25 0.005 0.0001')
        geom.set('solimp','0.95 0.95 0.01')
        geom.set('solref','0.01 1')
        geom.set('condim','4')
        geom.set('margin','0.0')
        # site = ET.SubElement(obj, "site")
        # site.set('name', 'target0')
        # site.set('size', '0.02 0.0001')
        # site.set('rgba', '0 1 0 1')
        # site.set('type', 'cylinder')
        # site.set('pos', '{} {} {}'.format(-0.1,-0.25,size_table[2]))
        
        #table legs                
        geom1 = ET.SubElement(obj, "geom")
        geom1.set('size','{} {}'.format(0.0135,0.2))
        geom1.set('pos','{} {} {}'.format(-(size_table[0]-0.025), -(size_table[1]-0.025), -0.2))
        geom1.set('type','cylinder')
        geom1.set('rgba','0.8 0.2 0.1 1')
        
        geom2 = ET.SubElement(obj, "geom")
        geom2.set('size','{} {}'.format(0.0135,0.2))
        geom2.set('pos','{} {} {}'.format(-(size_table[0]-0.025), (size_table[1]-0.025), -0.2))
        geom2.set('type','cylinder')
        geom2.set('rgba','0.8 0.2 0.1 1')
        
        geom3 = ET.SubElement(obj, "geom")
        geom3.set('size','{} {}'.format(0.0135,0.2))
        geom3.set('pos','{} {} {}'.format((size_table[0]-0.025), -(size_table[1]-0.025), -0.2))
        geom3.set('type','cylinder')
        geom3.set('rgba','0.8 0.2 0.1 1')
        
        geom4 = ET.SubElement(obj, "geom")
        geom4.set('size','{} {}'.format(0.0135,0.2))
        geom4.set('pos','{} {} {}'.format((size_table[0]-0.025), (size_table[1]-0.025), -0.2))
        geom4.set('type','cylinder')
        geom4.set('rgba','0.8 0.2 0.1 1')
        
        '''
        generate spheres above the table
        '''                
        for i in range(100):
            tmp_pos = np.array([np.random.uniform(-size_table[0], size_table[0]), np.random.uniform(-size_table[1], size_table[1]), size_table[2]])
            geom = ET.SubElement(obj, "geom")
            geom.set('name','spheret{}'.format(i))
            geom.set('type','sphere')
            geom.set('size','{}'.format(np.random.uniform(0.00005,0.0002)))
            geom.set('pos','{} {} {}'.format(tmp_pos[0],tmp_pos[1],tmp_pos[2]))
            geom.set('rgba','1.0 1.0 1.0 1')
            geom.set('friction','0.25 0.005 0.0001')
            geom.set('solimp','0.95 0.95 0.01')
            geom.set('solref','0.01 1')
            geom.set('condim','4')
            geom.set('margin','0.0') 
        
    def isInHull(self, point, hull):
        tolerance=1e-12
        flag = all((np.dot(eq[:-1], point) + eq[-1] <= tolerance)
            for eq in hull.equations)
        return flag
        
    def update_v(self):
        self.lock.acquire()
        if len(self.queue) != 0:
            v_tgt_new = self.queue.popleft()            
            for i in range(self.nv):
                self.v_tgt[i] = v_tgt_new[i]
        self.lock.release()

    def cv2_viewer(self):
        title = "Camera Output"
        cv2.namedWindow(title)
        
        inc_pos_v = 0.15
        inc_ang_v = 15/180*np.pi
        
        qtgt = np.array([-0.07370902, 0.18526047, -3.05346724, -1.93002792, -0.01739147, -1.04480512, 1.59032335])
        
        self.lock1.acquire()
        for i in range(7):
            self.sim.data.qpos[i] = qtgt[i]
            self.sim.data.qpos[i+15] = qtgt[i]
        self.lock1.release()          
        
        gqtgt = float(self.sim.data.ctrl[self.nv])
        self.initQpos = [self.sim.data.qpos[i] for i in range(self.sim.model.nq)]
        
        # initial speeds
        ang_v = np.array([1,0,0,0])
        pos_v = np.array([0,0,0])
        self.twist_ee = np.array([0, 0, 0, 0, 0, 0])        

        if self.action == "pap":
            gqtgt = self.pick_and_place(gqtgt)
        elif self.action == "push":
            gqtgt = self.push(gqtgt)
        elif self.action == "slide":
            gqtgt, ang_v = self.slide(gqtgt)
        
        while True:
            twist_ee = self.twist_ee
            is_cmd_received = False
            keypressed = cv2.waitKey(1)
            
            if keypressed == 27:
                break
            elif keypressed == ord('\\'):#92: #\, up
                is_cmd_received = True
                twist_ee = twist_ee + np.array([0,0,inc_pos_v,0,0,0])
            elif keypressed == 13: #return, down
                is_cmd_received = True
                twist_ee = twist_ee + np.array([0,0,-inc_pos_v,0,0,0])
            elif keypressed == ord('i'): # forward
                is_cmd_received = True
                twist_ee = twist_ee + np.array([inc_pos_v,0,0,0,0,0])
            elif keypressed == ord('k'): # backward
                is_cmd_received = True
                twist_ee = twist_ee + np.array([-inc_pos_v,0,0,0,0,0])
            elif keypressed == ord('j'): # left
                is_cmd_received = True
                twist_ee = twist_ee + np.array([0,inc_pos_v,0,0,0,0])
            elif keypressed == ord('l'): # right
                is_cmd_received = True
                twist_ee = twist_ee + np.array([0,-inc_pos_v,0,0,0,0])
                
            elif keypressed == ord('q'): # roll
                is_cmd_received = True
                ang_v = quatmultiply(np.array([np.cos(inc_ang_v/2),np.sin(inc_ang_v/2),0,0]),ang_v)
                axang = quat2axang(ang_v)
                tmp = axang[3]*axang[0:3]
                twist_ee = np.array([twist_ee[0],twist_ee[1],twist_ee[2],tmp[0],tmp[1],tmp[2]])
                
            elif keypressed == ord('a'): # -roll
                is_cmd_received = True
                ang_v = quatmultiply(np.array([np.cos(-inc_ang_v/2),np.sin(-inc_ang_v/2),0,0]),ang_v)
                axang = quat2axang(ang_v)
                tmp = axang[3]*axang[0:3]
                twist_ee = np.array([twist_ee[0],twist_ee[1],twist_ee[2],tmp[0],tmp[1],tmp[2]])
                
            elif keypressed == ord('w'): # pitch
                is_cmd_received = True
                ang_v = quatmultiply(np.array([np.cos(inc_ang_v/2),0,np.sin(inc_ang_v/2),0]),ang_v)
                axang = quat2axang(ang_v)
                tmp = axang[3]*axang[0:3]
                twist_ee = np.array([twist_ee[0],twist_ee[1],twist_ee[2],tmp[0],tmp[1],tmp[2]])
                
            elif keypressed == ord('s'): # -pitch
                is_cmd_received = True
                ang_v = quatmultiply(np.array([np.cos(-inc_ang_v/2),0,np.sin(-inc_ang_v/2),0]),ang_v)
                axang = quat2axang(ang_v)
                tmp = axang[3]*axang[0:3]
                twist_ee = np.array([twist_ee[0],twist_ee[1],twist_ee[2],tmp[0],tmp[1],tmp[2]])
                
            elif keypressed == ord('e'): # yaw
                is_cmd_received = True
                ang_v = quatmultiply(np.array([np.cos(inc_ang_v/2),0,0,np.sin(inc_ang_v/2)]),ang_v)
                axang = quat2axang(ang_v)
                tmp = axang[3]*axang[0:3]
                twist_ee = np.array([twist_ee[0],twist_ee[1],twist_ee[2],tmp[0],tmp[1],tmp[2]])
                
            elif keypressed == ord('d'): # -yaw
                is_cmd_received = True
                ang_v = quatmultiply(np.array([np.cos(-inc_ang_v/2),0,0,np.sin(-inc_ang_v/2)]),ang_v)
                axang = quat2axang(ang_v)
                tmp = axang[3]*axang[0:3]                
                twist_ee = np.array([twist_ee[0],twist_ee[1],twist_ee[2],tmp[0],tmp[1],tmp[2]])
                
            elif keypressed == ord('c'): # gripper close
                is_cmd_received = True
                tmp = gqtgt + 0.1
                if tmp < 1.5:
                    gqtgt = tmp
                else:
                    gqtgt = 1.5
            elif keypressed == ord('v'): # gripper open
                is_cmd_received = True
                tmp = gqtgt - 0.1
                if tmp > 0:
                    gqtgt = tmp
                else:
                    gqtgt = 0
            elif keypressed == ord('p'): # pause robot
                is_cmd_received = True
                ang_v = np.array([1,0,0,0])
                pos_v = np.array([0,0,0])
                twist_ee = np.array([0, 0, 0, 0, 0, 0])
           
            if is_cmd_received:
                self.twist_ee = twist_ee
                #gripper
                self.lock1.acquire()
                self.sim.data.ctrl[self.nv] = gqtgt 
                self.sim.data.ctrl[self.nv+1] = gqtgt
                self.lock1.release()
            #joint
            self.lock.acquire()                    
            vtgt = self.velocityCtrl.get_joint_vel_worldframe(self.twist_ee, np.array(self.sim.data.qpos[0:7]), np.array(self.sim.data.qvel[0:7]))   
            self.queue.append(vtgt)                    
            self.lock.release()
            
            self.show_image(title)
        cv2.destroyAllWindows()

    def pick_and_place(self, gqtgt):
        time.sleep(3)

        # move down for 1 seconds
        self.twist_ee = np.array([0.88,0,-0.99,0,0,0])
        t0 = time.time()
        duration = 1
        while time.time() - t0 < duration:
            self.lock.acquire()
            vtgt = self.velocityCtrl.get_joint_vel_worldframe(self.twist_ee, np.array(self.sim.data.qpos[0:7]), np.array(self.sim.data.qvel[0:7]))   
            self.queue.append(vtgt*14)                    
            self.lock.release()
            time.sleep(0.1)

        # close gripper
        while gqtgt <= 1.2:
            gqtgt = gqtgt + 0.1
            self.lock1.acquire()
            self.sim.data.ctrl[self.nv] = gqtgt 
            self.sim.data.ctrl[self.nv+1] = gqtgt
            self.sim.data.ctrl[self.nv+9] = gqtgt 
            self.sim.data.ctrl[self.nv+1+9] = gqtgt
            self.lock1.release()
        
        self.twist_ee = np.array([0,0,0.01,0,0,0])
        t0 = time.time()
        duration = 1 # up duration
        while time.time() - t0 < duration:
            self.lock.acquire()
            self.twist_ee *= 2
            vtgt = self.velocityCtrl.get_joint_vel_worldframe(self.twist_ee, np.array(self.sim.data.qpos[0:7]), np.array(self.sim.data.qvel[0:7]))   
            self.queue.append(vtgt*14)
            self.lock.release()
            time.sleep(0.1)

        # move down for 1 seconds
        self.twist_ee = np.array([-0.99,0,-0.77,0,0,0])
        t0 = time.time()
        duration = 2
        while time.time() - t0 < duration:
            self.lock.acquire()
            vtgt = self.velocityCtrl.get_joint_vel_worldframe(self.twist_ee, np.array(self.sim.data.qpos[0:7]), np.array(self.sim.data.qvel[0:7]))   
            self.queue.append(vtgt*12)                    
            self.lock.release()
            time.sleep(0.1)

        # open gripper
        while gqtgt >= 0.0:
            gqtgt = gqtgt - 0.1
            self.lock1.acquire()
            self.sim.data.ctrl[self.nv] = gqtgt 
            self.sim.data.ctrl[self.nv+1] = gqtgt
            self.sim.data.ctrl[self.nv+9] = gqtgt 
            self.sim.data.ctrl[self.nv+1+9] = gqtgt
            self.lock1.release()

        self.twist_ee = np.array([0, 0, 0, 0, 0, 0])
        self.lock.acquire()                    
        vtgt = self.velocityCtrl.get_joint_vel_worldframe(self.twist_ee, np.array(self.sim.data.qpos[0:7]), np.array(self.sim.data.qvel[0:7]))   
        self.queue.append(vtgt)                    
        self.lock.release()
        return gqtgt
    


    def push(self, gqtgt):
        time.sleep(3)

        # move down for 1 seconds
        self.twist_ee = np.array([0.45,0,-0.99,0,0,0])
        t0 = time.time()
        duration = 1
        while time.time() - t0 < duration:
            self.lock.acquire()
            vtgt = self.velocityCtrl.get_joint_vel_worldframe(self.twist_ee, np.array(self.sim.data.qpos[0:7]), np.array(self.sim.data.qvel[0:7]))   
            self.queue.append(vtgt*14)                    
            self.lock.release()
            time.sleep(0.1)

        # close gripper
        while gqtgt < 1.5:
            gqtgt = gqtgt + 0.1
            self.lock1.acquire()
            self.sim.data.ctrl[self.nv] = gqtgt 
            self.sim.data.ctrl[self.nv+1] = gqtgt
            self.lock1.release()

        self.twist_ee = np.array([0, 0, 0, 0, 0, 0])
        self.lock.acquire()                    
        vtgt = self.velocityCtrl.get_joint_vel_worldframe(self.twist_ee, np.array(self.sim.data.qpos[0:7]), np.array(self.sim.data.qvel[0:7]))   
        self.queue.append(vtgt)                    
        self.lock.release()
        time.sleep(0.1)

        self.twist_ee = np.array([0.99, 0, 0, 0, 0, 0])
        t0 = time.time()
        duration = 4
        while time.time() - t0 < duration:
            self.lock.acquire()
            vtgt = self.velocityCtrl.get_joint_vel_worldframe(self.twist_ee, np.array(self.sim.data.qpos[0:7]), np.array(self.sim.data.qvel[0:7]))   
            self.queue.append(vtgt*4)
            self.lock.release()
            time.sleep(0.01)

        self.twist_ee = np.array([0, 0, 0, 0, 0, 0])
        self.lock.acquire()                    
        vtgt = self.velocityCtrl.get_joint_vel_worldframe(self.twist_ee, np.array(self.sim.data.qpos[0:7]), np.array(self.sim.data.qvel[0:7]))   
        self.queue.append(vtgt)                    
        self.lock.release()
        return gqtgt

    def slide(self, gqtgt):
        time.sleep(3)   

        # move down for 1 seconds
        self.twist_ee = np.array([0.45,0,-0.99,0,0,0])
        t0 = time.time()
        duration = 1
        while time.time() - t0 < duration:
            self.lock.acquire()
            vtgt = self.velocityCtrl.get_joint_vel_worldframe(self.twist_ee, np.array(self.sim.data.qpos[0:7]), np.array(self.sim.data.qvel[0:7]))   
            self.queue.append(vtgt*14)                    
            self.lock.release()
            time.sleep(0.1)

        # close gripper
        while gqtgt < 1.5:
            gqtgt = gqtgt + 0.1
            self.lock1.acquire()
            self.sim.data.ctrl[self.nv] = gqtgt 
            self.sim.data.ctrl[self.nv+1] = gqtgt
            self.lock1.release()

        self.twist_ee = np.array([0, 0, 0, 0, 0, 0])
        self.lock.acquire()                    
        vtgt = self.velocityCtrl.get_joint_vel_worldframe(self.twist_ee, np.array(self.sim.data.qpos[0:7]), np.array(self.sim.data.qvel[0:7]))   
        self.queue.append(vtgt)                    
        self.lock.release()
        time.sleep(0.1)

        ang_v = [np.cos(-(15/180*np.pi)),0,np.sin(-(15/180*np.pi)),0]
        axang = quat2axang(ang_v)
        tmp = axang[3]*axang[0:3]
        self.twist_ee = np.array([0.44, 0, 0, 0, 0, 0])
        self.twist_ee = np.array([self.twist_ee[0],self.twist_ee[1],self.twist_ee[2],tmp[0]*3,tmp[1]*3,tmp[2]*3])
        t0 = time.time()
        duration = 0.9
        while time.time() - t0 < duration:
            self.lock.acquire()
            vtgt = self.velocityCtrl.get_joint_vel_worldframe(self.twist_ee, np.array(self.sim.data.qpos[0:7]), np.array(self.sim.data.qvel[0:7]))   
            self.queue.append(vtgt*32)
            self.lock.release()
            time.sleep(0.01)

        self.twist_ee = np.array([0, 0, 0, 0, 0, 0])
        self.lock.acquire()                    
        vtgt = self.velocityCtrl.get_joint_vel_worldframe(self.twist_ee, np.array(self.sim.data.qpos[0:7]), np.array(self.sim.data.qvel[0:7]))   
        self.queue.append(vtgt)                    
        self.lock.release()
        return gqtgt,ang_v
        
    def show_image(self, title):
        znear = 0.01
        zfar = 50.0

        image_data = []  
        self.lockcv2.acquire()
        if len(self.queue_img) != 0:
            image_data = self.queue_img.popleft() 
        self.lockcv2.release()
        if len(image_data) != 0:
            div_near = 1/(znear*self.model.stat.extent)
            div_far = 1/(zfar*self.model.stat.extent)
            s = div_far-div_near
            image_data = 1/(s*image_data + div_near)
            
            #limit measurement range
            dplim_upper = 0.7
            dplim_lower = 0.16
            image_data[image_data<=dplim_lower]=0
            image_data[image_data>=dplim_upper]=1.0
            
            #add noise
            image_noise_1=stats.distributions.norm.rvs(0,0.00005,size=image_data.shape)
            image_noise_2=np.random.normal(0,0.00015,size=image_data.shape)
            image_data = image_data + image_noise_1 + image_noise_2
            norm_image = cv2.normalize(image_data, None, alpha = 0, beta = 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)            
            cv2.imshow(title, norm_image)

    def start(self):
        ct = 0 
        while True:
            self.lock1.acquire()
            for i in range(self.nv):
                self.sim.data.qfrc_applied[i] = self.sim.data.qfrc_bias[i]
                self.sim.data.qvel[i] = self.v_tgt[i]
                self.sim.data.qfrc_applied[i+GAP] = self.sim.data.qfrc_bias[i+GAP]
                self.sim.data.qvel[i+GAP] = self.v_tgt[i]
            if  (ct*self.sim.model.opt.timestep/0.01).is_integer():
                self.update_v()
            ct = ct + 1
            self.sim.step()
            self.lock1.release()
            self.viewer.render()
            if ct%17 == 1:
                self.offscreen.render(width=424, height=240, camera_id=0)
                a=self.offscreen.read_pixels(width=424, height=240, depth=True)
                rgb_img = a[0]
                rgb_img = rgb_img[:, ::-1, ::-1]
                depth_img = a[1]
                depth_img = depth_img[:, ::-1]
                self.lockcv2.acquire()
                self.queue_img.append(depth_img)
                self.lockcv2.release()
            
'''
convert a unit quaternion to angle/axis representation
'''                                                                                                            
def quat2axang(q): 
    s = np.linalg.norm(q[1:4])
    if s >= 10*np.finfo(float).eps:#10*np.finfo(q.dtype).eps:
        vector = q[1:4]/s
        theta = 2*np.arctan2(s,q[0])
    else:
        vector = np.array([0,0,1])
        theta = 0
    
    axang = np.hstack((vector,theta))
    return axang
    
    
'''
multiply two quaternions (numpy arrays)
'''
def quatmultiply(q1, q2):
    # scalar = s1*s2 - dot(v1,v2)
    scalar = q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3]

    # vector = s1*v2 + s2*v1 + cross(v1,v2)
    vector = np.array([q1[0]*q2[1], q1[0]*q2[2], q1[0]*q2[3]]) + \
             np.array([q2[0]*q1[1], q2[0]*q1[2], q2[0]*q1[3]]) + \
             np.array([ q1[2]*q2[3]-q1[3]*q2[2], \
                        q1[3]*q2[1]-q1[1]*q2[3], \
                        q1[1]*q2[2]-q1[2]*q2[1]])

    rslt = np.hstack((scalar, vector))
    return rslt

'''
from OpenAI gym
''' 
_FLOAT_EPS = np.finfo(np.float64).eps
_EPS4 = _FLOAT_EPS * 4.0

def mat2euler(mat):
    """ Convert Rotation Matrix to Euler Angles.  See rotation.py for notes """
    mat = np.asarray(mat, dtype=np.float64)
    assert mat.shape[-2:] == (3, 3), "Invalid shape matrix {}".format(mat)

    cy = np.sqrt(mat[..., 2, 2] * mat[..., 2, 2] + mat[..., 1, 2] * mat[..., 1, 2])
    condition = cy > _EPS4
    euler = np.empty(mat.shape[:-1], dtype=np.float64)
    euler[..., 2] = np.where(condition,
                             -np.arctan2(mat[..., 0, 1], mat[..., 0, 0]),
                             -np.arctan2(-mat[..., 1, 0], mat[..., 1, 1]))
    euler[..., 1] = np.where(condition,
                             -np.arctan2(-mat[..., 0, 2], cy),
                             -np.arctan2(-mat[..., 0, 2], cy))
    euler[..., 0] = np.where(condition,
                             -np.arctan2(mat[..., 1, 2], mat[..., 2, 2]),
                             0.0)
    return euler
    
if __name__ == "__main__":
    # get parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str, default='pap')
    args = parser.parse_args()
    action = args.action
    
    simulator = SimulatorVelCtrl()
    simulator.init("gen3_robotiq_2f_85_cellphone_table_grasp_multi.xml", 7, action)
    simulator.start()