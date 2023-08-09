import os
import cv2
import random
import math
import numpy as np
import shutil

show_info = False

#历遍文件夹
def findfile(path, ret,file_state):
    filelist = os.listdir(path)
    for filename in filelist:
        de_path = os.path.join(path, filename)
        if os.path.isfile(de_path):
            if de_path.endswith(file_state):
                ret.append(de_path)
        else:
            findfile(de_path, ret,file_state)

#可视化结果image_MOT
def plot_tracking(filename,tracking_data_root,results, save_image_tracking=False):
    text_scale = 2
    text_thickness = 2
    line_thickness = 3

    num_zero = ["00000","0000","000","00","0"]
    if not os.path.exists(filename):
        os.mkdir(filename)
    ret = []
    id_color = {}
    id_point = {}
    for i in range(len(os.listdir(tracking_data_root))):
        ret.append(tracking_data_root+"/"+num_zero[len(str(i+1))-1] + str(i+1)+".jpg")
    for i in range(len(results)):
        x = int(float(results[i][2]))
        y = int(float(results[i][3]))
        w = int(float(results[i][4]))
        h = int(float(results[i][5]))
        conf = float(results[i][6])
        conf  = round(conf,2)
        unc = float(results[i][7])
        unc  = round(unc,2)

        caption =' CF:' + str(conf) +  ' SU:' + str(unc)

        id = int(results[i][1])
        id_text = '{}'.format(int(id))
        if id not in id_color.keys():
            id_color[id] = [random.randint(0,255),random.randint(0,255),random.randint(0,255)]
        if i == 0:
            frame_id = int(results[i][0])
            img = cv2.imread(ret[frame_id-1])
            cv2.putText(img, 'frame: %d' % (frame_id),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)
        if frame_id == int(results[i][0]):
            cv2.rectangle(img, (x, y), (x+w, y+h), (id_color[id][0], id_color[id][1], id_color[id][2]), 4)
            #show ID
           
         
            cv2.putText(img, str(id), (x, y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            if show_info:
                cv2.putText(img, ' CF: ' + str(conf), (x, y+h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (id_color[id][0], id_color[id][1], id_color[id][2]), 2)
                cv2.putText(img, ' SU: ' + str(unc), (x, y+h//2+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (id_color[id][0], id_color[id][1], id_color[id][2]), 2)
            if save_image_tracking:
                if id not in id_point.keys():
                    id_point[id] = [[int(x+w/2),int(y+h)]]
                else:
                    id_point[id] += [[int(x + w / 2), int(y + h)]]
                    for i_Track in range(1,len(id_point[id])):
                        ptStart = (id_point[id][i_Track-1][0],id_point[id][i_Track-1][1])
                        ptEnd = (id_point[id][i_Track][0], id_point[id][i_Track][1])
                        cv2.line(img, ptStart, ptEnd, (id_color[id][0], id_color[id][1], id_color[id][2]), 2, 4)
        else:
            cv2.imwrite(filename+"/"+str(frame_id)+".jpg",img)
            
            frame_id = int(results[i][0])
            img = cv2.imread(ret[frame_id-1])
            cv2.putText(img, 'frame: %d' % (frame_id),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)
            cv2.rectangle(img, (x, y), (x+w, y+h), (id_color[id][0], id_color[id][1], id_color[id][2]), 4)
            

           
            cv2.putText(img, str(id), (x, y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(img, ' CF: ' + str(conf), (x , y+h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (id_color[id][0], id_color[id][1], id_color[id][2]), 2)
            cv2.putText(img, ' SU: ' + str(unc), (x, y+h//2+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (id_color[id][0], id_color[id][1], id_color[id][2]), 2)
            if save_image_tracking:
                if id not in id_point.keys():
                    id_point[id] = [[int(x+w/2),int(y+h)]]
                else:
                    id_point[id] += [[int(x + w / 2), int(y + h)]]
                    for i_Track in range(1,len(id_point[id])):
                        ptStart = (id_point[id][i_Track-1][0],id_point[id][i_Track-1][1])
                        ptEnd = (id_point[id][i_Track - 1][0], id_point[id][i_Track - 1][1])
                        cv2.line(img, ptStart, ptEnd, (id_color[id][0], id_color[id][1], id_color[id][2]), 1, 4)
    if len(results) != 0:
        cv2.imwrite(filename+"/"+str(frame_id)+".jpg",img)
    print('save image to {}'.format(filename))







#将图片合成视频
def image_T_video(im_dir,video_dir,filename):
    if not os.path.exists(video_dir):
        os.mkdir(video_dir)
    video_dir = video_dir+"/"+filename+".mp4"
    fps = 25
    num = len(os.listdir(im_dir))
    img = cv2.imread(im_dir+"/10.jpg")
    img_size = (len(img[0]),len(img))
    #fourcc = cv2.cv.CV_FOURCC('M','J','P','G')#opencv2.4
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') #opencv3.0
    videoWriter = cv2.VideoWriter(video_dir, fourcc, fps, img_size)
    for i in range(num):
        im_name = im_dir+"/"+str(i+1)+'.jpg'
        frame = cv2.imread(im_name)
        videoWriter.write(frame)
    videoWriter.release()
    print('finish '+str(filename)+".mp4")

if __name__ == '__main__':
    input_root = "./vis_images"
    new_root = ""
    output_root = "./visual/image_trans"
    video_root = "./visual/video_trans"
    tracking_data = "MOT17"
    if not os.path.exists(output_root):
        os.mkdir(output_root)
    if not os.path.exists(video_root):
        os.mkdir(video_root)

    ret = ['./outs/mot17_trans/MOT17-01.txt', './outs/mot17_trans/MOT17-09.txt']
    #findfile("./outs/mot17_trans", ret,".txt")
    #print(ret)
    for path in ret:

        print(path)
        path_name = path.split("/")[3].split(".")[0]
        print(path_name)
        output_root_sub = output_root+"/"+path_name
        # input_root_sub = input_root + "/" + path_name + '-SDP' + "/img1/"
        input_root_sub = input_root + "/" + path_name  + "-SDP/img1/"
        result = []
        with open(path, "a", encoding="utf-8")as f:
            print('====================================')
            print("Loading tracking results form", path)
            print('====================================')
            f = open(path, "r", encoding="utf-8")
            for line in f:
                data = line.split(',')
                result.append(data)
        plot_tracking(output_root_sub,input_root_sub, result, False)
        image_T_video(output_root_sub, video_root,path_name)

