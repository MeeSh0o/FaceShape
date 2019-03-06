#-*- coding: utf-8 -*-
import sys
import apiutil
import json

app_key = '9sXAAQmcwi8rZcFA'
app_id = '2111336478'

ai_obj = apiutil.AiPlat(app_id, app_key)

file_path = 'F:/FaceShapeJson/'

def GetFaceShapeJson(path):
    #提取文件名
    name = path
    if "/" in name:
        name = path.split("/")[-1]
    if "\\" in name:
        name = name.split("\\")[-1]
    if "." in name:
        name = name.split(".")[0]

    #获取图像特征点
    with open(path,"rb") as image:
        file_name = file_path + name + '.json' # F:/FaceShapeJson/0.jpg
        rsp = ai_obj.getFaceShape(image.read(), 1)
        rsp_loads = json.loads(rsp)
        rsp_json = json.dumps(rsp_loads, encoding="UTF-8", ensure_ascii=False, sort_keys=False, indent=0)
    
    #返回图像数据
    if rsp_loads['ret'] == 0:
        #print (json.dumps(rsp_loads, encoding="UTF-8", ensure_ascii=False, sort_keys=False, indent=4))
        #print ('----------------------API SUCC----------------------')
        print ("Get FaceShapeData SUCC!\n")
        with open(file_name, 'w') as file_obj:
            json.dump(rsp_loads, file_obj)
            print("Write Down Json File!")
        
    else:
        print (json.dumps(rsp_loads, encoding="UTF-8", ensure_ascii=False, sort_keys=False, indent=4))
        # print rsp
        print ('----------------------API FAIL----------------------')

    return rsp_json

#GetFaceShapeJson('./Images/0.JPG')
