#-*- coding: utf-8 -*-
import hashlib
import urllib
import urllib2
import base64
import json
import time

url_preffix='https://api.ai.qq.com/fcgi-bin/'

def setParams(array, key, value):
    array[key] = value

def genSignString(parser):
    uri_str = ''
    for key in sorted(parser.keys()):
        if key == 'app_key':
            continue
        uri_str += "%s=%s&" % (key, urllib.quote(str(parser[key]), safe = ''))
    sign_str = uri_str + 'app_key=' + parser['app_key']

    hash_md5 = hashlib.md5(sign_str)
    return hash_md5.hexdigest().upper()


class AiPlat(object):
    def __init__(self, app_id, app_key):
        self.app_id = app_id
        self.app_key = app_key
        self.data = {}

    def invoke(self, params):
        self.url_data = urllib.urlencode(params)
        req = urllib2.Request(self.url, self.url_data)
        try:
            rsp = urllib2.urlopen(req)
            str_rsp = rsp.read()
            dict_rsp = json.loads(str_rsp)
            #return dict_rsp
            return str_rsp
        except urllib2.URLError, e:
            dict_error = {}
            if hasattr(e, "code"):
                dict_error = {}
                dict_error['ret'] = -1
                dict_error['httpcode'] = e.code
                dict_error['msg'] = "sdk http post err"
                return dict_error
            if hasattr(e,"reason"):
                dict_error['msg'] = 'sdk http post err'
                dict_error['httpcode'] = -1
                dict_error['ret'] = -1
                return dict_error
        else:
            dict_error = {}
            dict_error['ret'] = -1
            dict_error['httpcode'] = -1
            dict_error['msg'] = "system error"
            return dict_error

    #获得五官定位
    def getFaceShape(self, image, mode):
        self.url = url_preffix + 'face/face_faceshape'
        setParams(self.data, 'app_id', self.app_id)
        setParams(self.data, 'app_key', self.app_key)
        setParams(self.data, 'time_stamp', int(time.time()))
        setParams(self.data, 'nonce_str', int(time.time()))
        image_data = base64.b64encode(image)
        setParams(self.data, 'image', image_data)
        setParams(self.data, 'mode', mode)
        sign_str = genSignString(self.data)
        setParams(self.data, 'sign', sign_str)
        return self.invoke(self.data)