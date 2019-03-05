package com.xs.face;

import java.util.ArrayList;
import java.util.List;


import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONObject;
import com.xs.pojo.face.FaceV3Bean;
import com.xs.pojo.face.FaceV3MatchBean;
import com.xs.util.baidu.Base64Util;
import com.xs.util.baidu.FileUtil;
import com.xs.util.baidu.HttpUtil;

public class FaceMatchV3Sample {
	//接口地址
	public static String FACE_MATCH ="https://aip.baidubce.com/rest/2.0/face/v3/match";
	public static void main(String[] args) {
		String filePath1 = "G:/dcg.jpg";
        String filePath2 = "G:/dc.jpg";
        String token = "自己应用获取的AccessToken";
        String result = FaceMatch(filePath1,filePath2,token);
        FaceV3MatchBean faceV3MatchBean = JSONObject.toJavaObject(JSON.parseObject(result), FaceV3MatchBean.class);
        System.out.println("对比分数值:"+faceV3MatchBean.getResult().getScore());
	}
	/**
	 * 人脸对比示例代码
	 * @param path1 图片本地路径1
	 * @param path2 图片本地路径1
	 * @param token AccessToken
	 * @return
	 */
	public static String FaceMatch(String path1,String path2,String token) {
        try {
            // 本地文件路径
            String filePath1 = path1;
            String filePath2 = path2;
            byte[] imgData1 = FileUtil.readFileByBytes(filePath1);
            byte[] imgData2 = FileUtil.readFileByBytes(filePath2);
            String imgStr1 = Base64Util.encode(imgData1);
            String imgStr2 = Base64Util.encode(imgData2);
//            String imgParam1 = URLEncoder.encode(imgStr1, "UTF-8");
//            String imgParam2 = URLEncoder.encode(imgStr2, "UTF-8");
            String url = "https://www.xsshome.cn/test2.jpg";
            List<FaceV3Bean> faceMatchs = new ArrayList<FaceV3Bean>();
//            FaceMatchV3Bean faceMatch1 = new FaceMatchV3Bean("f7ec8ecd441886371b9749d1fc853f44","FACE_TOKEN");
//            FaceMatchV3Bean faceMatch2 = new FaceMatchV3Bean("f7ec8ecd441886371b9749d1fc853f44","FACE_TOKEN");
            FaceV3Bean faceMatch1 = new FaceV3Bean(imgStr1,"BASE64","LIVE","LOW","NONE");
            FaceV3Bean faceMatch2 = new FaceV3Bean(imgStr2,"BASE64","LIVE","LOW","NONE");
//            FaceMatchV3Bean faceMatch1 = new FaceMatchV3Bean(url,"URL");
//            FaceMatchV3Bean faceMatch2 = new FaceMatchV3Bean(url,"URL");
            faceMatchs.add(faceMatch1);
            faceMatchs.add(faceMatch2);
            String param = JSONObject.toJSONString(faceMatchs);
            System.out.println("======"+param);
            // 注意这里仅为了简化编码每一次请求都去获取access_token，线上环境access_token有过期时间， 客户端可自行缓存，过期后重新获取。
            String accessToken = token;
            String result = HttpUtil.post(FACE_MATCH, accessToken, param);
            System.out.println(result);
            return result;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }
}
