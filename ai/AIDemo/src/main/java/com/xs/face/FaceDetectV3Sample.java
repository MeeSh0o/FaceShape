package com.xs.face;

import java.net.URLEncoder;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONObject;
import com.xs.pojo.face.FaceV3DetectBean;
import com.xs.util.baidu.Base64Util;
import com.xs.util.baidu.FileUtil;
import com.xs.util.baidu.HttpUtil;
/**
 * 人脸检测V3版本JAVA-API调用示例代码
 * @author 小帅丶
 *
 */
public class FaceDetectV3Sample {
	/**
	 * 人脸检测V3版本URL
	 */
	public static String FACE_DETECT_URL="https://aip.baidubce.com/rest/2.0/face/v3/detect";
	
	public static void main(String[] args) throws Exception {
        String filePath ="G:/face.jpg";
		String result = faceDetect(filePath,"1","自己的AccessToken");
		System.out.println("=====接口返回内容:"+result);
		JSON json = JSON.parseObject(result);
		FaceV3DetectBean bean = JSONObject.toJavaObject(json, FaceV3DetectBean.class);
		System.out.println("人脸数量:"+bean.getResult().getFace_num());
		System.out.println("美丑打分:"+bean.getResult().getFace_list().get(0).getBeauty());
		System.out.println("年龄:"+bean.getResult().getFace_list().get(0).getAge());
	}
	/**
	 * 所有参数人脸检测
	 * @param filePath 图片路径数据，图片大小不超过2M。
	 * @param max_face_num 最多处理人脸数目，默认值1
	 * @param face_fields 包括age,beauty,expression,faceshape,gender,glasses,landmark,race,qualities信息，逗号分隔，默认只返回人脸框、概率和旋转角度。
	 * @return
	 * @throws Exception 
	 */
	public static String faceDetect(String filePath,String max_face_num,String accessToken) throws Exception {
		byte[] imgData = FileUtil.readFileByBytes(filePath);
        String imgStr = Base64Util.encode(imgData);
        //image_type为FACE_TOKEN image参数为faceToken
        String faceToken = "f7ec8ecd441886371b9749d1fc853f88";
        //image_type为URL image参数为url
        String url = "https://www.xxx.com/face.jpg";
      /**
       * image_type 新加参数 与image结合使用  URL BASE64 FACE_TOKEN
       */
        String params = "image=" + URLEncoder.encode(imgStr,"UTF-8")+"&image_type=BASE64&max_face_num="+max_face_num+"&face_field=age,beauty,expression,faceshape,gender,glasses,landmark,race,qualities";
//      String params = "image=" + url+"&image_type=URL&max_face_num="+max_face_num+"&face_field=age,beauty,expression,faceshape,gender,glasses,landmark,race,qualities";
//      String params = "image=" + faceToken+"&image_type=FACE_TOKEN&max_face_num="+max_face_num+"&face_field=age,beauty,expression,faceshape,gender,glasses,landmark,race,qualities";
		String result = HttpUtil.post(FACE_DETECT_URL, accessToken, params);
		return result;
	}
}
