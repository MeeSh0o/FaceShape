package com.xs.face;




import java.util.ArrayList;
import java.util.List;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONObject;
import com.baidu.aip.util.Base64Util;
import com.xs.pojo.face.FaceV3Bean;
import com.xs.pojo.face.FaceV3VerifyBean;
import com.xs.util.baidu.FileUtil;
import com.xs.util.baidu.HttpUtil;
/**
 * 人脸在线活体检测V3示例代码
 * @author 小帅丶
 *
 */
public class FaceVerifyV3Sample {
	//接口地址
	public static String FACE_VERIFY ="https://aip.baidubce.com/rest/2.0/face/v3/faceverify";
	public static void main(String[] args) {
		String filePath1 = "G:/32801.jpg";
        String token = "自己的token";
        String result = FaceVerify(filePath1,token);
        System.out.println(result);
        FaceV3VerifyBean bean = JSONObject.toJavaObject(JSON.parseObject(result), FaceV3VerifyBean.class);
        System.out.println(bean.getResult().getFace_liveness());
	}
	/**
	 * 人脸在线活体检测示例代码
	 * @param path1 图片本地路径1
	 * @param token AccessToken
	 * @return
	 */
	public static String FaceVerify(String path1,String token) {
        try {
        	// 本地文件路径
            String filePath1 = path1;
            byte[] imgData1 = FileUtil.readFileByBytes(filePath1);
            String imgStr1 = Base64Util.encode(imgData1);
            List<FaceV3Bean> faceMatchs = new ArrayList<FaceV3Bean>();
            FaceV3Bean faceMatch1 = new FaceV3Bean(imgStr1,"BASE64");
            faceMatchs.add(faceMatch1);
            String param = JSONObject.toJSONString(faceMatchs);
            System.out.println("======"+param);
            // 注意这里仅为了简化编码每一次请求都去获取access_token，线上环境access_token有过期时间， 客户端可自行缓存，过期后重新获取。
            String accessToken = token;
            String result = HttpUtil.post(FACE_VERIFY, accessToken, param);
            return result;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }
}
