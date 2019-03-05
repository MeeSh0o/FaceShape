package com.xs.image;

import java.net.URLEncoder;

import com.xs.util.baidu.Base64Util;
import com.xs.util.baidu.FileUtil;
import com.xs.util.baidu.HttpUtil;
/**
 * 地标(景点)识别示例代码
 * @author 小帅丶
 *
 */
public class LandMarkAPISample {
	//地标(景点)识别接口地址
	private static String LANDMARK_URL = "https://aip.baidubce.com/rest/2.0/image-classify/v1/landmark";
  	public static void main(String[] args) throws Exception {
  		String imagePath = "G:/testimg/dsb.jpg";//图片本地路径
  		String accessToken = "应用获取的AccessToken";//应用获取的AccessToken
		String result = getLandMarkResult(imagePath, accessToken);
		System.out.println(result);
	}
	/**
	 * 地标识别Demo
	 * @param imagePath
	 * @param accessToken
	 * @return 字符串
	 * @throws Exception
	 */
	public static String getLandMarkResult(String imagePath,String accessToken) throws Exception{
		byte[] imgData = FileUtil.readFileByBytes(imagePath);
        String imgStr = Base64Util.encode(imgData);
		String param = "image=" + URLEncoder.encode(imgStr,"UTF-8");
        // 注意这里仅为了简化编码每一次请求都去获取access_token，线上环境access_token有过期时间， 客户端可自行缓存，过期后重新获取。
		String result = HttpUtil.post(LANDMARK_URL, accessToken, param);
        System.out.println(result);
        return result;
	}
}
