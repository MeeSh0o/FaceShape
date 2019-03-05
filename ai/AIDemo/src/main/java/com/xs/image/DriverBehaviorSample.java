package com.xs.image;

import java.net.URLEncoder;

import com.xs.util.baidu.Base64Util;
import com.xs.util.baidu.FileUtil;
import com.xs.util.baidu.HttpUtil;
/**
 * 驾驶行为分析Sample
 * @author 小帅丶
 * 2018年6月8日
 */
public class DriverBehaviorSample {
  //驾驶行为分析 此接口为内测。需要单独申请权限哦
  private static String ICR_DRIVER_BEHAVIOR_URL = "https://aip.baidubce.com/rest/2.0/image-classify/v1/driver_behavior";
  	
  	public static void main(String[] args) throws Exception {
  		String imagePath = "";//图片本地路径
  		String accessToken = "";//应用获取的AccessToken
		String result = getDriverBehaviorResult(imagePath, accessToken);
		System.out.println(result);
	}
	/**
	 * 驾驶行为分析Demo
	 * @param imagePath 图片路径
	 * @param accessToken token
	 * @return 字符串
	 * @throws Exception
	 */
	public static String getDriverBehaviorResult(String imagePath,String accessToken) throws Exception{
		byte[] imgData = FileUtil.readFileByBytes(imagePath);
		String imgStr = Base64Util.encode(imgData);
		String param = "image=" + URLEncoder.encode(imgStr, "UTF-8");
		// 注意这里仅为了简化编码每一次请求都去获取access_token，线上环境access_token有过期时间，
		// 客户端可自行缓存，过期后重新获取。
		String result = HttpUtil.post(ICR_DRIVER_BEHAVIOR_URL, accessToken, param);
		return result;
	}
}
