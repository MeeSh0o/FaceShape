package com.xs.image;

import java.net.URLEncoder;

import com.xs.util.baidu.Base64Util;
import com.xs.util.baidu.FileUtil;
import com.xs.util.baidu.HttpUtil;
/**
 * 图像无损放大
 * @author 小帅丶
 *
 */
public class ImageQualityEnhanceSample {
	  //图像无损放大接口
	  private static String IMAGE_QUALITY_ENHANCE_URL = "https://aip.baidubce.com/rest/2.0/image-classify/v1/image_quality_enhance";
	  	
	  	public static void main(String[] args) throws Exception {
	  		String imagePath = "G:/facecut.jpg";//图片本地路径
	  		String accessToken = "";//应用获取的AccessToken
			String result = getImageQualityEnhanceResult(imagePath, accessToken);
			System.out.println(result);
		}
		/**
		 * 图像无损放大
		 * @param imagePath 图片路径
		 * @param accessToken token
		 * @return 字符串
		 * @throws Exception
		 */
		public static String getImageQualityEnhanceResult(String imagePath,String accessToken) throws Exception{
			byte[] imgData = FileUtil.readFileByBytes(imagePath);
			String imgStr = Base64Util.encode(imgData);
			String param = "image=" + URLEncoder.encode(imgStr, "UTF-8");
			// 注意这里仅为了简化编码每一次请求都去获取access_token，线上环境access_token有过期时间，
			// 客户端可自行缓存，过期后重新获取。
			String result = HttpUtil.post(IMAGE_QUALITY_ENHANCE_URL, accessToken, param);
			return result;
		}
}
