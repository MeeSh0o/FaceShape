package com.xs.face;

import java.io.IOException;
import java.net.URLEncoder;

import com.xs.util.baidu.Base64Util;
import com.xs.util.baidu.FileUtil;
import com.xs.util.baidu.HttpUtil;
/**
 * 人脸核身在线接口调用demo
 * @author 小帅丶
 *
 */
public class PersonVerifyV3APISample {
	private static String PERSON_VERIFY_URL="https://aip.baidubce.com/rest/2.0/face/v3/person/verify";
	public static void main(String[] args) throws Exception {
		String accessToken = "自己应用获取的AccessToken";
		String filePath = "带有人脸的图片本地路径";
		String id_card_number = "身份证号码";
		String name = "身份证上的名称";
		String result = getPersonVerifyResult(filePath, accessToken, id_card_number, name);
		System.out.println(result);
	}
	/**
	 * 人脸核身在线接口调用demo
	 * @param imagePath 图片本地路径
	 * @param accessToken 应用AccessToken
	 * @return String
	 * @throws Exception
	 */
	private static String getPersonVerifyResult(String imagePath, String accessToken,String id_card_number,String name)
			throws Exception {
		byte[] imgData = FileUtil.readFileByBytes(imagePath);
		String imgStr = Base64Util.encode(imgData);
		String param = "image=" + URLEncoder.encode(imgStr, "UTF-8")+"&image_type=BASE64"+"&id_card_number="+id_card_number+"&name="+URLEncoder.encode(name, "UTF-8");
		// 注意这里仅为了简化编码每一次请求都去获取access_token，线上环境access_token有过期时间，
		// 客户端可自行缓存，过期后重新获取。
		String result = HttpUtil.post(PERSON_VERIFY_URL, accessToken, param);
		return result;
	}
}
