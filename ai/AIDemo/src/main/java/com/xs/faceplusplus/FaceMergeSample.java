package com.xs.faceplusplus;

import java.net.URLEncoder;
import java.util.Base64;
import java.util.Base64.Encoder;
import java.util.HashMap;
import java.util.Map;

import com.xs.faceplusplus.common.FacePPConstants;
import com.xs.util.baidu.FileUtil;
import com.xs.util.baidu.HttpUtil;
/**
 * 人脸融合FPP版本
 * 具体参数可以填写什么 请自己看文档
 * @author 小帅丶
 *
 */
public class FaceMergeSample {
	public static void main(String[] args) throws Exception {
		Encoder base64Encoder = Base64.getEncoder();
		//要融合的图片 
		String image_base64 = base64Encoder.encodeToString(FileUtil.readFileByBytes("G:/testimg/meinv.jpg"));
		//模板图片 
		String template_url = "https://xxxx/facemerge/46.png";
		//模板图片的人脸框位置  top left width height
		String template_rectangle = "145,142,157,157";
		String merge_base64 = image_base64;
		Map<String, Object> paramMap = new HashMap<String, Object>();
		paramMap.put("api_key", FacePPConstants.API_KEY );
		paramMap.put("api_secret", FacePPConstants.API_SECRET );
		paramMap.put("template_url", template_url);
		paramMap.put("template_rectangle", template_rectangle);
		paramMap.put("merge_base64", merge_base64);
		String params = "api_key=" + FacePPConstants.API_KEY + "&api_secret="
				+ FacePPConstants.API_SECRET + "&template_url="
				+ URLEncoder.encode(template_url, "utf-8")
				+ "&template_rectangle=" + template_rectangle
				+ "&merge_base64=" + URLEncoder.encode(merge_base64, "utf-8");
		String result = HttpUtil.post(FacePPConstants.FACE_MERGE_URL, params);
		System.out.println(result);
	}
}
