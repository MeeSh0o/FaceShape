package com.xs.aliet.face;

import java.util.HashMap;
import java.util.Map;

import org.apache.http.HttpResponse;
import org.apache.http.util.EntityUtils;

import com.xs.aliet.util.AliYunFaceHttpUtils;

import cn.xsshome.taip.util.Base64Util;
import cn.xsshome.taip.util.FileUtil;

/**
 * 人脸属性识别接口示例代码
 * @author 小帅丶
 *
 */
public class AliFaceAttributeSampleByAPI {
	private static String FACE_ATTRIBUTE_HOST = "http://rlsxsb.market.alicloudapi.com";
	private static String FACE_ATTRIBUTE_PATH = "/face/attribute";
	public static void main(String[] args) throws Exception {
		String appCode = "自己的appCode";
		Map<String, String> headers = new HashMap<String, String>();
	    //最后在header中的格式(中间是英文空格)为Authorization:APPCODE 83359fd73fe94948385f570e3c139105
	    headers.put("Authorization", "APPCODE " + appCode);
	    //根据API的要求，定义相对应的Content-Type
	    headers.put("Content-Type", "application/json; charset=UTF-8");
	    Map<String, String> querys = new HashMap<String, String>();
	    String base64 = Base64Util.encode(FileUtil.readFileByBytes("G:/test2.jpg"));
	    //type 1 为图片的base64数据
	    String bodys = "{\"type\":1,\"content\":\""+base64+"\"}";
	    HttpResponse httpResponse = AliYunFaceHttpUtils.doPost(FACE_ATTRIBUTE_HOST, FACE_ATTRIBUTE_PATH, "POST", headers, querys, bodys);
	    System.out.println(EntityUtils.toString(httpResponse.getEntity()));
	}
}
