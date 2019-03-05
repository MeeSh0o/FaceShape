package com.xs.imageserach.sdk;

import java.util.HashMap;
import java.util.Map;

import org.json.JSONObject;

import com.baidu.aip.imagesearch.AipImageSearch;
import com.xiaoleilu.hutool.http.HttpUtil;

public class SDKSample {
	public static void main(String[] args) {
//		Map<String, Object> paramMap = new HashMap<String, Object>();
		AipImageSearch aipImageSearch = new AipImageSearch("APPID", "APIKEY", "SECRETKEY");
		HashMap<String, String> options = new HashMap<String, String>();
		options.put("brief", "{\"name\":\"测试开发者\", \"id\":\"777\"}");
		JSONObject object = aipImageSearch.sameHqAddUrl("https://www.xsshome.cn/wehcatpay.jpg", options );
		System.out.println(object);
//		paramMap.put("Content-Type", "image/png");
//		String result = HttpUtil.get("https://www.csnhw.com/pic/10363/QV13oRU050IFRVt/QV13oRU050IFRVt0.jpg");
//		System.out.println(result);
	}
}
