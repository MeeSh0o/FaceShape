package com.xs.imageserach.same_hq;

import java.net.URLEncoder;
import java.util.Date;

import com.xs.util.baidu.Base64Util;
import com.xs.util.baidu.FileUtil;
import com.xs.util.baidu.HttpUtil;

public class SameSample {
	/**
	 * 相似图检索-入库
	 */
	public static String IMAGESEARCH_ADD = "https://aip.baidubce.com/rest/2.0/image-classify/v1/realtime_search/similar/add";
	/**
	 * 相似图检索-检索
	 */
	public static String IMAGESEARCH_SEARCH = "https://aip.baidubce.com/rest/2.0/image-classify/v1/realtime_search/similar/search";
	/**
	 * 相似图检索-删除
	 */
	public static String IMAGESEARCH_DELETE = "https://aip.baidubce.com/rest/2.0/image-classify/v1/realtime_search/similar/delete";
	
	public static void main(String[] args) throws Exception {
		String Filepath = "G:/aaaa.png";
		String image = Base64Util.encode(FileUtil.readFileByBytes(Filepath));
		String id = new Date().getTime()/1000+"";
		String params =  URLEncoder.encode("image", "UTF-8") + "=" + URLEncoder.encode(image, "UTF-8")+"&brief={\"name\":\"宝马汽车X5\", \"id\":\""+id+"\"}";
//		String params =  URLEncoder.encode("image", "UTF-8") + "=" + URLEncoder.encode(image, "UTF-8");
		System.out.println("请求的参数为:"+params);
//		{"log_id": 7853134874782377575, "cont_sign": "4146738871,2244601894"}
		String result = HttpUtil.post(IMAGESEARCH_ADD, "自己应用获取的AccessToken", params);
		System.out.println("检索返回的数据:"+result);
	}
}
