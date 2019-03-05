package com.xs.imageserach.product;

import java.net.URLEncoder;
import java.util.Date;

import com.xs.util.baidu.Base64Util;
import com.xs.util.baidu.FileUtil;
import com.xs.util.baidu.HttpUtil;
/**
 * 商品检索 入库-检索-删除 Java-API方式示例代码
 * @author 小帅丶
 * @nickname 7叔
 *
 */
public class ProductTest {
	/**
	 * 商品检索-入库
	 */
	public static String IMAGESEARCH_PRODUCT_ADD = "https://aip.baidubce.com/rest/2.0/image-classify/v1/realtime_search/product/add";
	/**
	 * 商品检索-检索
	 */
	public static String IMAGESEARCH_PRODUCT_SEARCH = "https://aip.baidubce.com/rest/2.0/realtime_search/same_hq/search";
	/**
	 * 商品检索-删除
	 */
	public static String IMAGESEARCH_PRODUCT_DELETE = "https://aip.baidubce.com/rest/2.0/image-classify/v1/realtime_search/product/delete";
	
	public static void main(String[] args) throws Exception {
		String Filepath = "G:/hw.jpg";
		String image = Base64Util.encode(FileUtil.readFileByBytes(Filepath));
		String id = new Date().getTime()/1000+"";
//		String params =  URLEncoder.encode("image", "UTF-8") + "=" + URLEncoder.encode(image, "UTF-8")+"&class_id1=59&class_id1=1"+"&brief={\"name\":\"01\", \"id\":\""+id+"\"}";
//		String params =  URLEncoder.encode("image", "UTF-8") + "=" + URLEncoder.encode(image, "UTF-8")+"&class_id1=59&class_id1=1";
		String params = "cont_sign=2994812473,4149866635";
		System.out.println("请求的参数为:"+params);
		String result = HttpUtil.post(IMAGESEARCH_PRODUCT_SEARCH, "自己应用获取的AccessToken", params);
		System.out.println("返回的数据:"+result);
	}
}
