package com.xs.ocr;

import java.net.URLEncoder;

import com.alibaba.fastjson.JSON;
import com.xs.pojo.ocr.invoice.QuoTaBean;
import com.xs.util.baidu.Base64Util;
import com.xs.util.baidu.FileUtil;
import com.xs.util.baidu.HttpUtil;
/**
 * 定额发票API示例代码
 * @author 小帅丶
 *
 */
public class QuotaInvoiceAPISample {
	//接口地址
	public static String QUOTAINVOICE_URL = "https://aip.baidubce.com/rest/2.0/ocr/v1/quota_invoice";
	//根据应用APIKEY SECRETKEY 获取的accesstoken 文档 https://ai.baidu.com/docs#/Auth/75d80ed1
	public static String ACCESSTOKEN = "";
	
	public static void main(String[] args) {
		//返回字符串
//		String result=getQuoTaText("G:/testimg/quota.jpg");
		//返回bean对象 方便取值
		QuoTaBean bean = getQuoTaBean("G:/testimg/quota.jpg");
		System.out.println("发票代码==>"+bean.getWords_result().getInvoice_code());
		System.out.println("发票号码==>"+bean.getWords_result().getInvoice_number());
		System.out.println("发票金额==>"+bean.getWords_result().getInvoice_rate());
	}
		
	/**
	 * 定额发票识别 通过图片获取手写文字内容
	 * 返回字符串
	 * @param filePath 图片文件目录
	 * @return text
	 */
	public  static String getQuoTaText(String filePath){
		String result = "";
		 try {
	            byte[] imgData = FileUtil.readFileByBytes(filePath);
	            String imgStr = Base64Util.encode(imgData);
	           //识别图片上的手写文字
	            String params = URLEncoder.encode("image", "UTF-8") + "=" + URLEncoder.encode(imgStr, "UTF-8");
	            result = HttpUtil.post(QUOTAINVOICE_URL,ACCESSTOKEN, params);
	        } catch (Exception e) {
	            e.printStackTrace();
	        }
		 return result;
	}
	/**
	 * 定额发票识别 通过图片获取手写文字内容
	 * 返回对象
	 * @param filePath 图片文件目录
	 * @return QuoTaBean
	 */
	public static QuoTaBean getQuoTaBean(String filePath){
		String result = getQuoTaText(filePath);
		System.out.println(result);
		QuoTaBean bean = JSON.parseObject(result,QuoTaBean.class);
		return bean;
	}
}
