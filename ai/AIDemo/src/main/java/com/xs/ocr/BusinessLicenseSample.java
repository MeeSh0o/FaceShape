package com.xs.ocr;

import java.net.URLEncoder;

import com.alibaba.fastjson.JSON;
import com.xs.pojo.ocr.businesslincense.BusinessLicenseBean;
import com.xs.util.baidu.Base64Util;
import com.xs.util.baidu.FileUtil;
import com.xs.util.baidu.HttpUtil;

public class BusinessLicenseSample {
	//接口地址
		public static String BUSINESS_LICENSE_URL = "https://aip.baidubce.com/rest/2.0/ocr/v1/business_license";
		public static void main(String[] args) {
//			String result=getText("G:/biz.jpg");
//			System.out.println(result);
			BusinessLicenseBean bean = getBean("G:/biz.jpg");
			System.out.println(bean.getWords_result().getSocialCreditCode().getWords());
		}
		
		/**
		 * 营业执照识别 通过图片获取文字内容
		 * 返回字符串
		 * @param filePath 图片文件目录
		 * @return text
		 */
		public static String getText(String filePath){
			String result = "";
			 try {
		            byte[] imgData = FileUtil.readFileByBytes(filePath);
		            String imgStr = Base64Util.encode(imgData);
		           //识别图片上的手写文字
		            String params = URLEncoder.encode("image", "UTF-8") + "=" + URLEncoder.encode(imgStr, "UTF-8")+"&detect_direction=true";
		            /**
		             * 线上环境access_token有过期时间， 客户端可自行缓存，过期后重新获取。
		             * 为了方便。直接调用了获取accesstoken的方法
		             */
//		            BDAccessToken bdAccessToken= util.getAccessToken(APIContants.IDCARD_API_KEY, APIContants.IDCARD_SERCET_KEY);
//		            String accessToken = bdAccessToken.getAccess_token().toString();
		            result = HttpUtil.post(BUSINESS_LICENSE_URL,"自己的AccessToken", params);
		        } catch (Exception e) {
		            e.printStackTrace();
		        }
			 return result;
		}
		/**
		 * 营业执照识别 通过图片获取文字内容
		 * 返回字符串
		 * @param filePath 图片文件目录
		 * @return text
		 */
		public static BusinessLicenseBean getBean(String filePath){
			 try {
		            byte[] imgData = FileUtil.readFileByBytes(filePath);
		            String imgStr = Base64Util.encode(imgData);
		           //识别图片上的手写文字
		            String params = URLEncoder.encode("image", "UTF-8") + "=" + URLEncoder.encode(imgStr, "UTF-8")+"&detect_direction=true";
		            /**
		             * 线上环境access_token有过期时间， 客户端可自行缓存，过期后重新获取。
		             * 为了方便。直接调用了获取accesstoken的方法
		             */
		            String result = "";
//		            BDAccessToken bdAccessToken= util.getAccessToken(APIContants.IDCARD_API_KEY, APIContants.IDCARD_SERCET_KEY);
//		            String accessToken = bdAccessToken.getAccess_token().toString();
		            result = HttpUtil.post(BUSINESS_LICENSE_URL,"自己的AccessToken", params);
		            BusinessLicenseBean bean = JSON.parseObject(result, BusinessLicenseBean.class);
		            return bean;
		        } catch (Exception e) {
		            e.printStackTrace();
		        }
			return null;
		}
}
