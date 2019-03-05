package com.xs.ocr;

import java.net.URLEncoder;

import com.alibaba.fastjson.JSON;
import com.xs.pojo.ocr.iocr.IOCRBean;
import com.xs.util.baidu.HttpUtil;
import com.xs.util.baidu.official.Base64Util;
import com.xs.util.baidu.official.FileUtil;

public class IOCRSample {
	public static void main(String[] args) throws Exception {
		//图片本地路径
		String filePath = "C:/Users/Administrator/Desktop/query.jpg";
		//根据apikey secretkey获取token
		String accessToken = "";
		//自定义模块templateSign 
		String templateSign = "";
		//返回字符串内容
		String result = getResidencePermitOCR(filePath, accessToken, templateSign);
		System.out.println(result);
		//返回指定的对象属性内容
		IOCRBean iocrBean = getResidencePermitOCRBean(filePath, accessToken, templateSign);
		for (int i = 0; i < iocrBean.getData().getRet().size(); i++) {
			System.out.println(iocrBean.getData().getRet().get(i).getWord_name()+":"+iocrBean.getData().getRet().get(i).getWord());
		}
	}
	/**
	 * 返回字符串结果
	 * @param filePath 图片文件路径
	 * @param accessToken 应用token
	 * @param templateSign 自定义模板id 
	 * @return String
	 * @throws Exception
	 */
	public static String getResidencePermitOCR(String filePath,String accessToken,String templateSign) throws Exception{
		//接口地址
		String iocrurl = "https://aip.baidubce.com/rest/2.0/solution/v1/iocr/recognise";
		//获取图片的base64数据 先获取图片的数组数据再进行base64 编码
		byte[] imgaebinary = FileUtil.readFileByBytes(filePath);
		String imagebase64 = Base64Util.encode(imgaebinary);
		//拼接接口所需参数 image参数需要urlencode哦
		String param = "image="+URLEncoder.encode(imagebase64,"UTF-8")+"&templateSign="+templateSign;
		//开始请求接口
		String result = HttpUtil.post(iocrurl, accessToken, param);
		//输出接口返回的内容
		return result;	
	}
	/**
	 * 返回对象
	 * @param filePath 图片文件路径
	 * @param accessToken 应用token
	 * @param templateSign 自定义模板id 
	 * @return IOCRBean
	 * @throws Exception
	 */
	public static IOCRBean getResidencePermitOCRBean(String filePath,String accessToken,String templateSign) throws Exception{
		//接口地址
		String iocrurl = "https://aip.baidubce.com/rest/2.0/solution/v1/iocr/recognise";
		//获取图片的base64数据 先获取图片的数组数据再进行base64 编码
		byte[] imgaebinary = FileUtil.readFileByBytes(filePath);
		String imagebase64 = Base64Util.encode(imgaebinary);
		//拼接接口所需参数 image参数需要urlencode哦
		String param = "image="+URLEncoder.encode(imagebase64,"UTF-8")+"&templateSign="+templateSign;
		//开始请求接口
		String result = HttpUtil.post(iocrurl, accessToken, param);
		//输出接口返回的内容
		IOCRBean bean = JSON.parseObject(result,IOCRBean.class);
		return bean;
	}
}
