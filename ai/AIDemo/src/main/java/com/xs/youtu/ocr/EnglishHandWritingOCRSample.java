package com.xs.youtu.ocr;

import com.alibaba.fastjson.JSON;
import com.xs.util.baidu.Base64Util;
import com.xs.util.baidu.FileUtil;
import com.xs.youtu.common.YouTuAPIContants;
import com.xs.youtu.common.YouTuAppContants;
import com.xs.youtu.pojo.HandWritingOCRBean;
import com.xs.youtu.util.HttpUtil4YouTu;
import com.xs.youtu.util.YouTuSign;
/**
 * 手写英文字体识别
 * @author 小帅丶
 * 2018年9月17日
 */
public class EnglishHandWritingOCRSample {
	public static void main(String[] args) throws Exception {
		String imagePath = "G:/testimg/eocr2.jpg";
		String result = getHandWritingOCR(imagePath);
		System.out.println(result);
		//string结果转bean读取
		HandWritingOCRBean writingOCRBean = JSON.parseObject(result, HandWritingOCRBean.class);
		for (int i = 0; i < writingOCRBean.getItems().size(); i++) {
				System.out.println(writingOCRBean.getItems().get(i).getItemstring());
		}
		
	}
	/**
	 * 返回字符串
	 * @param imagePath 图片路径
	 * @return
	 * @throws Exception
	 */
	public static String getHandWritingOCR(String imagePath) throws Exception{
		//获取SIGN值
		String sign = YouTuSign.getSign(YouTuAppContants.userQQ,
				YouTuAppContants.AppID, YouTuAppContants.SecretID,
				YouTuAppContants.SecretKey);
		byte[] imgData = FileUtil.readFileByBytes(imagePath);
		String image = Base64Util.encode(imgData);
		String params = "{\"app_id\":\""+YouTuAppContants.AppID+"\",\"image\":\""+image+"\"}";
		String result = HttpUtil4YouTu.post(YouTuAPIContants.OCRAPI_EHOCR, params, sign);
		return result;
	}
}
