package com.xs.tencent.ocr;

import java.util.HashMap;

import com.xs.tencent.TencentAPI;
import com.xs.tencent.sign.Base64Util;
import com.xs.tencent.sign.TencentAISign;
import com.xs.tencent.sign.TencentAISignSort;
import com.xs.util.baidu.FileUtil;
import com.xs.util.baidu.HttpUtil;
/**
 * 手写体文字识别
 * @author 小帅丶
 *
 */
public class HandWritingSample {
	/**
	 * 图片文件识别示例
	 * @param args
	 * @throws Exception
	 */
	public static void main3(String[] args) throws Exception {
		//时间戳
		String time_stamp = System.currentTimeMillis()/1000+"";
		//图片的二进制数组数据
		byte [] imageData = FileUtil.readFileByBytes("G:/biz.jpg");
		//图片的base64编码数据
		String image = Base64Util.encode(imageData);
		//随机字符串
		String nonce_str = TencentAISign.getRandomString(10);
		HashMap<String, String> headers = new HashMap<String, String>();
		HashMap<String, String> bodys = new HashMap<String, String>();
		headers.put("Content-Type", "application/x-www-form-urlencoded");
		bodys.put("app_id",TencentAPI.APP_ID_AI.toString());
		bodys.put("time_stamp",time_stamp);
		bodys.put("nonce_str", nonce_str);
		bodys.put("image", image);
		//计算SIGN
		String sign = TencentAISignSort.getSignature(bodys);
		bodys.put("sign", sign);
		String params = TencentAISignSort.getParams(bodys);
		String result = HttpUtil.post(TencentAPI.OCR_HANDWRITING, params);
		System.out.println(result);
	}
	/**
	 * 网络图片识别示例
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
		//时间戳
		String time_stamp = System.currentTimeMillis()/1000+"";
		//随机字符串
		String nonce_str = TencentAISign.getRandomString(10);
		HashMap<String, String> headers = new HashMap<String, String>();
		HashMap<String, String> bodys = new HashMap<String, String>();
		headers.put("Content-Type", "application/x-www-form-urlencoded");
		bodys.put("app_id",TencentAPI.APP_ID_AI.toString());
		bodys.put("time_stamp",time_stamp);
		bodys.put("nonce_str", nonce_str);
		bodys.put("image_url", "https://ss0.bdstatic.com/70cFvHSh_Q1YnxGkpoWK1HF6hhy/it/u=2607793944,4253943040&fm=27&gp=0.jpg");
		//计算SIGN
		String sign = TencentAISignSort.getSignature(bodys);
		bodys.put("sign", sign);
		String params = TencentAISignSort.getParams(bodys);
		String result = HttpUtil.post(TencentAPI.OCR_HANDWRITING, params);
		System.out.println(result);
	}
}
