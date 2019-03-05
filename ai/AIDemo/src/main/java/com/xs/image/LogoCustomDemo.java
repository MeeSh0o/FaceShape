package com.xs.image;

import java.net.URLEncoder;

import com.alibaba.fastjson.JSON;
import com.xs.common.image.ImageAPI;
import com.xs.pojo.image.LogoAddBrief;
import com.xs.util.baidu.Base64Util;
import com.xs.util.baidu.FileUtil;
import com.xs.util.baidu.HttpUtil;
/**
 * 自定义LOGO入库
 * @author 小帅丶
 *
 */
public class LogoCustomDemo {
	public static void main(String[] args) throws Exception {
		//返回字符串
		//LogoAddBrief 根据自己的实际情况进行创建
		LogoAddBrief logoAddBrief = new LogoAddBrief();
		logoAddBrief.setCode("xiaoshuailogo");
		logoAddBrief.setName("小帅网络");
//		logoAddBrief.setFilePath("G:/weimei_logo.png");
		String result = getLogoAddResult("G:/logo.png", "应用获取的AccessToken",logoAddBrief);
		System.out.println(result);
	}
	/**
	 * LOGO子库入库Demo
	 * @param imagePath
	 * @param accessToken
	 * @param logoAddBrief
	 * @return 字符串
	 * @throws Exception
	 */
	public static String getLogoAddResult(String imagePath,String accessToken,LogoAddBrief logoAddBrief) throws Exception{
		byte[] imgData = FileUtil.readFileByBytes(imagePath);
        String imgStr = Base64Util.encode(imgData);
        String brief = JSON.toJSONString(logoAddBrief);
		String param = "image=" + URLEncoder.encode(imgStr,"UTF-8")+"&brief="+brief;
        // 注意这里仅为了简化编码每一次请求都去获取access_token，线上环境access_token有过期时间， 客户端可自行缓存，过期后重新获取。
		String result = HttpUtil.post(ImageAPI.LOGO_API_ADD, accessToken, param);
        System.out.println(result);
        return result;
	}
}
