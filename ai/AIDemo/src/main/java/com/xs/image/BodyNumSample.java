package com.xs.image;

import java.io.ByteArrayInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URLEncoder;

import sun.misc.BASE64Decoder;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONObject;
import com.xiaoleilu.hutool.lang.Base64;
import com.xs.util.baidu.Base64Util;
import com.xs.util.baidu.FileUtil;
import com.xs.util.baidu.HttpUtil;


/**
 * 人流统计-JavaAPI示例代码
 * @author 小帅丶
 * @date 2018年3月30日
 */
public class BodyNumSample {
	//接口地址
	private static final String BODY_NUM = "https://aip.baidubce.com/rest/2.0/image-classify/v1/body_num";
	//更换为自己应用的AccessToken
	static String token= "更换为自己应用的AccessToken";
	//调用示例代码
	public static void main(String[] args) throws Exception {
		//图片本地路径
		String imagePath = "G:/mall.jpg";
		String result = CountBodyNum(imagePath, token);
		JSONObject jsonObject = JSON.parseObject(result);
		//拿到图片的base64字符串
		String image = jsonObject.getString("image");
		BASE64Decoder decoder = new BASE64Decoder();
		//base64decode
		byte [] imageByte = decoder.decodeBuffer(image);
		System.out.println("图中的人脸个数为:"+jsonObject.getString("person_num"));
		//处理imageByte存入指定文件
		OutputStream outputStream = new FileOutputStream("G:/mallrender.jpg");
		InputStream inputStream = new ByteArrayInputStream(imageByte);
        byte[] buff = new byte[1024];
        int len = 0;
        while((len=inputStream.read(buff))!=-1){
        	outputStream.write(buff, 0, len);
        }
        inputStream.close();
        outputStream.close();
	}
	/**
	 * 对于输入的一张图片（可正常解码，且长宽比适宜），识别和统计图像中的人体个数，以俯拍角度为主要识别视角，支持特定框选区域的人数统计，并可以输出渲染图片（人体头顶热力图）
	 * @param path 图片路径
	 * @param token AccessToken
	 * @return
	 */
	public static String CountBodyNum(String path,String token) {
		String result="";
		try {
            String filePath = path;
            byte[] imgData = FileUtil.readFileByBytes(filePath);
            String imgStr = Base64Util.encode(imgData);
            String imgParam = URLEncoder.encode(imgStr, "UTF-8");
            String param = "image=" + imgParam+"&show=true";
            result = HttpUtil.post(BODY_NUM, token, param);
        } catch (Exception e) {
        	result = e.getMessage();
        }
        return result;
    }
}
