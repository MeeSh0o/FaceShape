package com.xs.imageprocess;

import java.io.FileOutputStream;
import java.io.OutputStream;
import java.net.URLEncoder;

import sun.misc.BASE64Decoder;

import com.alibaba.fastjson.JSON;
import com.xs.pojo.imageprocess.ImageProcessGeneralBean;
import com.xs.util.baidu.Base64Util;
import com.xs.util.baidu.FileUtil;
import com.xs.util.baidu.HttpUtil;
/**
 * 图像对比度增强
 * @author 小帅丶
 */
public class ContrastEnhanceAPISample {
	//图像对比度增强接口URL
	public static String CONTRAST_ENHANCE_URL="https://aip.baidubce.com/rest/2.0/image-classify/v1/contrast_enhance";
	public static void main(String[] args) throws Exception {
		String accessToken = "自己应用获取的AccessToken";
		String imagefilePath = "G:/x5.jpg";
		ImageProcessGeneralBean bean = getContrastEnhanceBean(imagefilePath, accessToken);
		GenerateImage(bean.getImage(), "G:/conhn.jpg");
	}
	/**
	 * 图像对比度增强Demo
	 * @param imagePath 图片本地路径
	 * @param accessToken 应用token
	 * @return 字符串
	 * @throws Exception
	 */
	public static String getContrastEnhanceResult(String imagePath,String accessToken) throws Exception{
		byte[] imgData = FileUtil.readFileByBytes(imagePath);
        String imgStr = Base64Util.encode(imgData);
		String param = "image=" + URLEncoder.encode(imgStr,"UTF-8");
        // 注意这里仅为了简化编码每一次请求都去获取access_token，线上环境access_token有过期时间， 客户端可自行缓存，过期后重新获取。
		String result = HttpUtil.post(CONTRAST_ENHANCE_URL, accessToken, param);
        System.out.println(result);
        return result;
	}
	/**
	 * 图像对比度增强Demo
	 * @param imagePath 图片本地路径
	 * @param accessToken 应用token
	 * @return ImageProcessGeneralBean
	 * @throws Exception
	 */
	public static ImageProcessGeneralBean getContrastEnhanceBean(String imagePath,String accessToken) throws Exception{
		String result = getContrastEnhanceResult(imagePath,accessToken);
		ImageProcessGeneralBean imageProcessGeneralBean = JSON.parseObject(result,ImageProcessGeneralBean.class);
		return imageProcessGeneralBean;
	}
	 /**
     * base64字符串转化成图片  
     * @param imgStr base64
     * @param imgFilePath 保存的图片路径
     * @return
     */
	public static boolean GenerateImage(String imgStr, String imgFilePath) { // 对字节数组字符串进行Base64解码并生成图片
		if (imgStr == null) // 图像数据为空
			return false;
		BASE64Decoder decoder = new BASE64Decoder();
		try {
			// Base64解码
			byte[] b = decoder.decodeBuffer(imgStr);
			for (int i = 0; i < b.length; ++i) {
				if (b[i] < 0) {// 调整异常数据
					b[i] += 256;
				}
			}
			// 生成jpeg图片
			OutputStream out = new FileOutputStream(imgFilePath);// 新生成的图片
			out.write(b);
			out.flush();
			out.close();
			return true;
		} catch (Exception e) {
			System.out.println("出错了" + e.getMessage());
			return false;
		}
	}
}
