package com.xs.body;

import java.net.URLEncoder;

import com.alibaba.fastjson.JSON;
import com.xs.body.bean.GestureBean;
import com.xs.util.baidu.Base64Util;
import com.xs.util.baidu.FileUtil;
import com.xs.util.baidu.HttpUtil;

/**
 *  手势识别Java-API调用示例代码
 *  Base64Util、FileUtil、HttpUtil工具类下载地址
 *  http://aixiaoshuai.mydoc.io/?t=234826
 *  @author 小帅丶
 */
public class GestureAPISample {
	    public static String GESTURE_URL = "https://aip.baidubce.com/rest/2.0/image-classify/v1/gesture";
		public static void main(String[] args) throws Exception {
		//返回字符串
//		String result = getGestureResult("G:/gesture/摇滚.jpg", "自己的token");
//		System.out.println(result);
		//返回java对象
		GestureBean  gestureBean = getGestureBean("G:/htt.jpg", "自己的token");
		System.out.println("识别个数:"+gestureBean.getResult_num());
		String gestureName = "";
		for (int i = 0; i < gestureBean.getResult().size(); i++) {
			if(gestureBean.getResult().get(i).getClassname().equals("Face")){
				System.out.println("此图中识别出人脸");
			}else{
				gestureName += GestureClassNameWorkBook.getGestureName(gestureBean.getResult().get(i).getClassname())+",";
			}
		}
		System.out.println("识别的手势："+gestureName.substring(0, gestureName.length()-1));
	}
    /**
     * 识别图片中的手势类型，返回手势名称、手势矩形框、概率分数，可识别15种手势，支持动态手势识别，适用于手势特效、智能家居手势交互等场景；
     * 支持的15类手势列表：手指、掌心向前、拳头、OK、祈祷、作揖、作别、单手比心、点赞、diss、rock、掌心向上、双手比心（3种）。
     * @param imagePath 图片本地路径
     * @param accessToken 应用AccessToken
     * @return String
     * @throws Exception
     */
	private static String getGestureResult(String imagePath, String accessToken) throws Exception {
		byte[] imgData = FileUtil.readFileByBytes(imagePath);
        String imgStr = Base64Util.encode(imgData);
		String param = "image=" + URLEncoder.encode(imgStr,"UTF-8");
        // 注意这里仅为了简化编码每一次请求都去获取access_token，线上环境access_token有过期时间， 客户端可自行缓存，过期后重新获取。
		String result = HttpUtil.post(GESTURE_URL, accessToken, param);
        return result;
	}
    /**
     * 识别图片中的手势类型，返回手势名称、手势矩形框、概率分数，可识别15种手势，支持动态手势识别，适用于手势特效、智能家居手势交互等场景；
     * 支持的15类手势列表：手指、掌心向前、拳头、OK、祈祷、作揖、作别、单手比心、点赞、diss、rock、掌心向上、双手比心（3种）。
     * @param imagePath 图片本地路径
     * @param accessToken 应用AccessToken
     * @return GestureBean
     * @throws Exception
     */
	private static GestureBean getGestureBean(String imagePath, String accessToken) throws Exception {
		byte[] imgData = FileUtil.readFileByBytes(imagePath);
        String imgStr = Base64Util.encode(imgData);
		String param = "image=" + URLEncoder.encode(imgStr,"UTF-8");
        // 注意这里仅为了简化编码每一次请求都去获取access_token，线上环境access_token有过期时间， 客户端可自行缓存，过期后重新获取。
		String result = HttpUtil.post(GESTURE_URL, accessToken, param);
		System.out.println(result);
		GestureBean gestureBean = JSON.parseObject(result,GestureBean.class);
        return gestureBean;
	}
}
