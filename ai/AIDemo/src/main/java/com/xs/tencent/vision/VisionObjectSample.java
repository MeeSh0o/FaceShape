package com.xs.tencent.vision;

import java.util.HashMap;

import com.xs.tencent.TencentAPI;
import com.xs.tencent.sign.TencentAISign;
import com.xs.tencent.sign.TencentAISignSort;
import com.xs.util.baidu.Base64Util;
import com.xs.util.baidu.FileUtil;
import com.xs.util.baidu.HttpUtil;
/**
 * 物体识别接口示例代码
 * @author 小帅丶
 *
 */
public class VisionObjectSample {
	public static void main(String[] args) throws Exception {
		//时间戳
		String time_stamp = System.currentTimeMillis()/1000+"";
		//图片的二进制数组数据
		byte [] imageData = FileUtil.readFileByBytes("G:/onion.jpg");
		//图片的base64编码数据
		String image = Base64Util.encode(imageData);
		//随机字符串
		String nonce_str = TencentAISign.getRandomString(10);
		HashMap<String, String> bodys = new HashMap<String, String>();
		bodys.put("app_id",TencentAPI.APP_ID_AI.toString());
		bodys.put("time_stamp",time_stamp);
		bodys.put("nonce_str", nonce_str);
		bodys.put("format", "1");
		/**
		 * 图片格式，1 JPG格式（image/jpeg） 
		 * bodys.put("format", "1");
		 * 亲测发现这个参数并不是必须填写参数 如果填写则会16388鉴权错误
		 * 2018年4月23日 更新 如不填写提示4096错误 接口服务已经更新为必填参数
		 * **/
		//topk 返回结果个数（已按置信度倒排） 1-5
		bodys.put("topk", "1");
		bodys.put("image", image);
	   //计算SIGN 
		String sign = TencentAISignSort.getSignature(bodys);
		bodys.put("sign", sign);
		//拼接一下带sign的参数
		String params = TencentAISignSort.getParams(bodys);
		//自行在TencentAPI 物体识别 API地址即可
		String result = HttpUtil.post(TencentAPI.VISION_OBJECTR, params);
		System.out.println(result);
	}
}
