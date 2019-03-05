package com.xs.aliet.beebot;
import java.util.Date;
import java.util.HashMap;
import java.util.UUID;

import com.xs.aliet.beebot.request.BeeBotRequest;
import com.xs.aliet.util.AliETSignUtil;
import com.xs.aliet.util.AliYunHttpUtils;
/**
 * 云小蜜 示例代码
 * @author 小帅丶
 *
 */
public class BeeBotSample {
	//接口地址
	private static final String CHATBOT_URL= "https://chatbot.cn-shanghai.aliyuncs.com/";
	public static void main(String[] args) throws Exception {
		String akId = "自己的akid";
		String akSecret ="自己的akSecret";
		String timeStamp = AliETSignUtil.getSolrDate(new Date());
		String nonce_str = UUID.randomUUID().toString();
		System.out.println("时间："+timeStamp+"\n"+"随机数:"+nonce_str);
		//全部参数
		HashMap<String, Object> hashMap = AliETSignUtil.transBean2Map(new BeeBotRequest("JSON","2017-10-11",akId,""," HMAC-SHA1",timeStamp,"1.0",nonce_str,"Chat", "chatbot-cn-0pp0qmhav000h8", "电脑黑屏了怎么办"));
		//省略参数
//		HashMap<String, Object> hashMap = AliETSignUtil.transBean2Map(new BeeBotRequest(akId,timeStamp,nonce_str,"Chat", "chatbot-cn-0pp0qmhav000h8", "电脑黑屏了怎么办"));
		String sign = AliETSignUtil.getSignature("GET", hashMap, akSecret);
		hashMap.put("Signature", sign);
		String param = AliETSignUtil.getParams(hashMap);
		String result  = AliYunHttpUtils.sendGet(CHATBOT_URL, param);
		System.out.println(result);
	}
}
