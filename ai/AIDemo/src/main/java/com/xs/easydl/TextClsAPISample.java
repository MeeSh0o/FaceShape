package com.xs.easydl;


import com.xs.util.baidu.HttpUtil;
/**
 * 文本分类接口示例代码 -JavaAPI
 * @author 小帅丶
 *
 */
public class TextClsAPISample {
	static final String TEXT_CLS_API ="https://aip.baidubce.com/rpc/2.0/ai_custom/v1/text_cls/xstext";
	static final String ACCESS_TOKEN = "";
	public static void main(String[] args) throws Exception {
			String text = "今天老师不在";
			String result = getTextClsResult(ACCESS_TOKEN, text);
			System.out.println(result);
	}
	/**
	 * 请求EasyDL文本分类接口
	 * @param accessToken 鉴权的token
	 * @param text 
	 * @return String
	 * @throws Exception
	 */
	public static String getTextClsResult(String accessToken,String text) throws Exception {
		// 请将API地址替换为EasyDL所提供的API地址
		String url = TEXT_CLS_API;
		// access_token获取方法请详见API使用说明，请注意access_token有效期为30天
		String access_token = accessToken;
		// 返回分类数量top_num设置为5 如不设置默认返回6条结果
		String params = "{\"text\":\"" + text + "\",\"top_num\":5}";
		String result = HttpUtil.post(url, access_token, params);
		// 输出识别结果
		System.out.println(result);
		return result;
	}
}
