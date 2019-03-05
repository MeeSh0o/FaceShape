package com.xs.aliet.asr;
import com.xs.aliet.util.AliETHttpUtil;

import cn.xsshome.taip.util.FileUtil;

/**
 * 语音识别示例代码
 * @author 小帅丶
 *
 */
public class AliASRSampleByAPI {
	private static String url = "http://nlsapi.aliyun.com/recognize?";
	public static void main(String[] args) throws Exception {
		String akId = "替换成自己的";
		String akSecret ="替换成自己的";
        String model = "chat";
        url = url+"model="+model;
        byte [] data = FileUtil.readFileByBytes("G:/16.pcm");
        com.xs.aliet.util.HttpResponse response = AliETHttpUtil.sendAsrPost(data, "pcm", 16000, url, akId, akSecret);
        System.out.println(response.getResult());
	}
}
