package com.xs.faceplusplus;

import java.net.URLEncoder;
import java.util.Base64;
import java.util.Base64.Encoder;
import java.util.HashMap;
import java.util.Map;

import com.alibaba.fastjson.JSON;
import com.xs.faceplusplus.bean.FacePPBean;
import com.xs.faceplusplus.common.FacePPConstants;
import com.xs.util.baidu.FileUtil;
import com.xs.util.baidu.HttpUtil;
/**
 * FacePlusPlus 人脸检测示例代码
 * @author 小帅丶
 *
 */
public class FaceSkinSample {
	public static void main(String[] args) throws Exception {
		
		Encoder base64Encoder = Base64.getEncoder();
		String image_base64 = base64Encoder.encodeToString(FileUtil.readFileByBytes("G:/testimg/zdx.jpg"));
		String return_attributes = "gender,age,smiling,headpose,facequality,blur,eyestatus,emotion,ethnicity,beauty,mouthstatus,eyegaze,skinstatus";
		
		String params = "api_key="+FacePPConstants.API_KEY+"&api_secret="+FacePPConstants.API_SECRET+"&image_base64="+URLEncoder.encode(image_base64,"utf-8")+"&return_attributes="+return_attributes;
		String result = HttpUtil.post(FacePPConstants.FACE_DETECT_URL, params);
		System.out.println(result);
		FacePPBean facePP = JSON.parseObject(result, FacePPBean.class);
		if(facePP.getFaces().size()>0){
			System.out.println("肤质健康分数:"+facePP.getFaces().get(0).getAttributes().getSkinstatus().getHealth());
			System.out.println("肤质色斑分数:"+facePP.getFaces().get(0).getAttributes().getSkinstatus().getStain());
			System.out.println("肤质青春痘分数:"+facePP.getFaces().get(0).getAttributes().getSkinstatus().getAcne());
			System.out.println("黑眼圈分数:"+facePP.getFaces().get(0).getAttributes().getSkinstatus().getDark_circle());
		}else{
			System.out.println("没有人脸");
		}
	}
}
