package com.xs.tencent.aai.controller;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONObject;
import com.xs.tencent.aai.bean.TXDetectWordNotifyBean;
import com.xs.tencent.aai.bean.TXResponse;

//import org.springframework.context.annotation.Scope;
//import org.springframework.stereotype.Controller;
//import org.springframework.web.bind.annotation.RequestMapping;
//import org.springframework.web.bind.annotation.RequestMethod;
//import org.springframework.web.bind.annotation.RequestParam;
//import org.springframework.web.bind.annotation.ResponseBody;
//import org.springframework.web.servlet.ModelAndView;
/**
 * 关键词检索识别回调方法
 * @author 小帅丶
 *
 */
//@Controller
//@Scope("prototype") 
public class DetectWordNotifyController {
	/**
	 * 关键词检索识别回调方法
	 * @param request
	 * @param response
	 * @throws IOException
	 */
//	@RequestMapping(value="/txnotifydetectword",method={RequestMethod.POST,RequestMethod.GET},produces="text/html;charset=UTF-8")
//	@ResponseBody
	public String TxNotifyDetectWord(HttpServletRequest request,HttpServletResponse response) throws Exception {
		TXResponse txResponse = null;
		InputStream inputStream;
		String result = "";
		String response_result = "";
		response.setContentType("text/xml;charset=UTF-8");
		response.setCharacterEncoding("UTF-8");
		try {
			inputStream = request.getInputStream();
			String s = null;
			BufferedReader in = new BufferedReader(new InputStreamReader(inputStream, "UTF-8"));
			while ((s = in.readLine()) != null) {
				result = result + s;
			}
			System.out.println("tx回调返回的数据==="+result);
			TXDetectWordNotifyBean notify = JSONObject.toJavaObject(JSON.parseObject(result), TXDetectWordNotifyBean.class);
			if(notify.getRet()==0){
				txResponse = new TXResponse();
				txResponse.setMsg("ok");
				txResponse.setRet(0);
				result = JSONObject.toJSONString(txResponse);
				System.out.println("返回的数据==="+result);
				return result;
			}else{
				txResponse = new TXResponse();
				txResponse.setMsg("fail");
				txResponse.setRet(1);
				result = JSONObject.toJSONString(txResponse);
				System.out.println("返回的数据==="+result);
				return result;
			}
		} catch (Exception e) {
			System.out.println("====TX回调出错了"+e.getMessage()+"=========响应结果"+response_result);
			txResponse = new TXResponse();
			txResponse.setMsg("error");
			txResponse.setRet(5);
			result = JSONObject.toJSONString(txResponse);
			System.out.println("返回的数据==="+result);
			return result;
		}
	}
}
