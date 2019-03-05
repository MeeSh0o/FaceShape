package com.xs.easydl;

import java.util.ArrayList;
import java.util.List;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONObject;
import com.xs.easydl.DatasetAddRequestParam.Labels;
import com.xs.util.baidu.Base64Util;
import com.xs.util.baidu.FileUtil;
import com.xs.util.baidu.HttpUtil;

/**
 * 数据集管理API
 * @author 小帅丶
 *
 */
public class EasyDLManagerAPISample {
//	数据集创建API 
	static final String DATASET_CREATE_API ="https://aip.baidubce.com/rpc/2.0/easydl/dataset/create";
//	数据集列表API 
	static final String DATASET_LIST_API ="https://aip.baidubce.com/rpc/2.0/easydl/dataset/list";
//	添加数据API
	static final String DATASET_ADD_API ="https://aip.baidubce.com/rpc/2.0/easydl/dataset/addentity";
//	数据集删除API
	static final String DATASET_DELETE_API ="https://aip.baidubce.com/rpc/2.0/easydl/dataset/delete";
//	分类（标签）列表API 
	static final String LABEL_LIST_API ="https://aip.baidubce.com/rpc/2.0/easydl/label/list";
//	分类（标签）删除API 
	static final String LABEL_DELETE_API ="https://aip.baidubce.com/rpc/2.0/easydl/label/delete";
	//测试使用的AccessToken  具体需要替换成自己的哦
	static final String ACCESS_TOKEN = "";
	/**
	 * 测试main方法
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
		String OBJECT_DETECTION = "OBJECT_DETECTION";//物体检测
		String IMAGE_CLASSIFICATION = "IMAGE_CLASSIFICATION";//图像分类
		String TEXT_CLASSIFICATION = "TEXT_CLASSIFICATION";//文本分类
		String SOUND_CLASSIFICATION = "SOUND_CLASSIFICATION";//声音分类
		
//		String result = datasetList(ACCESS_TOKEN, TEXT_CLASSIFICATION);//获取数据集列表  
		//{"total_num":2,"results":[{"dataset_id":20598,"dataset_name":"apitest","type":"TEXT_CLASSIFICATION","status":"normal"},{"dataset_id":20567,"dataset_name":"textdemo","type":"TEXT_CLASSIFICATION","status":"normal"}],"log_id":1324820120}
//		String result = datasetCreate(ACCESS_TOKEN, TEXT_CLASSIFICATION,"campus");//数据集创建   
		//{"dataset_id":20598,"log_id":1276713049}
//		String result = labelList(ACCESS_TOKEN, TEXT_CLASSIFICATION,20567);//分类（标签）列表
		//{"total_num":4,"results":[{"label_id":"5c075681ea934f0001324ae3","label_name":"woman","entity_count":949},{"label_id":"5c075672ea934f0001324ae1","label_name":"sport","entity_count":1083},{"label_id":"5c07566815d32b0001a275bc","label_name":"publish","entity_count":760},{"label_id":"5c075663ea934f0001324adc","label_name":"campus","entity_count":247}],"log_id":3295199978}
/***************************文本分类示例参数拼接******************************/		
		/**
			{
		    "appendLabel": true, 
		    "dataset_id": 20598, 
		    "entity_content": "今天老师不在", 
		    "entity_name": "20181205.txt", 
		    "labels": [
		        {
		            "label_name": "campus"
		        }
		    ], 
		    "type": "TEXT_CLASSIFICATION"
			}
		 */
//		DatasetAddRequestParam addRequestParam = new DatasetAddRequestParam();
//		addRequestParam.setType(TEXT_CLASSIFICATION);
//		addRequestParam.setDataset_id(20598);
//		addRequestParam.setAppendLabel(true);
//		addRequestParam.setEntity_content("今天老师不在");
//		addRequestParam.setEntity_name("20181205.txt");
//		List<Labels> labelsList = new ArrayList<DatasetAddRequestParam.Labels>();
//		Labels labels = new Labels();
//		labels.setLabel_name("campus");
//		labelsList.add(labels);
//		addRequestParam.setLabels(labelsList);
/***************************物体检测示例参数拼接******************************/	
		/**
		{
		    "appendLabel": true, 
		    "dataset_id": 2803, 
		    "entity_content": "图片的base64 不需要urlencode", 
		    "entity_name": "whitefly127.jpg", 
		    "labels": [
		        {
		            "height": 103, 
		            "label_name": "whitefly", 
		            "left": 221, 
		            "top": 221, 
		            "width": 208
		        }
		    ], 
		    "type": "OBJECT_DETECTION"
		}
		 */
		DatasetAddRequestParam addRequestParam = new DatasetAddRequestParam();
		addRequestParam.setType(OBJECT_DETECTION);
		addRequestParam.setDataset_id(2803);
		addRequestParam.setAppendLabel(true);
		//读取本地图片文件  并转base64 不需要urlencode
		String imgPath = "G:/mnist_png/whitefly/whitefly127.jpg";
		String imgbase64 = Base64Util.encode(FileUtil.readFileByBytes(imgPath));
		addRequestParam.setEntity_content(imgbase64);
		addRequestParam.setEntity_name("whitefly127.jpg");
		List<Labels> labelsList = new ArrayList<DatasetAddRequestParam.Labels>();
		Labels labels = new Labels();
		labels.setLabel_name("whitefly");
		labelsList.add(labels);
		labels.setLeft(221);
		labels.setTop(221);
		labels.setWidth(208);
		labels.setHeight(103);
		addRequestParam.setLabels(labelsList);
		//添加数据 测试为文本分类数据添加 
//		String result = datasetAdd(ACCESS_TOKEN,addRequestParam);//添加数据
		
//		String result = datasetDel(ACCESS_TOKEN,TEXT_CLASSIFICATION,20642);//数据集删除
		//{"log_id":2595666927}
//		String result = labelDel(ACCESS_TOKEN,TEXT_CLASSIFICATION,20598,"campus");//分类（标签）删除
		//{"log_id":119766394}
//		System.out.println(result);
	}
	/**
	 * 分类（标签）删除
	 * @param accessToken 应用鉴权的token 
	 * @param type
	 *            数据集类型，可包括： IMAGE_CLASSIFICATION, OBJECT_DETECTION,
	 *            SOUND_CLASSIFICATION, TEXT_CLASSIFICATION
	 *            分别对应：图像分类、物体检测、声音分类、文本分类
	 * @param dataset_id 数据集ID
	 * @param label_name 标签/分类名称
	 * @return String
	 * @throws Exception
	 */
	public static String labelDel(String accessToken,String type,int dataset_id,String label_name) throws Exception{
		//接口地址
		String url = LABEL_DELETE_API;
		String access_token = accessToken;
		// JSON格式参数拼接
		String params = "{\"type\":\"" + type + "\",\"dataset_id\":"+dataset_id+",\"label_name\":\"" + label_name + "\"}";
		String result = HttpUtil.post(url,access_token, params);
		// 输出识别结果
		return result;
	}
	/**
	 * 数据集删除
	 * @param accessToken 应用鉴权的token 
	 * @param type
	 *            数据集类型，可包括： IMAGE_CLASSIFICATION, OBJECT_DETECTION,
	 *            SOUND_CLASSIFICATION, TEXT_CLASSIFICATION
	 *            分别对应：图像分类、物体检测、声音分类、文本分类
	 * @param dataset_id 数据集ID
	 * @return String
	 * @throws Exception
	 */
	public static String datasetDel(String accessToken,String type,int dataset_id) throws Exception{
		//接口地址
		String url = DATASET_DELETE_API;
		String access_token = accessToken;
		// JSON格式参数拼接
		String params = "{\"type\":\"" + type + "\",\"dataset_id\":"+dataset_id+"}";
		String result = HttpUtil.post(url,access_token, params);
		// 输出识别结果
		return result;
	}
	/**
	 * 添加数据
	 * @param accessToken 应用鉴权的token 
	 * @param addRequestParam 参数对象
	 * @return String
	 * @throws Exception
	 */
	public static String datasetAdd(String accessToken,DatasetAddRequestParam addRequestParam) throws Exception{
		//接口地址
		String url = DATASET_ADD_API;
		String access_token = accessToken;
		// JSON格式参数拼接
		String params = JSON.toJSONString(addRequestParam);
		String result = HttpUtil.post(url,access_token, params);
		// 输出识别结果
		return result;
	}
	
	/**
	 * 分类（标签）列表获取 所有参数
	 * @param accessToken 应用鉴权的token 
	 * @param type
	 *            数据集类型，可包括： IMAGE_CLASSIFICATION, OBJECT_DETECTION,
	 *            SOUND_CLASSIFICATION, TEXT_CLASSIFICATION
	 *            分别对应：图像分类、物体检测、声音分类、文本分类
	 * @param dataset_id 数据集ID
	 * @param start 起始序号，默认为0
	 * @param num 数量，默认20，最多100
	 * @return String
	 * @throws Exception
	 */
	public static String labelList(String accessToken,String type,int dataset_id,int start,int num) throws Exception{
		//接口地址
		String url = LABEL_LIST_API;
		String access_token = accessToken;
		// JSON格式参数拼接
		String params = "{\"type\":\"" + type + "\",\"dataset_id\":"+dataset_id+",\"start\":"+start+",\"num\":"+num+"}";
		String result = HttpUtil.post(url, access_token, params);
		// 输出识别结果
		return result;
	}
	/**
	 * 分类（标签）列表获取
	 * @param accessToken 应用鉴权的token 
	 * @param type
	 *            数据集类型，可包括： IMAGE_CLASSIFICATION, OBJECT_DETECTION,
	 *            SOUND_CLASSIFICATION, TEXT_CLASSIFICATION
	 *            分别对应：图像分类、物体检测、声音分类、文本分类
	 * @param dataset_id 数据集ID
	 * @return String
	 * @throws Exception
	 */
	public static String labelList(String accessToken,String type,int dataset_id) throws Exception{
		return labelList(accessToken, type, dataset_id, 0, 20);
	}
	/**
	 * 数据集创建
	 * @param accessToken 应用鉴权的token 
	 * @param type
	 *            数据集类型，可包括： IMAGE_CLASSIFICATION, OBJECT_DETECTION,
	 *            SOUND_CLASSIFICATION, TEXT_CLASSIFICATION
	 *            分别对应：图像分类、物体检测、声音分类、文本分类
	 * @param dataset_name 数据集名称，长度不超过20个utf-8字符
	 * @return String
	 * @throws Exception
	 */
	public static String datasetCreate(String accessToken,String type,String dataset_name) throws Exception{
		//接口地址
		String url = DATASET_CREATE_API;
		String access_token = accessToken;
		// JSON格式参数拼接
		String params = "{\"type\":\"" + type + "\",\"dataset_name\":\""+dataset_name+"\"}";
		String result = HttpUtil.post(url, access_token, params);
		// 输出识别结果
		return result;
	}
	/**
	 * 数据集列表获取 所有参数
	 * @param accessToken 应用鉴权的token 
	 * @param type
	 *            数据集类型，可包括： IMAGE_CLASSIFICATION, OBJECT_DETECTION,
	 *            SOUND_CLASSIFICATION, TEXT_CLASSIFICATION
	 *            分别对应：图像分类、物体检测、声音分类、文本分类
	 * @param start 起始序号，默认为0
	 * @param num 数量，默认20，最多100
	 * @return String
	 * @throws Exception
	 */
	public static String datasetList(String accessToken,String type,int start,int num) throws Exception{
		//接口地址
		String url = DATASET_LIST_API;
		String access_token = accessToken;
		// JSON格式参数拼接
		String params = "{\"type\":\"" + type + "\",\"start\":"+start+",\"num\":"+num+"}";
		String result = HttpUtil.post(url, access_token, params);
		// 输出识别结果
		return result;
	}
	/**
	 * 数据集列表获取 默认必填参数
	 * @param accessToken 应用鉴权的token 
	 * @param type
	 *            数据集类型，可包括： IMAGE_CLASSIFICATION, OBJECT_DETECTION,
	 *            SOUND_CLASSIFICATION, TEXT_CLASSIFICATION
	 *            分别对应：图像分类、物体检测、声音分类、文本分类
	 * @return String
	 * @throws Exception
	 */
	public static String datasetList(String accessToken,String type) throws Exception{
		return datasetList(accessToken, type, 0, 20);
	}
}
