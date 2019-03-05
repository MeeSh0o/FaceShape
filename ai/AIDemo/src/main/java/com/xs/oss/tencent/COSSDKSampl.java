package com.xs.oss.tencent;

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;

import com.alibaba.fastjson.JSONObject;
import com.qcloud.cos.COSClient;
import com.qcloud.cos.ClientConfig;
import com.qcloud.cos.auth.BasicCOSCredentials;
import com.qcloud.cos.auth.COSCredentials;
import com.qcloud.cos.model.ObjectMetadata;
import com.qcloud.cos.model.PutObjectRequest;
import com.qcloud.cos.model.PutObjectResult;
import com.qcloud.cos.region.Region;
/**
 * 腾讯云COS上传文件示例
 * @author 小帅丶
 *
 */
public class COSSDKSampl {
	
	public static void main(String[] args) throws Exception {
		String accessKey = "";//自己的SecretId
		String secretKey = "";//自己的SecretKey  
		// 1 初始化用户身份信息(secretId, secretKey)
		COSCredentials cred = new BasicCOSCredentials(accessKey, secretKey);
		// 2 设置bucket的区域 自己的哦
		ClientConfig clientConfig = new ClientConfig(new Region(""));
		// 3 生成cos客户端
		COSClient cosClient = new COSClient(cred, clientConfig);
		// bucket的命名 此处填写存储桶名称
		String bucketName = "";
		//4.测试上传的文件 并给一个文件名称包含后缀名
		File file = new File("G:/testimg/xcx.jpg");
		String key = "xcx.jpg";
		InputStream input = new FileInputStream(file);
		ObjectMetadata metadata = new ObjectMetadata();
		metadata.setContentLength(file.length());
		PutObjectRequest putObjectRequest = new PutObjectRequest(bucketName, key, input,metadata );
		PutObjectResult putObjectResult = cosClient.putObject(putObjectRequest);
		//上传返回的内容
		System.out.println(JSONObject.toJSONString(putObjectResult));
	}
}
