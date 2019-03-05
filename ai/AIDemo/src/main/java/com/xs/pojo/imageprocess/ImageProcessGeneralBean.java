package com.xs.pojo.imageprocess;
/**
 * 图像处理接口返回的bean对象
 * @author 小帅丶
 */
public class ImageProcessGeneralBean {
	private long log_id;//唯一的log id，用于问题定位
	private String image;//base64编码图片
	public long getLog_id() {
		return log_id;
	}
	public void setLog_id(long log_id) {
		this.log_id = log_id;
	}
	public String getImage() {
		return image;
	}
	public void setImage(String image) {
		this.image = image;
	}
}
