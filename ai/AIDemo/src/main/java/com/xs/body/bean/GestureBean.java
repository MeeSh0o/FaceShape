package com.xs.body.bean;

import java.util.List;

/**
 * 手势Bean
 * @author 小帅丶
 */
public class GestureBean {
	//手势数目
    private int result_num;
    //检测到的目标，手势、人脸
    private List<Result> result;
    //唯一的log id，用于问题定位
    private long log_id;
    
    public int getResult_num() {
		return result_num;
	}

	public void setResult_num(int result_num) {
		this.result_num = result_num;
	}

	public List<Result> getResult() {
		return result;
	}

	public void setResult(List<Result> result) {
		this.result = result;
	}

	public long getLog_id() {
		return log_id;
	}

	public void setLog_id(long log_id) {
		this.log_id = log_id;
	}

	/**
     * classname	目标所属类别，15种手势、other、face
     * top	目标框上坐标
     * widt	目标框的宽
     * left	目标框最左坐标
     * height	目标框的高
     * probability	目标属于该类别的概率
     * @author 小帅丶
     */
    public static class Result{
    	private double probability;
    	private int top;
    	private int height;
    	private String classname;
    	private int width;
    	private int left;
		public double getProbability() {
			return probability;
		}
		public void setProbability(double probability) {
			this.probability = probability;
		}
		public int getTop() {
			return top;
		}
		public void setTop(int top) {
			this.top = top;
		}
		public int getHeight() {
			return height;
		}
		public void setHeight(int height) {
			this.height = height;
		}
		public String getClassname() {
			return classname;
		}
		public void setClassname(String classname) {
			this.classname = classname;
		}
		public int getWidth() {
			return width;
		}
		public void setWidth(int width) {
			this.width = width;
		}
		public int getLeft() {
			return left;
		}
		public void setLeft(int left) {
			this.left = left;
		}
    }
}
