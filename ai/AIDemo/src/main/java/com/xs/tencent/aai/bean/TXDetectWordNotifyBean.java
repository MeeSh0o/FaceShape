package com.xs.tencent.aai.bean;

import java.util.List;
/**
 * @author 小帅丶
 * @类名称  TXDetectWordNotifyBean
 * @remark 
 * @date  2018-9-4
 */
public class TXDetectWordNotifyBean {
	public int ret;//0表示成功，非0表示出错
	private String msg;//返回信息；ret非0时表示出错时错误原因
	private Data data;//结果数据；ret为0时有意义
	
	public int getRet() {
		return ret;
	}
	public void setRet(int ret) {
		this.ret = ret;
	}
	public String getMsg() {
		return msg;
	}
	public void setMsg(String msg) {
		this.msg = msg;
	}
	public Data getData() {
		return data;
	}
	public void setData(Data data) {
		this.data = data;
	}
	public static class Data{
		private String task_id;// 	本次识别任务ID
		private int is_end;//是否结束
		private int seg_index;//识别语音片段
		private int bps;//语音片段开始时间（单位毫秒）
		private int eps;//语音片段结束时间（单位毫秒）
		private Res res;//关键词
		public String getTask_id() {
			return task_id;
		}
		public void setTask_id(String task_id) {
			this.task_id = task_id;
		}
		public int getIs_end() {
			return is_end;
		}
		public void setIs_end(int is_end) {
			this.is_end = is_end;
		}
		public int getSeg_index() {
			return seg_index;
		}
		public void setSeg_index(int seg_index) {
			this.seg_index = seg_index;
		}
		public int getBps() {
			return bps;
		}
		public void setBps(int bps) {
			this.bps = bps;
		}
		public int getEps() {
			return eps;
		}
		public void setEps(int eps) {
			this.eps = eps;
		}
		public Res getRes() {
			return res;
		}
		public void setRes(Res res) {
			this.res = res;
		}
		
	}
	public static class Res{
		private int key_words_size;//关键词数量
		private List<KeyWords> key_words;//关键词列表
		public int getKey_words_size() {
			return key_words_size;
		}
		public void setKey_words_size(int key_words_size) {
			this.key_words_size = key_words_size;
		}
		public List<KeyWords> getKey_words() {
			return key_words;
		}
		public void setKey_words(List<KeyWords> key_words) {
			this.key_words = key_words;
		}
		
	}
	public static class KeyWords{
		private String key_word;//关键词
		private float score;//可靠性得分
		private float mbtm;//当前关键词相对于该语音片段的开始时间（单位毫秒）
		private float metm;//当前关键词相对于该语音片段的结束时间（单位毫秒）
		public String getKey_word() {
			return key_word;
		}
		public void setKey_word(String key_word) {
			this.key_word = key_word;
		}
		public float getScore() {
			return score;
		}
		public void setScore(float score) {
			this.score = score;
		}
		public float getMbtm() {
			return mbtm;
		}
		public void setMbtm(float mbtm) {
			this.mbtm = mbtm;
		}
		public float getMetm() {
			return metm;
		}
		public void setMetm(float metm) {
			this.metm = metm;
		}
		
	}
}
