package com.xs.pojo.ocr.iocr;

import java.util.List;
/**
 * 自定义模板文字识别Javabean 
 * 只针对于templateSign参数测试过 classifierId并不一定完全适用哦
 * @author 小帅丶
 */
public class IOCRBean {
    private Data data;//识别返回的结果
    private int error_code;//0代表成功，如果有错误码返回可以参考官方错误码列表排查问题
    private String error_msg;//如果error_code具体的失败信息，可以官方错误码列表排查问题
    
    public Data getData() {
		return data;
	}
	public void setData(Data data) {
		this.data = data;
	}
	public int getError_code() {
		return error_code;
	}
	public void setError_code(int error_code) {
		this.error_code = error_code;
	}
	public String getError_msg() {
		return error_msg;
	}
	public void setError_msg(String error_msg) {
		this.error_msg = error_msg;
	}
	public static class Data {
        private List<Ret> ret;//识别出来的字段数组，每一个单元里包含以下几个元素
        private String templateSign;//使用的模板ID
        private int scores;//分类器的得分，直接选择模板ID识别时为1
        private boolean isStructured;//表示是否结构话成功，true为成功，false为失败
        private String logId;//调用的日志id
		public List<Ret> getRet() {
			return ret;
		}
		public void setRet(List<Ret> ret) {
			this.ret = ret;
		}
		public String getTemplateSign() {
			return templateSign;
		}
		public void setTemplateSign(String templateSign) {
			this.templateSign = templateSign;
		}
		public int getScores() {
			return scores;
		}
		public void setScores(int scores) {
			this.scores = scores;
		}
		public boolean isStructured() {
			return isStructured;
		}
		public void setStructured(boolean isStructured) {
			this.isStructured = isStructured;
		}
		public String getLogId() {
			return logId;
		}
		public void setLogId(String logId) {
			this.logId = logId;
		}
        
    }
    
    public static class Ret {
        private List<Charset> charset;//单个字的数组，每个元素包含一个word 和 rect
        private String word_name;//表示字段的名字
        private String word;//识别的字符串
		public List<Charset> getCharset() {
			return charset;
		}
		public void setCharset(List<Charset> charset) {
			this.charset = charset;
		}
		public String getWord_name() {
			return word_name;
		}
		public void setWord_name(String word_name) {
			this.word_name = word_name;
		}
		public String getWord() {
			return word;
		}
		public void setWord(String word) {
			this.word = word;
		}
        
    }
    public static class Charset {
        private Rect rect;//字符串或单字所在矩形框
        private String word;//识别的单字
		public Rect getRect() {
			return rect;
		}
		public void setRect(Rect rect) {
			this.rect = rect;
		}
		public String getWord() {
			return word;
		}
		public void setWord(String word) {
			this.word = word;
		}
        
    }
    public static class Rect {
        private int top;//表示定位位置的长方形左上顶点的垂直坐标
        private int left;//表示定位位置的长方形左上顶点的水平坐标
        private int width;//表示定位位置的长方形的宽度
        private int height;//表示定位位置的长方形的高度
		public int getTop() {
			return top;
		}
		public void setTop(int top) {
			this.top = top;
		}
		public int getLeft() {
			return left;
		}
		public void setLeft(int left) {
			this.left = left;
		}
		public int getWidth() {
			return width;
		}
		public void setWidth(int width) {
			this.width = width;
		}
		public int getHeight() {
			return height;
		}
		public void setHeight(int height) {
			this.height = height;
		}
    }
}
