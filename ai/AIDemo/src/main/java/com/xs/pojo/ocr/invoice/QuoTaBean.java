package com.xs.pojo.ocr.invoice;
/**
 * 定额发票识别Bean
 * @author 小帅丶
 *
 */
public class QuoTaBean {
	//日志id
    private long log_id;
    //识别结果个数 
    private int words_result_num;
    //识别结果集
    private WordsResult words_result;
    
    public long getLog_id() {
		return log_id;
	}

	public void setLog_id(long log_id) {
		this.log_id = log_id;
	}

	public int getWords_result_num() {
		return words_result_num;
	}

	public void setWords_result_num(int words_result_num) {
		this.words_result_num = words_result_num;
	}

	public WordsResult getWords_result() {
		return words_result;
	}

	public void setWords_result(WordsResult words_result) {
		this.words_result = words_result;
	}
	/**
	 * WordsResult 识别具体的内容
	 * @author 小帅丶
	 *
	 */
	public static class WordsResult{
		//发票代码
        private String invoice_code;
        //发票号码
        private String invoice_number;
        //发票金额
        private String invoice_rate;
        
		public String getInvoice_code() {
			return invoice_code;
		}
		public void setInvoice_code(String invoice_code) {
			this.invoice_code = invoice_code;
		}
		public String getInvoice_number() {
			return invoice_number;
		}
		public void setInvoice_number(String invoice_number) {
			this.invoice_number = invoice_number;
		}
		public String getInvoice_rate() {
			return invoice_rate;
		}
		public void setInvoice_rate(String invoice_rate) {
			this.invoice_rate = invoice_rate;
		}
    }
}
