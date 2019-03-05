package com.xs.pojo.image;

import java.math.BigDecimal;
import java.util.List;
/**
 * 定制化图像平台JavaBean
 * @author 小帅丶
 * BigDecimal 支持科学计数
 */
public class CustomBean {
	private Long log_id;
	private List<Results> results;
	public Long getLog_id() {
		return log_id;
	}

	public void setLog_id(Long log_id) {
		this.log_id = log_id;
	}

	public List<Results> getResults() {
		return results;
	}

	public void setResults(List<Results> results) {
		this.results = results;
	}

	public static class Results {
		private String name;
		private BigDecimal score;
		public void setName(String name) {
			this.name = name;
		}
		public BigDecimal getScore() {
			return score;
		}
		public void setScore(BigDecimal score) {
			this.score = score;
		}
		public String getName() {
			return name;
		}
	}
}
