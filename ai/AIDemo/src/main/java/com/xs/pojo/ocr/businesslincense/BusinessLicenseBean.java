package com.xs.pojo.ocr.businesslincense;

import com.alibaba.fastjson.annotation.JSONField;

public class BusinessLicenseBean {
	    private long log_id;
	    private int direction;
	    private int words_result_num;
	    private Words_result words_result;
	    
	    public long getLog_id() {
			return log_id;
		}
		public void setLog_id(long log_id) {
			this.log_id = log_id;
		}
		public int getDirection() {
			return direction;
		}
		public void setDirection(int direction) {
			this.direction = direction;
		}
		public int getWords_result_num() {
			return words_result_num;
		}
		public void setWords_result_num(int words_result_num) {
			this.words_result_num = words_result_num;
		}
		public Words_result getWords_result() {
			return words_result;
		}
		public void setWords_result(Words_result words_result) {
			this.words_result = words_result;
		}
		public static class Location {
	        private int width;
	        private int top;
	        private int height;
	        private int left;
			public int getWidth() {
				return width;
			}
			public void setWidth(int width) {
				this.width = width;
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
			public int getLeft() {
				return left;
			}
			public void setLeft(int left) {
				this.left = left;
			}
	    }
	    public class Words_result {
	    	@JSONField(name = "成立日期")
	        private EstablishDate establishDate;
	    	@JSONField(name = "地址")
	        private Address address;
	        @JSONField(name = "单位名称")
	        private UnitName unitName;
	        @JSONField(name = "法人")
	        private LegalPerson legalPerson;
	        @JSONField(name = "有效期")
	        private TermValidity termValidity;
	        @JSONField(name = "证件编号")
	        private IdNum idNum;
	        @JSONField(name = "社会信用代码")
	        private SocialCreditCode socialCreditCode;
			public EstablishDate getEstablishDate() {
				return establishDate;
			}
			public void setEstablishDate(EstablishDate establishDate) {
				this.establishDate = establishDate;
			}
			public Address getAddress() {
				return address;
			}
			public void setAddress(Address address) {
				this.address = address;
			}
			public UnitName getUnitName() {
				return unitName;
			}
			public void setUnitName(UnitName unitName) {
				this.unitName = unitName;
			}
			public LegalPerson getLegalPerson() {
				return legalPerson;
			}
			public void setLegalPerson(LegalPerson legalPerson) {
				this.legalPerson = legalPerson;
			}
			public TermValidity getTermValidity() {
				return termValidity;
			}
			public void setTermValidity(TermValidity termValidity) {
				this.termValidity = termValidity;
			}
			public IdNum getIdNum() {
				return idNum;
			}
			public void setIdNum(IdNum idNum) {
				this.idNum = idNum;
			}
			public SocialCreditCode getSocialCreditCode() {
				return socialCreditCode;
			}
			public void setSocialCreditCode(SocialCreditCode socialCreditCode) {
				this.socialCreditCode = socialCreditCode;
			}
	        
	    }
	    public static class EstablishDate {
	        private Location location;
	        private String words;
			public Location getLocation() {
				return location;
			}
			public void setLocation(Location location) {
				this.location = location;
			}
			public String getWords() {
				return words;
			}
			public void setWords(String words) {
				this.words = words;
			}	        
	    }
	    public static class Address {
	        private Location location;
	        private String words;
			public Location getLocation() {
				return location;
			}
			public void setLocation(Location location) {
				this.location = location;
			}
			public String getWords() {
				return words;
			}
			public void setWords(String words) {
				this.words = words;
			}	        
	    }
	    public static class UnitName {
	        private Location location;
	        private String words;
			public Location getLocation() {
				return location;
			}
			public void setLocation(Location location) {
				this.location = location;
			}
			public String getWords() {
				return words;
			}
			public void setWords(String words) {
				this.words = words;
			}	        
	    }
	    public static class LegalPerson {
	        private Location location;
	        private String words;
			public Location getLocation() {
				return location;
			}
			public void setLocation(Location location) {
				this.location = location;
			}
			public String getWords() {
				return words;
			}
			public void setWords(String words) {
				this.words = words;
			}	        
	    }
	    public static class TermValidity {
	        private Location location;
	        private String words;
			public Location getLocation() {
				return location;
			}
			public void setLocation(Location location) {
				this.location = location;
			}
			public String getWords() {
				return words;
			}
			public void setWords(String words) {
				this.words = words;
			}	        
	    }
	    public static class IdNum {
	        private Location location;
	        private String words;
			public Location getLocation() {
				return location;
			}
			public void setLocation(Location location) {
				this.location = location;
			}
			public String getWords() {
				return words;
			}
			public void setWords(String words) {
				this.words = words;
			}	        
	    }
	    public static class SocialCreditCode {
	        private Location location;
	        private String words;
			public Location getLocation() {
				return location;
			}
			public void setLocation(Location location) {
				this.location = location;
			}
			public String getWords() {
				return words;
			}
			public void setWords(String words) {
				this.words = words;
			}
	    }
}
