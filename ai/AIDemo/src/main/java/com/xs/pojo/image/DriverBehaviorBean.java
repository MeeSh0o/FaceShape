package com.xs.pojo.image;

import java.util.List;
/**
 * 驾驶行为Bean
 * @author 小帅丶
 */
public class DriverBehaviorBean {
	private int person_num;
    private List<Person_info> person_info;
    private long log_id;
    public int getPerson_num() {
		return person_num;
	}
	public void setPerson_num(int person_num) {
		this.person_num = person_num;
	}
	public List<Person_info> getPerson_info() {
		return person_info;
	}
	public void setPerson_info(List<Person_info> person_info) {
		this.person_info = person_info;
	}
	public long getLog_id() {
		return log_id;
	}
	public void setLog_id(long log_id) {
		this.log_id = log_id;
	}
	public static class Attributes {
        private Cellphone cellphone;
        private Smoke smoke;
		public Cellphone getCellphone() {
			return cellphone;
		}
		public void setCellphone(Cellphone cellphone) {
			this.cellphone = cellphone;
		}
		public Smoke getSmoke() {
			return smoke;
		}
		public void setSmoke(Smoke smoke) {
			this.smoke = smoke;
		}
    }
    public static class Cellphone {
        private double threshold;
        private double score;
		public double getThreshold() {
			return threshold;
		}
		public void setThreshold(double threshold) {
			this.threshold = threshold;
		}
		public double getScore() {
			return score;
		}
		public void setScore(double score) {
			this.score = score;
		}
    }
    public static class Smoke {
        private double threshold;
        private double score;
		public double getThreshold() {
			return threshold;
		}
		public void setThreshold(double threshold) {
			this.threshold = threshold;
		}
		public double getScore() {
			return score;
		}
		public void setScore(double score) {
			this.score = score;
		}
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
    public class Person_info {
        private Attributes attributes;
        private Location location;
		public Attributes getAttributes() {
			return attributes;
		}
		public void setAttributes(Attributes attributes) {
			this.attributes = attributes;
		}
		public Location getLocation() {
			return location;
		}
		public void setLocation(Location location) {
			this.location = location;
		}
    }
}
