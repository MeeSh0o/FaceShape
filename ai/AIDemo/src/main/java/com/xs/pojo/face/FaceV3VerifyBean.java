package com.xs.pojo.face;

import java.util.List;

import com.alibaba.fastjson.annotation.JSONField;

/**
 * 人脸V3对象
 * @author 小帅丶
 *
 */
public class FaceV3VerifyBean {
    private int error_code;
    private String error_msg;
    private long log_id;
    private long timestamp;
    private int cached;
    private Result result;
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
	public long getLog_id() {
		return log_id;
	}
	public void setLog_id(long log_id) {
		this.log_id = log_id;
	}
	public long getTimestamp() {
		return timestamp;
	}
	public void setTimestamp(long timestamp) {
		this.timestamp = timestamp;
	}
	public int getCached() {
		return cached;
	}
	public void setCached(int cached) {
		this.cached = cached;
	}
	public Result getResult() {
		return result;
	}
	public void setResult(Result result) {
		this.result = result;
	}
	public static class Result{
        private Thresholds thresholds;
        private double face_liveness;
        private List<Face_list> face_list;
		public Thresholds getThresholds() {
			return thresholds;
		}
		public void setThresholds(Thresholds thresholds) {
			this.thresholds = thresholds;
		}
		public double getFace_liveness() {
			return face_liveness;
		}
		public void setFace_liveness(double face_liveness) {
			this.face_liveness = face_liveness;
		}
		public List<Face_list> getFace_list() {
			return face_list;
		}
		public void setFace_list(List<Face_list> face_list) {
			this.face_list = face_list;
		}
    }
    public static class Face_list{
        private String face_token;
        private Location location;
        private int face_probability;
        private Angle angle;
        private Liveness liveness;
		public String getFace_token() {
			return face_token;
		}
		public void setFace_token(String face_token) {
			this.face_token = face_token;
		}
		public Location getLocation() {
			return location;
		}
		public void setLocation(Location location) {
			this.location = location;
		}
		public int getFace_probability() {
			return face_probability;
		}
		public void setFace_probability(int face_probability) {
			this.face_probability = face_probability;
		}
		public Angle getAngle() {
			return angle;
		}
		public void setAngle(Angle angle) {
			this.angle = angle;
		}
		public Liveness getLiveness() {
			return liveness;
		}
		public void setLiveness(Liveness liveness) {
			this.liveness = liveness;
		}
    }
    public static class Liveness{
        private double faceliveness;
        private double livemapscore;
    }
    public static class Angle{
    	private double yaw;
	    private double pitch;
	    private double roll;
		public double getYaw() {
			return yaw;
		}
		public void setYaw(double yaw) {
			this.yaw = yaw;
		}
		public double getPitch() {
			return pitch;
		}
		public void setPitch(double pitch) {
			this.pitch = pitch;
		}
		public double getRoll() {
			return roll;
		}
		public void setRoll(double roll) {
			this.roll = roll;
		}
    }
    public static class Location{
        private double left;
        private double top;
        private int width;
        private int height;
        private int rotation;
		public double getLeft() {
			return left;
		}
		public void setLeft(double left) {
			this.left = left;
		}
		public double getTop() {
			return top;
		}
		public void setTop(double top) {
			this.top = top;
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
		public int getRotation() {
			return rotation;
		}
		public void setRotation(int rotation) {
			this.rotation = rotation;
		}
        
    }
    public static class Thresholds{
    	@JSONField(name="frr_1e-4")
        private double frr1e4;
    	@JSONField(name="frr_1e-3")
        private double frr1e3;
    	@JSONField(name="frr_1e-2")
        private double frr1e2;
		public double getFrr1e4() {
			return frr1e4;
		}
		public void setFrr1e4(double frr1e4) {
			this.frr1e4 = frr1e4;
		}
		public double getFrr1e3() {
			return frr1e3;
		}
		public void setFrr1e3(double frr1e3) {
			this.frr1e3 = frr1e3;
		}
		public double getFrr1e2() {
			return frr1e2;
		}
		public void setFrr1e2(double frr1e2) {
			this.frr1e2 = frr1e2;
		}
    	
    }
}
