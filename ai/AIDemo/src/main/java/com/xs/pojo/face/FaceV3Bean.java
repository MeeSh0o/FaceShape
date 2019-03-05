package com.xs.pojo.face;

import org.json.JSONObject;



public class FaceV3Bean {
		private String image;
		private String image_type;
		private String face_type;
		private String quality_control;
		private String liveness_control;
		public FaceV3Bean() {
		}
		
		public FaceV3Bean(String image, String image_type) {
			super();
			this.image = image;
			this.image_type = image_type;
		}

		public FaceV3Bean(String image, String image_type, String face_type,
				String quality_control, String liveness_control) {
			super();
			this.image = image;
			this.image_type = image_type;
			this.face_type = face_type;
			this.quality_control = quality_control;
			this.liveness_control = liveness_control;
		}
		public String getImage() {
			return image;
		}
		public void setImage(String image) {
			this.image = image;
		}
		public String getImage_type() {
			return image_type;
		}
		public void setImage_type(String image_type) {
			this.image_type = image_type;
		}
		public String getFace_type() {
			return face_type;
		}
		public void setFace_type(String face_type) {
			this.face_type = face_type;
		}
		public String getQuality_control() {
			return quality_control;
		}
		public void setQuality_control(String quality_control) {
			this.quality_control = quality_control;
		}
		public String getLiveness_control() {
			return liveness_control;
		}
		public void setLiveness_control(String liveness_control) {
			this.liveness_control = liveness_control;
		}
		  public JSONObject toJsonObject() {
		        JSONObject obj = new JSONObject();
		        obj.put("image", this.image);
		        obj.put("image_type", this.image_type);
		        if (this.face_type != null) {
		            obj.put("face_type", this.face_type);
		        }
		        if (this.quality_control != null) {
		            obj.put("quality_control", this.quality_control);
		        }
		        if (this.liveness_control != null) {
		            obj.put("liveness_control", this.liveness_control);
		        }
		        return obj;
		    }
}
