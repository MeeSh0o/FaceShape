package com.xs.faceplusplus.bean;

import java.util.List;
/**
 * 人脸检测对象 
 * 非全部属性包含哦 只包含必须和：
 * gender,age,smiling,headpose,facequality,blur,eyestatus,emotion,ethnicity,beauty,mouthstatus,eyegaze,skinstatus
 * @author 小帅丶
 *
 */
public class FacePPBean {
	private String image_id;
    private String request_id;
    private int time_used;
    private List<Faces> faces;
    public String getImage_id() {
		return image_id;
	}
	public void setImage_id(String image_id) {
		this.image_id = image_id;
	}
	public String getRequest_id() {
		return request_id;
	}
	public void setRequest_id(String request_id) {
		this.request_id = request_id;
	}
	public int getTime_used() {
		return time_used;
	}
	public void setTime_used(int time_used) {
		this.time_used = time_used;
	}
	public List<Faces> getFaces() {
		return faces;
	}
	public void setFaces(List<Faces> faces) {
		this.faces = faces;
	}
	public static class Faces{
    	private Attributes attributes;
	    private Face_rectangle face_rectangle;
	    private String face_token;
		public Attributes getAttributes() {
			return attributes;
		}
		public void setAttributes(Attributes attributes) {
			this.attributes = attributes;
		}
		public Face_rectangle getFace_rectangle() {
			return face_rectangle;
		}
		public void setFace_rectangle(Face_rectangle face_rectangle) {
			this.face_rectangle = face_rectangle;
		}
		public String getFace_token() {
			return face_token;
		}
		public void setFace_token(String face_token) {
			this.face_token = face_token;
		}
    }
    public static class Face_rectangle{
        private int width;
        private int top;
        private int left;
        private int height;
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
		public int getLeft() {
			return left;
		}
		public void setLeft(int left) {
			this.left = left;
		}
		public int getHeight() {
			return height;
		}
		public void setHeight(int height) {
			this.height = height;
		}
    }
    //属性
    public static class Attributes{
        private Emotion emotion;
        private Beauty beauty;
        private Gender gender;
        private Age age;
        private Mouthstatus mouthstatus;
        private Glass glass;
        private Skinstatus skinstatus;
        private Headpose headpose;
        private Blur blur;
        private Smile smile;
        private Eyestatus eyestatus;
        private Facequality facequality;
        private Ethnicity ethnicity;
        private Eyegaze eyegaze;
		public Emotion getEmotion() {
			return emotion;
		}
		public void setEmotion(Emotion emotion) {
			this.emotion = emotion;
		}
		public Beauty getBeauty() {
			return beauty;
		}
		public void setBeauty(Beauty beauty) {
			this.beauty = beauty;
		}
		public Gender getGender() {
			return gender;
		}
		public void setGender(Gender gender) {
			this.gender = gender;
		}
		public Age getAge() {
			return age;
		}
		public void setAge(Age age) {
			this.age = age;
		}
		public Mouthstatus getMouthstatus() {
			return mouthstatus;
		}
		public void setMouthstatus(Mouthstatus mouthstatus) {
			this.mouthstatus = mouthstatus;
		}
		public Glass getGlass() {
			return glass;
		}
		public void setGlass(Glass glass) {
			this.glass = glass;
		}
		public Skinstatus getSkinstatus() {
			return skinstatus;
		}
		public void setSkinstatus(Skinstatus skinstatus) {
			this.skinstatus = skinstatus;
		}
		public Headpose getHeadpose() {
			return headpose;
		}
		public void setHeadpose(Headpose headpose) {
			this.headpose = headpose;
		}
		public Blur getBlur() {
			return blur;
		}
		public void setBlur(Blur blur) {
			this.blur = blur;
		}
		public Smile getSmile() {
			return smile;
		}
		public void setSmile(Smile smile) {
			this.smile = smile;
		}
		public Eyestatus getEyestatus() {
			return eyestatus;
		}
		public void setEyestatus(Eyestatus eyestatus) {
			this.eyestatus = eyestatus;
		}
		public Facequality getFacequality() {
			return facequality;
		}
		public void setFacequality(Facequality facequality) {
			this.facequality = facequality;
		}
		public Ethnicity getEthnicity() {
			return ethnicity;
		}
		public void setEthnicity(Ethnicity ethnicity) {
			this.ethnicity = ethnicity;
		}
		public Eyegaze getEyegaze() {
			return eyegaze;
		}
		public void setEyegaze(Eyegaze eyegaze) {
			this.eyegaze = eyegaze;
		}
    }
    //情绪
    public static class Emotion{
        private double sadness;
        private double neutral;
        private double disgust;
        private double anger;
        private double surprise;
        private double fear;
        private double happiness;
		public double getSadness() {
			return sadness;
		}
		public void setSadness(double sadness) {
			this.sadness = sadness;
		}
		public double getNeutral() {
			return neutral;
		}
		public void setNeutral(double neutral) {
			this.neutral = neutral;
		}
		public double getDisgust() {
			return disgust;
		}
		public void setDisgust(double disgust) {
			this.disgust = disgust;
		}
		public double getAnger() {
			return anger;
		}
		public void setAnger(double anger) {
			this.anger = anger;
		}
		public double getSurprise() {
			return surprise;
		}
		public void setSurprise(double surprise) {
			this.surprise = surprise;
		}
		public double getFear() {
			return fear;
		}
		public void setFear(double fear) {
			this.fear = fear;
		}
		public double getHappiness() {
			return happiness;
		}
		public void setHappiness(double happiness) {
			this.happiness = happiness;
		} 
    }
    //颜值
    public static class Beauty{
        private double female_score;
        private double male_score;
		public double getFemale_score() {
			return female_score;
		}
		public void setFemale_score(double female_score) {
			this.female_score = female_score;
		}
		public double getMale_score() {
			return male_score;
		}
		public void setMale_score(double male_score) {
			this.male_score = male_score;
		}    
    }
    //性别
    public static class Gender{
        private String value;

		public String getValue() {
			return value;
		}

		public void setValue(String value) {
			this.value = value;
		}    	
    }
    //年龄
    public static class Age{
        private int value;

		public int getValue() {
			return value;
		}

		public void setValue(int value) {
			this.value = value;
		}    
    }
    //嘴部状态
    public static class Mouthstatus{
        private double close;
        private double surgical_mask_or_respirator;
        private double open;
        private double other_occlusion;
		public double getClose() {
			return close;
		}
		public void setClose(double close) {
			this.close = close;
		}
		public double getSurgical_mask_or_respirator() {
			return surgical_mask_or_respirator;
		}
		public void setSurgical_mask_or_respirator(double surgical_mask_or_respirator) {
			this.surgical_mask_or_respirator = surgical_mask_or_respirator;
		}
		public double getOpen() {
			return open;
		}
		public void setOpen(double open) {
			this.open = open;
		}
		public double getOther_occlusion() {
			return other_occlusion;
		}
		public void setOther_occlusion(double other_occlusion) {
			this.other_occlusion = other_occlusion;
		}   
    }
    //眼镜
    public static class Glass{
        private String value;

		public String getValue() {
			return value;
		}

		public void setValue(String value) {
			this.value = value;
		}    	
    }
    //面部特征
    public static class Skinstatus{
        private double dark_circle;
        private double stain;
        private double acne;
        private double health;
		public double getDark_circle() {
			return dark_circle;
		}
		public void setDark_circle(double dark_circle) {
			this.dark_circle = dark_circle;
		}
		public double getStain() {
			return stain;
		}
		public void setStain(double stain) {
			this.stain = stain;
		}
		public double getAcne() {
			return acne;
		}
		public void setAcne(double acne) {
			this.acne = acne;
		}
		public double getHealth() {
			return health;
		}
		public void setHealth(double health) {
			this.health = health;
		}    	
    }
    //人脸姿势
    public static class Headpose{
        private double yaw_angle;
        private double pitch_angle;
        private double roll_angle;
		public double getYaw_angle() {
			return yaw_angle;
		}
		public void setYaw_angle(double yaw_angle) {
			this.yaw_angle = yaw_angle;
		}
		public double getPitch_angle() {
			return pitch_angle;
		}
		public void setPitch_angle(double pitch_angle) {
			this.pitch_angle = pitch_angle;
		}
		public double getRoll_angle() {
			return roll_angle;
		}
		public void setRoll_angle(double roll_angle) {
			this.roll_angle = roll_angle;
		}    
    }
    //人脸模糊
    public static class Blur{
        private Blurness blurness;
        private Motionblur motionblur;
        private Gaussianblur gaussianblur;
		public Blurness getBlurness() {
			return blurness;
		}
		public void setBlurness(Blurness blurness) {
			this.blurness = blurness;
		}
		public Motionblur getMotionblur() {
			return motionblur;
		}
		public void setMotionblur(Motionblur motionblur) {
			this.motionblur = motionblur;
		}
		public Gaussianblur getGaussianblur() {
			return gaussianblur;
		}
		public void setGaussianblur(Gaussianblur gaussianblur) {
			this.gaussianblur = gaussianblur;
		}    
    }
    public static class Blurness{
        private int threshold;
        private double value;
		public int getThreshold() {
			return threshold;
		}
		public void setThreshold(int threshold) {
			this.threshold = threshold;
		}
		public double getValue() {
			return value;
		}
		public void setValue(double value) {
			this.value = value;
		}    
    }
    public static class Motionblur{
        private int threshold;
        private double value;
		public int getThreshold() {
			return threshold;
		}
		public void setThreshold(int threshold) {
			this.threshold = threshold;
		}
		public double getValue() {
			return value;
		}
		public void setValue(double value) {
			this.value = value;
		}    
    }
    public static class Gaussianblur{
        private int threshold;
        private double value;
		public int getThreshold() {
			return threshold;
		}
		public void setThreshold(int threshold) {
			this.threshold = threshold;
		}
		public double getValue() {
			return value;
		}
		public void setValue(double value) {
			this.value = value;
		}    
    }
    //笑容
    public static class Smile{
        private int threshold;
        private double value;
		public int getThreshold() {
			return threshold;
		}
		public void setThreshold(int threshold) {
			this.threshold = threshold;
		}
		public double getValue() {
			return value;
		}
		public void setValue(double value) {
			this.value = value;
		}  
    }
    //眼睛状态
    public static class Eyestatus{
        private Left_eye_status left_eye_status;
        private Right_eye_status right_eye_status;
		public Left_eye_status getLeft_eye_status() {
			return left_eye_status;
		}
		public void setLeft_eye_status(Left_eye_status left_eye_status) {
			this.left_eye_status = left_eye_status;
		}
		public Right_eye_status getRight_eye_status() {
			return right_eye_status;
		}
		public void setRight_eye_status(Right_eye_status right_eye_status) {
			this.right_eye_status = right_eye_status;
		}  
    }
    public static class Left_eye_status{
        private double normal_glass_eye_open;
        private double no_glass_eye_close;
        private double occlusion;
        private double no_glass_eye_open;
        private int normal_glass_eye_close;
        private double dark_glasses;
		public double getNormal_glass_eye_open() {
			return normal_glass_eye_open;
		}
		public void setNormal_glass_eye_open(double normal_glass_eye_open) {
			this.normal_glass_eye_open = normal_glass_eye_open;
		}
		public double getNo_glass_eye_close() {
			return no_glass_eye_close;
		}
		public void setNo_glass_eye_close(double no_glass_eye_close) {
			this.no_glass_eye_close = no_glass_eye_close;
		}
		public double getOcclusion() {
			return occlusion;
		}
		public void setOcclusion(double occlusion) {
			this.occlusion = occlusion;
		}
		public double getNo_glass_eye_open() {
			return no_glass_eye_open;
		}
		public void setNo_glass_eye_open(double no_glass_eye_open) {
			this.no_glass_eye_open = no_glass_eye_open;
		}
		public int getNormal_glass_eye_close() {
			return normal_glass_eye_close;
		}
		public void setNormal_glass_eye_close(int normal_glass_eye_close) {
			this.normal_glass_eye_close = normal_glass_eye_close;
		}
		public double getDark_glasses() {
			return dark_glasses;
		}
		public void setDark_glasses(double dark_glasses) {
			this.dark_glasses = dark_glasses;
		}    	
    }
    public static class Right_eye_status{
        private double normal_glass_eye_open;
        private int no_glass_eye_close;
        private int occlusion;
        private double no_glass_eye_open;
        private int normal_glass_eye_close;
        private int dark_glasses;
		public double getNormal_glass_eye_open() {
			return normal_glass_eye_open;
		}
		public void setNormal_glass_eye_open(double normal_glass_eye_open) {
			this.normal_glass_eye_open = normal_glass_eye_open;
		}
		public int getNo_glass_eye_close() {
			return no_glass_eye_close;
		}
		public void setNo_glass_eye_close(int no_glass_eye_close) {
			this.no_glass_eye_close = no_glass_eye_close;
		}
		public int getOcclusion() {
			return occlusion;
		}
		public void setOcclusion(int occlusion) {
			this.occlusion = occlusion;
		}
		public double getNo_glass_eye_open() {
			return no_glass_eye_open;
		}
		public void setNo_glass_eye_open(double no_glass_eye_open) {
			this.no_glass_eye_open = no_glass_eye_open;
		}
		public int getNormal_glass_eye_close() {
			return normal_glass_eye_close;
		}
		public void setNormal_glass_eye_close(int normal_glass_eye_close) {
			this.normal_glass_eye_close = normal_glass_eye_close;
		}
		public int getDark_glasses() {
			return dark_glasses;
		}
		public void setDark_glasses(int dark_glasses) {
			this.dark_glasses = dark_glasses;
		}   
    }
    //人脸质量
    public static class Facequality{
        private double threshold;
        private double value;
		public double getThreshold() {
			return threshold;
		}
		public void setThreshold(double threshold) {
			this.threshold = threshold;
		}
		public double getValue() {
			return value;
		}
		public void setValue(double value) {
			this.value = value;
		}    
    }
    //人种分析
    public static class Ethnicity{
        private String value;

		public String getValue() {
			return value;
		}

		public void setValue(String value) {
			this.value = value;
		}   
    }
    //眼球位置与视线方向
    public static class Eyegaze{
        private Right_eye_gaze right_eye_gaze;
        private Left_eye_gaze left_eye_gaze;
		public Right_eye_gaze getRight_eye_gaze() {
			return right_eye_gaze;
		}
		public void setRight_eye_gaze(Right_eye_gaze right_eye_gaze) {
			this.right_eye_gaze = right_eye_gaze;
		}
		public Left_eye_gaze getLeft_eye_gaze() {
			return left_eye_gaze;
		}
		public void setLeft_eye_gaze(Left_eye_gaze left_eye_gaze) {
			this.left_eye_gaze = left_eye_gaze;
		}    	
    }
    public static class Right_eye_gaze{
        private double position_x_coordinate;
        private double vector_z_component;
        private double vector_x_component;
        private double vector_y_component;
        private double position_y_coordinate;
		public double getPosition_x_coordinate() {
			return position_x_coordinate;
		}
		public void setPosition_x_coordinate(double position_x_coordinate) {
			this.position_x_coordinate = position_x_coordinate;
		}
		public double getVector_z_component() {
			return vector_z_component;
		}
		public void setVector_z_component(double vector_z_component) {
			this.vector_z_component = vector_z_component;
		}
		public double getVector_x_component() {
			return vector_x_component;
		}
		public void setVector_x_component(double vector_x_component) {
			this.vector_x_component = vector_x_component;
		}
		public double getVector_y_component() {
			return vector_y_component;
		}
		public void setVector_y_component(double vector_y_component) {
			this.vector_y_component = vector_y_component;
		}
		public double getPosition_y_coordinate() {
			return position_y_coordinate;
		}
		public void setPosition_y_coordinate(double position_y_coordinate) {
			this.position_y_coordinate = position_y_coordinate;
		}  
    }
    public static class Left_eye_gaze{
        private double position_x_coordinate;
        private double vector_z_component;
        private double vector_x_component;
        private double vector_y_component;
        private double position_y_coordinate;
		public double getPosition_x_coordinate() {
			return position_x_coordinate;
		}
		public void setPosition_x_coordinate(double position_x_coordinate) {
			this.position_x_coordinate = position_x_coordinate;
		}
		public double getVector_z_component() {
			return vector_z_component;
		}
		public void setVector_z_component(double vector_z_component) {
			this.vector_z_component = vector_z_component;
		}
		public double getVector_x_component() {
			return vector_x_component;
		}
		public void setVector_x_component(double vector_x_component) {
			this.vector_x_component = vector_x_component;
		}
		public double getVector_y_component() {
			return vector_y_component;
		}
		public void setVector_y_component(double vector_y_component) {
			this.vector_y_component = vector_y_component;
		}
		public double getPosition_y_coordinate() {
			return position_y_coordinate;
		}
		public void setPosition_y_coordinate(double position_y_coordinate) {
			this.position_y_coordinate = position_y_coordinate;
		}
    }
}
