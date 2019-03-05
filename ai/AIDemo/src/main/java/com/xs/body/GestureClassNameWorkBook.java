package com.xs.body;
/**
 * 手势中文说明
 * @author 小帅丶
 *
 */
public class GestureClassNameWorkBook {
	/**
	 * 根据classname返回中文对应的名称
	 * @param classname 手势所属类别，15种手势、other、face
	 * @return String
	 * @throws Exception
	 */
	public static String getGestureName(String classname) throws Exception {
		String gestureName ="";
		if(classname.equals("Thumb_down")){
			gestureName="踩|拇指向下";
			return gestureName;
		}else if (classname.equals("Ok")) {
			gestureName="OK";
			return gestureName;
		}else if (classname.equals("ILY")) {
			gestureName="我爱你";
			return gestureName;
		}else if (classname.equals("Heart_single")) {
			gestureName="单手比心";
			return gestureName;
		}else if (classname.equals("Thumb_up")) {
			gestureName="点赞|拇指向上";
			return gestureName;
		}else if (classname.equals("Prayer")) {
			gestureName="祈祷";
			return gestureName;
		}else if (classname.equals("Fist")) {
			gestureName="拳头";
			return gestureName;
		}else if (classname.equals("Point")) {
			gestureName="点|食指";
			return gestureName;
		}else if (classname.equals("Heart_1")||classname.equals("Heart_2")||classname.equals("Heart_3")) {
			gestureName="双手比心";
			return gestureName;
		}else if (classname.equals("Palm")) {
			gestureName="掌心向前";
			return gestureName;
		}else if (classname.equals("Palm_up")) {
			gestureName="掌心向上";
			return gestureName;
		}else if (classname.equals("Honour")) {
			gestureName="作别|告别";
			return gestureName;
		}else if (classname.equals("Rock")) {
			gestureName="摇滚";
			return gestureName;
		}else if (classname.equals("Congratulation")) {
			gestureName="作揖|祝贺";
			return gestureName;
		}else{
			gestureName ="未能匹配的手势:"+classname;
		    return gestureName;
		}
	}

}
