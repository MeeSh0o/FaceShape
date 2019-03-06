#include "pch.h"
#include <iostream>
#include <Python.h>
#include <direct.h>
//#include <windows.data.json.h>
#include <opencv2\opencv.hpp>
#include <string>
#include <json\json.h>

using namespace std;
using namespace cv;


string path = "F:\\Graduation Project\\SoftWare\\FaceShape_Tencent\\FaceShape";
string ImagePath = "F:\\Graduation Project\\SoftWare\\FaceModuel\\FaceModuel\\Images\\";
string chdir_cmd = string("sys.path.append(\"") + path + "\")";
const char* cstr_cmd = chdir_cmd.c_str();

class Pnt {
public:
	int x;
	int y;
};
class FaceShapeList {
public:
	Pnt face_profile[21];
	Pnt left_eye[8];
	Pnt left_eyebrow[8];
	Pnt right_eye[8];
	Pnt mouth[22];
	Pnt nose[13];
	Pnt right_eyebrow[8];
};
class Data {
public:
	int image_height;
	int image_width;
	FaceShapeList face_shape_list[1];
};
class FaceShapeData {
public:
	string msg;
	Data data;
	int ret;
};

int Json_ReadInt(Json::Value JV, int ori_value = 0);
double Json_ReadDouble(Json::Value JV, double ori_value = 0.0);
string Json_ReadString(Json::Value JV, string ori_value = "");
string FaceShape_Python(char* image);
FaceShapeData Json2Class(string data);
void ShowImage(string path,FaceShapeData data);

int main()
{
	string fileName; // 文件名
	cout << "输入文件名：" << endl;
	cin >> fileName;

	string fullPath = ImagePath + fileName;

	const char* path = fullPath.c_str();
	char* path_char = new char[99];
	path_char = const_cast<char*>(path);

	string data_str = FaceShape_Python(path_char); // str格式的json数据

	FaceShapeData faceShapeData = Json2Class(data_str); // FaceShapeData数据

	ShowImage(path_char, faceShapeData); // 下一步：ShowImage，传入faceShapeData，输出图像并标点
}

// 实现特征点标定的python方法，需传入.JPG图片完整路径
string FaceShape_Python(char* image)
{
	// 初始化Python
	Py_Initialize();
	//if (!Py_IsInitialized)
	//{
	//	Py_Finalize();
	//	return;
	//}

	// 加载目录
	PyRun_SimpleString("import sys");
	PyRun_SimpleString(cstr_cmd);

	// 加载模块
	PyObject* moduleName = PyString_FromString("FaceShape");
	PyObject* pModule = PyImport_Import(moduleName);
	if (!pModule) {
		cout << "加载人脸特征标定Python模块失败！\n";
		PyErr_Print();
		Py_Finalize(); // 释放资源
		return NULL;
	}
	cout << "加载人脸特征标定Python模块成功！\n";

	// 加载方法
	PyObject* pv = PyObject_GetAttrString(pModule, "GetFaceShapeJson");
	if (!pv || !PyCallable_Check(pv)) {
		cout << "加载人脸特征标定Python方法失败！\n";
		PyErr_Print();
		Py_Finalize(); // 释放资源
		return NULL;
	}
	cout << "加载人脸特征标定Python方法成功！\n";

	// 调用方法
	char s_char[] = "s";
	char* s = s_char;
	PyObject *pValue = PyObject_CallFunction(pv, s, image);

	//处理数据
	string callBack3 = "空的";
	callBack3 = PyString_AsString(pValue);
	//cout << callBack3 << endl;
	// 释放资源
	Py_Finalize();;
	return callBack3;
};

// 将string格式的json数据转换为class
FaceShapeData Json2Class(string data)
{
	//char jsonData[3000];
	//int i;
	//for (i = 0; i < data.length(); i++) {
	//	jsonData[i] = data[i];
	//}
	//jsonData[i] = '\0';

	FaceShapeData faceShapeData;

	Json::CharReaderBuilder builder;
	Json::CharReader* JsonReader(builder.newCharReader());
	Json::Value faceShapeData_json; // 数据根节点，对应结构体FaceShapeData
	Json::Value ObjectTmp;
	JSONCPP_STRING errs;
	const char* pstr = data.c_str();
	if (!JsonReader->parse(pstr, pstr + strlen(pstr), &faceShapeData_json, &errs))
	{
		cout << "ERROR" << endl;
		return faceShapeData;
	}
	else cout << "READ JSON SUCC" << endl;

	faceShapeData.msg = Json_ReadString(faceShapeData_json["msg"]);
	faceShapeData.ret = Json_ReadInt(faceShapeData_json["ret"]);
	if (faceShapeData.ret != 0) {
		cout << "人脸识别失败！" << endl;
		return faceShapeData;
	}
	else cout << "faceShapeData.msg = " << faceShapeData.msg << endl;
	// 写入json数据根节点

	Json::Value data_josn = faceShapeData_json["data"]; // data对象
	faceShapeData.data.image_height = Json_ReadInt(data_josn["image_height"]);
	faceShapeData.data.image_width = Json_ReadInt(data_josn["image_width"]);
	Json::Value face_shape_list = data_josn["face_shape_list"]; // data对象
	// 写入json数据data节点

	face_shape_list[0]; // 暂存一个引用

	faceShapeData.data.face_shape_list[0]; // 暂存一个引用

	Json::Value face_profile_json = face_shape_list[0]["face_profile"];
	for (size_t i = 0; i < face_profile_json.size(); i++) {
		faceShapeData.data.face_shape_list[0].face_profile[i].x
			= Json_ReadInt(face_profile_json[(Json::ArrayIndex)i]["x"]);
		faceShapeData.data.face_shape_list[0].face_profile[i].y
			= Json_ReadInt(face_profile_json[(Json::ArrayIndex)i]["y"]);
		//cout << "face_profile" << faceShapeData.data.face_shape_list[0].face_profile[i].x << faceShapeData.data.face_shape_list[0].face_profile[i].y << endl;
	}

	Json::Value left_eye_json = face_shape_list[0]["left_eye"];
	for (size_t i = 0; i < left_eye_json.size(); i++) {
		faceShapeData.data.face_shape_list[0].left_eye[i].x
			= Json_ReadInt(left_eye_json[(Json::ArrayIndex)i]["x"]);
		faceShapeData.data.face_shape_list[0].left_eye[i].y
			= Json_ReadInt(left_eye_json[(Json::ArrayIndex)i]["y"]);
		//cout << "left_eye" << faceShapeData.data.face_shape_list[0].left_eye[i].x << faceShapeData.data.face_shape_list[0].left_eye[i].y << endl;
	}

	Json::Value left_eyebrow_json = face_shape_list[0]["left_eyebrow"];
	for (size_t i = 0; i < left_eyebrow_json.size(); i++) {
		faceShapeData.data.face_shape_list[0].left_eyebrow[i].x
			= Json_ReadInt(left_eyebrow_json[(Json::ArrayIndex)i]["x"]);
		faceShapeData.data.face_shape_list[0].left_eyebrow[i].y
			= Json_ReadInt(left_eyebrow_json[(Json::ArrayIndex)i]["y"]);
		//cout << "left_eyebrow" << faceShapeData.data.face_shape_list[0].left_eyebrow[i].x << faceShapeData.data.face_shape_list[0].left_eyebrow[i].y << endl;
	}

	Json::Value right_eye_json = face_shape_list[0]["right_eye"];
	for (size_t i = 0; i < right_eye_json.size(); i++) {
		faceShapeData.data.face_shape_list[0].right_eye[i].x
			= Json_ReadInt(right_eye_json[(Json::ArrayIndex)i]["x"]);
		faceShapeData.data.face_shape_list[0].right_eye[i].y
			= Json_ReadInt(right_eye_json[(Json::ArrayIndex)i]["y"]);
		//cout << "right_eye" << faceShapeData.data.face_shape_list[0].right_eye[i].x << faceShapeData.data.face_shape_list[0].right_eye[i].y << endl;
	}

	Json::Value mouth_json = face_shape_list[0]["mouth"];
	for (size_t i = 0; i < mouth_json.size(); i++) {
		faceShapeData.data.face_shape_list[0].mouth[i].x
			= Json_ReadInt(mouth_json[(Json::ArrayIndex)i]["x"]);
		faceShapeData.data.face_shape_list[0].mouth[i].y
			= Json_ReadInt(mouth_json[(Json::ArrayIndex)i]["y"]);
		//cout << "mouth" << faceShapeData.data.face_shape_list[0].mouth[i].x << faceShapeData.data.face_shape_list[0].mouth[i].y << endl;
	}

	Json::Value nose_json = face_shape_list[0]["nose"];
	for (size_t i = 0; i < nose_json.size(); i++) {
		faceShapeData.data.face_shape_list[0].nose[i].x
			= Json_ReadInt(nose_json[(Json::ArrayIndex)i]["x"]);
		faceShapeData.data.face_shape_list[0].nose[i].y
			= Json_ReadInt(nose_json[(Json::ArrayIndex)i]["y"]);
		//cout << "nose" << faceShapeData.data.face_shape_list[0].nose[i].x << faceShapeData.data.face_shape_list[0].nose[i].y << endl;
	}

	Json::Value right_eyebrow_json = face_shape_list[0]["right_eyebrow"];
	for (size_t i = 0; i < right_eyebrow_json.size(); i++) {
		faceShapeData.data.face_shape_list[0].right_eyebrow[i].x
			= Json_ReadInt(right_eyebrow_json[(Json::ArrayIndex)i]["x"]);
		faceShapeData.data.face_shape_list[0].right_eyebrow[i].y
			= Json_ReadInt(right_eyebrow_json[(Json::ArrayIndex)i]["y"]);
		//cout << "right_eyebrow" << faceShapeData.data.face_shape_list[0].right_eyebrow[i].x << faceShapeData.data.face_shape_list[0].right_eyebrow[i].y << endl;
	}
	// Json数据写入Class完成！

	return faceShapeData;
}

// 绘图
void ShowImage(string path,FaceShapeData data)
{
	Mat read = imread(path);//实例化一个Mat对象,这里使用的是局部路径，图片和工程文件放在同一目录（不是输出文件的目录）
	assert(read.data);//如果数据为空就终止执行

	Scalar pointColor = Scalar(0, 255, 0);

	for (int i = 0; i < 21; i++) {
		Point p;
		p.x = data.data.face_shape_list[0].face_profile[i].x;
		p.y = data.data.face_shape_list[0].face_profile[i].y;
		circle(read, p, 3, pointColor, -1);
	}
	for (int i = 0; i < 8; i++) {
		Point p;
		p.x = data.data.face_shape_list[0].left_eye[i].x;
		p.y = data.data.face_shape_list[0].left_eye[i].y;
		circle(read, p, 3, pointColor, -1);
	}
	for (int i = 0; i < 8; i++) {
		Point p;
		p.x = data.data.face_shape_list[0].left_eyebrow[i].x;
		p.y = data.data.face_shape_list[0].left_eyebrow[i].y;
		circle(read, p, 3, pointColor, -1);
	}
	for (int i = 0; i < 8; i++) {
		Point p;
		p.x = data.data.face_shape_list[0].right_eye[i].x;
		p.y = data.data.face_shape_list[0].right_eye[i].y;
		circle(read, p, 3, pointColor, -1);
	}
	for (int i = 0; i < 8; i++) {
		Point p;
		p.x = data.data.face_shape_list[0].right_eyebrow[i].x;
		p.y = data.data.face_shape_list[0].right_eyebrow[i].y;
		circle(read, p, 3, pointColor, -1);
	}
	for (int i = 0; i < 22; i++) {
		Point p;
		p.x = data.data.face_shape_list[0].mouth[i].x;
		p.y = data.data.face_shape_list[0].mouth[i].y;
		circle(read, p, 3, pointColor, -1);
	}
	for (int i = 0; i < 13; i++) {
		Point p;
		p.x = data.data.face_shape_list[0].nose[i].x;
		p.y = data.data.face_shape_list[0].nose[i].y;
		circle(read, p, 3, pointColor, -1);
	}

	namedWindow("FACE", WINDOW_NORMAL);//建立一个窗口，大小自适应图片：WINDOW_AUTOSIZE  大小可拖动：WINDOW_NORMAL	
	imshow("FACE", read);
	waitKey(0);
	return;
}

int Json_ReadInt(Json::Value JV, int ori_value)
{
	int result = ori_value;
	Json::ValueType VT = JV.type();
	if (VT == Json::ValueType::intValue)
		result = JV.asInt();
	return result;
}
double Json_ReadDouble(Json::Value JV, double ori_value)
{
	double result = ori_value;
	Json::ValueType VT = JV.type();
	if (VT == Json::ValueType::realValue)
		result = JV.asDouble();
	return result;
}
string Json_ReadString(Json::Value JV, string ori_value)
{
	string result = ori_value;
	Json::ValueType VT = JV.type();
	if (VT == Json::ValueType::stringValue)
		result = JV.asCString();
	return result;
}