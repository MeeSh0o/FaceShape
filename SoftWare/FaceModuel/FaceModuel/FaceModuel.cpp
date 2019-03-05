#include "pch.h"
#include <iostream>
#include <Python.h>
#include <direct.h>
#include <windows.data.json.h>
#include <opencv2\opencv.hpp>
#include <string>

using namespace std;
using namespace cv;


string path = "F:\\Graduation Project\\SoftWare\\FaceShape_Tencent\\FaceShape";
string ImagePath = "F:\\Graduation Project\\SoftWare\\FaceModuel\\FaceModuel\\Images\\";
string chdir_cmd = string("sys.path.append(\"") + path + "\")";
const char* cstr_cmd = chdir_cmd.c_str();

typedef struct
{
	int x;
	int y;
} Pnt;

typedef struct {
	Pnt face_profile[21];
	Pnt left_eye[8];
	Pnt left_eyebrow[8];
	Pnt right_eye[8];
	Pnt mouth[22];
	Pnt nose[13];
	Pnt right_eyebrow[8];
} FaceShapeList;

typedef struct {
	int image_height;
	int image_width;
	FaceShapeList face_shape_list[1];
} Data;

typedef struct {
	char msg[100];
	Data data;
	char ret[100];
} FaceShapeData;


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
	printf("%s\n", callBack3.c_str());
	// 释放资源
	Py_Finalize();
	return callBack3;
};

string RunFaceShape(string name) {
	const char* path = name.c_str();
	char* path_char = new char[99];
	path_char = const_cast<char*>(path);
	return FaceShape_Python(path_char);
}

// 绘图
void ShowImage(string name) {
	Mat read = imread(name);//实例化一个Mat对象,这里使用的是局部路径，图片和工程文件放在同一目录（不是输出文件的目录）
	assert(read.data);//如果数据为空就终止执行

	Point p2;
	p2.x = 100;
	p2.y = 100;
	//画实心点
	circle(read, p2, 3, Scalar(0, 0, 255), -1); //第五个参数我设为-1，表明这是个实点。


	namedWindow("FACE", WINDOW_NORMAL);//建立一个窗口，大小自适应图片：WINDOW_AUTOSIZE  大小可拖动：WINDOW_NORMAL	
	imshow("FACE", read);
	waitKey(0);
	return;
}


void Json2FSD(string data)
{
	char jsonData[3000];
	int i;
	for (i = 0; i < data.length(); i++) {
		jsonData[i] = data[i];
	}
	jsonData[i] = '\0';

}

int main()
{
	string fileName;

	cout << "输入文件名：\n";
	cin >> fileName;
	//ShowImage(ImagePath + fileName);
	Json2FSD(RunFaceShape(ImagePath + fileName));
}