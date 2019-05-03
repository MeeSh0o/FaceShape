#include "pch.h"
#include <iostream>
#include <fstream>
#include <Python.h>
#include <direct.h>
#include <opencv2\opencv.hpp>
#include <string>
#include <json\json.h>
#include <OBJ_Loader.h>
#include <GL/glut.h>
#include <Eigen/Dense>


#include <vector>
#include <cmath>
#include <cstdio>
#include <cstring>

using namespace std;
using namespace cv;
using namespace Eigen;

/*数据结构声明*/

class FaceShapeList {
public:
	objl::Vector2 face_profile[21];
	objl::Vector2 left_eye[8];
	objl::Vector2 left_eyebrow[8];
	objl::Vector2 right_eye[8];
	objl::Vector2 mouth[22];
	objl::Vector2 nose[13];
	objl::Vector2 right_eyebrow[8];
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
// 储存模型坐标点和特征点及其索引
class VertexF
{
public:
	// 构造函数
	VertexF();
	// 带参数的构造函数
	VertexF(objl::Vector3 _Position, unsigned int _Indice, objl::Vector2 _Feature, unsigned int _FIndice);
	~VertexF();
	// 点坐标
	objl::Vector3 Position;
	// 点在模型中的索引号
	unsigned int Indice;
	// 特征点到鼻尖点的正面位移量
	objl::Vector2 Feature;
	// 特征点在特征数据中的索引号
	unsigned int FIndice;

};
class FeaturePoint
{
public:
	objl::Vector2 Position;
	unsigned int FIndice, Indice;
	double X, Y;
	FeaturePoint();
	FeaturePoint(objl::Vector2 _Position, unsigned int _Indice, unsigned int _FIndice);
	FeaturePoint(double x, double y, unsigned int _Indice, unsigned int _FIndice);
	~FeaturePoint();

};

/*函数声明*/

// 读取json的int格式数据
int Json_ReadInt(Json::Value JV, int ori_value = 0);
// 读取json的double格式数据
double Json_ReadDouble(Json::Value JV, double ori_value = 0.0);
// 读取json的string格式数据
string Json_ReadString(Json::Value JV, string ori_value = "");
// 实现特征点标定的python方法，需传入.JPG图片完整路径
std::string FaceShape_Python(char* image);
// 将string格式的json数据转换为class
FaceShapeData Json2Class(string data);
// 绘图
void ShowImage(string path, FaceShapeData data);
// 读取Obj格式模型演示方法
void LoadObjTest(string path, string out);
// 读取Obj格式模型方法及其算法调用
void LoadObj(string path);
// 标准模型特征点初始化
void FeatureVerticlesInitiate(std::vector <objl::Vector3> positions);
// 目标特征点顶点集
std::vector<VertexF> FeatureOut(FaceShapeData faceShapeData);
// 径向基函数2D
std::vector <objl::Vector2> RadialBasisFunction2D(std::vector<VertexF> FeatureData);
// 解线性方程组2D
MatrixXd Calc_tps2D(std::vector< VertexF > control_points, double regularization = 0.0);
// 输出OBj文件
void WriteObj(std::vector <objl::Vector3> OutModelPositions, FaceShapeData faceShapeData,string path, string name);
void WriteObj(std::vector<objl::Vector2> OutModelPositions);
// 输出mtl文件
void WriteMtl(string path, string name);
// 薄样板条函数
static double thin_plate_splines(double r);
// 对齐函数
vector<objl::Vector3> Align(std::vector <objl::Vector2> Q_, FaceShapeData faceShapeData);
// 分割字符串
void SplitString(const string& s, vector<string>& v, const string& c);
// 计算法向量
std::vector <objl::Vector3> CaculateNormal(std::vector <objl::Vector3> P);

/*变量声明*/

// 标准模型的特征点顶点数据
VertexF Face_profile_Indices[15];
VertexF Left_eye_Indices[4];
VertexF Right_eye_Indices[4];
VertexF Left_eyebrow_Indices[7];
VertexF Right_eyebrow_Indices[7];
VertexF Mouth_Indices[14];
VertexF Nose_Indices[13];
std::vector<VertexF> F;


// 标准模型的点坐标
std::vector <objl::Vector3> P;
//// 目标模型的点坐标中间变量
//std::vector <objl::Vector3> OutModelPositions;
// 标准模型的面
std::vector <objl::Vector3> StandardModelFaces;

//int face_profile_Indices[30]
//{
//	817,0,
//	827,2,
//	805,4,
//	789,5,
//	783,6,
//	139,8,
//	140,9,
//	51,10,
//	23,11,
//	13,12,
//	2,14,
//	777,15,
//	455,16,
//	629,18,
//	622,20
//};
FeaturePoint face_profile_Indices[15]{
	FeaturePoint(-74.1267,31.5881,358,0),
	FeaturePoint(-71.4071,1.17975,365,2),
	FeaturePoint(-64.8084,-28.1877,388,4),
	FeaturePoint(-59.0352,-41.4368,352,5),
	FeaturePoint(-50.2794,-54.9622,1,6),
	FeaturePoint(-28.0597,-73.9578,21,8),
	FeaturePoint(-16.407,-79.322,20,9),
	FeaturePoint(0,-82.787,23,10),
	FeaturePoint(16.407,-79.322,463,11),
	FeaturePoint(28.0597,-73.9578,464,12),
	FeaturePoint(50.2794,-54.9622,445,14),
	FeaturePoint(59.0352,-41.4368,769,15),
	FeaturePoint(64.8084,-28.1877,802,16),
	FeaturePoint(71.4071,1.17975,782,18),
	FeaturePoint(74.1267,31.5881,775,20)
};
//int left_eye_Indices[8]
//{
//	292,0,
//	277,2,
//	271,4,
//	287,6
//};
FeaturePoint left_eye_Indices[4]{
	FeaturePoint(-45.7558,35.2404,177,0),
	FeaturePoint(-34.5137,30.7566,174,2),
	FeaturePoint(-18.8728,31.4413,181,4),
	FeaturePoint(-34.9494,37.7824,172,6)
};
//int right_eye_Indices[8]
//{
//	651,0,
//	672,2,
//	727,4,
//	665,6
//};
FeaturePoint right_eye_Indices[4]{
	FeaturePoint(45.7558,35.2404,610,0),
	FeaturePoint(34.5137,30.7566,607,2),
	FeaturePoint(18.8728,31.4413,614,4),
	FeaturePoint(34.9494,37.7824,605,6)
};
//int left_eyebrow_Indices[14]
//{
//	276,0,
//	237,1,
//	303,3,
//	305,4,
//	313,5,
//	228,6,
//	239,7
//};
FeaturePoint left_eyebrow_Indices[7]{
	FeaturePoint(-55.9635,45.8081,225,0),
	FeaturePoint(-41.1773,48.1978,230,1),
	FeaturePoint(-18.7694,46.2841,160,3),
	FeaturePoint(-9.6074,43.7701,156,4),
	FeaturePoint(-19.7024,52.5089,241,5),
	FeaturePoint(-29.4305,55.0616,235,6),
	FeaturePoint(-43.153,53.4884,231,7)
};
//int right_eyebrow_Indices[14]
//{
//	732,0,
//	835,1,
//	636,3,
//	632,4,
//	642,5,
//	643,6,
//	836,7
//};
FeaturePoint right_eyebrow_Indices[7]{
	FeaturePoint(55.9635,45.8081,658,0),
	FeaturePoint(41.1773,48.1978,663,1),
	FeaturePoint(18.7694,46.2841,593,3),
	FeaturePoint(9.6074,43.7701,589,4),
	FeaturePoint(19.7024,52.5089,674,5),
	FeaturePoint(29.4305,55.0616,668,6),
	FeaturePoint(43.153,53.4884,664,7)
};
//int mouth_Indices[28]
//{
//	70,0,
//	85,2,
//	101,3,
//	189,4,
//	176,6,
//	510,7,
//	490,9,
//	470,11,
//	203,13,
//	200,14,
//	197,15,
//	500,18,
//	496,19,
//	489,20
//};
FeaturePoint mouth_Indices[14]{
	FeaturePoint(-22.2323,-35.0152,398,0),
	FeaturePoint(-8.06929,-44.0214,264,2),
	FeaturePoint(0,-41.0681,443,3),
	FeaturePoint(8.06929,-44.0214,693,4),
	FeaturePoint(22.2323,-35.0152,812,6),
	FeaturePoint(13.3161,-30.6713,824,7),
	FeaturePoint(0,-31.0293,432,9),
	FeaturePoint(-13.3161,-30.6713,412,11),
	FeaturePoint(-7.25357,-34.9263,402,13),
	FeaturePoint(0,-35.1599,442,14),
	FeaturePoint(7.25357,-34.9263,815,15),
	FeaturePoint(7.25357,-34.7663,830,18),
	FeaturePoint(0,-34.9999,423,19),
	FeaturePoint(-7.25357,-34.7663,418,20)
};
//int nose_Indices[26]
//{
//	350,0,
//	312,1,
//	319,2,
//	320,3,
//	394,4,
//	388,5,
//	357,6,
//	365,7,
//	537,8,
//	558,9,
//	562,10,
//	575,11,
//	832,12
//};
FeaturePoint nose_Indices[13]{
	FeaturePoint(0,-1.921,114,0),
	FeaturePoint(0,32.0337,289,1),
	FeaturePoint(-3.75229,24.0425,244,2),
	FeaturePoint(-7.62703,14.775,151,3),
	FeaturePoint(-14.2093,3.0488,104,4),
	FeaturePoint(-19.0289,-3.88021,79,5),
	FeaturePoint(-10.2238,-9.5218,93,6),
	FeaturePoint(0,-11.7102,102,7),
	FeaturePoint(10.2238,-9.5218,530,8),
	FeaturePoint(19.0289,-3.88021,516,9),
	FeaturePoint(14.2093,3.0488,540,10),
	FeaturePoint(7.62703,14.775,584,11),
	FeaturePoint(3.75229,24.0425,676,12),
};



string path = "F:\\Graduation Project\\SoftWare\\FaceShape_Tencent\\FaceShape";
string ImagePath = "F:\\Graduation Project\\SoftWare\\FaceModuel\\FaceModuel\\Images\\";
string chdir_cmd = string("sys.path.append(\"") + path + "\")";
string faces_path = "F:\\Graduation Project\\Faces";
string candide_3_path = "F:\\Graduation Project\\Faces\\Candide-3.obj";
string TreeDRestruntion_path = "F:\\Graduation Project\\Faces\\3D Restruntion.obj";
const char* cstr_cmd = chdir_cmd.c_str();

float StandardXY = 0; // 模型整体形变系数，控制Z轴坐标变化


/*主函数*/

int main()
{
	LoadObj(TreeDRestruntion_path); // 初始化标准模型数据

	string fileName; // 文件名
	std::cout << "输入文件名：" << endl;
	cin >> fileName;
	vector<string> v;
	SplitString(fileName, v, ".");
	string Filepath = faces_path + "\\" + v[0]; // 输出文件路径

	string fullPath = ImagePath + fileName;

	const char* path = fullPath.c_str();
	char* path_char = new char[99];
	path_char = const_cast<char*>(path);

	string data_str = FaceShape_Python(path_char); // str格式的json数据

	FaceShapeData faceShapeData = Json2Class(data_str); // FaceShapeData数据

	ShowImage(path_char, faceShapeData); // 传入faceShapeData，输出图像并标点

	std::vector<VertexF> F_ = FeatureOut(faceShapeData); // 获得目标模型的特征点的模型顶点数据

	cout << "有效特征点数量：" << F_.size() << endl;

	std::vector <objl::Vector2> P_2 = RadialBasisFunction2D(F_); // 用径向基函数计算模型每个点的位置，不包含深度信息

	std::vector <objl::Vector3> P_3 = Align(P_2, faceShapeData); // 将输出结果对齐回原来的比例

	// 写成最简单的obj文件看看
	WriteObj(P_3, faceShapeData, Filepath, fileName);
	WriteMtl(Filepath, fileName);
}


/*数据结构定义*/

VertexF::VertexF(void)
{
	Position = objl::Vector3();
	Indice = 0;
	Feature = objl::Vector2();
	FIndice = 0;
}
VertexF::VertexF(objl::Vector3 _Position, unsigned int _Indice, objl::Vector2 _Feature, unsigned int _FIndice)
{
	Position = _Position;
	Indice = _Indice;
	Feature = _Feature;
	FIndice = _FIndice;
}
VertexF::~VertexF() {}
FeaturePoint::FeaturePoint(void) {
	Position = objl::Vector2();
	FIndice = 0;
	Indice = 0;
	X = Y = 0;
}
FeaturePoint::FeaturePoint(objl::Vector2 _Position, unsigned int _Indice, unsigned int _FIndice) {
	Position = _Position;
	Indice = _Indice;
	FIndice = _FIndice;
	X = _Position.X;
	Y = _Position.Y;
}
FeaturePoint::FeaturePoint(double x, double y, unsigned int _Indice, unsigned int _FIndice) {
	Position = objl::Vector2(x, y);
	Indice = _Indice;
	FIndice = _FIndice;
	X = x;
	Y = y;
}
FeaturePoint::~FeaturePoint() {}


/*函数定义*/

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
		std::cout << "加载人脸特征标定Python模块失败！\n";
		PyErr_Print();
		Py_Finalize(); // 释放资源
		return NULL;
	}
	std::cout << "加载人脸特征标定Python模块成功！\n";

	// 加载方法
	PyObject* pv = PyObject_GetAttrString(pModule, "GetFaceShapeJson");
	if (!pv || !PyCallable_Check(pv)) {
		std::cout << "加载人脸特征标定Python方法失败！\n";
		PyErr_Print();
		Py_Finalize(); // 释放资源
		return NULL;
	}
	std::cout << "加载人脸特征标定Python方法成功！\n";

	// 调用方法
	char s_char[] = "s";
	char* s = s_char;
	PyObject* pValue = PyObject_CallFunction(pv, s, image);

	//处理数据
	string callBack3 = "空的";
	callBack3 = PyString_AsString(pValue);
	//cout << callBack3 << endl;
	// 释放资源
	Py_Finalize();;
	return callBack3;
};

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
		std::cout << "ERROR" << endl;
		return faceShapeData;
	}
	else std::cout << "READ JSON SUCC" << endl;

	faceShapeData.msg = Json_ReadString(faceShapeData_json["msg"]);
	faceShapeData.ret = Json_ReadInt(faceShapeData_json["ret"]);
	if (faceShapeData.ret != 0) {
		std::cout << "人脸识别失败！" << endl;
		return faceShapeData;
	}
	else std::cout << "faceShapeData.msg = " << faceShapeData.msg << endl;
	// 写入json数据根节点

	Json::Value data_josn = faceShapeData_json["data"]; // data对象
	faceShapeData.data.image_height = Json_ReadInt(data_josn["image_height"]);
	faceShapeData.data.image_width = Json_ReadInt(data_josn["image_width"]);
	Json::Value face_shape_list = data_josn["face_shape_list"]; // data对象
	// 写入json数据data节点

	Json::Value face_profile_json = face_shape_list[0]["face_profile"];
	for (size_t i = 0; i < face_profile_json.size(); i++) {
		faceShapeData.data.face_shape_list[0].face_profile[i].X
			= Json_ReadInt(face_profile_json[(Json::ArrayIndex)i]["x"]);
		faceShapeData.data.face_shape_list[0].face_profile[i].Y
			= Json_ReadInt(face_profile_json[(Json::ArrayIndex)i]["y"]);
		//cout << "face_profile" << faceShapeData.data.face_shape_list[0].face_profile[i].x << faceShapeData.data.face_shape_list[0].face_profile[i].y << endl;
	}

	Json::Value left_eye_json = face_shape_list[0]["left_eye"];
	for (size_t i = 0; i < left_eye_json.size(); i++) {
		faceShapeData.data.face_shape_list[0].left_eye[i].X
			= Json_ReadInt(left_eye_json[(Json::ArrayIndex)i]["x"]);
		faceShapeData.data.face_shape_list[0].left_eye[i].Y
			= Json_ReadInt(left_eye_json[(Json::ArrayIndex)i]["y"]);
		//cout << "left_eye" << faceShapeData.data.face_shape_list[0].left_eye[i].x << faceShapeData.data.face_shape_list[0].left_eye[i].y << endl;
	}

	Json::Value left_eyebrow_json = face_shape_list[0]["left_eyebrow"];
	for (size_t i = 0; i < left_eyebrow_json.size(); i++) {
		faceShapeData.data.face_shape_list[0].left_eyebrow[i].X
			= Json_ReadInt(left_eyebrow_json[(Json::ArrayIndex)i]["x"]);
		faceShapeData.data.face_shape_list[0].left_eyebrow[i].Y
			= Json_ReadInt(left_eyebrow_json[(Json::ArrayIndex)i]["y"]);
		//cout << "left_eyebrow" << faceShapeData.data.face_shape_list[0].left_eyebrow[i].x << faceShapeData.data.face_shape_list[0].left_eyebrow[i].y << endl;
	}

	Json::Value right_eye_json = face_shape_list[0]["right_eye"];
	for (size_t i = 0; i < right_eye_json.size(); i++) {
		faceShapeData.data.face_shape_list[0].right_eye[i].X
			= Json_ReadInt(right_eye_json[(Json::ArrayIndex)i]["x"]);
		faceShapeData.data.face_shape_list[0].right_eye[i].Y
			= Json_ReadInt(right_eye_json[(Json::ArrayIndex)i]["y"]);
		//cout << "right_eye" << faceShapeData.data.face_shape_list[0].right_eye[i].x << faceShapeData.data.face_shape_list[0].right_eye[i].y << endl;
	}

	Json::Value mouth_json = face_shape_list[0]["mouth"];
	for (size_t i = 0; i < mouth_json.size(); i++) {
		faceShapeData.data.face_shape_list[0].mouth[i].X
			= Json_ReadInt(mouth_json[(Json::ArrayIndex)i]["x"]);
		faceShapeData.data.face_shape_list[0].mouth[i].Y
			= Json_ReadInt(mouth_json[(Json::ArrayIndex)i]["y"]);
		//cout << "mouth" << faceShapeData.data.face_shape_list[0].mouth[i].x << faceShapeData.data.face_shape_list[0].mouth[i].y << endl;
	}

	Json::Value nose_json = face_shape_list[0]["nose"];
	for (size_t i = 0; i < nose_json.size(); i++) {
		faceShapeData.data.face_shape_list[0].nose[i].X
			= Json_ReadInt(nose_json[(Json::ArrayIndex)i]["x"]);
		faceShapeData.data.face_shape_list[0].nose[i].Y
			= Json_ReadInt(nose_json[(Json::ArrayIndex)i]["y"]);
		//cout << "nose" << faceShapeData.data.face_shape_list[0].nose[i].x << faceShapeData.data.face_shape_list[0].nose[i].y << endl;
	}

	Json::Value right_eyebrow_json = face_shape_list[0]["right_eyebrow"];
	for (size_t i = 0; i < right_eyebrow_json.size(); i++) {
		faceShapeData.data.face_shape_list[0].right_eyebrow[i].X
			= Json_ReadInt(right_eyebrow_json[(Json::ArrayIndex)i]["x"]);
		faceShapeData.data.face_shape_list[0].right_eyebrow[i].Y
			= Json_ReadInt(right_eyebrow_json[(Json::ArrayIndex)i]["y"]);
		//cout << "right_eyebrow" << faceShapeData.data.face_shape_list[0].right_eyebrow[i].x << faceShapeData.data.face_shape_list[0].right_eyebrow[i].y << endl;
	}
	// Json数据写入Class完成！

	return faceShapeData;
}

void ShowImage(string path, FaceShapeData data)
{
	Mat read = imread(path);//实例化一个Mat对象,这里使用的是局部路径，图片和工程文件放在同一目录（不是输出文件的目录）
	assert(read.data);//如果数据为空就终止执行

	Scalar pointColor = Scalar(0, 255, 0);

	for (int i = 0; i < 21; i++) {
		Point p;
		p.x = data.data.face_shape_list[0].face_profile[i].X;
		p.y = data.data.face_shape_list[0].face_profile[i].Y;
		cv::circle(read, p, 3, pointColor, -1);
	}
	for (int i = 0; i < 8; i++) {
		Point p;
		p.x = data.data.face_shape_list[0].left_eye[i].X;
		p.y = data.data.face_shape_list[0].left_eye[i].Y;
		cv::circle(read, p, 3, pointColor, -1);
	}
	for (int i = 0; i < 8; i++) {
		Point p;
		p.x = data.data.face_shape_list[0].left_eyebrow[i].X;
		p.y = data.data.face_shape_list[0].left_eyebrow[i].Y;
		cv::circle(read, p, 3, pointColor, -1);
	}
	for (int i = 0; i < 8; i++) {
		Point p;
		p.x = data.data.face_shape_list[0].right_eye[i].X;
		p.y = data.data.face_shape_list[0].right_eye[i].Y;
		cv::circle(read, p, 3, pointColor, -1);
	}
	for (int i = 0; i < 8; i++) {
		Point p;
		p.x = data.data.face_shape_list[0].right_eyebrow[i].X;
		p.y = data.data.face_shape_list[0].right_eyebrow[i].Y;
		cv::circle(read, p, 3, pointColor, -1);
	}
	for (int i = 0; i < 22; i++) {
		Point p;
		p.x = data.data.face_shape_list[0].mouth[i].X;
		p.y = data.data.face_shape_list[0].mouth[i].Y;
		cv::circle(read, p, 3, pointColor, -1);
	}
	for (int i = 0; i < 13; i++) {
		Point p;
		p.x = data.data.face_shape_list[0].nose[i].X;
		p.y = data.data.face_shape_list[0].nose[i].Y;
		cv::circle(read, p, 3, pointColor, -1);
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

void LoadObjTest(string path, string out)
{
	// 初始化 Loader
	objl::Loader Loader;

	// 读取模型文件
	bool loadout = Loader.LoadFile(path);

	// 检查

	// 如果成功读取
	if (loadout)
	{
		// 打开输出文档
		std::ofstream file(out);

		// Go through each loaded mesh and out its contents
		for (int i = 0; i < Loader.LoadedMeshes.size(); i++)
		{
			// Copy one of the loaded meshes to be our current mesh
			objl::Mesh curMesh = Loader.LoadedMeshes[i];

			// Print Mesh Name
			file << "Mesh " << i << ": " << curMesh.MeshName << "\n";

			// Print Vertices
			file << "Vertices:\n";

			// Go through each vertex and print its number,
			//  position, normal, and texture coordinate
			for (int j = 0; j < curMesh.Vertices.size(); j++)
			{
				file << "V" << j << ": " <<
					"P(" << curMesh.Vertices[j].Position.X << ", " << curMesh.Vertices[j].Position.Y << ", " << curMesh.Vertices[j].Position.Z << ") " <<
					"N(" << curMesh.Vertices[j].Normal.X << ", " << curMesh.Vertices[j].Normal.Y << ", " << curMesh.Vertices[j].Normal.Z << ") " <<
					"TC(" << curMesh.Vertices[j].TextureCoordinate.X << ", " << curMesh.Vertices[j].TextureCoordinate.Y << ")\n";
			}

			// Print Indices
			file << "Indices:\n";

			// Go through every 3rd index and print the
			//	triangle that these indices represent
			for (int j = 0; j < curMesh.Indices.size(); j += 3)
			{
				file << "T" << j / 3 << ": " << curMesh.Indices[j] << ", " << curMesh.Indices[j + 1] << ", " << curMesh.Indices[j + 2] << "\n";
			}

			// Print Material
			file << "Material: " << curMesh.MeshMaterial.name << "\n";
			file << "Ambient Color: " << curMesh.MeshMaterial.Ka.X << ", " << curMesh.MeshMaterial.Ka.Y << ", " << curMesh.MeshMaterial.Ka.Z << "\n";
			file << "Diffuse Color: " << curMesh.MeshMaterial.Kd.X << ", " << curMesh.MeshMaterial.Kd.Y << ", " << curMesh.MeshMaterial.Kd.Z << "\n";
			file << "Specular Color: " << curMesh.MeshMaterial.Ks.X << ", " << curMesh.MeshMaterial.Ks.Y << ", " << curMesh.MeshMaterial.Ks.Z << "\n";
			file << "Specular Exponent: " << curMesh.MeshMaterial.Ns << "\n";
			file << "Optical Density: " << curMesh.MeshMaterial.Ni << "\n";
			file << "Dissolve: " << curMesh.MeshMaterial.d << "\n";
			file << "Illumination: " << curMesh.MeshMaterial.illum << "\n";
			file << "Ambient Texture Map: " << curMesh.MeshMaterial.map_Ka << "\n";
			file << "Diffuse Texture Map: " << curMesh.MeshMaterial.map_Kd << "\n";
			file << "Specular Texture Map: " << curMesh.MeshMaterial.map_Ks << "\n";
			file << "Alpha Texture Map: " << curMesh.MeshMaterial.map_d << "\n";
			file << "Bump Map: " << curMesh.MeshMaterial.map_bump << "\n";

			// Leave a space to separate from the next mesh
			file << "\n";
		}

		// 关闭文档
		file.close();
	}
	// 如果读取失败，则提示
	else
	{
		// 打开输出文档
		std::ofstream file(out);

		// 输出错误
		file << "Failed to Load File. May have failed to find it or it was not an .obj file.\n";

		// 关闭文档
		file.close();
	}
}

void LoadObj(string path)
{
	// 初始化 Loader
	objl::Loader Loader;

	// 读取模型文件
	bool loadout = Loader.LoadFile(path);

	// 检查

	// 如果成功读取
	if (loadout)
	{
		//// 输出一遍所有定点的坐标
		//for (int i = 0; i < Loader.LoadedVertices.size(); i++) {
		//	cout << Loader.LoadedVertices[i].Position.X << "," << Loader.LoadedVertices[i].Position.Y << "," << Loader.LoadedVertices[i].Position.Z << "\n";
		//}
		cout << "标准模型读取成功" << endl;

		std::vector <objl::Vector3> Faces(Loader.Faces.size());
		for (int i = 0; i < Loader.PointsPositions.size(); i++)
		{
			P.push_back(Loader.PointsPositions[i]);
			//OutModelPositions.push_back(Loader.PointsPositions[i]);
		}
		for (int i = 0; i < Loader.Faces.size(); i++)
		{
			StandardModelFaces.push_back(Loader.Faces[i]);
		}
		// 初始化标注模型特征点数据
		FeatureVerticlesInitiate(P);

		std::cout << "标准模型数据初始化完成" << endl;
		/*
		* 整一个向量来存所有顶点坐标
		* 找出所有顶点中对应的特征点Pi（如果没有特征点？-1找个最近的 -2创建一个新的）及对应的索引值
		* 求出特征点位移之后的位置P'i
		* 应用径向基函数求剩余所有点的值P'i
		*/
	}
	else
	{
		cout << "标准模型读取失败" << endl;
	}
}

void FeatureVerticlesInitiate(std::vector <objl::Vector3> positions)
{
	// 选取鼻尖作为特征点参考点
	objl::Vector2 NoseFeature(nose_Indices[0].X, nose_Indices[0].Y);
	for (int i = 0; i < sizeof(Face_profile_Indices) / sizeof(Face_profile_Indices[0]); i++)
	{
		unsigned int Indice = face_profile_Indices[i].Indice;
		objl::Vector3 Position = positions[Indice];
		unsigned int FIndice = face_profile_Indices[i].FIndice;
		objl::Vector2 Feature = face_profile_Indices[i].Position - NoseFeature;
		Face_profile_Indices[i] = VertexF(Position, Indice, Feature, FIndice);
		F.push_back(Face_profile_Indices[i]);
		//if (abs(face_profile_Indices[i].X - positions[Indice].X) > 0.001) cout << i << endl;
	}
	for (int i = 0; i < sizeof(Left_eye_Indices) / sizeof(Left_eye_Indices[0]); i++)
	{
		unsigned int Indice = left_eye_Indices[i].Indice;
		objl::Vector3 Position = positions[Indice];
		unsigned int FIndice = left_eye_Indices[i].FIndice;
		objl::Vector2 Feature = left_eye_Indices[i].Position - NoseFeature;
		Left_eye_Indices[i] = VertexF(Position, Indice, Feature, FIndice);
		F.push_back(Left_eye_Indices[i]);
		//if (abs(left_eye_Indices[i].X - positions[Indice].X) > 0.001) cout << i << endl;
	}
	for (int i = 0; i < sizeof(Right_eye_Indices) / sizeof(Right_eye_Indices[0]); i++)
	{
		unsigned int Indice = right_eye_Indices[i].Indice;
		objl::Vector3 Position = positions[Indice];
		unsigned int FIndice = right_eye_Indices[i].FIndice;
		objl::Vector2 Feature = right_eye_Indices[i].Position - NoseFeature;
		Right_eye_Indices[i] = VertexF(Position, Indice, Feature, FIndice);
		F.push_back(Right_eye_Indices[i]);
		//if (abs(right_eye_Indices[i].X - positions[Indice].X) > 0.001) cout << i << endl;
	}
	for (int i = 0; i < sizeof(Left_eyebrow_Indices) / sizeof(Left_eyebrow_Indices[0]); i++)
	{
		unsigned int Indice = left_eyebrow_Indices[i].Indice;
		objl::Vector3 Position = positions[Indice];
		unsigned int FIndice = left_eyebrow_Indices[i].FIndice;
		objl::Vector2 Feature = left_eyebrow_Indices[i].Position - NoseFeature;
		Left_eyebrow_Indices[i] = VertexF(Position, Indice, Feature, FIndice);
		F.push_back(Left_eyebrow_Indices[i]);
		//if (abs(left_eyebrow_Indices[i].X - positions[Indice].X) > 0.001) cout << i << endl;
	}
	for (int i = 0; i < sizeof(Right_eyebrow_Indices) / sizeof(Right_eyebrow_Indices[0]); i++)
	{
		unsigned int Indice = right_eyebrow_Indices[i].Indice;
		objl::Vector3 Position = positions[Indice];
		unsigned int FIndice = right_eyebrow_Indices[i].FIndice;
		objl::Vector2 Feature = right_eyebrow_Indices[i].Position - NoseFeature;
		Right_eyebrow_Indices[i] = VertexF(Position, Indice, Feature, FIndice);
		F.push_back(Right_eyebrow_Indices[i]);
		//if (abs(right_eyebrow_Indices[i].X - positions[Indice].X) > 0.001) cout << i << endl;
	}
	for (int i = 0; i < sizeof(Mouth_Indices) / sizeof(Mouth_Indices[0]); i++)
	{
		unsigned int Indice = mouth_Indices[i].Indice;
		objl::Vector3 Position = positions[Indice];
		unsigned int FIndice = mouth_Indices[i].FIndice;
		objl::Vector2 Feature = mouth_Indices[i].Position - NoseFeature;
		Mouth_Indices[i] = VertexF(Position, Indice, Feature, FIndice);
		F.push_back(Mouth_Indices[i]);
		//if (abs(mouth_Indices[i].X - positions[Indice].X) > 0.001) cout << i << endl;
	}
	for (int i = 0; i < sizeof(Nose_Indices) / sizeof(Nose_Indices[0]); i++)
	{
		unsigned int Indice = nose_Indices[i].Indice;
		objl::Vector3 Position = positions[Indice];
		unsigned int FIndice = nose_Indices[i].FIndice;
		objl::Vector2 Feature = nose_Indices[i].Position - NoseFeature;
		Nose_Indices[i] = VertexF(Position, Indice, Feature, FIndice);
		F.push_back(Nose_Indices[i]);
		//if (abs(nose_Indices[i].X - positions[Indice].X) > 0.001) cout << i << endl;
	}
	// 对特征点排序
	for (int i = 0; i < F.size() - 1; i++)
	{
		for (int j = 0; j < F.size() - i - 1; j++)
		{
			if (F[j].Indice > F[j + 1].Indice)
			{
				swap(F[j], F[j + 1]);
			}
		}
	}
}

std::vector<VertexF> FeatureOut(FaceShapeData faceShapeData)
{
	std::vector<VertexF> F_;
	//// 标准模型特征点到其鼻尖的长度
	//double StandardLength = 0;
	//// 目标特征点到其鼻尖的长度
	//double OutLength = 0;
	// X轴、Y轴的缩放比例
	double XChange = 0;
	double YChange = 0;
	double StandardX = 0;
	double StandardY = 0;
	double OutX = 0;
	double OutY = 0;
	// 取鼻尖作为参考点
	objl::Vector2 NoseF_ = faceShapeData.data.face_shape_list[0].nose[0];

	for (int i = 0; i < sizeof(Face_profile_Indices) / sizeof(Face_profile_Indices[0]); i++)
	{
		unsigned int Indice = Face_profile_Indices[i].Indice;
		unsigned int FIndice = Face_profile_Indices[i].FIndice;
		objl::Vector3 Position = objl::Vector3(-faceShapeData.data.face_shape_list[0].face_profile[FIndice].X, -faceShapeData.data.face_shape_list[0].face_profile[FIndice].Y, Face_profile_Indices[i].Position.Z);
		objl::Vector2 Feature = objl::Vector2(Position.X, Position.Y) - NoseF_;
		F_.push_back(VertexF(Position, Indice, Feature, FIndice));
		//StandardLength += Face_profile_Indices[i].Feature.magnitude();
		//OutLength += Feature.magnitude();
		//if(Face_profile_Indices[i].Position.X != 0)
		//XYChange += abs(Feature.X / Face_profile_Indices[i].Feature.X);
		//if (Face_profile_Indices[i].Position.Y != 0)
		//XYChange += abs(Feature.Y / Face_profile_Indices[i].Feature.Y);
		StandardX += abs(Face_profile_Indices[i].Feature.X);
		StandardY += abs(Face_profile_Indices[i].Feature.Y);
		OutX += abs(Feature.X);
		OutY += abs(Feature.Y);
	}
	for (int i = 0; i < sizeof(Left_eye_Indices) / sizeof(Left_eye_Indices[0]); i++)
	{
		unsigned int Indice = Left_eye_Indices[i].Indice;
		unsigned int FIndice = Left_eye_Indices[i].FIndice;
		objl::Vector3 Position = objl::Vector3(-faceShapeData.data.face_shape_list[0].left_eye[FIndice].X, -faceShapeData.data.face_shape_list[0].left_eye[FIndice].Y, Left_eye_Indices[i].Position.Z);
		objl::Vector2 Feature = objl::Vector2(Position.X, Position.Y) - NoseF_;
		F_.push_back(VertexF(Position, Indice, Feature, FIndice));
		//StandardLength += Face_profile_Indices[i].Feature.magnitude();
		//OutLength += Feature.magnitude();
		//if (Face_profile_Indices[i].Position.X != 0)
		//XYChange += abs(Feature.X / Face_profile_Indices[i].Feature.X);
		//if (Face_profile_Indices[i].Position.Y != 0)
		//XYChange += abs(Feature.Y / Face_profile_Indices[i].Feature.Y);
		StandardX += abs(Face_profile_Indices[i].Feature.X);
		StandardY += abs(Face_profile_Indices[i].Feature.Y);
		OutX += abs(Feature.X);
		OutY += abs(Feature.Y);
	}
	for (int i = 0; i < sizeof(Right_eye_Indices) / sizeof(Right_eye_Indices[0]); i++)
	{
		unsigned int Indice = Right_eye_Indices[i].Indice;
		unsigned int FIndice = Right_eye_Indices[i].FIndice;
		objl::Vector3 Position = objl::Vector3(-faceShapeData.data.face_shape_list[0].right_eye[FIndice].X, -faceShapeData.data.face_shape_list[0].right_eye[FIndice].Y, Right_eye_Indices[i].Position.Z);
		objl::Vector2 Feature = objl::Vector2(Position.X, Position.Y) - NoseF_;
		F_.push_back(VertexF(Position, Indice, Feature, FIndice));
		//StandardLength += Face_profile_Indices[i].Feature.magnitude();
		//OutLength += Feature.magnitude();
		//if (Face_profile_Indices[i].Position.X != 0)
		//XYChange += abs(Feature.X / Face_profile_Indices[i].Feature.X);
		//if (Face_profile_Indices[i].Position.Y != 0)
		//XYChange += abs(Feature.Y / Face_profile_Indices[i].Feature.Y);
		StandardX += abs(Face_profile_Indices[i].Feature.X);
		StandardY += abs(Face_profile_Indices[i].Feature.Y);
		OutX += abs(Feature.X);
		OutY += abs(Feature.Y);
	}
	for (int i = 0; i < sizeof(Left_eyebrow_Indices) / sizeof(Left_eyebrow_Indices[0]); i++)
	{
		unsigned int Indice = Left_eyebrow_Indices[i].Indice;
		unsigned int FIndice = Left_eyebrow_Indices[i].FIndice;
		objl::Vector3 Position = objl::Vector3(-faceShapeData.data.face_shape_list[0].left_eyebrow[FIndice].X, -faceShapeData.data.face_shape_list[0].left_eyebrow[FIndice].Y, Left_eyebrow_Indices[i].Position.Z);
		objl::Vector2 Feature = objl::Vector2(Position.X, Position.Y) - NoseF_;
		F_.push_back(VertexF(Position, Indice, Feature, FIndice));
		//StandardLength += Face_profile_Indices[i].Feature.magnitude();
		//OutLength += Feature.magnitude();
		//if (Face_profile_Indices[i].Position.X != 0)
		//XYChange += abs(Feature.X / Face_profile_Indices[i].Feature.X);
		//if (Face_profile_Indices[i].Position.Y != 0)
		//XYChange += abs(Feature.Y / Face_profile_Indices[i].Feature.Y);
		StandardX += abs(Face_profile_Indices[i].Feature.X);
		StandardY += abs(Face_profile_Indices[i].Feature.Y);
		OutX += abs(Feature.X);
		OutY += abs(Feature.Y);
	}
	for (int i = 0; i < sizeof(Right_eyebrow_Indices) / sizeof(Right_eyebrow_Indices[0]); i++)
	{
		unsigned int Indice = Right_eyebrow_Indices[i].Indice;
		unsigned int FIndice = Right_eyebrow_Indices[i].FIndice;
		objl::Vector3 Position = objl::Vector3(-faceShapeData.data.face_shape_list[0].right_eyebrow[FIndice].X, -faceShapeData.data.face_shape_list[0].right_eyebrow[FIndice].Y, Right_eyebrow_Indices[i].Position.Z);
		objl::Vector2 Feature = objl::Vector2(Position.X, Position.Y) - NoseF_;
		F_.push_back(VertexF(Position, Indice, Feature, FIndice));
		//StandardLength += Face_profile_Indices[i].Feature.magnitude();
		//OutLength += Feature.magnitude();
		//if (Face_profile_Indices[i].Position.X != 0)
		//XYChange += abs(Feature.X / Face_profile_Indices[i].Feature.X);
		//if (Face_profile_Indices[i].Position.Y != 0)
		//XYChange += abs(Feature.Y / Face_profile_Indices[i].Feature.Y);
		StandardX += abs(Face_profile_Indices[i].Feature.X);
		StandardY += abs(Face_profile_Indices[i].Feature.Y);
		OutX += abs(Feature.X);
		OutY += abs(Feature.Y);
	}
	for (int i = 0; i < sizeof(Mouth_Indices) / sizeof(Mouth_Indices[0]); i++)
	{
		unsigned int Indice = Mouth_Indices[i].Indice;
		unsigned int FIndice = Mouth_Indices[i].FIndice;
		objl::Vector3 Position = objl::Vector3(-faceShapeData.data.face_shape_list[0].mouth[FIndice].X, -faceShapeData.data.face_shape_list[0].mouth[FIndice].Y, Mouth_Indices[i].Position.Z);
		objl::Vector2 Feature = objl::Vector2(Position.X, Position.Y) - NoseF_;
		F_.push_back(VertexF(Position, Indice, Feature, FIndice));
		//StandardLength += Face_profile_Indices[i].Feature.magnitude();
		//OutLength += Feature.magnitude();
		//if (Face_profile_Indices[i].Position.X != 0)
		//XYChange += abs(Feature.X / Face_profile_Indices[i].Feature.X);
		//if (Face_profile_Indices[i].Position.Y != 0)
		//XYChange += abs(Feature.Y / Face_profile_Indices[i].Feature.Y);
		StandardX += abs(Face_profile_Indices[i].Feature.X);
		StandardY += abs(Face_profile_Indices[i].Feature.Y);
		OutX += abs(Feature.X);
		OutY += abs(Feature.Y);
	}
	for (int i = 0; i < sizeof(Nose_Indices) / sizeof(Nose_Indices[0]); i++)
	{
		unsigned int Indice = Nose_Indices[i].Indice;
		unsigned int FIndice = Nose_Indices[i].FIndice;
		objl::Vector3 Position = objl::Vector3(-faceShapeData.data.face_shape_list[0].nose[FIndice].X, -faceShapeData.data.face_shape_list[0].nose[FIndice].Y, Nose_Indices[i].Position.Z);
		objl::Vector2 Feature = objl::Vector2(Position.X, Position.Y) - NoseF_;
		F_.push_back(VertexF(Position, Indice, Feature, FIndice));
		//StandardLength += Face_profile_Indices[i].Feature.magnitude();
		//OutLength += Feature.magnitude();
		//if (Face_profile_Indices[i].Position.X != 0)
		//XYChange += abs(Feature.X / Face_profile_Indices[i].Feature.X);
		//if (Face_profile_Indices[i].Position.Y != 0)
		//XYChange += abs(Feature.Y / Face_profile_Indices[i].Feature.Y);
		StandardX += abs(Face_profile_Indices[i].Feature.X);
		StandardY += abs(Face_profile_Indices[i].Feature.Y);
		OutX += abs(Feature.X);
		OutY += abs(Feature.Y);
	}

	//XYChange = XYChange / (2 * Out.size());
	//Zchange = sqrt( OutLength / StandardLength);
	//std::cout << "模型特征移动比例 = " << Zchange << endl;
	//cout << "模型平均缩放比例 = " << XYChange << endl;
	XChange = 0.5 * (OutX / StandardX);
	YChange = 0.5 * (OutY / StandardY);
	//cout << "标准模型特征点X轴移动和:" << StandardX << "\n标准模型特征点Y轴移动和:" << StandardY << "\n目标型特征点X轴移动和:" << OutX << "\n目标模型特征点Y轴移动和:" << OutY << endl;
	//cout << "模型特征缩放比例X = " << OutX / StandardX << endl;
	//cout << "模型特征缩放比例Y = " << OutY / StandardY << endl;
	//cout << "\n模型特征X轴缩放比例 = " << XChange << endl;
	//cout << "模型特征Y轴缩放比例 = " << YChange << endl;

	// 对特征点系数排序

	for (int i = 0; i < F_.size() - 1; i++)
	{
		for (int j = 0; j < F_.size() - i - 1; j++)
		{
			if (F_[j].Indice > F_[j + 1].Indice)
			{
				swap(F_[j], F_[j + 1]);
			}
		}
	}

	double OutX_ = 0;
	double OutY_ = 0;
	XChange = (XChange + YChange) / 2;

	// 对目标特征移动量进行缩放，使之与标准模型基本对齐，便于形变
	for (int i = 0; i < F_.size(); i++)
	{
		F_[i].Feature.X /= XChange;
		F_[i].Feature.Y /= XChange;
		// 对目标特征点坐标进行缩放，使最终模型对齐
		F_[i].Position.X = NoseF_.X + F_[i].Feature.X;
		F_[i].Position.Y = NoseF_.Y + F_[i].Feature.Y;
		OutX_ += abs(F_[i].Feature.X);
		OutY_ += abs(F_[i].Feature.Y);
	}

	//XChange = 0.5 * (OutX_ / StandardX);
	//YChange = 0.5 * (OutY_ / StandardY);
	//cout << "\n对齐后模型特征X轴缩放比例 = " << XChange << endl;
	//cout << "对齐后模型特征Y轴缩放比例 = " << YChange << endl << endl;
	return F_;
}

static double thin_plate_splines(double r)
{
	if (r == 0.0)
		return 0.0;
	else
		return r * r * log(r);
}

MatrixXd Calc_tps2D(std::vector< VertexF > F_, double regularization)
{
	unsigned int p = unsigned int(F_.size());
	
	// 配置矩阵和向量
	MatrixXd mtx_l(p + 3, p + 3);
	MatrixXd mtx_v(p + 3, 2);
	MatrixXd mtx_orig_k(p, p);

	// 填充K (px, L左上角)
	// 从控制点计算平均边缘长度
	//
	// K是对称的
	// 所以我们只需要计算大约一半的系数。
	double a = 0.0;
	for (unsigned i = 0; i < p; ++i)
	{
		for (unsigned j = i + 1; j < p; ++j)
		{
			objl::Vector2 pt_i(F[i].Position.X, F[i].Position.Y);
			objl::Vector2 pt_j(F[j].Position.X, F[j].Position.Y);
			double elen = (pt_i - pt_j).magnitude();
			mtx_l(i, j) = mtx_l(j, i) =
				mtx_orig_k(i, j) = mtx_orig_k(j, i) =
				thin_plate_splines(elen);
			a += elen * 2; // 对于上、下tri也是一样
		}
	}
	a /= (double)(p * p);

	// 填充剩下的L
	for (unsigned i = 0; i < p; ++i)
	{
		// 对角线:正则化参数(lamda * a^2)
		mtx_l(i, i) = mtx_orig_k(i, i) =
			regularization * (a * a);

		// P (p x 3，右上角)
		mtx_l(i, p + 0) = F[i].Position.X;
		mtx_l(i, p + 1) = F[i].Position.Y;
		mtx_l(i, p + 2) = 1.0;

		// P 转置 (3 x p, 左下角)
		mtx_l(p + 0, i) = F[i].Position.X;
		mtx_l(p + 1, i) = F[i].Position.Y;
		mtx_l(p + 2, i) = 1.0;
	}

	// O (3 x 3, 右下角)
	for (unsigned i = p; i < p + 3; ++i)
		for (unsigned j = p; j < p + 3; ++j)
			mtx_l(i, j) = 0.0;

	// 向量 V( p+3 x 2 )
	unsigned i = 0;
	for (; i < p; ++i)
	{
		mtx_v(i, 0) = F_[i].Position.X - F[i].Position.X;
		mtx_v(i, 1) = F_[i].Position.Y - F[i].Position.Y;
	}
	for (; i < p + 3; ++i)
	{
		mtx_v(i, 0) = 0;
		mtx_v(i, 1) = 0;
	}


	// 解线性方程组
	MatrixXd X = mtx_l.fullPivLu().solve(mtx_v);


	// 输出一遍矩阵
	//cout << "X = \n" << X << endl;
	//cout << X << endl;

	return X;
}

std::vector <objl::Vector2> RadialBasisFunction2D(std::vector<VertexF> F_)
{
	std::vector <objl::Vector2> P_; // 最后的输出点结果

	MatrixXd X = Calc_tps2D(F_, 0);

	MatrixXd C(X.rows() - 3, 2);

	MatrixXd M(2, 2);
	M.row(0) = X.row(X.rows() - 3);
	M.row(1) = X.row(X.rows() - 2);

	Vector2d T = X.row(X.rows() - 1);

	for (int i = 0; i < X.rows() - 3; i++)
		C.row(i) = X.row(i);

	cout << "\nM矩阵：\n";
	cout << M << endl;
	cout << "T矩阵：\n";
	cout << T << endl;

	for (int i = 0; i < P.size(); i++)
	{
		objl::Vector2 deltaP;
		for (int j = 0; j < C.rows(); j++)
		{
			objl::Vector2 Cj(C(j, 0), C(j, 1));
			deltaP += Cj * thin_plate_splines(
				(
					objl::Vector2(P[i].X, P[i].Y)
					- objl::Vector2(F[j].Position.X, F[j].Position.Y)
					).magnitude()
			);
		}
		deltaP = deltaP + objl::Vector2
		(
			M(0, 0) * P[i].X + M(0, 1) * P[i].Y,
			M(1, 0) * P[i].X + M(1, 1) * P[i].Y
		);
		//deltaP = deltaP + objl::Vector2(T[0], T[1]);
		//cout << "deltaP" << i << ": " << deltaP.X << ", " << deltaP.Y << ", " << endl;
		P_.push_back(objl::Vector2(P[i].X + deltaP.X, P[i].Y + deltaP.Y));
	}
	return P_;
}

vector<objl::Vector3> Align(std::vector <objl::Vector2> Q_, FaceShapeData faceShapeData)
{
	vector<objl::Vector3> Q;

	objl::Vector2 Image(faceShapeData.data.image_width, faceShapeData.data.image_height);

	for (int i = 0; i < Q_.size(); i++) {
		Q_[i].X *= 1;
		//Q_[i].Y *= -1;
	}

	// 鼻尖（参考点）
	objl::Vector2 NoseP(faceShapeData.data.face_shape_list[0].nose[0]);
	objl::Vector2 NoseM(Q_[Nose_Indices[0].Indice].X, Q_[Nose_Indices[0].Indice].Y);
	NoseP = Image - NoseP;
	// 脸的长宽
	double FaceWP = (faceShapeData.data.face_shape_list[0].face_profile[0] - faceShapeData.data.face_shape_list[0].face_profile[20]).magnitude();
	double HalfFaceHP = (faceShapeData.data.face_shape_list[0].nose[1] - faceShapeData.data.face_shape_list[0].face_profile[10]).magnitude();
	double FaceWM = (Q_[Face_profile_Indices[0].Indice] - Q_[Face_profile_Indices[14].Indice]).magnitude();
	double HalfFaceHM = (Q_[Nose_Indices[1].Indice] - Q_[Face_profile_Indices[7].Indice]).magnitude();
	double FaceWS = (objl::Vector2(Face_profile_Indices[0].Position.X, Face_profile_Indices[0].Position.Y) - objl::Vector2(Face_profile_Indices[14].Position.X, Face_profile_Indices[14].Position.Y)).magnitude();
	double HalfFaceHS = (objl::Vector2(Nose_Indices[1].Position.X, Nose_Indices[1].Position.Y) - objl::Vector2(Face_profile_Indices[7].Position.X, Face_profile_Indices[7].Position.Y)).magnitude();
	cout << "FaceWP"<< FaceWP <<endl;
	cout << "HalfFaceHP"<< HalfFaceHP <<endl;
	cout << "FaceWM"<< FaceWM <<endl;
	cout << "HalfFaceHM"<< HalfFaceHM <<endl;
	cout << "FaceWS"<< FaceWS <<endl;
	cout << "HalfFaceHS"<< HalfFaceHS <<endl;

	// 脸的长宽比例   图片/模型 
	double WScale = FaceWP / FaceWM;
	double HScale = HalfFaceHP / HalfFaceHM;
	double SPScale = (FaceWP / FaceWS + HalfFaceHP / HalfFaceHS) / 2;
	cout << "WScale"<< WScale <<endl;
	cout << "HScale"<< HScale <<endl;
	cout << "SPScale"<< SPScale <<endl;

	for (int i = 0; i < Q_.size(); i++) {
		Q_[i].X = (Q_[i].X - NoseM.X) * WScale + NoseP.X;
		Q_[i].Y = (Q_[i].Y - NoseM.Y) * HScale + NoseP.Y;
		Q.push_back(objl::Vector3(Q_[i].X, Q_[i].Y, P[i].Z * SPScale));
	}

	return Q;
}

void WriteObj(std::vector <objl::Vector3> OutModelPositions, FaceShapeData faceShapeData,string path, string name)
{
	vector<string> v;
	SplitString(name, v, ".");

	int imageW = faceShapeData.data.image_width;
	int imageH = faceShapeData.data.image_height;
	
	//string OutObj = "F:\\Graduation Project\\Faces\\3D_Restruntion.obj";
	string OutObj = path + "/" + v[0] + ".obj";
	ofstream file(OutObj);

	if (file) {
		cout << "开始写入" << OutObj << endl;

		file << "mtllib " << v[0] << ".mtl" << endl << endl; // 使用材质库

		// 顶点
		file << "#" << OutModelPositions.size() << "Vertices" << endl;
		for (int i = 0; i < OutModelPositions.size(); i++) {
			//cout << OutModelPositions[i].X << "," << OutModelPositions[i].Y << "," << OutModelPositions[i].Z << endl;
			file <<"v " << OutModelPositions[i].X << " " << OutModelPositions[i].Y << " " << -OutModelPositions[i].Z << endl;
		}
		// 材质索引
		file << "#" << OutModelPositions.size() << "Texture Coordinates" << endl;
		for (int i = 0; i < OutModelPositions.size(); i++) {
			file << "vt " << 1 - float(OutModelPositions[i].X / imageW)
				<< " " << float(OutModelPositions[i].Y / imageH)
				<< endl;
		}
		// 法向量
		std::vector <objl::Vector3> Pnormal = CaculateNormal(OutModelPositions);
		file << "#" << Pnormal.size() << "Vertex Normals" << endl;
		for (int i = 0; i < Pnormal.size(); i++) {
			file << "vn " << Pnormal[i].X << " " << Pnormal[i].Y << " " << Pnormal[i].Z << endl;
			//file << "vn " << 0 << " " << 0 << " " << 0 << endl;
		}
		file << endl;
		file << "usemtl " << v[0] << endl << endl;

		// 面
		for (int j = 0; j < StandardModelFaces.size(); j++) {
			int x = int(StandardModelFaces[j].X);
			int y = int(StandardModelFaces[j].Y);
			int z = int(StandardModelFaces[j].Z);
			file << "f " << x << "/" << x << "/" << x				
				<< " " << y << "/" << y << "/" << y	
				<< " " << z << "/" << z << "/" << z	
				<< endl;
		}
		cout << "写入完成" << endl;
		file.close();
	}
	else {
		cout << "创建OBJ输出文件失败" << endl;
	}
}

void WriteObj(vector<objl::Vector2> OutModelPositions)
{
	string T = "F:\\Graduation Project\\Faces\\3D_Restruntion.obj";
	ofstream file(T);
	if (file) {
		cout << "开始写入" << T << endl;
		for (int i = 0; i < OutModelPositions.size(); i++) {
			//cout << OutModelPositions[i].X << "," << OutModelPositions[i].Y << "," << OutModelPositions[i].Z << endl;
			file << "v " << OutModelPositions[i].X << " " << OutModelPositions[i].Y << " " << P[i].Z << endl;
		}
		file << endl << endl;
		for (int j = 0; j < StandardModelFaces.size(); j++) {
			int x = int(StandardModelFaces[j].X);
			int y = int(StandardModelFaces[j].X);
			int z = int(StandardModelFaces[j].X);
			file << "f " << x << "/" 
				<< " " << y
				<< " " << z << endl;
		}
		cout << "写入完成" << endl;
		file.close();
	}
}

void WriteMtl(string path, string name) {
	vector<string> v;
	SplitString(name, v, ".");
	string OutMtl = path + "/" + v[0] + ".mtl";

	ofstream file(OutMtl);

	if (file) {
		cout << "开始写入" << OutMtl << endl;

		file << "newmtl " << v[0] << "\n" << "\tNs 400\n\td 1\n\tillum 2\n\tKd 0.784314 0.784314 0.784314\n\tKs 0.0 0.0 0.0\n\tKa 0.2 0.2 0.2\n\tmap_Kd " + name;

		cout << "写入完成" << endl;
		file.close();
	}
	else {
		cout << "创建MTL输出文件失败" << endl;
	}
}

void SplitString(const string& s, vector<string>& v, const string& c)
{
	string::size_type pos1, pos2;
	pos2 = s.find(c);
	pos1 = 0;
	while (string::npos != pos2)
	{
		v.push_back(s.substr(pos1, pos2 - pos1));

		pos1 = pos2 + c.size();
		pos2 = s.find(c, pos1);
	}
	if (pos1 != s.length())
		v.push_back(s.substr(pos1));
}

std::vector <objl::Vector3> CaculateNormal(std::vector <objl::Vector3> P) {
	std::vector <objl::Vector3> PNormal(P.size()); // P的法向量和
	std::vector<int> Pcount(P.size()); // P的法向量加和个数

	for (int i = 0; i < StandardModelFaces.size(); i++) {
		// 三个点的索引值
		int aindice = int(StandardModelFaces[i].X) - 1;
		int bindice = int(StandardModelFaces[i].Y) - 1;
		int cindice = int(StandardModelFaces[i].Z) - 1;
		// 三个点的坐标
		objl::Vector3 Pa = P[aindice];
		objl::Vector3 Pb = P[bindice];
		objl::Vector3 Pc = P[cindice];
		// a:
		PNormal[aindice] += objl::cross(Pa - Pb, Pa - Pc);
		Pcount[aindice]++;
		// b:
		PNormal[bindice] += objl::cross(Pb - Pc, Pb - Pa);
		Pcount[bindice]++;
		// c:
		PNormal[cindice] += objl::cross(Pc - Pa, Pc - Pb);
		Pcount[cindice]++;
	}
	
	for (int i = 0; i < PNormal.size(); i++) {
		if (Pcount[i] > 0) {
			PNormal[i] = PNormal[i] / Pcount[i];
		}
		PNormal[i] = PNormal[i].normalize();

	}

	return PNormal;
}