#include "pch.h"
#include <iostream>
#include <fstream>
#include <Python.h>
#include <direct.h>
#include <opencv2\opencv.hpp>
#include <string>
#include <json\json.h>
#include <OBJ_Loader.h>

using namespace std;
using namespace cv;


string path = "F:\\Graduation Project\\SoftWare\\FaceShape_Tencent\\FaceShape";
string ImagePath = "F:\\Graduation Project\\SoftWare\\FaceModuel\\FaceModuel\\Images\\";
string chdir_cmd = string("sys.path.append(\"") + path + "\")";
string candide_3_path = "F:\\Graduation Project\\Faces\\Candide-3.obj";
string TreeDRestruntion_path = "F:\\Graduation Project\\Faces\\3D Restruntion.obj";
const char* cstr_cmd = chdir_cmd.c_str();

namespace TDRF {
	int face_profile_Indices[30]
	{
		817,0,
		827,2,
		805,4,
		789,5,
		783,6,
		139,8,
		140,9,
		51,10,
		23,11,
		13,12,
		2,14,
		777,15,
		455,16,
		629,18,
		622,20
	};
	int left_eye_Indices[8]
	{
		292,0,
		277,2,
		271,4,
		287,6
	};
	int right_eye_Indices[8]
	{
		651,0,
		672,2,
		727,4,
		665,6
	};
	int left_eyebrow_Indices[16]
	{
		276,0,
		237,1,
		303,3,
		305,4,
		313,5,
		228,6,
		239,7
	};
	int right_eyebrow_Indices[16]
	{
		732,0,
		835,1,
		636,3,
		632,4,
		642,5,
		643,6,
		836,7
	};
	int mouth_Indices[28]
	{
		70,0,
		85,2,
		101,3,
		189,4,
		176,6,
		510,7,
		490,9,
		470,11,
		203,13,
		200,14,
		197,15,
		500,18,
		496,19,
		489,20
	};
	int nose_Indices[26]
	{
		350,0,
		312,1,
		319,2,
		320,3,
		394,4,
		388,5,
		357,6,
		365,7,
		537,8,
		558,9,
		562,10,
		575,11,
		832,12
	};
}

//class Pnt {
//public:
//	int X;
//	int Y;
//};
//class FaceShapeList {
//public:
//	Pnt face_profile[21];
//	Pnt left_eye[8];
//	Pnt left_eyebrow[8];
//	Pnt right_eye[8];
//	Pnt mouth[22];
//	Pnt nose[13];
//	Pnt right_eyebrow[8];
//};
//class Data {
//public:
//	int image_height;
//	int image_width;
//	FaceShapeList face_shape_list[1];
//};
//class FaceShapeData {
//public:
//	string msg;
//	Data data;
//	int ret;
//};
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

// 读取json的int格式数据
int Json_ReadInt(Json::Value JV, int ori_value = 0);
// 读取json的double格式数据
double Json_ReadDouble(Json::Value JV, double ori_value = 0.0);
// 读取json的string格式数据
string Json_ReadString(Json::Value JV, string ori_value = "");
// 实现特征点标定的python方法，需传入.JPG图片完整路径
string FaceShape_Python(char* image);
// 将string格式的json数据转换为class
FaceShapeData Json2Class(string data);
// 绘图
void ShowImage(string path, FaceShapeData data);
// 读取Obj格式模型演示方法
void LoadObjTest(string path, string out);
// 读取Obj格式模型方法及其算法调用
void LoadObj(string path, string out);
// 比对Vector3和Vector2，返回最相近的点的索引值
std::vector<int> GetNearestPosionsIndex(std::vector <objl::Vector3> vector3, FaceShapeData faceShapeData);
// 获取重建目标特征点的位置
void TargetFeature(std::vector <objl::Vector3> Positions, std::vector<objl::Mesh> Meshes);
// 径向基函数
void RadialBasisFunction();


int main()
{
	//string fileName; // 文件名
	//cout << "输入文件名：" << endl;
	//cin >> fileName;

	//string fullPath = ImagePath + fileName;

	//const char* path = fullPath.c_str();
	//char* path_char = new char[99];
	//path_char = const_cast<char*>(path);

	//string data_str = FaceShape_Python(path_char); // str格式的json数据

	//FaceShapeData faceShapeData = Json2Class(data_str); // FaceShapeData数据

	//ShowImage(path_char, faceShapeData); // 下一步：ShowImage，传入faceShapeData，输出图像并标点

	LoadObj(TreeDRestruntion_path, "F:\\Graduation Project\\Faces\\TestOut.obj");
}

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
		circle(read, p, 3, pointColor, -1);
	}
	for (int i = 0; i < 8; i++) {
		Point p;
		p.x = data.data.face_shape_list[0].left_eye[i].X;
		p.y = data.data.face_shape_list[0].left_eye[i].Y;
		circle(read, p, 3, pointColor, -1);
	}
	for (int i = 0; i < 8; i++) {
		Point p;
		p.x = data.data.face_shape_list[0].left_eyebrow[i].X;
		p.y = data.data.face_shape_list[0].left_eyebrow[i].Y;
		circle(read, p, 3, pointColor, -1);
	}
	for (int i = 0; i < 8; i++) {
		Point p;
		p.x = data.data.face_shape_list[0].right_eye[i].X;
		p.y = data.data.face_shape_list[0].right_eye[i].Y;
		circle(read, p, 3, pointColor, -1);
	}
	for (int i = 0; i < 8; i++) {
		Point p;
		p.x = data.data.face_shape_list[0].right_eyebrow[i].X;
		p.y = data.data.face_shape_list[0].right_eyebrow[i].Y;
		circle(read, p, 3, pointColor, -1);
	}
	for (int i = 0; i < 22; i++) {
		Point p;
		p.x = data.data.face_shape_list[0].mouth[i].X;
		p.y = data.data.face_shape_list[0].mouth[i].Y;
		circle(read, p, 3, pointColor, -1);
	}
	for (int i = 0; i < 13; i++) {
		Point p;
		p.x = data.data.face_shape_list[0].nose[i].X;
		p.y = data.data.face_shape_list[0].nose[i].Y;
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
void LoadObj(string path, string out) 
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

		std::vector <objl::Vector3> Positions(Loader.PointsPositions.size());
		std::vector <objl::Vector3> Faces(Loader.Faces.size());

		for (int i = 0; i < Loader.PointsPositions.size(); i++)
		{
			Positions[i] = Loader.PointsPositions[i];
		}
		for (int i = 0; i < Loader.Faces.size(); i++)
		{
			Faces[i] = Loader.Faces[i];
		}




		/* 
		* 整一个向量来存所有顶点坐标
		* 找出所有顶点中对应的特征点Pi（如果没有特征点？-1找个最近的 -2创建一个新的）及对应的索引值
		* 求出特征点位移之后的位置P'i
		* 应用径向基函数求剩余所有点的值P'i
		*/
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

}void TargetFeature(std::vector <objl::Vector3> Positions, std::vector<objl::Mesh> Meshes) {

}

void RadialBasisFunction() 
{

}