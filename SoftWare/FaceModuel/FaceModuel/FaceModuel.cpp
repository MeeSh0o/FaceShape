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
class Verticle
{		 
	public :
		// 构造函数
		Verticle();
		// 带参数的构造函数
		Verticle(objl::Vector3 _Position, unsigned int _Indice, objl::Vector2 _Feature, unsigned int _FIndice);
		~Verticle();
		// 点坐标
		objl::Vector3 Position;
		// 点在模型中的索引号
		unsigned int Indice;
		// 特征点到鼻尖点的正面位移量
		objl::Vector2 Feature;
		// 特征点在特征数据中的索引号
		unsigned int FIndice;

};


/*函数声明*/

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
void LoadObj(string path);
// 标准模型特征点初始化
void FeatureVerticlesInitiate(std::vector <objl::Vector3> positions);
// 目标特征点顶点集
std::vector<Verticle> FeatureOut(FaceShapeData faceShapeData);
// 径向基函数
void RadialBasisFunction(std::vector<Verticle> FeatureData);


/*变量声明*/

// 标准模型的点坐标
std::vector <objl::Vector3> StandardModelPositions;
// 目标模型的点坐标中间变量
std::vector <objl::Vector3> OutModelPositions;
// 标准模型的面
std::vector <objl::Vector3> StandardModelFaces;

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
int left_eyebrow_Indices[14]
{
	276,0,
	237,1,
	303,3,
	305,4,
	313,5,
	228,6,
	239,7
};
int right_eyebrow_Indices[14]
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

// 标准模型的特征点顶点数据
Verticle Face_profile_Indices[15];
Verticle Left_eye_Indices[4];
Verticle Right_eye_Indices[4];
Verticle Left_eyebrow_Indices[7];
Verticle Right_eyebrow_Indices[7];
Verticle Mouth_Indices[14];
Verticle Nose_Indices[13];
std::vector<Verticle> StandardFeaturePosiotons;

string path = "F:\\Graduation Project\\SoftWare\\FaceShape_Tencent\\FaceShape";
string ImagePath = "F:\\Graduation Project\\SoftWare\\FaceModuel\\FaceModuel\\Images\\";
string chdir_cmd = string("sys.path.append(\"") + path + "\")";
string candide_3_path = "F:\\Graduation Project\\Faces\\Candide-3.obj";
string TreeDRestruntion_path = "F:\\Graduation Project\\Faces\\3D Restruntion.obj";
const char* cstr_cmd = chdir_cmd.c_str();

float Zchange = 0; // 模型整体形变系数，控制Z轴坐标变化

double R = 80;

/*主函数*/

int main()
{
	LoadObj(TreeDRestruntion_path); // 初始化标准模型数据
	
	string fileName; // 文件名
	cout << "输入文件名：" << endl;
	cin >> fileName;

	string fullPath = ImagePath + fileName;

	const char* path = fullPath.c_str();
	char* path_char = new char[99];
	path_char = const_cast<char*>(path);

	string data_str = FaceShape_Python(path_char); // str格式的json数据

	FaceShapeData faceShapeData = Json2Class(data_str); // FaceShapeData数据

	//ShowImage(path_char, faceShapeData); // 传入faceShapeData，输出图像并标点

	std::vector<Verticle> FeatureData = FeatureOut(faceShapeData); // 获得目标模型的特征点的模型顶点数据

	cout << FeatureData.size() << endl;
	// 用径向基函数计算模型每个点的位置

	RadialBasisFunction(FeatureData);
}


/*数据结构定义*/

Verticle::Verticle(void)
{
	Position = objl::Vector3();
	Indice = 0;
	Feature = objl::Vector2();
	FIndice = 0;
}
Verticle::Verticle(objl::Vector3 _Position, unsigned int _Indice, objl::Vector2 _Feature, unsigned int _FIndice) 
{
	Position = _Position;
	Indice = _Indice;
	Feature = _Feature;
	FIndice = _FIndice;
}
Verticle::~Verticle() {}


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
			StandardModelPositions.push_back(Loader.PointsPositions[i]);
			OutModelPositions.push_back(Loader.PointsPositions[i]);
		}
		for (int i = 0; i < Loader.Faces.size(); i++)
		{
			StandardModelFaces.push_back(Loader.Faces[i]);
		}
		// 初始化标注模型特征点数据
		FeatureVerticlesInitiate(StandardModelPositions);

		cout << "标准模型数据初始化完成" << endl;
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
	objl::Vector2 NoseFeature(positions[nose_Indices[0]].X, positions[nose_Indices[0]].Y);

	for (int i = 0; i < sizeof(Face_profile_Indices) / sizeof(Face_profile_Indices[0]); i++)
	{
		unsigned int Indice = face_profile_Indices[2 * i];
		objl::Vector3 Position = positions[Indice];
		unsigned int FIndice = face_profile_Indices[2 * i + 1];
		objl::Vector2 Feature = objl::Vector2(Position.X, Position.Y) - NoseFeature;
		Face_profile_Indices[i] = Verticle(Position, Indice, Feature, FIndice);
		StandardFeaturePosiotons.push_back(Face_profile_Indices[i]);
	}
	for (int i = 0; i < sizeof(Left_eye_Indices) / sizeof(Left_eye_Indices[0]); i++)
	{

		unsigned int Indice = left_eye_Indices[2 * i];
		objl::Vector3 Position = positions[Indice];
		unsigned int FIndice = left_eye_Indices[2 * i + 1];
		objl::Vector2 Feature = objl::Vector2(Position.X, Position.Y) - NoseFeature;
		Left_eye_Indices[i] = Verticle(Position, Indice, Feature, FIndice);
		StandardFeaturePosiotons.push_back(Left_eye_Indices[i]);
	}	
	for (int i = 0; i < sizeof(Right_eye_Indices) / sizeof(Right_eye_Indices[0]); i++)
	{
		unsigned int Indice = right_eye_Indices[2 * i];
		objl::Vector3 Position = positions[Indice];
		unsigned int FIndice = right_eye_Indices[2 * i + 1];
		objl::Vector2 Feature = objl::Vector2(Position.X, Position.Y) - NoseFeature;
		Right_eye_Indices[i] = Verticle(Position, Indice, Feature, FIndice);
		StandardFeaturePosiotons.push_back(Right_eye_Indices[i]);
	}	
	for (int i = 0; i < sizeof(Left_eyebrow_Indices) / sizeof(Left_eyebrow_Indices[0]); i++)
	{
		unsigned int Indice = left_eyebrow_Indices[2 * i];
		objl::Vector3 Position = positions[Indice];
		unsigned int FIndice = left_eyebrow_Indices[2 * i + 1];
		objl::Vector2 Feature = objl::Vector2(Position.X, Position.Y) - NoseFeature;
		Left_eyebrow_Indices[i] = Verticle(Position, Indice, Feature, FIndice);
		StandardFeaturePosiotons.push_back(Left_eyebrow_Indices[i]);
	}	
	for (int i = 0; i < sizeof(Right_eyebrow_Indices) / sizeof(Right_eyebrow_Indices[0]); i++)
	{
		unsigned int Indice = right_eyebrow_Indices[2 * i];
		objl::Vector3 Position = positions[Indice];
		unsigned int FIndice = right_eyebrow_Indices[2 * i + 1];
		objl::Vector2 Feature = objl::Vector2(Position.X, Position.Y) - NoseFeature;
		Right_eyebrow_Indices[i] = Verticle(Position, Indice, Feature, FIndice);
		StandardFeaturePosiotons.push_back(Right_eyebrow_Indices[i]);
	}	
	for (int i = 0; i < sizeof(Mouth_Indices) / sizeof(Mouth_Indices[0]); i++)
	{
		unsigned int Indice = mouth_Indices[2 * i];
		objl::Vector3 Position = positions[Indice];
		unsigned int FIndice = mouth_Indices[2 * i + 1];
		objl::Vector2 Feature = objl::Vector2(Position.X, Position.Y) - NoseFeature;
		Mouth_Indices[i] = Verticle(Position, Indice, Feature, FIndice);
		StandardFeaturePosiotons.push_back(Mouth_Indices[i]);
	}	
	for (int i = 0; i < sizeof(Nose_Indices) / sizeof(Nose_Indices[0]); i++)
	{
		unsigned int Indice = nose_Indices[2 * i];
		objl::Vector3 Position = positions[Indice];
		unsigned int FIndice = nose_Indices[2 * i + 1];
		objl::Vector2 Feature = objl::Vector2(Position.X, Position.Y) - NoseFeature;
		Nose_Indices[i] = Verticle(Position, Indice, Feature, FIndice);
		StandardFeaturePosiotons.push_back(Nose_Indices[i]);
	}

	// 对特征点排序
	for (int i = 0; i < StandardFeaturePosiotons.size() - 1; i++)
	{
		for (int j = 0; j < StandardFeaturePosiotons.size() - i - 1; j++)
		{
			if (StandardFeaturePosiotons[j].Indice > StandardFeaturePosiotons[j + 1].Indice)
			{
				swap(StandardFeaturePosiotons[j], StandardFeaturePosiotons[j + 1]);
			}
		}
	}
}

std::vector<Verticle> FeatureOut(FaceShapeData faceShapeData)
{
	std::vector<Verticle> Out;
	double StandardLength = 0;
	double OutLength = 0;
	objl::Vector2 NoseF_(faceShapeData.data.face_shape_list[0].nose[0].X, faceShapeData.data.face_shape_list[0].nose[0].Y);

	for (int i = 0; i < sizeof(Face_profile_Indices) / sizeof(Face_profile_Indices[0]); i++)
	{
		unsigned int Indice = Face_profile_Indices[i].Indice;
		unsigned int FIndice = Face_profile_Indices[i].FIndice;
		objl::Vector3 Position = objl::Vector3(faceShapeData.data.face_shape_list[0].face_profile[FIndice].X, faceShapeData.data.face_shape_list[0].face_profile[FIndice].Y, Face_profile_Indices[i].Position.Z);
		objl::Vector2 Feature = objl::Vector2(Position.X, Position.Y) - NoseF_;
		Out.push_back(Verticle(Position, Indice, Feature, FIndice));
		StandardLength += Face_profile_Indices[i].Feature.magnitude();
		OutLength += Feature.magnitude();
	}
	for (int i = 0; i < sizeof(Left_eye_Indices) / sizeof(Left_eye_Indices[0]); i++)
	{
		unsigned int Indice = Left_eye_Indices[i].Indice;
		unsigned int FIndice = Left_eye_Indices[i].FIndice;
		objl::Vector3 Position = objl::Vector3(faceShapeData.data.face_shape_list[0].left_eye[FIndice].X, faceShapeData.data.face_shape_list[0].left_eye[FIndice].Y, Left_eye_Indices[i].Position.Z);
		objl::Vector2 Feature = objl::Vector2(Position.X, Position.Y) - NoseF_;
		Out.push_back(Verticle(Position, Indice, Feature, FIndice));
		StandardLength += Face_profile_Indices[i].Feature.magnitude();
		OutLength += Feature.magnitude();
	}
	for (int i = 0; i < sizeof(Right_eye_Indices) / sizeof(Right_eye_Indices[0]); i++)
	{
		unsigned int Indice = Right_eye_Indices[i].Indice;
		unsigned int FIndice = Right_eye_Indices[i].FIndice;
		objl::Vector3 Position = objl::Vector3(faceShapeData.data.face_shape_list[0].right_eye[FIndice].X, faceShapeData.data.face_shape_list[0].right_eye[FIndice].Y, Right_eye_Indices[i].Position.Z);
		objl::Vector2 Feature = objl::Vector2(Position.X, Position.Y) - NoseF_;
		Out.push_back(Verticle(Position, Indice, Feature, FIndice));
		StandardLength += Face_profile_Indices[i].Feature.magnitude();
		OutLength += Feature.magnitude();
	}
	for (int i = 0; i < sizeof(Left_eyebrow_Indices) / sizeof(Left_eyebrow_Indices[0]); i++)
	{
		unsigned int Indice = Left_eyebrow_Indices[i].Indice;
		unsigned int FIndice = Left_eyebrow_Indices[i].FIndice;
		objl::Vector3 Position = objl::Vector3(faceShapeData.data.face_shape_list[0].left_eyebrow[FIndice].X, faceShapeData.data.face_shape_list[0].left_eyebrow[FIndice].Y, Left_eyebrow_Indices[i].Position.Z);
		objl::Vector2 Feature = objl::Vector2(Position.X, Position.Y) - NoseF_;
		Out.push_back(Verticle(Position, Indice, Feature, FIndice));
		StandardLength += Face_profile_Indices[i].Feature.magnitude();
		OutLength += Feature.magnitude();
	}
	for (int i = 0; i < sizeof(Right_eyebrow_Indices) / sizeof(Right_eyebrow_Indices[0]); i++)
	{
		unsigned int Indice = Right_eyebrow_Indices[i].Indice;
		unsigned int FIndice = Right_eyebrow_Indices[i].FIndice;
		objl::Vector3 Position = objl::Vector3(faceShapeData.data.face_shape_list[0].right_eyebrow[FIndice].X, faceShapeData.data.face_shape_list[0].right_eyebrow[FIndice].Y, Right_eyebrow_Indices[i].Position.Z);
		objl::Vector2 Feature = objl::Vector2(Position.X, Position.Y) - NoseF_;
		Out.push_back(Verticle(Position, Indice, Feature, FIndice));
		StandardLength += Face_profile_Indices[i].Feature.magnitude();
		OutLength += Feature.magnitude();
	}
	for (int i = 0; i < sizeof(Mouth_Indices) / sizeof(Mouth_Indices[0]); i++)
	{
		unsigned int Indice = Mouth_Indices[i].Indice;
		unsigned int FIndice = Mouth_Indices[i].FIndice;
		objl::Vector3 Position = objl::Vector3(faceShapeData.data.face_shape_list[0].mouth[FIndice].X, faceShapeData.data.face_shape_list[0].mouth[FIndice].Y, Mouth_Indices[i].Position.Z);
		objl::Vector2 Feature = objl::Vector2(Position.X, Position.Y) - NoseF_;
		Out.push_back(Verticle(Position, Indice, Feature, FIndice));
		StandardLength += Face_profile_Indices[i].Feature.magnitude();
		OutLength += Feature.magnitude();
	}
	for (int i = 0; i < sizeof(Nose_Indices) / sizeof(Nose_Indices[0]); i++)
	{
		unsigned int Indice = Nose_Indices[i].Indice;
		unsigned int FIndice = Nose_Indices[i].FIndice;
		objl::Vector3 Position = objl::Vector3(faceShapeData.data.face_shape_list[0].nose[FIndice].X, faceShapeData.data.face_shape_list[0].nose[FIndice].Y, Nose_Indices[i].Position.Z);
		objl::Vector2 Feature = objl::Vector2(Position.X, Position.Y) - NoseF_;
		Out.push_back(Verticle(Position, Indice, Feature, FIndice));
		StandardLength += Face_profile_Indices[i].Feature.magnitude();
		OutLength += Feature.magnitude();
	}

	Zchange = OutLength / StandardLength;
	cout << "模型轴形变比例 = " << Zchange << endl;

	for (int i = 0; i < Out.size() - 1; i++)
	{
		for (int j = 0; j < Out.size() - i - 1; j++)
		{
			if (Out[j].Indice > Out[j + 1].Indice)
			{
				swap(Out[j], Out[j + 1]);
				swap(StandardFeaturePosiotons[j], StandardFeaturePosiotons[j + 1]);
			}
		}
	}

	return Out;
}

void RadialBasisFunction(std::vector<Verticle> FeatureData)
{
	std::vector<Verticle>::iterator it = FeatureData.begin();

	for (int i = 0; i < OutModelPositions.size(); i++)
	{
		// 这个点是 OutModelPositions[i],在未变换前与标准模型对应点相同
		//cout << FeatureData.end()->Indice << endl;

		// 如果该点是特征点
		if (i == it->Indice)
		{
			OutModelPositions[i] = it->Position;
			if(it != (FeatureData.end()-1))
				it++;
		}
		// 该点不是特征点
		else 
		{
			objl::Vector3 Move = objl:: Vector3();
			for (int j = 0; j < FeatureData.size(); j++) 
			{
				double r = (FeatureData[j].Position - OutModelPositions[i]).magnitude();
				objl::Vector3 Dir = FeatureData[j].Position - StandardFeaturePosiotons[j].Position;
				//cout << r << endl;
				//objl::Vector3 delta = Dir * exp(-r * r / R);
				objl::Vector3 delta = Dir * exp(-r * r / R);
				//cout << delta.X << "," << delta.Y << "," << delta.Z << "___";
				Move = Move + delta;
			}
		}
	}
}