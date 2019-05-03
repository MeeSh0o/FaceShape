#ifndef __GEOM_H
#define __GEOM_H

#include <OBJ_Loader.h>
#include <cmath>
#include <boost/numeric/ublas/matrix.hpp>

using namespace boost::numeric::ublas;


#define GRID_W 100
#define GRID_H 100
static float grid[GRID_W][GRID_H];


#define EPSILON 0.00001f
#define PI 3.1415926
#define Deg2Rad(Ang) ((float)( Ang * PI / 180.0 ))
#define Rad2Deg(Ang) ((float)( Ang * 180.0 / PI ))

int selected_cp = -1;

double regularization = 0.0;
double bending_energy = 0.0;

// =========================================
// 4 x 4 matrix
// =========================================
class Mtx
{
public:

	// 4x4, [[0 1 2 3] [4 5 6 7] [8 9 10 11] [12 13 14 15]]
	float data[16];

	// Creates an identity matrix
	Mtx()
	{
		for (int i = 0; i < 16; ++i)
			data[i] = 0;
		data[0 + 0] = data[4 + 1] = data[8 + 2] = data[12 + 3] = 1;
	}

	// Returns the transpose of this matrix
	Mtx transpose() const
	{
		Mtx m;
		int idx = 0;
		for (int row = 0; row < 4; ++row)
			for (int col = 0; col < 4; ++col, ++idx)
				m.data[idx] = data[row + (col * 4)];
		return m;
	}

	// Operators
	float operator () (unsigned column, unsigned row)
	{
		return data[column + (row * 4)];
	}
};

// Creates a scale matrix
Mtx scale(const objl::Vector3 & scale)
{
	Mtx m;
	m.data[0 + 0] = scale.X;
	m.data[4 + 1] = scale.Y;
	m.data[8 + 2] = scale.Z;
	return m;
}

// Creates a translation matrix
Mtx translate(const objl::Vector3& moveAmt)
{
	Mtx m;
	m.data[0 + 3] = moveAmt.X;
	m.data[4 + 3] = moveAmt.Y;
	m.data[8 + 3] = moveAmt.Z;
	return m;
}

// Creates an euler rotation matrix (by X-axis)
Mtx rotateX(float ang)
{
	float s = (float)sin(Deg2Rad(ang));
	float c = (float)cos(Deg2Rad(ang));

	Mtx m;
	m.data[4 + 1] = c; m.data[4 + 2] = -s;
	m.data[8 + 1] = s; m.data[8 + 2] = c;
	return m;
}

// Creates an euler rotation matrix (by Y-axis)
Mtx rotateY(float ang)
{
	float s = (float)sin(Deg2Rad(ang));
	float c = (float)cos(Deg2Rad(ang));

	Mtx m;
	m.data[0 + 0] = c; m.data[0 + 2] = s;
	m.data[8 + 0] = -s; m.data[8 + 2] = c;
	return m;
}

// Creates an euler rotation matrix (by Z-axis)
Mtx rotateZ(float ang)
{
	float s = (float)sin(Deg2Rad(ang));
	float c = (float)cos(Deg2Rad(ang));

	Mtx m;
	m.data[0 + 0] = c; m.data[0 + 1] = -s;
	m.data[4 + 0] = s; m.data[4 + 1] = c;
	return m;
}

// Creates an euler rotation matrix (pitch/head/roll (x/y/z))
Mtx rotate(float pitch, float head, float roll)
{
	float sp = (float)sin(Deg2Rad(pitch));
	float cp = (float)cos(Deg2Rad(pitch));
	float sh = (float)sin(Deg2Rad(head));
	float ch = (float)cos(Deg2Rad(head));
	float sr = (float)sin(Deg2Rad(roll));
	float cr = (float)cos(Deg2Rad(roll));

	Mtx m;
	m.data[0 + 0] = cr * ch - sr * sp * sh;
	m.data[0 + 1] = -sr * cp;
	m.data[0 + 2] = cr * sh + sr * sp * ch;

	m.data[4 + 0] = sr * ch + cr * sp * sh;
	m.data[4 + 1] = cr * cp;
	m.data[4 + 2] = sr * sh - cr * sp * ch;

	m.data[8 + 0] = -cp * sh;
	m.data[8 + 1] = sp;
	m.data[8 + 2] = cp * ch;
	return m;
}

// Creates an arbitraty rotation matrix
Mtx makeRotationMatrix(const objl::Vector3& dir, const objl::Vector3& up)
{
	objl::Vector3 x = cross(up, dir), y = cross(dir, x), z = dir;
	Mtx m;
	m.data[0] = x.X; m.data[1] = x.Y; m.data[2] = x.Z;
	m.data[4] = y.X; m.data[5] = y.Y; m.data[6] = y.Z;
	m.data[8] = z.X; m.data[9] = z.Y; m.data[10] = z.Z;
	return m;
}
// Transforms a vector by a matrix
inline objl::Vector3 operator * (const objl::Vector3& v, const Mtx& m)
{
	return objl::Vector3(
		m.data[0] * v.X + m.data[1] * v.Y + m.data[2] * v.Z + m.data[3],
		m.data[4] * v.X + m.data[5] * v.Y + m.data[6] * v.Z + m.data[7],
		m.data[8] * v.X + m.data[9] * v.Y + m.data[10] * v.Z + m.data[11]);
}

// Multiplies a matrix by another matrix
Mtx operator * (const Mtx& a, const Mtx& b)
{
	Mtx ans;
	for (int aRow = 0; aRow < 4; ++aRow)
		for (int bCol = 0; bCol < 4; ++bCol)
		{
			int aIdx = aRow * 4;
			int bIdx = bCol;

			float val = 0;
			for (int idx = 0; idx < 4; ++idx, ++aIdx, bIdx += 4)
				val += a.data[aIdx] * b.data[bIdx];
			ans.data[bCol + aRow * 4] = val;
		}
	return ans;
}


// =========================================
// Plane
// =========================================
class Plane
{
public:
	enum PLANE_EVAL
	{
		EVAL_COINCIDENT,
		EVAL_IN_BACK_OF,
		EVAL_IN_FRONT_OF,
		EVAL_SPANNING
	};

	objl::Vector3 normal;
	float d;

	// Default constructor
	Plane() : normal(0, 1, 0), d(0) {}

	// Vector form constructor
	//   normal = normalized normal of the plane
	//   pt = any point on the plane
	Plane(const objl::Vector3& normal, const objl::Vector3& pt)
		: normal(normal), d(dot(-normal, pt)) {}

	// Copy constructor
	Plane(const Plane& a)
		: normal(a.normal), d(a.d) {}

	// Classifies a point (<0 == back, 0 == on plane, >0 == front)
	float classify(const objl::Vector3& pt) const
	{
		float f = dot(normal, pt) + d;
		return (f > -EPSILON && f < EPSILON) ? 0 : f;
	}
};

static double thin_plate_splines(double r)
{
	if (r == 0.0)
		return 0.0;
	else
		return r * r * log(r);
}
// 从控制点计算薄板样条(TPS)权值，通过插值建立新的高度网格。3D
static void calc_tps(std::vector< objl::Vector3 > control_points)
{
	// 控制点最少有3个
	if (control_points.size() < 3)
		return;

	unsigned p = control_points.size();

	// 配置矩阵和向量
	matrix<double> mtx_l(p + 3, p + 3);
	matrix<double> mtx_v(p + 3, 1);
	matrix<double> mtx_orig_k(p, p);

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
			objl::Vector3 pt_i = control_points[i];
			objl::Vector3 pt_j = control_points[j];
			pt_i.Y = pt_j.Y = 0;
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
		mtx_l(i, p + 0) = 1.0;
		mtx_l(i, p + 1) = control_points[i].X;
		mtx_l(i, p + 2) = control_points[i].Z;

		// P 转置 (3 x p, 左下角)
		mtx_l(p + 0, i) = 1.0;
		mtx_l(p + 1, i) = control_points[i].X;
		mtx_l(p + 2, i) = control_points[i].Z;
	}
	// O (3 x 3, 右下角)
	for (unsigned i = p; i < p + 3; ++i)
		for (unsigned j = p; j < p + 3; ++j)
			mtx_l(i, j) = 0.0;


	// 向量 V
	for (unsigned i = 0; i < p; ++i)
		mtx_v(i, 0) = control_points[i].Y;
	mtx_v(p + 0, 0) = mtx_v(p + 1, 0) = mtx_v(p + 2, 0) = 0.0;

	// LU分解法解线性方程组 "inplace" 解存在mtx_v中
	if (0 != LU_Solve(mtx_l, mtx_v))
	{
		puts("Singular matrix! Aborting.");
		exit(1);
	}

	// 设置网格高度
	for (int x = -GRID_W / 2; x < GRID_W / 2; ++x)
	{
		for (int z = -GRID_H / 2; z < GRID_H / 2; ++z)
		{
			double h = mtx_v(p + 0, 0) + mtx_v(p + 1, 0) * x + mtx_v(p + 2, 0) * z;
			objl::Vector3 pt_i, pt_cur(x, 0, z);
			for (unsigned i = 0; i < p; ++i)
			{
				pt_i = control_points[i];
				pt_i.Y = 0;
				h += mtx_v(i, 0) * thin_plate_splines((pt_i - pt_cur).magnitude());
			}
			grid[x + GRID_W / 2][z + GRID_H / 2] = h;
		}
	}

	// 计算弯曲能量
	matrix<double> w(p, 1);
	for (int i = 0; i < p; ++i)
		w(i, 0) = mtx_v(i, 0);
	matrix<double> be = prod(prod<matrix<double> >(trans(w), mtx_orig_k), w);
	bending_energy = be(0, 0);
}
// 从控制点计算薄板样条(TPS)权值，通过插值建立新的高度网格。2D，仅X方向插值量
static void calc_tps2D(std::vector< objl::Vector2 > control_points)
{
	// 控制点最少有3个
	if (control_points.size() < 3)
		return;

	unsigned p = control_points.size();

	// 配置矩阵和向量
	matrix<double> mtx_l(p + 3, p + 3);
	matrix<double> mtx_v(p + 3, 1);
	matrix<double> mtx_orig_k(p, p);

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
			objl::Vector2 pt_i = control_points[i];
			objl::Vector2 pt_j = control_points[j];
			pt_i.Y = pt_j.Y = 0;
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

		// P (p3，右上角)
		mtx_l(i, p + 0) = 1.0;
		mtx_l(i, p + 1) = control_points[i].X;
		mtx_l(i, p + 2) = control_points[i].Y;

		// P 转置 (3 x p, 左下角)
		mtx_l(p + 0, i) = 1.0;
		mtx_l(p + 1, i) = control_points[i].X;
		mtx_l(p + 2, i) = control_points[i].Y;
	}
	// O (3 x 3, 右下角)
	for (unsigned i = p; i < p + 3; ++i)
		for (unsigned j = p; j < p + 3; ++j)
			mtx_l(i, j) = 0.0;


	// Fill the right hand vector V
	for (unsigned i = 0; i < p; ++i)
		mtx_v(i, 0) = control_points[i].X;
	mtx_v(p + 0, 0) = mtx_v(p + 1, 0) = mtx_v(p + 2, 0) = 0.0;

	// Solve the linear system "inplace"
	if (0 != LU_Solve(mtx_l, mtx_v))
	{
		puts("Singular matrix! Aborting.");
		exit(1);
	}

	// Interpolate grid heights
	for (int x = -GRID_W / 2; x < GRID_W / 2; ++x)
	{
		for (int z = -GRID_H / 2; z < GRID_H / 2; ++z)
		{
			double h = mtx_v(p + 0, 0) + mtx_v(p + 1, 0) * x + mtx_v(p + 2, 0) * z;
			objl::Vector2 pt_i, pt_cur(x, z);
			for (unsigned i = 0; i < p; ++i)
			{
				pt_i = control_points[i];
				pt_i.Y = 0;
				h += mtx_v(i, 0) * thin_plate_splines((pt_i - pt_cur).magnitude());
			}
			grid[x + GRID_W / 2][z + GRID_H / 2] = h;
		}
	}

	// Calc bending energy
	matrix<double> w(p, 1);
	for (int i = 0; i < p; ++i)
		w(i, 0) = mtx_v(i, 0);
	matrix<double> be = prod(prod<matrix<double> >(trans(w), mtx_orig_k), w);
	bending_energy = be(0, 0);
}

#endif