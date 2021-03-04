/**************************************************************
MIT License
Copyright (c) 2018 SEU-SuperNova-CVRA
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
Authors:	Florentino Zhang, <danteseu@126.com>
**************************************************************/
#include "AngleSolver.hpp"
#include "opencv2/opencv.hpp"
#include "math.h"

using namespace cv;
using namespace std;

namespace rm
{

/*bool solvePnP(InputArray objectPoints, InputArray imagePoints, InputArray cameraMatrix, InputArray distCoeffs,
 OutputArray rvec, OutputArray tvec, bool useExtrinsicGuess=false, int flags=ITERATIVE )

　　objectPoints：特征点的世界坐标，坐标值需为float型，不能为double型，可以为mat类型，也可以直接输入vector
　　imagePoints：特征点在图像中的像素坐标，可以输入mat类型，也可以直接输入vector，
	注意输入点的顺序要与前面的特征点的世界坐标一一对应
　　cameraMatrix：相机内参矩阵
　　distCoeffs：相机的畸变参数【Mat_(5, 1)】
　　rvec：输出的旋转向量
　　tvec：输出的平移矩阵
　　最后的输入参数有三个可选项：
　　CV_ITERATIVE，默认值，它通过迭代求出重投影误差最小的解作为问题的最优解。
　　CV_P3P则是使用非常经典的Gao的P3P问题求解算法。
　　CV_EPNP使用文章《EPnP: Efficient Perspective-n-Point Camera Pose Estimation》中的方法求解。*/

std::vector<cv::Point3f> AngleSolverParam::POINT_3D_OF_ARMOR_BIG = std::vector<cv::Point3f>
{
/*
大装甲板
objectPoints特征点世界坐标
以特征点(装甲板的四个角点)所在平面为世界坐标XY平面，并在该平面中确定世界坐标的原点
这里以装甲板的中心为世界坐标的原点，并按照逆时针方向逐个输入四个角点的世界坐标
输入需要为float或point3f类型
*/
		cv::Point3f(-105, -30, 0),	//tl
		cv::Point3f(105, -30, 0),	//tr
		cv::Point3f(105, 30, 0),	//br
		cv::Point3f(-105, 30, 0)	//bl
};
std::vector<cv::Point3f> AngleSolverParam::POINT_3D_OF_RUNE = std::vector<cv::Point3f>
{
	/*大符部分*/
	cv::Point3f(-370, -220, 0),
	cv::Point3f(0, -220, 0),
	cv::Point3f(370, -220, 0),
	cv::Point3f(-370, 0, 0),
	cv::Point3f(0, 0, 0),
	cv::Point3f(370, 0, 0),
	cv::Point3f(-370, 220, 0),
	cv::Point3f(0, 220, 0),
	cv::Point3f(370, 220, 0)
};

std::vector<cv::Point3f> AngleSolverParam::POINT_3D_OF_ARMOR_SMALL = std::vector<cv::Point3f>
{
	/*小装甲板角点世界坐标*/
	cv::Point3f(-65, -35, 0),	//tl
	cv::Point3f(65, -35, 0),	//tr
	cv::Point3f(65, 35, 0),		//br
	cv::Point3f(-65, 35, 0)		//bl
};

AngleSolver::AngleSolver()
{
	//将四个像素点坐标初始化为(0, 0)
	for(int ll = 0; ll <= 3; ll++)
		target_nothing.push_back(cv::Point2f(0.0, 0.0));
}

// 构造函数，进行成员变量的赋值和初始化
AngleSolver::AngleSolver(const AngleSolverParam & AngleSolverParam)
{
	_params = AngleSolverParam;
	_cam_instant_matrix = _params.CAM_MATRIX.clone(); //复制相机内参矩阵
	for(int ll = 0; ll <= 3; ll++) //初始化
		target_nothing.push_back(cv::Point2f(0.0, 0.0));
}

// 进行_params和_cam_instant_matrix的赋值
void AngleSolver::init(const AngleSolverParam& AngleSolverParam)
{
	_params = AngleSolverParam;
	_cam_instant_matrix = _params.CAM_MATRIX.clone();
}

void AngleSolver::setTarget(const std::vector<cv::Point2f> objectPoints, int objectType)
{
	if(objectType == 1 || objectType == 2)
	{
		// 将算法重置为PNP算法
		if(angle_solver_algorithm == 0 || angle_solver_algorithm == 2)
		{
			angle_solver_algorithm = 1; cout << "algorithm is reset to PNP Solution" << endl;
		}
		point_2d_of_armor = objectPoints;
		if(objectType == 1)
			enemy_type = 0;
		else
			enemy_type = 1;
		return;
	}
	if(objectType == 3 || objectType == 4)
	{
		// 算法设为PNP解算大符
		angle_solver_algorithm = 2;
		point_2d_of_rune = objectPoints;
	}

}

void AngleSolver::setTarget(const cv::Point2f Center_of_armor, int objectPoint)
{
	if(angle_solver_algorithm == 1 || angle_solver_algorithm == 2)
	{
		angle_solver_algorithm = 0; cout << "algorithm is reset to One Point Solution" << endl;
	}
	centerPoint = Center_of_armor;
	if(objectPoint == 3 || objectPoint == 4)
		is_shooting_rune = 1;
	else
	{
		is_shooting_rune = 0;
		_rune_compensated_angle = 0;
	}

}

/*
void AngleSolver::setTarget(const std::vector<cv::Point2f> runePoints)
{
	angle_solver_algorithm = 2;
	point_2d_of_rune = runePoints;
}
*/

//debug模式执行下方代码
#ifdef DEBUG

void AngleSolver::showPoints2dOfArmor()
{
	cout << "the point 2D of armor is" << point_2d_of_armor << endl;
}


void AngleSolver::showTvec()
{
	cv::Mat tvect;
	transpose(_tVec, tvect);
	cout << "the current _tVec is:" << endl << tvect << endl;
}

void AngleSolver::showEDistance()
{
	cout << "  _euclideanDistance is  " << _euclideanDistance / 1000 << "m" << endl;
}

void AngleSolver::showcenter_of_armor()
{
	cout << "the center of armor is" << centerPoint << endl;
}

void AngleSolver::showAngle()
{
	cout << "_xErr is  " << _xErr << "  _yErr is  " << _yErr << endl;
}

int AngleSolver::showAlgorithm()
{
	return angle_solver_algorithm;
}
#endif // DEBUG


AngleSolver::AngleFlag AngleSolver::solve()
//类外定义AngleSolver类的成员函数solve，返回AngleSolver::AngleFlag类型的数据
{
	if(angle_solver_algorithm == 1) //PNP算法
	{
		if(enemy_type == 1)
			//使用solvePNP求解相机姿态和位置信息  rvec：输出的旋转向量，tvec：输出的平移相量
			//平移向量就是以当前摄像头中心为原点时物体原点所在的位置。
			solvePnP(_params.POINT_3D_OF_ARMOR_BIG, point_2d_of_armor, _cam_instant_matrix, _params.DISTORTION_COEFF, _rVec, _tVec, false, CV_ITERATIVE);
		if(enemy_type == 0)
			solvePnP(_params.POINT_3D_OF_ARMOR_SMALL, point_2d_of_armor, _cam_instant_matrix, _params.DISTORTION_COEFF, _rVec, _tVec, false, CV_ITERATIVE);
		//因为我们的目的是让枪管瞄准装甲板的中央，而得到的坐标是以摄像头中心为原点的，要得到正确的转角需要对这个坐标进行平移，将原点平移到云台的轴上
		//at<double>(1, 0) 返回1行0列对应元素的引用，这里减去相机和gun间的偏移量，进行修正
		_tVec.at<double>(1, 0) -= _params.Y_DISTANCE_BETWEEN_GUN_AND_CAM;
		_xErr = atan(_tVec.at<double>(0, 0) / _tVec.at<double>(2, 0)) / 2 / CV_PI * 360;
		_yErr = atan(_tVec.at<double>(1, 0) / _tVec.at<double>(2, 0)) / 2 / CV_PI * 360;
		_euclideanDistance = sqrt(_tVec.at<double>(0, 0)*_tVec.at<double>(0, 0) + _tVec.at<double>(1, 0)*_tVec.at<double>(1, 0) + _tVec.at<double>(2, 0)* _tVec.at<double>(2, 0));
		if(_euclideanDistance >= 8500)
		{
			return TOO_FAR;
		}
		/*
						if (_x_time_points.size < 5) _x_time_points.push_back(cv::Point2f(framecount,_xErr));
				else {
					_x_time_points[0] = _x_time_points[1];
					_x_time_points[1] = _x_time_points[2];
					_x_time_points[2] = _x_time_points[3];
					_x_time_points[3] = _x_time_points[4];
					_x_time_points[4] = cv::Point2f(framecount, _xErr);
				}
				if (_y_time_points.size < 5) _y_time_points.push_back(cv::Point2f(framecount, _yErr));
				else {
					_y_time_points[0] = _y_time_points[1];
					_y_time_points[1] = _y_time_points[2];
					_y_time_points[2] = _y_time_points[3];
					_y_time_points[3] = _y_time_points[4];
					_y_time_points[4] = cv::Point2f(framecount, _yErr);
				}
		*/
		return ANGLES_AND_DISTANCE;
	}
	if(angle_solver_algorithm == 0)
	{
		double x1, x2, y1, y2, r2, k1, k2, p1, p2, y_ture;
		x1 = (centerPoint.x - _cam_instant_matrix.at<double>(0, 2)) / _cam_instant_matrix.at<double>(0, 0);
		y1 = (centerPoint.y - _cam_instant_matrix.at<double>(1, 2)) / _cam_instant_matrix.at<double>(1, 1);
		r2 = x1 * x1 + y1 * y1;
		k1 = _params.DISTORTION_COEFF.at<double>(0, 0);
		k2 = _params.DISTORTION_COEFF.at<double>(1, 0);
		p1 = _params.DISTORTION_COEFF.at<double>(2, 0);
		p2 = _params.DISTORTION_COEFF.at<double>(3, 0);
		x2 = x1 * (1 + k1 * r2 + k2 * r2*r2) + 2 * p1*x1*y1 + p2 * (r2 + 2 * x1*x1);
		y2 = y1 * (1 + k1 * r2 + k2 * r2*r2) + 2 * p2*x1*y1 + p1 * (r2 + 2 * y1*y1);
		y_ture = y2 - _params.Y_DISTANCE_BETWEEN_GUN_AND_CAM / 1000;
		_xErr = atan(x2) / 2 / CV_PI * 360;
		_yErr = atan(y_ture) / 2 / CV_PI * 360;
		if(is_shooting_rune) _yErr -= _rune_compensated_angle;
		/*
						if (_x_time_points.size < 5) _x_time_points.push_back(cv::Point2f(framecount, _xErr));
				else {
					_x_time_points[0] = _x_time_points[1];
					_x_time_points[1] = _x_time_points[2];
					_x_time_points[2] = _x_time_points[3];
					_x_time_points[3] = _x_time_points[4];
					_x_time_points[4] = cv::Point2f(framecount, _xErr);
				}
				if (_y_time_points.size < 5) _y_time_points.push_back(cv::Point2f(framecount, _yErr));
				else {
					_y_time_points[0] = _y_time_points[1];
					_y_time_points[1] = _y_time_points[2];
					_y_time_points[2] = _y_time_points[3];
					_y_time_points[3] = _y_time_points[4];
					_y_time_points[4] = cv::Point2f(framecount, _yErr);
				}
		*/
		return ONLY_ANGLES;
	}
	if(angle_solver_algorithm == 2)
	{
		std::vector<Point2f> runeCenters;
		std::vector<Point3f> realCenters;
		for(size_t i = 0; i < 9; i++)
		{
			if(point_2d_of_rune[i].x > 0 && point_2d_of_rune[i].y > 0)
			{
				runeCenters.push_back(point_2d_of_rune[i]);
				realCenters.push_back(_params.POINT_3D_OF_RUNE[i]);
			}
		}

		solvePnP(realCenters, runeCenters, _cam_instant_matrix, _params.DISTORTION_COEFF, _rVec, _tVec, false, CV_ITERATIVE);
		_tVec.at<double>(1, 0) -= _params.Y_DISTANCE_BETWEEN_GUN_AND_CAM;
		_xErr = atan(_tVec.at<double>(0, 0) / _tVec.at<double>(2, 0)) / 2 / CV_PI * 360;
		_yErr = atan(_tVec.at<double>(1, 0) / _tVec.at<double>(2, 0)) / 2 / CV_PI * 360;
		_euclideanDistance = sqrt(_tVec.at<double>(0, 0)*_tVec.at<double>(0, 0) + _tVec.at<double>(1, 0)*_tVec.at<double>(1, 0) + _tVec.at<double>(2, 0)* _tVec.at<double>(2, 0));
		if(_euclideanDistance >= 8500)
		{
			return TOO_FAR;
		}
		/*
						if (_x_time_points.size < 5) _x_time_points.push_back(cv::Point2f(framecount, _xErr));
				else {
					_x_time_points[0] = _x_time_points[1];
					_x_time_points[1] = _x_time_points[2];
					_x_time_points[2] = _x_time_points[3];
					_x_time_points[3] = _x_time_points[4];
					_x_time_points[4] = cv::Point2f(framecount, _xErr);
				}
				if (_y_time_points.size < 5) _y_time_points.push_back(cv::Point2f(framecount, _yErr));
				else {
					_y_time_points[0] = _y_time_points[1];
					_y_time_points[1] = _y_time_points[2];
					_y_time_points[2] = _y_time_points[3];
					_y_time_points[3] = _y_time_points[4];
					_y_time_points[4] = cv::Point2f(framecount, _yErr);
				}
		*/
		return ANGLES_AND_DISTANCE;
	}
	return ANGLE_ERROR;
}

//偏移量补偿
void AngleSolver::compensateOffset()
{
	/* z of the camera COMS */
	const auto offset_z = 115.0;
	const auto& d = _euclideanDistance;
	const auto theta_y = _xErr / 180 * CV_PI;
	const auto theta_p = _yErr / 180 * CV_PI;
	const auto theta_y_prime = atan((d*sin(theta_y)) / (d*cos(theta_y) + offset_z));
	const auto theta_p_prime = atan((d*sin(theta_p)) / (d*cos(theta_p) + offset_z));
	const auto d_prime = sqrt(pow(offset_z + d * cos(theta_y), 2) + pow(d*sin(theta_y), 2));
	_xErr = theta_y_prime / CV_PI * 180;
	_yErr = theta_p_prime / CV_PI * 180;
	_euclideanDistance = d_prime;
}

//重力补偿
void AngleSolver::compensateGravity()
{
	const auto& theta_p_prime = _yErr / 180 * CV_PI;
	const auto& d_prime = _euclideanDistance;
	const auto& v = _bullet_speed;
	const auto theta_p_prime2 = atan((sin(theta_p_prime) - 0.5*9.8*d_prime / pow(v, 2)) / cos(theta_p_prime));
	_yErr = theta_p_prime2 / CV_PI * 180;
}


void AngleSolver::setResolution(const cv::Size2i& image_resolution)
{
	//好像有问题
	image_size = image_resolution;
	//cout << _cam_instant_matrix.at<double>(0, 2) << "  " << _cam_instant_matrix.at<double>(1, 2) << endl;
	_cam_instant_matrix.at<double>(0, 2) = _params.CAM_MATRIX.at<double>(0, 2) - (1920 / 2 - image_size.width / 2);
	_cam_instant_matrix.at<double>(1, 2) = _params.CAM_MATRIX.at<double>(1, 2) - (1080 / 2 - image_size.height / 2);
	_cam_instant_matrix.at<double>(0, 0) = _params.CAM_MATRIX.at<double>(0, 0) / (1080 / image_size.height);
	_cam_instant_matrix.at<double>(1, 1) = _params.CAM_MATRIX.at<double>(1, 1) / (1080 / image_size.height);
	//cout << _cam_instant_matrix.at<double>(0, 2) << "  " << _cam_instant_matrix.at<double>(1, 2) << endl;
//cout << _params.CAM_MATRIX.at<double>(0, 2) << "  " << _params.CAM_MATRIX.at<double>(1, 2) << endl;
//cout << endl;
}

void AngleSolver::setUserType(int usertype)
{
	user_type = usertype;
}

void AngleSolver::setEnemyType(int enemytype)
{
	enemy_type = enemytype;
}

//const cv::Vec2f AngleSolver::getCompensateAngle()
//{
//	return cv::Vec2f(_xErr, _yErr);
//}

void AngleSolver::setBulletSpeed(int bulletSpeed)
{
	_bullet_speed = bulletSpeed;
}

const cv::Vec2f AngleSolver::getAngle()
{
	return cv::Vec2f(_xErr, _yErr);
}


double AngleSolver::getDistance()
{
	return _euclideanDistance;
}

//void AngleSolver::selectAlgorithm(const int t)
//{
//	if (t == 0 || t == 1)
//		angle_solver_algorithm = t;
//
//}

//从xml文件中获取枪口和相机间偏移量
void AngleSolverParam::readFile(const int id)
{
	cv::FileStorage fsread("../Pose/angle_solver_params.xml", cv::FileStorage::READ);
	if(!fsread.isOpened())
	{
		std::cerr << "failed to open xml" << std::endl;
		return;
	}
	fsread["Y_DISTANCE_BETWEEN_GUN_AND_CAM"] >> Y_DISTANCE_BETWEEN_GUN_AND_CAM;

#ifdef DEBUG
	std::cout << Y_DISTANCE_BETWEEN_GUN_AND_CAM << std::endl;
#endif // DEBUG

	//通过id来指定获取不同类型的相机内参矩阵和畸变参数
	switch(id)
	{
	case 0:
	{
		fsread["CAMERA_MARTRIX_0"] >> CAM_MATRIX;
		fsread["DISTORTION_COEFF_0"] >> DISTORTION_COEFF;
		return;
	}
	case 1:
	{
		fsread["CAMERA_MARTRIX_1"] >> CAM_MATRIX;
		fsread["DISTORTION_COEFF_1"] >> DISTORTION_COEFF;
		return;
	}
	case 2:
	{
		fsread["CAMERA_MARTRIX_2"] >> CAM_MATRIX;
		fsread["DISTORTION_COEFF_2"] >> DISTORTION_COEFF;
		return;
	}
	case 3:
	{
		fsread["CAMERA_MARTRIX_3"] >> CAM_MATRIX;
		fsread["DISTORTION_COEFF_3"] >> DISTORTION_COEFF;
		return;
	}
	case 4:
	{
		fsread["CAMERA_MARTRIX_4"] >> CAM_MATRIX;
		fsread["DISTORTION_COEFF_4"] >> DISTORTION_COEFF;
		return;
	}
	case 5:
	{
		fsread["CAMERA_MARTRIX_OFFICIAL"] >> CAM_MATRIX;
		fsread["DISTORTION_COEFF_OFFICIAL"] >> DISTORTION_COEFF;
		return;
	}
	case 6:
	{
		fsread["CAMERA_MARTRIX_6"] >> CAM_MATRIX;
		fsread["DISTORTION_COEFF_6"] >> DISTORTION_COEFF;
		return;
	}
	case 7:
	{
		fsread["CAMERA_MARTRIX_7"] >> CAM_MATRIX;
		fsread["DISTORTION_COEFF_7"] >> DISTORTION_COEFF;
		return;
	}
	case 8:
	{
		fsread["CAMERA_MARTRIX_8"] >> CAM_MATRIX;
		fsread["DISTORTION_COEFF_8"] >> DISTORTION_COEFF;
		return;
	}
	case 9:
	{
		fsread["CAMERA_MARTRIX_9"] >> CAM_MATRIX;
		fsread["DISTORTION_COEFF_9"] >> DISTORTION_COEFF;
		return;
	}
	default:
		std::cout << "wrong cam number given." << std::endl;
		return;
	}
	/*
	if (id == 1) {
	fsread["CAMERA_MARTRIX_1"] >> CAM_MATRIX;
	fsread["DISTORTION_COEFF_1"] >> DISTORTION_COEFF;
	return;
	}
	if (id == 2) {
	fsread["CAMERA_MARTRIX_2"] >> CAM_MATRIX;
	fsread["DISTORTION_COEFF_2"] >> DISTORTION_COEFF;
	return;
	}
	if (id == 3) {
	fsread["CAMERA_MARTRIX_3"] >> CAM_MATRIX;
	fsread["DISTORTION_COEFF_3"] >> DISTORTION_COEFF;
	return;
	}
	if (id == 4) {
	fsread["CAMERA_MARTRIX_4"] >> CAM_MATRIX;
	fsread["DISTORTION_COEFF_4"] >> DISTORTION_COEFF;
	return;
	}
	if (id == 5) {
	fsread["CAMERA_MARTRIX_OFFICIAL"] >> CAM_MATRIX;
	fsread["DISTORTION_COEFF_OFFICIAL"] >> DISTORTION_COEFF;
	return;
	}
	if (id == 0) {
	fsread["CAMERA_MARTRIX_0"] >> CAM_MATRIX;
	fsread["DISTORTION_COEFF_0"] >> DISTORTION_COEFF;
	return;
	}
	std::cout << "wrong cam number given." << std::endl;*/
}
}

