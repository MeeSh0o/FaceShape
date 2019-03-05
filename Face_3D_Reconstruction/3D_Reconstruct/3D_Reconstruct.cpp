// 3D_Reconstruct.cpp : 定义控制台应用程序的入口点。
//

#if 1 
/*
* eos - A 3D Morphable Model fitting library written in modern C++11/14.
*
* File: examples/fit-model.cpp
*
* Copyright 2016 Patrik Huber
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
#include "stdafx.h"
#include "eos/core/Landmark.hpp"
#include "eos/core/LandmarkMapper.hpp"
#include "eos/morphablemodel/MorphableModel.hpp"
#include "eos/morphablemodel/Blendshape.hpp"
#include "eos/fitting/fitting.hpp"
#include "eos/render/utils.hpp"
#include "eos/render/texture_extraction.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "boost/program_options.hpp"
#include "boost/filesystem.hpp"

#include <vector>
#include <iostream>
#include <fstream>

using namespace eos;
namespace po = boost::program_options;
namespace fs = boost::filesystem;
using eos::core::Landmark;
using eos::core::LandmarkCollection;
using cv::Mat;
using cv::Vec2f;
using cv::Vec3f;
using cv::Vec4f;
using std::cout;
using std::endl;
using std::vector;
using std::string;

/**
* Reads an ibug .pts landmark file and returns an ordered vector with
* the 68 2D landmark coordinates.
*
* @param[in] filename Path to a .pts file.
* @return An ordered vector with the 68 ibug landmarks.
*/
LandmarkCollection<cv::Vec2f> read_pts_landmarks(std::string filename)
{
	using std::getline;
	using cv::Vec2f;
	using std::string;
	LandmarkCollection<Vec2f> landmarks;
	landmarks.reserve(68);

	std::ifstream file(filename);
	if (!file.is_open()) {
		throw std::runtime_error(string("Could not open landmark file: " + filename));
	}

	string line;
	// Skip the first 3 lines, they're header lines:
	getline(file, line); // 'version: 1'
	getline(file, line); // 'n_points : 68'
	getline(file, line); // '{'

	int ibugId = 1;
	while (getline(file, line))
	{
		if (line == "}") { // end of the file
			break;
		}
		std::stringstream lineStream(line);

		Landmark<Vec2f> landmark;
		landmark.name = std::to_string(ibugId);
		if (!(lineStream >> landmark.coordinates[0] >> landmark.coordinates[1])) {
			throw std::runtime_error(string("Landmark format error while parsing the line: " + line));
		}
		// From the iBug website:
		// "Please note that the re-annotated data for this challenge are saved in the Matlab convention of 1 being
		// the first index, i.e. the coordinates of the top left pixel in an image are x=1, y=1."
		// ==> So we shift every point by 1:
		landmark.coordinates[0] -= 1.0f;
		landmark.coordinates[1] -= 1.0f;
		landmarks.emplace_back(landmark);
		++ibugId;
	}
	return landmarks;
};

/**
* Draws the given mesh as wireframe into the image.
*
* It does backface culling, i.e. draws only vertices in CCW order.
*
* @param[in] image An image to draw into.
* @param[in] mesh The mesh to draw.
* @param[in] modelview Model-view matrix to draw the mesh.
* @param[in] projection Projection matrix to draw the mesh.
* @param[in] viewport Viewport to draw the mesh.
* @param[in] colour Colour of the mesh to be drawn.
*/
void draw_wireframe(cv::Mat image, const core::Mesh& mesh, glm::mat4x4 modelview, glm::mat4x4 projection, glm::vec4 viewport, cv::Scalar colour = cv::Scalar(0, 255, 0, 255))
{
	for (const auto& triangle : mesh.tvi)
	{
		const auto p1 = glm::project({ mesh.vertices[triangle[0]][0], mesh.vertices[triangle[0]][1], mesh.vertices[triangle[0]][2] }, modelview, projection, viewport);
		const auto p2 = glm::project({ mesh.vertices[triangle[1]][0], mesh.vertices[triangle[1]][1], mesh.vertices[triangle[1]][2] }, modelview, projection, viewport);
		const auto p3 = glm::project({ mesh.vertices[triangle[2]][0], mesh.vertices[triangle[2]][1], mesh.vertices[triangle[2]][2] }, modelview, projection, viewport);
		if (render::detail::are_vertices_ccw_in_screen_space(glm::vec2(p1), glm::vec2(p2), glm::vec2(p3)))
		{
			cv::line(image, cv::Point(p1.x, p1.y), cv::Point(p2.x, p2.y), colour);
			cv::line(image, cv::Point(p2.x, p2.y), cv::Point(p3.x, p3.y), colour);
			cv::line(image, cv::Point(p3.x, p3.y), cv::Point(p1.x, p1.y), colour);
		}
	}
};

/**
* This app demonstrates estimation of the camera and fitting of the shape
* model of a 3D Morphable Model from an ibug LFPW image with its landmarks.
* In addition to fit-model-simple, this example uses blendshapes, contour-
* fitting, and can iterate the fitting.
*
* 68 ibug landmarks are loaded from the .pts file and converted
* to vertex indices using the LandmarkMapper.
*/
int main(int argc, char *argv[])
{
	fs::path modelfile, isomapfile, imagefile, landmarksfile, mappingsfile, contourfile, edgetopologyfile, blendshapesfile, outputfile;
	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
				"display the help message")
				("model,m", po::value<fs::path>(&modelfile)->required()->default_value("../share/sfm_shape_3448.bin"),
					"a Morphable Model stored as cereal BinaryArchive")
					("image,i", po::value<fs::path>(&imagefile)->required()->default_value("data/image_0129.png"),
						"an input image")
						("landmarks,l", po::value<fs::path>(&landmarksfile)->required()->default_value("data/image_0129.pts"),
							"2D landmarks for the image, in ibug .pts format")
							("mapping,p", po::value<fs::path>(&mappingsfile)->required()->default_value("../share/ibug_to_sfm.txt"),
								"landmark identifier to model vertex number mapping")
								("model-contour,c", po::value<fs::path>(&contourfile)->required()->default_value("../share/model_contours.json"),
									"file with model contour indices")
									("edge-topology,e", po::value<fs::path>(&edgetopologyfile)->required()->default_value("../share/sfm_3448_edge_topology.json"),
										"file with model's precomputed edge topology")
										("blendshapes,b", po::value<fs::path>(&blendshapesfile)->required()->default_value("../share/expression_blendshapes_3448.bin"),
											"file with blendshapes")
											("output,o", po::value<fs::path>(&outputfile)->required()->default_value("out"),
												"basename for the output rendering and obj files")
			;
		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
		if (vm.count("help")) {
			cout << "Usage: fit-model [options]" << endl;
			cout << desc;
			return EXIT_SUCCESS;
		}
		po::notify(vm);
	}
	catch (const po::error& e) {
		cout << "Error while parsing command-line arguments: " << e.what() << endl;
		cout << "Use --help to display a list of options." << endl;
		return EXIT_FAILURE;
	}

	// Load the image, landmarks, LandmarkMapper and the Morphable Model:
	Mat image = cv::imread(imagefile.string());
	LandmarkCollection<cv::Vec2f> landmarks;
	try {
		landmarks = read_pts_landmarks(landmarksfile.string());
	}

	catch (const std::runtime_error& e) {
		cout << "Error reading the landmarks: " << e.what() << endl;
		return EXIT_FAILURE;
	}

	cv::imshow("Original image", image);
	morphablemodel::MorphableModel morphable_model;
	try {
		morphable_model = morphablemodel::load_model(modelfile.string());
	}
	catch (const std::runtime_error& e) {
		cout << "Error loading the Morphable Model: " << e.what() << endl;
		return EXIT_FAILURE;
	}
	// The landmark mapper is used to map ibug landmark identifiers to vertex ids:
	core::LandmarkMapper landmark_mapper = mappingsfile.empty() ? core::LandmarkMapper() : core::LandmarkMapper(mappingsfile);

	// The expression blendshapes:
	vector<morphablemodel::Blendshape> blendshapes = morphablemodel::load_blendshapes(blendshapesfile.string());

	// These two are used to fit the front-facing contour to the ibug contour landmarks:
	fitting::ModelContour model_contour = contourfile.empty() ? fitting::ModelContour() : fitting::ModelContour::load(contourfile.string());
	fitting::ContourLandmarks ibug_contour = fitting::ContourLandmarks::load(mappingsfile.string());

	// The edge topology is used to speed up computation of the occluding face contour fitting:
	morphablemodel::EdgeTopology edge_topology = morphablemodel::load_edge_topology(edgetopologyfile.string());

	// Draw the loaded landmarks:
	Mat outimg = image.clone();
	for (auto&& lm : landmarks) {
		cv::rectangle(outimg, cv::Point2f(lm.coordinates[0] - 2.0f, lm.coordinates[1] - 2.0f), cv::Point2f(lm.coordinates[0] + 2.0f, lm.coordinates[1] + 2.0f), { 255, 0, 0 });
	}

	// Fit the model, get back a mesh and the pose:
	core::Mesh mesh;
	fitting::RenderingParameters rendering_params;
	std::tie(mesh, rendering_params) = fitting::fit_shape_and_pose(morphable_model, blendshapes, landmarks, landmark_mapper, image.cols, image.rows, edge_topology, ibug_contour, model_contour, 50, boost::none, 30.0f);

	// The 3D head pose can be recovered as follows:
	float yaw_angle = glm::degrees(glm::yaw(rendering_params.get_rotation()));
	// and similarly for pitch and roll.

	// Extract the texture from the image using given mesh and camera parameters:
	Mat affine_from_ortho = fitting::get_3x4_affine_camera_matrix(rendering_params, image.cols, image.rows);
	Mat isomap = render::extract_texture(mesh, affine_from_ortho, image);

	// Draw the fitted mesh as wireframe, and save the image:
	draw_wireframe(outimg, mesh, rendering_params.get_modelview(), rendering_params.get_projection(), fitting::get_opencv_viewport(image.cols, image.rows));
	outputfile += fs::path(".png");
	cv::imwrite(outputfile.string(), outimg);

	// Save the mesh as textured obj:
	outputfile.replace_extension(".obj");
	core::write_textured_obj(mesh, outputfile.string());

	// And save the isomap:
	outputfile.replace_extension(".isomap.png");
	cv::imwrite(outputfile.string(), isomap);

	cout << "Finished fitting and wrote result mesh and isomap to files with basename " << outputfile.stem().stem() << "." << endl;

	return EXIT_SUCCESS;
}

#endif

#if 0
#include "eos/core/Landmark.hpp"
#include "eos/core/LandmarkMapper.hpp"
#include "eos/fitting/orthographic_camera_estimation_linear.hpp"
#include "eos/fitting/RenderingParameters.hpp"
#include "eos/fitting/linear_shape_fitting.hpp"
#include "eos/render/utils.hpp"
#include "eos/render/texture_extraction.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "boost/program_options.hpp"
#include "boost/filesystem.hpp"

#include <vector>
#include <iostream>
#include <fstream>

using namespace eos;
namespace po = boost::program_options;
namespace fs = boost::filesystem;
using eos::core::Landmark;
using eos::core::LandmarkCollection;
using cv::Mat;
using cv::Vec2f;
using cv::Vec3f;
using cv::Vec4f;
using std::cout;
using std::endl;
using std::vector;
using std::string;

/**
* Reads an ibug .pts landmark file and returns an ordered vector with
* the 68 2D landmark coordinates.
*
* @param[in] filename Path to a .pts file.
* @return An ordered vector with the 68 ibug landmarks.
*/
LandmarkCollection<cv::Vec2f> read_pts_landmarks(std::string filename)
{
	using std::getline;
	using cv::Vec2f;
	using std::string;
	LandmarkCollection<Vec2f> landmarks;
	landmarks.reserve(68);

	std::ifstream file(filename);
	if (!file.is_open()) {
		throw std::runtime_error(string("Could not open landmark file: " + filename));
	}

	string line;
	// Skip the first 3 lines, they're header lines:
	getline(file, line); // 'version: 1'
	getline(file, line); // 'n_points : 68'
	getline(file, line); // '{'

	int ibugId = 1;
	while (getline(file, line))
	{
		if (line == "}") { // end of the file
			break;
		}
		std::stringstream lineStream(line);

		Landmark<Vec2f> landmark;
		landmark.name = std::to_string(ibugId);
		if (!(lineStream >> landmark.coordinates[0] >> landmark.coordinates[1])) {
			throw std::runtime_error(string("Landmark format error while parsing the line: " + line));
		}
		// From the iBug website:
		// "Please note that the re-annotated data for this challenge are saved in the Matlab convention of 1 being
		// the first index, i.e. the coordinates of the top left pixel in an image are x=1, y=1."
		// ==> So we shift every point by 1:
		landmark.coordinates[0] -= 1.0f;
		landmark.coordinates[1] -= 1.0f;
		landmarks.emplace_back(landmark);
		++ibugId;
	}
	return landmarks;
};

/**
* This app demonstrates estimation of the camera and fitting of the shape
* model of a 3D Morphable Model from an ibug LFPW image with its landmarks.
*
* First, the 68 ibug landmarks are loaded from the .pts file and converted
* to vertex indices using the LandmarkMapper. Then, an orthographic camera
* is estimated, and then, using this camera matrix, the shape is fitted
* to the landmarks.
*/
int main(int argc, char *argv[])
{
	fs::path modelfile, isomapfile, imagefile, landmarksfile, mappingsfile, outputfile;
	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
				"display the help message")
				("model,m", po::value<fs::path>(&modelfile)->required()->default_value("../share/sfm_shape_3448.bin"),
					"a Morphable Model stored as cereal BinaryArchive")
					("image,i", po::value<fs::path>(&imagefile)->required()->default_value("data/image_0129.png"),
						"an input image")
						("landmarks,l", po::value<fs::path>(&landmarksfile)->required()->default_value("data/image_0129.pts"),
							"2D landmarks for the image, in ibug .pts format")
							("mapping,p", po::value<fs::path>(&mappingsfile)->required()->default_value("../share/ibug_to_sfm.txt"),
								"landmark identifier to model vertex number mapping")
								("output,o", po::value<fs::path>(&outputfile)->required()->default_value("out"),
									"basename for the output rendering and obj files")
			;
		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
		if (vm.count("help")) {
			cout << "Usage: fit-model-simple [options]" << endl;
			cout << desc;
			return EXIT_SUCCESS;
		}
		po::notify(vm);
	}
	catch (const po::error& e) {
		cout << "Error while parsing command-line arguments: " << e.what() << endl;
		cout << "Use --help to display a list of options." << endl;
		return EXIT_FAILURE;
	}

	// Load the image, landmarks, LandmarkMapper and the Morphable Model:
	Mat image = cv::imread(imagefile.string());
	LandmarkCollection<cv::Vec2f> landmarks;
	try {
		landmarks = read_pts_landmarks(landmarksfile.string());
	}
	catch (const std::runtime_error& e) {
		cout << "Error reading the landmarks: " << e.what() << endl;
		return EXIT_FAILURE;
	}
	morphablemodel::MorphableModel morphable_model;
	try {
		morphable_model = morphablemodel::load_model(modelfile.string());
	}
	catch (const std::runtime_error& e) {
		cout << "Error loading the Morphable Model: " << e.what() << endl;
		return EXIT_FAILURE;
	}
	core::LandmarkMapper landmark_mapper = mappingsfile.empty() ? core::LandmarkMapper() : core::LandmarkMapper(mappingsfile);

	// Draw the loaded landmarks:
	Mat outimg = image.clone();
	for (auto&& lm : landmarks) {
		cv::rectangle(outimg, cv::Point2f(lm.coordinates[0] - 2.0f, lm.coordinates[1] - 2.0f), cv::Point2f(lm.coordinates[0] + 2.0f, lm.coordinates[1] + 2.0f), { 255, 0, 0 });
	}

	// These will be the final 2D and 3D points used for the fitting:
	vector<Vec4f> model_points; // the points in the 3D shape model
	vector<int> vertex_indices; // their vertex indices
	vector<Vec2f> image_points; // the corresponding 2D landmark points

								// Sub-select all the landmarks which we have a mapping for (i.e. that are defined in the 3DMM):
	for (int i = 0; i < landmarks.size(); ++i) {
		auto converted_name = landmark_mapper.convert(landmarks[i].name);
		if (!converted_name) { // no mapping defined for the current landmark
			continue;
		}
		int vertex_idx = std::stoi(converted_name.get());
		auto vertex = morphable_model.get_shape_model().get_mean_at_point(vertex_idx);
		model_points.emplace_back(Vec4f(vertex.x(), vertex.y(), vertex.z(), 1.0f));
		vertex_indices.emplace_back(vertex_idx);
		image_points.emplace_back(landmarks[i].coordinates);
	}

	// Estimate the camera (pose) from the 2D - 3D point correspondences
	fitting::ScaledOrthoProjectionParameters pose = fitting::estimate_orthographic_projection_linear(image_points, 
		model_points,
		true,
		image.rows);
	fitting::RenderingParameters rendering_params(pose, image.cols, image.rows);

	// The 3D head pose can be recovered as follows:
	float yaw_angle = glm::degrees(glm::yaw(rendering_params.get_rotation()));
	// and similarly for pitch and roll.

	// Estimate the shape coefficients by fitting the shape to the landmarks:
	Mat affine_from_ortho = fitting::get_3x4_affine_camera_matrix(rendering_params, image.cols, image.rows);
	vector<float> fitted_coeffs = fitting::fit_shape_to_landmarks_linear(morphable_model, 
																			affine_from_ortho,
																			image_points,
																			vertex_indices);

	// Obtain the full mesh with the estimated coefficients:
	core::Mesh mesh = morphable_model.draw_sample(fitted_coeffs, vector<float>());

	// Extract the texture from the image using given mesh and camera parameters:
	Mat isomap = render::extract_texture(mesh, affine_from_ortho, image);

	// Save the mesh as textured obj:
	outputfile += fs::path(".obj");
	core::write_textured_obj(mesh, outputfile.string());

	// And save the isomap:
	outputfile.replace_extension(".isomap.png");
	cv::imwrite(outputfile.string(), isomap);

	cout << "Finished fitting and wrote result mesh and isomap to files with basename " << outputfile.stem().stem() << "." << endl;

	return EXIT_SUCCESS;
}
#endif

#if 0
//dlib include
#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
//eos library include
#include "eos/core/Landmark.hpp"
#include "eos/core/LandmarkMapper.hpp"
#include "eos/fitting/nonlinear_camera_estimation.hpp"
#include "eos/fitting/linear_shape_fitting.hpp"
#include "eos/render/utils.hpp"
#include "eos/render/texture_extraction.hpp"
//OpenCV include 
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"


#if 0
#ifdef WIN32
#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#endif
#include "boost/program_options.hpp"
#include <boost/filesystem.hpp>

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

using namespace eos;
namespace po = boost::program_options;
namespace fs = boost::filesystem;
using eos::core::Landmark;
using eos::core::LandmarkCollection;
using cv::Mat;
using cv::Vec2f;
using cv::Vec3f;
using cv::Vec4f;
using std::cout;
using std::endl;
using std::vector;
using std::string;
using Eigen::Vector4f;

int main(int argc, char *argv[])
{
	/// read eos file
	fs::path modelfile, isomapfile,mappingsfile, outputfilename, outputfilepath;
	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h", "display the help message")
			("model,m", po::value<fs::path>(&modelfile)->required()->default_value("../share1/sfm_shape_3448.bin"), "a Morphable Model stored as cereal BinaryArchive")
			("mapping,p", po::value<fs::path>(&mappingsfile)->required()->default_value("../share1/ibug2did.txt"), "landmark identifier to model vertex number mapping")
			("outputfilename,o", po::value<fs::path>(&outputfilename)->required()->default_value("out"), "basename for the output rendering and obj files")
			("outputfilepath,o", po::value<fs::path>(&outputfilepath)->required()->default_value("output/"), "basename for the output rendering and obj files")
			;
		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
		if (vm.count("help")) {
			cout << "Usage: webcam_face_fit_model_keegan [options]" << endl;
			cout << desc;
			return EXIT_SUCCESS;
		}
		po::notify(vm);
	}
	catch (const po::error& e) {
		cout << "Error while parsing command-line arguments: " << e.what() << endl;
		cout << "Use --help to display a list of options." << endl;
		return EXIT_SUCCESS;
	}

	try
	{
		cv::VideoCapture cap(0);
		dlib::image_window win;

		// Load face detection and pose estimation models.
		dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
		dlib::shape_predictor pose_model;
		dlib::deserialize("../share1/shape_predictor_68_face_landmarks.dat") >> pose_model;


#define TEST_FRAME
		cv::Mat frame_capture;
#ifdef TEST_FRAME
		frame_capture = cv::imread("./data/image_0129.png");
		cv::imshow("input", frame_capture);
		cv::imwrite("frame_capture.png", frame_capture);
		cv::waitKey(1);
#endif

		// Grab and process frames until the main window is closed by the user.
		int frame_count = 0;
		while (!win.is_closed())
		{
			CAPTURE_FRAME:
			Mat image;
#ifndef TEST_FRAME
			cap >> frame_capture;
#endif
			frame_capture.copyTo(image);

			// Turn OpenCV's Mat into something dlib can deal with.  Note that this just
			// wraps the Mat object, it doesn't copy anything.  So cimg is only valid as
			// long as frame_capture is valid.  Also don't do anything to frame_capture that would cause it
			// to reallocate the memory which stores the image as that will make cimg
			// contain dangling pointers.  This basically means you shouldn't modify frame_capture
			// while using cimg.
			dlib::cv_image<dlib::bgr_pixel> cimg(frame_capture);

			// Detect faces 
			std::vector<dlib::rectangle> faces = detector(cimg);
			if (faces.size() == 0) goto CAPTURE_FRAME;
			for (size_t i = 0; i < faces.size(); ++i)
			{
				cout << faces[i] << endl;
			}
			// Find the pose of each face.
			std::vector<dlib::full_object_detection> shapes;
			for (unsigned long i = 0; i < faces.size(); ++i)
				shapes.push_back(pose_model(cimg, faces[i]));

			/// face 68 pointers
			for (size_t i = 0; i < shapes.size(); ++i) 
			{
				morphablemodel::MorphableModel morphable_model;
				try 
				{
					morphable_model = morphablemodel::load_model(modelfile.string());
				}
				catch (const std::runtime_error& e) 
				{
					cout << "Error loading the Morphable Model: " << e.what() << endl;
					return EXIT_FAILURE;
				}
				core::LandmarkMapper landmark_mapper = mappingsfile.empty() ? core::LandmarkMapper() : core::LandmarkMapper(mappingsfile);

				/// every face
				LandmarkCollection<Vec2f> landmarks;
				landmarks.reserve(68);
				cout << "point_num = " << shapes[i].num_parts() << endl;
				int num_face = shapes[i].num_parts();
				for (size_t j = 0; j < num_face; ++j) 
				{
					dlib::point pt_save = shapes[i].part(j);
					Landmark<Vec2f> landmark;
					/// input
					landmark.name = std::to_string(j + 1);
					landmark.coordinates[0] = pt_save.x();
					landmark.coordinates[1] = pt_save.y();
					//cout << shapes[i].part(j) << "\t";
					landmark.coordinates[0] -= 1.0f;
					landmark.coordinates[1] -= 1.0f;
					landmarks.emplace_back(landmark);
				}

				// Draw the loaded landmarks:
				Mat outimg = image.clone();
				cv::imshow("image", image);
#ifdef TEST_FRAME
				cv::imwrite("./image.png", image);
#endif
				cv::waitKey(10);

				int face_point_i = 1;
				for (auto&& lm : landmarks) 
				{
					cv::Point numPoint(lm.coordinates[0] - 2.0f, lm.coordinates[1] - 2.0f);
					cv::rectangle(outimg, cv::Point2f(lm.coordinates[0] - 2.0f, lm.coordinates[1] - 2.0f), cv::Point2f(lm.coordinates[0] + 2.0f, lm.coordinates[1] + 2.0f), { 255, 0, 0 });
					char str_i[11];
					sprintf(str_i, "%d", face_point_i);
					cv::putText(outimg, str_i, numPoint, CV_FONT_HERSHEY_COMPLEX, 0.3, cv::Scalar(0, 0, 255));
					++i;
				}
				//cout << "face_point_i = " << face_point_i << endl;
				cv::imshow("rect_outimg", outimg);
#ifdef TEST_FRAME
				cv::imwrite("./rect_outimg.png", outimg);
#endif
				cv::waitKey(1);

				// These will be the final 2D and 3D points used for the fitting:
				std::vector<Vec4f> model_points; // the points in the 3D shape model
				std::vector<int> vertex_indices; // their vertex indices
				std::vector<Vec2f> image_points; // the corresponding 2D landmark points

				// Sub-select all the landmarks which we have a mapping for (i.e. that are defined in the 3DMM):
				for (int i = 0; i < landmarks.size(); ++i) 
				{
					auto converted_name = landmark_mapper.convert(landmarks[i].name);
					if (!converted_name) 
					{ 
						// no mapping defined for the current landmark
						continue;
					}
					int vertex_idx = std::stoi(converted_name.get());
					//Vec4f vertex = morphable_model.get_shape_model().get_mean_at_point(vertex_idx);
					auto vertex = morphable_model.get_shape_model().get_mean_at_point(vertex_idx);
					model_points.emplace_back(Vec4f(vertex.x(), vertex.y(), vertex.z(), 1.0f));
					vertex_indices.emplace_back(vertex_idx);
					image_points.emplace_back(landmarks[i].coordinates);
				}

				// Estimate the camera (pose) from the 2D - 3D point correspondences
				fitting::RenderingParameters rendering_params = fitting::estimate_orthographic_camera(image_points, 
																										model_points, 
																										image.cols, 
																										image.rows);
				Mat affine_from_ortho = get_3x4_affine_camera_matrix(rendering_params, 
																		image.cols,
																		image.rows);
				// 	cv::imshow("affine_from_ortho", affine_from_ortho);
				// 	cv::waitKey();

				// The 3D head pose can be recovered as follows:
				float yaw_angle = glm::degrees(glm::yaw(rendering_params.get_rotation()));


				// Estimate the shape coefficients by fitting the shape to the landmarks:
				std::vector<float> fitted_coeffs = fitting::fit_shape_to_landmarks_linear(morphable_model, 
																							affine_from_ortho,
																							image_points,
																							vertex_indices);
#if 0
				cout << "size = " << fitted_coeffs.size() << endl;
				for (int i = 0; i < fitted_coeffs.size(); ++i)
					cout << fitted_coeffs[i] << endl;
#endif

				// Obtain the full mesh with the estimated coefficients:
				core::Mesh mesh = morphable_model.draw_sample(fitted_coeffs, std::vector<float>());

				// Extract the texture from the image using given mesh and camera parameters:
				Mat isomap = render::extract_texture(mesh, affine_from_ortho, image);

				///// save obj
				std::stringstream strOBJ;
				strOBJ << std::setw(10) << std::setfill('0') << frame_count << ".obj";

				// Save the mesh as textured obj:
				outputfilename = strOBJ.str();
				std::cout << outputfilename << std::endl;
				auto outputfile =  outputfilepath.string() + outputfilename.string();
				core::write_textured_obj(mesh, outputfile);

				// And save the isomap:
				outputfilename.replace_extension(".isomap.png");
				cv::imwrite(outputfilepath.string() + outputfilename.string(), isomap);

				cv::imshow("isomap_png", isomap);
				cv::waitKey(1);

				outputfilename.clear();
#ifdef TEST_FRAME
				break;
#endif
			}
			frame_count++;

			// Display it all on the screen
			win.clear_overlay();
			win.set_image(cimg);
			win.add_overlay(render_face_detections(shapes));
		}
	}
	catch (dlib::serialization_error& e)
	{
		cout << "You need dlib's default face landmarking model file to run this example." << endl;
		cout << "You can get it from the following URL: " << endl;
		cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
		cout << endl << e.what() << endl;
	}
	catch (std::exception& e)
	{
		cout << e.what() << endl;
	}

	return EXIT_SUCCESS;
}
#endif

#if 0
/*
* 4dface: Real-time 3D face tracking and reconstruction from 2D video.
*
* File: apps/4dface.cpp
*
* Copyright 2015, 2016 Patrik Huber
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
#include "helpers.hpp"
#include "eos/core/Landmark.hpp"
#include "eos/core/LandmarkMapper.hpp"
#include "eos/fitting/fitting.hpp"
#include "eos/fitting/orthographic_camera_estimation_linear.hpp"
#include "eos/fitting/contour_correspondence.hpp"
#include "eos/fitting/closest_edge_fitting.hpp"
#include "eos/fitting/RenderingParameters.hpp"
#include "eos/render/utils.hpp"
#include "eos/render/render.hpp"
#include "eos/render/texture_extraction.hpp"

#include "rcr/model.hpp"
#include "cereal/cereal.hpp"

#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/quaternion.hpp"

#include "Eigen/Dense"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include "boost/program_options.hpp"
#include "boost/filesystem.hpp"

#include <vector>
#include <iostream>

#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>

//using namespace dlib;
using namespace eos;
namespace po = boost::program_options;
namespace fs = boost::filesystem;
using cv::Mat;
using cv::Vec2f;
using cv::Vec3f;
using cv::Vec4f;
using cv::Rect;
using std::cout;
using std::endl;
using std::vector;
using std::string;


void draw_axes_topright(float r_x, float r_y, float r_z, cv::Mat image);


/**
* This app demonstrates facial landmark tracking, estimation of the 3D pose
* and fitting of the shape model of a 3D Morphable Model from a video stream,
* and merging of the face texture.
*/
int main(int argc, char *argv[])
{
	dlib::image_window win;
	dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
	dlib::shape_predictor pose_model;
	dlib::deserialize("../share2/shape_predictor_68_face_landmarks.dat") >> pose_model;
	fs::path modelfile, inputvideo, facedetector, landmarkdetector, mappingsfile, contourfile, edgetopologyfile, blendshapesfile;
	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
				"display the help message")
				("morphablemodel,m", po::value<fs::path>(&modelfile)->required()->default_value("../share2/sfm_shape_3448.bin"),
					"a Morphable Model stored as cereal BinaryArchive")
					("facedetector,f", po::value<fs::path>(&facedetector)->required()->default_value("../share2/haarcascade_frontalface_alt2.xml"),
						"full path to OpenCV's face detector (haarcascade_frontalface_alt2.xml)")
						("landmarkdetector,l", po::value<fs::path>(&landmarkdetector)->required()->default_value("../share2/face_landmarks_model_rcr_68.bin"),
							"learned landmark detection model")
							("mapping,p", po::value<fs::path>(&mappingsfile)->required()->default_value("../share2/ibug2did.txt"),
								"landmark identifier to model vertex number mapping")
								("model-contour,c", po::value<fs::path>(&contourfile)->required()->default_value("../share2/model_contours.json"),
									"file with model contour indices")
									("edge-topology,e", po::value<fs::path>(&edgetopologyfile)->required()->default_value("../share2/sfm_3448_edge_topology.json"),
										"file with model's precomputed edge topology")
										("blendshapes,b", po::value<fs::path>(&blendshapesfile)->required()->default_value("../share2/expression_blendshapes_3448.bin"),
											"file with blendshapes")
											("input,i", po::value<fs::path>(&inputvideo),
												"input video file. If not specified, camera 0 will be used.")
			;
		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
		if (vm.count("help")) {
			cout << "Usage: 4dface [options]" << endl;
			cout << desc;
			return EXIT_FAILURE;
		}
		po::notify(vm);
	}
	catch (const po::error& e) {
		cout << "Error while parsing command-line arguments: " << e.what() << endl;
		cout << "Use --help to display a list of options." << endl;
		return EXIT_FAILURE;
	}

	// Load the Morphable Model and the LandmarkMapper:
	morphablemodel::MorphableModel morphable_model = morphablemodel::load_model(modelfile.string());
	core::LandmarkMapper landmark_mapper = mappingsfile.empty() ? core::LandmarkMapper() : core::LandmarkMapper(mappingsfile);

	fitting::ModelContour model_contour = contourfile.empty() ? fitting::ModelContour() : fitting::ModelContour::load(contourfile.string());
	fitting::ContourLandmarks ibug_contour = fitting::ContourLandmarks::load(mappingsfile.string());

	rcr::detection_model rcr_model;
	// Load the landmark detection model:
	try 
	{
		rcr_model = rcr::load_detection_model(landmarkdetector.string());
	}
	catch (const cereal::Exception& e) 
	{
		cout << "Error reading the RCR model " << landmarkdetector << ": " << e.what() << endl;
		return EXIT_FAILURE;
	}

	// Load the face detector from OpenCV:
	cv::CascadeClassifier face_cascade;
	if (!face_cascade.load(facedetector.string()))
	{
		cout << "Error loading the face detector " << facedetector << "." << endl;
		return EXIT_FAILURE;
	}

	cv::VideoCapture cap;
	if (inputvideo.empty()) {
		cap.open(0); // no file given, open the default camera
	}
	else {
		cap.open(inputvideo.string());
	}
	if (!cap.isOpened()) {
		cout << "Couldn't open the given file or camera 0." << endl;
		return EXIT_FAILURE;
	}

	vector<morphablemodel::Blendshape> blendshapes = morphablemodel::load_blendshapes(blendshapesfile.string());

	morphablemodel::EdgeTopology edge_topology = morphablemodel::load_edge_topology(edgetopologyfile.string());

	cv::namedWindow("video", 1);
	cv::namedWindow("render", 1);

	Mat frame, unmodified_frame;

	bool have_face = false;
	rcr::LandmarkCollection<Vec2f> current_landmarks;
	Rect current_facebox;
	WeightedIsomapAveraging isomap_averaging(60.f); // merge all triangles that are facing <60° towards the camera
	PcaCoefficientMerging pca_shape_merging;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty()) { // stop if we're at the end of the video
			break;
		}

		// We do a quick check if the current face's width is <= 50 pixel. If it is, we re-initialise the tracking with the face detector.
		if (have_face && get_enclosing_bbox(rcr::to_row(current_landmarks)).width <= 50) {
			cout << "Reinitialising because the face bounding-box width is <= 50 px" << endl;
			have_face = false;
		}

		unmodified_frame = frame.clone();
	//	unmodified_frame = cv::imread("D:\\Spoor\\face_recognition\\3D_Reconstruction\\FaceReconstruction\\imgs\\image.png");
	//	cv::imshow("input", unmodified_frame);


		if (!have_face) 
		{
			// Run the face detector and obtain the initial estimate using the mean landmarks:
			vector<Rect> detected_faces;
			face_cascade.detectMultiScale(unmodified_frame, detected_faces, 1.2, 2, 0, cv::Size(110, 110));
			if (detected_faces.empty()) 
			{
				cv::imshow("video", frame);
				cv::waitKey(20);
				continue;
			}
			//vector<rectangle> faces = detector(cimg);
			cv::rectangle(frame, detected_faces[0], { 255, 0, 0 });
			dlib::cv_image<dlib::bgr_pixel> cimg(frame);
			//detect a face
			std::vector<dlib::rectangle> faces = detector(cimg);
			// Find the pose of each face.
			std::vector<dlib::full_object_detection> shapes;
			for (unsigned long i = 0; i < faces.size(); ++i)
				shapes.push_back(pose_model(cimg, faces[i]));

			win.clear_overlay();
			win.set_image(cimg);
			win.add_overlay(dlib::render_face_detections(shapes));

			//cv::imshow("detected face", frame);
			// Rescale the V&J facebox to make it more like an ibug-facebox:
			// (also make sure the bounding box is square, V&J's is square)
			Rect ibug_facebox = rescale_facebox(detected_faces[0], 0.85, 0.2);

			current_landmarks = rcr_model.detect(unmodified_frame, ibug_facebox);
			rcr::draw_landmarks(frame, current_landmarks, { 0, 0, 255 }); // red, initial landmarks

			have_face = true;
		}
		else {
			// We already have a face - track and initialise using the enclosing bounding
			// box from the landmarks from the last frame:
			auto enclosing_bbox = get_enclosing_bbox(rcr::to_row(current_landmarks));
			enclosing_bbox = make_bbox_square(enclosing_bbox);
			current_landmarks = rcr_model.detect(unmodified_frame, enclosing_bbox);
			rcr::draw_landmarks(frame, current_landmarks, { 255, 0, 0 }); // blue, the new optimised landmarks
		}

		// Fit the 3DMM:
		fitting::RenderingParameters rendering_params;
		vector<float> shape_coefficients, blendshape_coefficients;
		vector<Vec2f> image_points;
		core::Mesh mesh;
		std::tie(mesh, rendering_params) = fitting::fit_shape_and_pose(  morphable_model, 
																			blendshapes, 
																			rcr_to_eos_landmark_collection(current_landmarks), 
																			landmark_mapper, 
																			unmodified_frame.cols, 
																			unmodified_frame.rows, 
																			edge_topology, 
																			ibug_contour, 
																			model_contour, 3, 5, 15.0f, 
																			boost::none, 
																			shape_coefficients, 
																			blendshape_coefficients, 
																			image_points);

		// Draw the 3D pose of the face:
		draw_axes_topright( glm::eulerAngles(rendering_params.get_rotation())[0], 
							glm::eulerAngles(rendering_params.get_rotation())[1],
							glm::eulerAngles(rendering_params.get_rotation())[2],
							frame);

		// Wireframe rendering of mesh of this frame (non-averaged):
		draw_wireframe( frame, mesh, 
						rendering_params.get_modelview(), 
						rendering_params.get_projection(), 
						fitting::get_opencv_viewport(frame.cols, frame.rows));

		// Extract the texture using the fitted mesh from this frame:
		Mat affine_cam = fitting::get_3x4_affine_camera_matrix(rendering_params, frame.cols, frame.rows);
		Mat isomap = render::extract_texture(	mesh, 
												affine_cam, 
												unmodified_frame, 
												true, 
												render::TextureInterpolation::NearestNeighbour,
												512);

		// Merge the isomaps - add the current one to the already merged ones:
		Mat merged_isomap = isomap_averaging.add_and_merge(isomap);
		// Same for the shape:
		shape_coefficients = pca_shape_merging.add_and_merge(shape_coefficients);
		auto merged_shape = morphable_model.get_shape_model().draw_sample(shape_coefficients) + 
			//morphablemodel::to_matrix(blendshapes) * Mat(blendshape_coefficients);
			Eigen::Map<const Eigen::VectorXf>(blendshape_coefficients.data(), blendshape_coefficients.size());
		core::Mesh merged_mesh = morphablemodel::sample_to_mesh(merged_shape, 
																	morphable_model.get_color_model().get_mean(),
																	morphable_model.get_shape_model().get_triangle_list(),
																	morphable_model.get_color_model().get_triangle_list(),
																	morphable_model.get_texture_coordinates());

		// Render the model in a separate window using the estimated pose, shape and merged texture:
		Mat rendering;
		auto modelview_no_translation = rendering_params.get_modelview();
		modelview_no_translation[3][0] = 0;
		modelview_no_translation[3][1] = 0;


		try {
			rcr_model = rcr::load_detection_model(landmarkdetector.string());
		}
		catch (const cereal::Exception& e) {
			cout << "Error reading the RCR model " << landmarkdetector << ": " << e.what() << endl;
			return EXIT_FAILURE;
		}

		try {
			std::tie(rendering, std::ignore) = render::render(merged_mesh,
				modelview_no_translation,
				glm::ortho(-130.0f, 130.0f, -130.0f, 130.0f), 256, 256,
				render::create_mipmapped_texture(merged_isomap),
				true, false, false);
		}
		catch (std::exception& e) {
			cout << "render exception" << e.what() << endl;
			return EXIT_FAILURE;
		}

		//cv::imshow("render", rendering);
		//cv::imwrite("rendering.png", rendering);
		//break;
		
		cv::imshow("video", frame);
		auto key = cv::waitKey(30);
		if (key == 'q') break;
		if (key == 'r') {
			have_face = false;
			isomap_averaging = WeightedIsomapAveraging(60.f);
		}
		if (key == 's') {
			// save an obj + current merged isomap to the disk:
 			core::Mesh neutral_expression = morphablemodel::sample_to_mesh( morphable_model.get_shape_model().draw_sample(shape_coefficients), 
																			morphable_model.get_color_model().get_mean(), 
																			morphable_model.get_shape_model().get_triangle_list(), 
																			morphable_model.get_color_model().get_triangle_list(), 
																			morphable_model.get_texture_coordinates());
			core::write_textured_obj(neutral_expression, "current_merged.obj");
			cv::imwrite("current_merged.isomap.png", merged_isomap);
		}
	}

	return EXIT_SUCCESS;
};

/**
* @brief Draws 3D axes onto the top-right corner of the image. The
* axes are oriented corresponding to the given angles.
*
* @param[in] r_x Pitch angle, in radians.
* @param[in] r_y Yaw angle, in radians.
* @param[in] r_z Roll angle, in radians.
* @param[in] image The image to draw onto.
*/
void draw_axes_topright(float r_x, float r_y, float r_z, cv::Mat image)
{
	const glm::vec3 origin(0.0f, 0.0f, 0.0f);
	const glm::vec3 x_axis(1.0f, 0.0f, 0.0f);
	const glm::vec3 y_axis(0.0f, 1.0f, 0.0f);
	const glm::vec3 z_axis(0.0f, 0.0f, 1.0f);

	const auto rot_mtx_x = glm::rotate(glm::mat4(1.0f), r_x, glm::vec3{ 1.0f, 0.0f, 0.0f });
	const auto rot_mtx_y = glm::rotate(glm::mat4(1.0f), r_y, glm::vec3{ 0.0f, 1.0f, 0.0f });
	const auto rot_mtx_z = glm::rotate(glm::mat4(1.0f), r_z, glm::vec3{ 0.0f, 0.0f, 1.0f });
	const auto modelview = rot_mtx_z * rot_mtx_x * rot_mtx_y;

	const auto viewport = fitting::get_opencv_viewport(image.cols, image.rows);
	const float aspect = static_cast<float>(image.cols) / image.rows;
	const auto ortho_projection = glm::ortho(-3.0f * aspect, 3.0f * aspect, -3.0f, 3.0f);
	const auto translate_topright = glm::translate(glm::mat4(1.0f), glm::vec3(0.7f, 0.65f, 0.0f));
	const auto o_2d = glm::project(origin, modelview, translate_topright * ortho_projection, viewport);
	const auto x_2d = glm::project(x_axis, modelview, translate_topright * ortho_projection, viewport);
	const auto y_2d = glm::project(y_axis, modelview, translate_topright * ortho_projection, viewport);
	const auto z_2d = glm::project(z_axis, modelview, translate_topright * ortho_projection, viewport);
	cv::line(image, cv::Point2f{ o_2d.x, o_2d.y }, cv::Point2f{ x_2d.x, x_2d.y }, { 0, 0, 255 });
	cv::line(image, cv::Point2f{ o_2d.x, o_2d.y }, cv::Point2f{ y_2d.x, y_2d.y }, { 0, 255, 0 });
	cv::line(image, cv::Point2f{ o_2d.x, o_2d.y }, cv::Point2f{ z_2d.x, z_2d.y }, { 255, 0, 0 });
};

#endif