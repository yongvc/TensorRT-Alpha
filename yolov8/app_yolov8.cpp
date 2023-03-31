#include "../utils/yolo.h"
#include "yolov8.h"

// using namespace cv;

void setParameters(utils::InitParameter &initParameters)
{
	initParameters.class_names = utils::dataSets::pole;
	// initParameters.class_names = utils::dataSets::voc20;
	initParameters.num_class = 80; // for coco
	// initParameters.num_class = 20; // for voc2012

	initParameters.batch_size = 1;
	initParameters.dst_h = 640;
	initParameters.dst_w = 640;
	initParameters.input_output_names = {"images", "output0"};
	initParameters.conf_thresh = 0.60f;
	initParameters.iou_thresh = 0.45f;
	initParameters.save_path = "";
}

void task(YOLOV8 &yolo, const utils::InitParameter &param, std::vector<cv::Mat> &imgsBatch, const int &delayTime, const int &batchi,
		  const bool &isShow, const bool &isSave)
{
	utils::DeviceTimer d_t0;
	yolo.copy(imgsBatch);
	float t0 = d_t0.getUsedTime();
	utils::DeviceTimer d_t1;
	yolo.preprocess(imgsBatch);
	float t1 = d_t1.getUsedTime();
	utils::DeviceTimer d_t2;
	yolo.infer();
	float t2 = d_t2.getUsedTime();
	utils::DeviceTimer d_t3;
	yolo.postprocess(imgsBatch);
	float t3 = d_t3.getUsedTime();
	sample::gLogInfo <<
		//"copy time = " << t0 / param.batch_size << "; "
		"preprocess time = " << t1 / param.batch_size << "; "
														 "infer time = "
					 << t2 / param.batch_size << "; "
												 "postprocess time = "
					 << t3 / param.batch_size << std::endl;

	// if (isShow)
	// utils::show(yolo.getObjectss(), param.class_names, delayTime, imgsBatch);
	if (isSave)
		utils::save(yolo.getObjectss(), param.class_names, param.save_path, imgsBatch, param.batch_size, batchi);
}

int main(int argc, char **argv)
{
	cv::CommandLineParser parser(argc, argv,
								 {"{model 	|| tensorrt model file	   }"
								  "{size      || image (h, w), eg: 640   }"
								  "{batch_size|| batch size              }"
								  "{video     || video's path			   }"
								  "{img       || image's path			   }"
								  "{cam_id    || camera's device id	   }"
								  "{show      || if show the result	   }"
								  "{savePath  || save path, can be ignore}"});

	/************************************************************************************************
	 * init
	 *************************************************************************************************/
	// parameters
	utils::InitParameter param;
	setParameters(param);
	// path
	std::string model_path = "../../data/yolov8/best.trt";
	std::string video_path = "../../data/people.mp4";
	std::string image_path = "../../data/bus.jpg";
	// camera' id
	int camera_id = 0;

	// get input
	utils::InputStream source;
	// source = utils::InputStream::IMAGE;
	// source = utils::InputStream::VIDEO;
	source = utils::InputStream::CAMERA;

	// update params from command line parser
	int size = 640; // w or h
	int batch_size = 1;
	bool is_show = true;
	bool is_save = false;
	if (parser.has("model"))
	{
		model_path = parser.get<std::string>("model");
		sample::gLogInfo << "model_path = " << model_path << std::endl;
	}
	if (parser.has("size"))
	{
		size = parser.get<int>("size");
		sample::gLogInfo << "size = " << size << std::endl;
		param.dst_h = param.dst_w = size;
	}
	if (parser.has("batch_size"))
	{
		batch_size = parser.get<int>("batch_size");
		sample::gLogInfo << "batch_size = " << batch_size << std::endl;
		param.batch_size = batch_size;
	}
	if (parser.has("video"))
	{
		source = utils::InputStream::VIDEO;
		video_path = parser.get<std::string>("video");
		sample::gLogInfo << "video_path = " << video_path << std::endl;
	}
	if (parser.has("img"))
	{
		source = utils::InputStream::IMAGE;
		image_path = parser.get<std::string>("img");
		sample::gLogInfo << "image_path = " << image_path << std::endl;
	}
	if (parser.has("cam_id"))
	{
		source = utils::InputStream::CAMERA;
		camera_id = parser.get<int>("cam_id");
		sample::gLogInfo << "camera_id = " << camera_id << std::endl;
	}
	if (parser.has("show"))
	{
		is_show = true;
		sample::gLogInfo << "is_show = " << is_show << std::endl;
	}
	if (parser.has("savePath"))
	{
		is_save = true;
		param.save_path = parser.get<std::string>("savePath");
		sample::gLogInfo << "save_path = " << param.save_path << std::endl;
	}

	int total_batches = 0;
	int delay_time = 1;
	float fps, fpstime;

	cv::VideoCapture capture;
	capture.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
	capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
	capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
	capture.set(cv::CAP_PROP_FPS, 120);

	if (!setInputStream(source, image_path, video_path, camera_id,
						capture, total_batches, delay_time, param))
	{
		sample::gLogError << "read the input data errors!" << std::endl;
		return -1;
	}

	YOLOV8 yolo(param);

	// read model
	std::vector<unsigned char> trt_file = utils::loadModel(model_path);
	if (trt_file.empty())
	{
		sample::gLogError << "trt_file is empty!" << std::endl;
		return -1;
	}
	// init model
	if (!yolo.init(trt_file))
	{
		sample::gLogError << "initEngine() ocur errors!" << std::endl;
		return -1;
	}
	yolo.check();
	/************************************************************************************************
	 * recycle
	 *************************************************************************************************/
	cv::Mat frame;
	cv::Rect_<double> roi;
	std::vector<cv::Mat> imgs_batch;
	imgs_batch.reserve(param.batch_size);
	sample::gLogInfo << imgs_batch.capacity() << std::endl;
	int i = 0; // debug
	int batchi = 0;
	int track_mode = 0;
	char key = 0;
	cv::Ptr<cv::Tracker> tracker = cv::TrackerKCF::create();
	tracker->init(frame, roi);
	while (capture.isOpened() && key != 'q')
	{
		utils::DeviceTimer fps_t;
		if (batchi >= total_batches && source != utils::InputStream::CAMERA)
		{
			break;
		}
		if (imgs_batch.size() < param.batch_size) // get input
		{
			if (source != utils::InputStream::IMAGE)
			{
				capture.read(frame);
			}
			else
			{
				frame = cv::imread(image_path);
			}

			if (frame.empty())
			{
				sample::gLogWarning << "no more video or camera frame" << std::endl;
				// task(yolo, param, imgs_batch, delay_time, batchi, is_show, is_save);

				// imgs_batch.clear(); // clear
				// sample::gLogInfo << imgs_batch.capacity() << std::endl;
				// batchi++;
				break;
			}
			else
			{
				imgs_batch.emplace_back(frame.clone());
			}
		}
		else // infer
		{
			task(yolo, param, imgs_batch, delay_time, batchi, is_show, is_save);

			if (key == 't')
			{
				track_mode = 1;
				tracker->init(frame, roi);
			}
			else if (key == 'r')
			{
				track_mode = 0;
			}

			// Add tracking here
			if (!track_mode)
			{
				// get detected position in x and y
				std::vector<std::vector<utils::Box>> objectss = yolo.getObjectss();
				for (auto &objects : objectss)
				{
					if (objects.size() > 0)
					{
						int i = 0, a = 0;
						float highest_mark = 0, mark;
						for (auto &object : objects)
						{
							// find object with higest confidence and closest to center
							mark = object.confidence * 1 + (1 - abs((object.left + object.right) / 2 - param.dst_w / 2) / (param.dst_w * 0.5));
							if (mark > highest_mark)
							{
								highest_mark = mark;
								a = i;
							}
							i++;
						}
						roi = cv::Rect(objects[a].left, objects[a].top, objects[a].right - objects[a].left, objects[a].bottom - objects[a].top);
						cv::rectangle(imgs_batch[0], roi, utils::Colors::color4[objects[a].label], 2, cv::LINE_AA);
						// cv::rectangle(imgs_batch[0], cv::Point(objects[a].left, objects[a].top), cv::Point(objects[a].right, objects[a].bottom), utils::Colors::color4[objects[a].label], 2, cv::LINE_AA);
						cv::circle(imgs_batch[0], cv::Point((objects[a].left + objects[a].right) / 2, objects[a].top), 5, cv::Scalar(0, 0, 255), -1);
						sample::gLogInfo
							<< mark << "  " << objects[a].confidence * 1.0 << "  " << (1 - abs((objects[a].left + objects[a].right) / 2 - param.dst_w / 2) / (param.dst_w * 0.5)) << "  "
							<< "x: " << (objects[a].left + objects[a].right) << " y: " << objects[a].top << std::endl;
					}
				}
				cv::putText(imgs_batch[0], "Tracker: OFF", cv::Point(150, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
			}
			else
			{
				tracker->update(imgs_batch[0], roi);
				cv::rectangle(imgs_batch[0], roi, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
				cv::putText(imgs_batch[0], "Tracker: ON", cv::Point(150, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
				sample::gLogInfo << "Tracking...   " << roi << std::endl;
			}

			// display fps
			fps = 1000.0 / fps_t.getUsedTime();
			sample::gLogInfo << fps << " " << fps_t.getUsedTime() << std::endl;
			cv::putText(imgs_batch[0], cv::format("fps: %.2f", fps), cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);

			cv::imshow("yolov8 detection", imgs_batch[0]);
			key = cv::waitKey(delay_time);
			imgs_batch.clear(); // clear
			yolo.reset();
			// sample::gLogInfo << imgs_batch.capacity() << std::endl;
			batchi++;
		}
	}

	return -1;
}
