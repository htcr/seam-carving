#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <climits>
#include <vector>
#include <string>
#include <opencv\cv.h>
#include <opencv\highgui.h>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\opencv.hpp>
#define MAX(a,b) ((a)<(b)?(b):(a))

class Seam{
public:
	float energy;
	//int pos_carve;
	//int pos_ori;
	//std::vector<signed char>* trace;
	std::vector<int>* trace_carve;
	std::vector<int>* trace_ori;
	Seam(float energy){
		//this->pos_carve = pos_carve;
		//this->pos_ori = pos_ori;
		this->energy = energy;
		trace_carve = new std::vector<int>();
		trace_ori = new std::vector<int>();
	}
	~Seam(){
		delete trace_ori;
		delete trace_carve;
	}
	void push(int pos_carve, int pos_ori){
		trace_carve->push_back(pos_carve);
		trace_ori->push_back(pos_ori);
	}
	int length(){
		return trace_carve->size();
	}
};

//input must be a three-channel image
//output must be a single-channel image
void rgb2grey(cv::Mat &in, cv::Mat &out){
	//int size = in.rows*in.cols;
	//for (int i = 0; i < size; i++){
		//out.data[i] = (unsigned char)((in.data[3 * i] + in.data[3 * i + 1] + in.data[3 * i + 2]) / 3.0);
	//}
	int r = in.rows;
	int c = in.cols;
	unsigned char *p_in_row;
	unsigned char *p_out_row;
	for (int i = 0; i < r; i++){
		p_in_row = in.ptr<unsigned char>(i);
		p_out_row = out.ptr<unsigned char>(i);
		for (int j = 0; j < c; j++){
			p_out_row[j] = (unsigned char)((p_in_row[3 * j] + p_in_row[3 * j + 1] + p_in_row[3 * j + 2]) / 3.0);
		}
	}
}

//input must be a single-channel image
//output must be a single-channel float image
//input must have same size with output
void grey2energy_diff(cv::Mat &in, cv::Mat &out, int r_max, int c_max){
	int r = r_max;
	int c = c_max;
	unsigned char *p_in_row, *p_in_next_row;
	float *p_out_row;
	//unsigned char *p_out_row;
	for (int i = 0; i < r-1; i++){
		p_in_row = in.ptr<unsigned char>(i);
		p_in_next_row = in.ptr<unsigned char>(i+1);
		p_out_row = out.ptr<float>(i);
		//p_out_row = out.ptr<unsigned char>(i);
		for (int j = 0; j < c-1; j++){
			p_out_row[j] = abs(p_in_row[j] - p_in_row[j + 1]) + abs(p_in_row[j] - p_in_next_row[j]);
			//printf("%lf\n", p_out_row[j]);
		}
		p_out_row[c - 1] = p_out_row[c - 2];
	}
	float *p_out_prev_row = out.ptr<float>(r - 2);
	p_out_row = out.ptr<float>(r - 1);
	for (int j = 0; j < c - 1; j++){
		p_out_row[j] = p_out_prev_row[j];
	}
	p_out_row[c - 1] = p_out_row[c - 2];
}

void grey2energy_canny(cv::Mat &in, cv::Mat &out, int r_max, int c_max){
	cv::Mat canny_in = in.colRange(0, c_max);
	canny_in = canny_in.rowRange(0, r_max);
	cv::Mat canny_out(r_max, c_max, CV_8UC1, cv::Scalar(0));
	//cv::Mat canny_out = out.colRange(0, c_max - 1);
	//canny_out = canny_out.rowRange(0, r_max - 1);
	cv::GaussianBlur(canny_in, canny_out,cv::Size(3,3),0,0,cv::BORDER_DEFAULT);
	cv::Canny(canny_out, canny_out, 50, 150);
	unsigned char* p_canny_row;
	float* p_out_row;
	for (int i = 0; i < r_max; i++){
		p_canny_row = canny_out.ptr<unsigned char>(i);
		p_out_row = out.ptr<float>(i);
		for (int j = 0; j < c_max; j++){
			p_out_row[j] = p_canny_row[j];
		}
	}
}

//energy: 32FC1
void show_energy(cv::Mat &energy){
	int r = energy.rows;
	int c = energy.cols;
	cv::Mat show(r, c, CV_8UC3, cv::Scalar(0));
	float *p_in_row;
	float u;
	unsigned char *p_out_row;
	for (int i = 0; i < r - 1; i++){
		p_in_row = energy.ptr<float>(i);
		p_out_row = show.ptr<unsigned char>(i);
		for (int j = 0; j < c - 1; j++){
			u = 1.0 / (1.0 + exp(-p_in_row[j] /	20));
			//b
			p_out_row[3 * j] = (unsigned char)(255 * (1 - u));
			//g
			p_out_row[3 * j + 1] = 0;
			//r
			p_out_row[3 * j + 2] = (unsigned char)(255 * u);
		}
	}
	cv::namedWindow("energy");
	imshow("energy", show);
}


//all mats should be of same size
//energy: 32FC1 value: 32FC1 track: 8SC1
//seam: bottom->up
void energy2dp_vertical(cv::Mat &energy, cv::Mat &value, cv::Mat &track, int r_max, int c_max){
	int r = r_max;
	int c = c_max;
	float *plv, *pe, *pv;
	signed char *pt;
	pe = energy.ptr<float>(0);
	pv = value.ptr<float>(0);
	for (int j = 0; j < c; j++){
		pv[j] = pe[j];
	}
	if (r < 2) return;
	for (int i = 1; i < r; i++){
		//ple = energy.ptr<float>(i-1);
		plv = value.ptr<float>(i - 1);
		pe = energy.ptr<float>(i);
		pv = value.ptr<float>(i);
		pt = track.ptr<signed char>(i);
		for (int j = 0; j < c; j++){			
			float left = (j == 0) ? FLT_MAX : pe[j] + plv[j - 1];
			float mid = pe[j] + plv[j];
			float right = (j == c - 1) ? FLT_MAX : pe[j] + plv[j + 1];
			if (left < mid && left < right){
				pv[j] = left;
				pt[j] = -1;
			}
			else if (right<mid&&right<left){
				pv[j] = right;
				pt[j] = 1;
			}
			else{
				pv[j] = mid;
				pt[j] = 0;
			}
		}
	}
}

Seam* locate_seam(cv::Mat &dpv, cv::Mat &dpt, std::vector<Seam*>* seam_rec, std::vector<std::vector<int>*>* carve2ori,int r_max,int c_max){
	float min_e = FLT_MAX;
	int min_j = 0;
	float *p = dpv.ptr<float>(r_max - 1);
	for (int j = 0; j < c_max; j++){
		if (p[j] < min_e) {
			min_e = p[j];
			min_j = j;
		}
	}
	//record seam
	Seam *seam = new Seam(min_e);
	std::vector<int>* p_vec = carve2ori->at(r_max - 1);
	signed char *p_t;
	int seam_j = min_j;
	seam->push(seam_j, p_vec->at(seam_j));
	p_vec->erase((p_vec->begin()) + seam_j);

	for (int i = r_max - 1; i >= 1; i--){
		p_t = dpt.ptr<signed char>(i);
		seam_j = seam_j + p_t[seam_j];
		p_vec = carve2ori->at(i - 1);
		seam->push(seam_j, p_vec->at(seam_j));
		p_vec->erase(p_vec->begin() + seam_j);
	}
	seam_rec->push_back(seam);
	return seam;
}

//carve a seam from Mat
template <typename T>
void carve_seam_vertical(cv::Mat &grey, Seam* seam, int r_max, int c_max){
	int n = seam->length();
	if (n != r_max){
		printf("fatal: seam length not match\n");
		exit(-1);
	}
	int seam_j_carve;
	T *p_row;
	for (int i = r_max-1; i >= 0; i--){
		p_row = grey.ptr<T>(i);
		seam_j_carve = seam->trace_carve->at(r_max - 1 - i);
		//carve
		for (int j = seam_j_carve; j < c_max - 1; j++){
			p_row[j] = p_row[j + 1];
		}
	}
}

void draw_seam(cv::Mat &ori, Seam* seam, bool is_vertical, unsigned char b = 0, unsigned char g = 0, unsigned char r = 255){
	if (is_vertical){
		int n = seam->length();
		int seam_j_ori;
		for (int i = n-1; i >= 0; i--){
			seam_j_ori = seam->trace_ori->at(n-1-i);
			ori.at<cv::Vec3b>(i, seam_j_ori)[0] = b;
			ori.at<cv::Vec3b>(i, seam_j_ori)[1] = g;
			ori.at<cv::Vec3b>(i, seam_j_ori)[2] = r;
		}
	}
}

void draw_all_seams(cv::Mat &ori, std::vector<Seam*>* seam_rec, bool is_vertical){
	float max_e = 0.1;
	for (int i = 0; i < seam_rec->size(); i++){
		if (((seam_rec->at(i))->energy)>max_e){
			max_e = (seam_rec->at(i))->energy;
		}
	}
	float u;
	for (int i = 0; i < seam_rec->size(); i++){
		u = ((seam_rec->at(i))->energy)/max_e;
		draw_seam(ori, (*seam_rec)[i], is_vertical,(unsigned char)((1-u)*255),0,(unsigned char)(u*255));
	}
}

void mark_seam(cv::Mat &mark, Seam* seam, bool is_vertical){
	if (is_vertical){
		int n = seam->length();
		int seam_j_ori;
		for (int i = n - 1; i >= 0; i--){
			seam_j_ori = seam->trace_ori->at(n-1-i);
			mark.at<unsigned char>(i, seam_j_ori) = 255;
		}
	}
}

void mark_all_seams(cv::Mat &ori, std::vector<Seam*>* seam_rec, bool is_vertical){
	for (int i = 0; i < seam_rec->size(); i++){
		mark_seam(ori, (*seam_rec)[i], is_vertical);
	}
}

void carve_ori(cv::Mat &ori, cv::Mat &mark, cv::Mat &dst){
	int r_max = ori.rows;
	int c_max = ori.cols;
	unsigned char *p_row_ori;
	unsigned char *p_row_mark;
	unsigned char *p_row_dst;
	int j_dst;
	for (int i = 0; i < r_max; i++){
		p_row_ori = ori.ptr<unsigned char>(i);
		p_row_mark = mark.ptr<unsigned char>(i);
		p_row_dst = dst.ptr<unsigned char>(i);
		j_dst = 0;
		for (int j = 0; j < c_max; j++){
			if (p_row_mark[j] > 0){
				//carve
				
			}
			else{
				//keep
				p_row_dst[3 * j_dst] = p_row_ori[3 * j];
				p_row_dst[3 * j_dst+1] = p_row_ori[3 * j+1];
				p_row_dst[3 * j_dst+2] = p_row_ori[3 * j+2];
				j_dst++;
			}
		}
	}
}

void dilate_ori(cv::Mat &ori, cv::Mat &mark, cv::Mat &dst, int dilate, bool interpolate){
	int r_ori = ori.rows;
	int c_ori = ori.cols;
	unsigned char *p_row_ori;
	unsigned char *p_row_mark;
	unsigned char *p_row_dst;
	int j_dst;
	if (interpolate){
		for (int i = 0; i < r_ori; i++){
			p_row_ori = ori.ptr<unsigned char>(i);
			p_row_mark = mark.ptr<unsigned char>(i);
			p_row_dst = dst.ptr<unsigned char>(i);
			j_dst = 0;
			float b0, b1, g0, g1, r0, r1;
			float kb, kg, kr;
			for (int j = 0; j < c_ori; j++){
				if (p_row_mark[j] > 0){
					//dilate
					b0 = p_row_ori[3 * j];
					g0 = p_row_ori[3 * j + 1];
					r0 = p_row_ori[3 * j + 2];
					if (j < c_ori - 1){
						b1 = p_row_ori[3 * j + 3];
						g1 = p_row_ori[3 * j + 4];
						r1 = p_row_ori[3 * j + 5];
					}
					else{
						b1 = b0;
						g1 = g0;
						r1 = r0;
					}
					kb = (b1 - b0) / (dilate + 1);
					kg = (g1 - g0) / (dilate + 1);
					kr = (r1 - r0) / (dilate + 1);
					for (int k = 0; k <= dilate; k++){
						p_row_dst[3 * j_dst] = (unsigned char)(b0+kb*k);
						p_row_dst[3 * j_dst + 1] = (unsigned char)(g0 + kg*k);
						p_row_dst[3 * j_dst + 2] = (unsigned char)(r0 + kr*k);
						j_dst++;
					}
				}
				else{
					//copy
					p_row_dst[3 * j_dst] = p_row_ori[3 * j];
					p_row_dst[3 * j_dst + 1] = p_row_ori[3 * j + 1];
					p_row_dst[3 * j_dst + 2] = p_row_ori[3 * j + 2];
					j_dst++;
				}
			}
		}
	}
	else{		
		for (int i = 0; i < r_ori; i++){
			p_row_ori = ori.ptr<unsigned char>(i);
			p_row_mark = mark.ptr<unsigned char>(i);
			p_row_dst = dst.ptr<unsigned char>(i);
			j_dst = 0;
			for (int j = 0; j < c_ori; j++){
				if (p_row_mark[j] > 0){
					//dilate
					for (int k = 0; k <= dilate; k++){
						p_row_dst[3 * j_dst] = p_row_ori[3 * j];
						p_row_dst[3 * j_dst + 1] = p_row_ori[3 * j + 1];
						p_row_dst[3 * j_dst + 2] = p_row_ori[3 * j + 2];
						j_dst++;
					}
				}
				else{
					//copy
					p_row_dst[3 * j_dst] = p_row_ori[3 * j];
					p_row_dst[3 * j_dst + 1] = p_row_ori[3 * j + 1];
					p_row_dst[3 * j_dst + 2] = p_row_ori[3 * j + 2];
					j_dst++;
				}
			}
		}
	}
}

void dilate_prot(cv::Mat &ori, cv::Mat &mark, cv::Mat &dst, int dilate){
	int r_ori = ori.rows;
	int c_ori = ori.cols;
	unsigned char *p_row_ori;
	unsigned char *p_row_mark;
	unsigned char *p_row_dst;
	int j_dst;
	for (int i = 0; i < r_ori; i++){
		p_row_ori = ori.ptr<unsigned char>(i);
		p_row_mark = mark.ptr<unsigned char>(i);
		p_row_dst = dst.ptr<unsigned char>(i);
		j_dst = 0;
		for (int j = 0; j < c_ori; j++){
			if (p_row_mark[j] > 0){
				//dilate
				for (int k = 0; k <= dilate; k++){
					p_row_dst[j_dst] = p_row_ori[j];
					j_dst++;
				}
			}
			else{
				//copy
				p_row_dst[j_dst] = p_row_ori[j];
				j_dst++;
			}
		}
	}
}

void apply_protection(cv::Mat &energy, cv::Mat &protection, int r_max, int c_max){
	unsigned char *p_row_prot;
	float *p_row_e;
	for (int i = 0; i < r_max; i++){
		p_row_e = energy.ptr<float>(i);
		p_row_prot = protection.ptr<unsigned char>(i);
		for (int j = 0; j < c_max; j++){
			if (p_row_prot[j] > 250){
				p_row_e[j] = 100000;
			}
		}
	}
}

cv::Mat read_protection(char* path){
	cv::Mat prot_rgb = cv::imread(path);
	cv::Mat prot(prot_rgb.rows, prot_rgb.cols, CV_8UC1);
	rgb2grey(prot_rgb, prot);
	return prot;
}

cv::Mat alter_cols(cv::Mat &input, cv::Mat *protection, bool use_protection, int c_target, bool interpolate,char mode){		
	if (c_target < 10){
		printf("fatal: target size too small.\n");
		exit(-1);
	}
	
	cv::Mat img = input;
	int r_ori = img.rows;
	int c_ori = img.cols;
	int r_target = r_ori;
	//int c_target = c_ori - 300;
	int r_space = r_ori;
	int c_space;
	
	int seam_num;
	int max_seam_num = c_ori / 2;
	int dilation;
	if (c_target <= c_ori){
		seam_num = c_ori - c_target;
		c_space = c_target;
	}
	else{
		dilation = (c_target - c_ori) / max_seam_num + 1;
		seam_num = (c_target - c_ori) / dilation + 1;
		c_space = c_ori + dilation*seam_num;
	}	

	//dp spaces
	cv::Mat grey(r_ori, c_ori, CV_8UC1, cv::Scalar(0));
	cv::Mat energy(r_ori, c_ori, CV_32FC1, cv::Scalar(0));
	cv::Mat dpv(r_ori, c_ori, CV_32FC1, cv::Scalar(0));
	cv::Mat dpt(r_ori, c_ori, CV_8SC1, cv::Scalar(0));

	//seam recorder
	std::vector<Seam*>* seam_rec = new std::vector<Seam*>();
	//for converting carve position to original position
	std::vector<std::vector<int>*>* carve2ori = new std::vector<std::vector<int>*>();
	for (int i = 0; i < r_ori; i++){
		std::vector<int>* vec = new std::vector<int>();
		carve2ori->push_back(vec);
		for (int j = 0; j < c_ori; j++){
			vec->push_back(j);
		}
	}

	//find vertical seams
	rgb2grey(img, grey);
	cv::Mat prot_alter;
	cv::Mat prot_cpy;
	if (use_protection){
		prot_alter.create(r_target, c_target, CV_8UC1);
		protection->copyTo(prot_cpy);
	}
	int c = c_ori;
	for (int k = 0; k < seam_num; k++){
		if (mode == 'd'){
			grey2energy_diff(grey, energy, r_ori, c);
		}			
		else{
			grey2energy_canny(grey, energy, r_ori, c);
		}			
		if (use_protection){
			apply_protection(energy, prot_cpy, r_ori, c);
		}
		
		energy2dp_vertical(energy, dpv, dpt, r_ori, c);
		Seam* seam = locate_seam(dpv, dpt, seam_rec, carve2ori, r_ori, c);
		carve_seam_vertical<unsigned char>(grey, seam, r_ori, c);
		if (use_protection){
			carve_seam_vertical<unsigned char>(prot_cpy, seam, r_ori, c);
		}
		c--;
	}

	//show_energy(energy);
	//cv::Mat show;
	//img.copyTo(show);
	//draw_all_seams(show, seam_rec, true);
	//cv::imwrite(std::string("seams.jpg"), show);
	//

	//found required number of seams
	cv::Mat result;
	cv::Mat seam_mark(r_ori, c_ori, CV_8UC1, cv::Scalar(0));
	mark_all_seams(seam_mark, seam_rec, true);
	if (c_target < c_ori){		
		result.create(r_target, c_target, CV_8UC3);
		dilate_ori(img, seam_mark, result,-1,false);
		if (use_protection){
			dilate_prot(*protection, seam_mark, prot_alter, -1);
			(*protection) = prot_alter;
		}
	}
	else if (c_target>c_ori){
		result.create(r_target, c_space,CV_8UC3);
		dilate_ori(img, seam_mark, result, dilation, interpolate);
		if (use_protection){
			dilate_prot(*protection, seam_mark, prot_alter, dilation);
			(*protection) = prot_alter;
		}
		result = result.colRange(0, c_target);
	}
	else{
		img.copyTo(result);
	}
	return result;
}

cv::Mat alter_img(cv::Mat &input, int r_target, int c_target, bool interpolate, cv::Mat protection, bool use_protection,char mode){
	cv::Mat col_result = alter_cols(input, &protection,use_protection, c_target,interpolate,mode);
	/*
	cv::Mat tmp1 = col_result.t();	
	cv::Mat prot_t;
	if (use_protection){
		prot_t = protection.t();
		//p = p->t();
	}
	cv::Mat tmp2 = alter_cols(tmp1, &prot_t, use_protection, r_target, interpolate,mode);
	cv::Mat result = tmp2.t();
	return result;
	*/
	//temp
	return col_result;
}

int main()
{
	
	while (true){
		char input_path[256] = { 0 };
		int r_target, c_target;
		printf("input image path:\n");
		scanf("%s", &input_path);
		cv::Mat img = cv::imread(input_path);
		printf("original size: r: %d c: %d\n", img.rows, img.cols);
		printf("input target size: \n");
		scanf("%d %d", &r_target, &c_target);
		bool use_protection = false;
		cv::Mat prot;
		char protection_path[256] = { 0 };
		printf("input protection map path:\n");
		scanf("%s", &protection_path);
		if (protection_path[0] == '-'){
			use_protection = false;
			printf("no protection;\n");
		}
		else{
			use_protection = true;
			prot = read_protection(protection_path);
			printf("protection map size: r: %d c: %d\n", prot.rows, prot.cols);
			if (prot.rows != img.rows || prot.cols != img.cols){
				printf("fatal: protection map must be of same size as original image.\n");
				continue;
			}
		}
		printf("select energy operator. 'c' for canny, 'd' for differentiation:\n");
		char mode[10];
		scanf("%s", &mode);
		cv::Mat result = alter_img(img, r_target, c_target, true,prot,use_protection,mode[0]);
		cv::namedWindow("result");
		imshow("result", result);
		char out_name[256] = { 0 };
		sprintf(out_name, "ret_%d_%d.jpg", r_target, c_target);
		cv::imwrite(std::string(out_name), result);
		cv::waitKey();
	}
	return 0;
}