//Network construction function, will be modified after all functional blocks are tested.
// Created by yaochen on 19/12/16.
//TODO: change all functions to store data in array, array dimensions are defined by network model
//

#ifndef _WEIGHT_BIAS_H_
#define _WEIGHT_BIAS_H_

#include <iostream>
//#include <vector>
#include <fstream>
//#include <algorithm>
//#include "data_type.h"
#include "config.h"
#include <string>

//template<int size>
void load_weight_conv(
	    const char* weight_src,
        data_type_w conv_weight2D[],
        int& weight_bias_record,
        int nn_channel_size_conv[],
        int nn_in_number_conv[],
        int nn_out_number_conv[],
        int in_number_conv) {

    //std::cout << "Loading CONV layer weights ..." << std::endl;
    std::ifstream ifs(weight_src);
    std::string str;
    if (!ifs) {
        std::cout << "weight file not found !" << std::endl;
    }
    int layer_weight_num = nn_channel_size_conv[in_number_conv] 
		               * nn_channel_size_conv[in_number_conv]
                       * nn_in_number_conv[in_number_conv]
                       * nn_out_number_conv[in_number_conv];
    int weight_bias_count = 0;


    /*ifs >> str;
    if(ifs.eof())  cout << "file error" << endl;
    else cout << str << endl;*/
    while (ifs >> str && weight_bias_count < layer_weight_num + weight_bias_record) {
		if (str != "weights:"&&str != "bias:") {
			//cout << str << "  ";
			if (weight_bias_count >= weight_bias_record) {
				int serial_no = weight_bias_count - weight_bias_record;
				data_type_w f = atof(str.c_str());
				int channel = serial_no / (nn_channel_size_conv[in_number_conv] * nn_channel_size_conv[in_number_conv]);
				int y_index = (serial_no % (nn_channel_size_conv[in_number_conv] * nn_channel_size_conv[in_number_conv])) / nn_channel_size_conv[in_number_conv];
				int x_index = serial_no % nn_channel_size_conv[in_number_conv];
				conv_weight2D[serial_no] = f;
			}
			weight_bias_count++;
		} 
    }
    weight_bias_record = weight_bias_count;
    //std::cout << std::endl;
    //std::cout << "conv layer weights number in total is = " << weight_bias_count << std::endl;
    //std::cout << "network weights in total is = " << weight_bias_count << std::endl;
}

void load_weight_pooling(
	const char* weight_src,
	data_type_w pooling_1_weight[],
	int& weight_bias_record,
	int nn_channel_size_pooling[],
	int nn_in_number_pooling[],
	int in_number_pooling) {

    //std::cout << "Loading POOL layer weights ..." << std::endl;

	std::ifstream ifs(weight_src);
	std::string str;

	if (!ifs) {
		std::cout << "weight file not found !" << std::endl;
	}
	int layer_weight_num = nn_channel_size_pooling[in_number_pooling] 
		*nn_channel_size_pooling[in_number_pooling] 
		*nn_in_number_pooling[in_number_pooling];
	int weight_bias_count = 0;
	while (ifs >> str&&weight_bias_count<layer_weight_num + weight_bias_record)
	{
		if (str != "weights:"&&str != "bias:") {
			//cout << str << " ";
			if (weight_bias_count >= weight_bias_record) {
				//
				int serial_no = weight_bias_count - weight_bias_record;
				data_type_w f = atof(str.c_str());
				//weight_v.push_back(f);
				pooling_1_weight[serial_no] = f;
			}
			weight_bias_count++;
		}
	}

	weight_bias_record = weight_bias_count;
	//std::cout << "pooling layer weights number in total is = " << weight_bias_count << std::endl;
	ifs.close();
	//return weight_v;
}

//template<int size>
void load_weight_fc(
	const char* weight_src,
	data_type_w fc_1_weight2D[],
	int& weight_bias_record,
	int nn_channel_size_fc[],
	int nn_in_number_fc[],
	int nn_out_number_fc[],
	int in_number_fc) {

    //std::cout << "Loading FC layer weights ..." << std::endl;

	std::ifstream ifs(weight_src);
	std::string str;

	if (!ifs) {
		std::cout << "weight file not found !" << std::endl;
	}
	int layer_weight_num = nn_channel_size_fc[in_number_fc] *nn_channel_size_fc[in_number_fc] *nn_in_number_fc[in_number_fc]
		* nn_out_number_fc[in_number_fc];
	int weight_bias_count = 0;
	while (ifs >> str && weight_bias_count < layer_weight_num + weight_bias_record)
	{
		if (str != "weights:"&&str != "bias:") {
			//cout << str << " ";
			if (weight_bias_count >= weight_bias_record) {
				//
				if (nn_channel_size_fc[in_number_fc] == 1) {
					int serial_no = weight_bias_count - weight_bias_record;
					data_type_w f = atof(str.c_str());
					int channel = serial_no / (nn_channel_size_fc[in_number_fc] * nn_channel_size_fc[in_number_fc]);
					fc_1_weight2D[serial_no] = f;
				}
				else {
					int serial_no = weight_bias_count - weight_bias_record;
					data_type_w f = atof(str.c_str());
					int channel = serial_no / (nn_channel_size_fc[in_number_fc] * nn_channel_size_fc[in_number_fc]);
					int y_index = (serial_no % (nn_channel_size_fc[in_number_fc] *nn_channel_size_fc[in_number_fc])) / nn_channel_size_fc[in_number_fc];
					int x_index = serial_no%nn_channel_size_fc[in_number_fc];
					fc_1_weight2D[serial_no] = f;
				}
			}
			weight_bias_count++;
		}
	}
	weight_bias_record = weight_bias_count;
	//std::cout << "fc layer weights number in total is = " << weight_bias_count << std::endl;
	ifs.close();
}

void load_bias_conv(
	const char* weight_src,
	data_type_w conv_1_bias2D[],
	int& weight_bias_record,
	int nn_channel_size_conv[],
	int nn_in_number_conv[],
	int nn_out_number_conv[],
	int in_number_conv) {

	//std::cout << "Loading CONV layer bias ..." << std::endl;

	std::ifstream ifs(weight_src);
    if (!ifs) {
        std::cout << "weight file not found !" << std::endl;
    }
	std::string str;
	int layer_bias_num = nn_out_number_conv[in_number_conv];
	int weight_bias_count = 0;
	/*ifs >> str;
	if (ifs.eof())  cout << "file error" << endl;
	else cout << str << endl;*/
	while (ifs >> str && weight_bias_count < layer_bias_num + weight_bias_record)
	{
		if (str != "weights:"&&str != "bias:") {
			if (weight_bias_count >= weight_bias_record) {
				//cout << str << "  ";
				data_type_w f = atof(str.c_str());
				conv_1_bias2D[weight_bias_count - weight_bias_record] = f;
			}
			weight_bias_count++;
		}
	}
	weight_bias_record = weight_bias_count;
	//std::cout << std::endl;
	//std::cout << "conv layer bias number in total is = " << weight_bias_count << std::endl;
	ifs.close();
	//return bias2D;
}

void load_bias_pooling(
	const char* weight_src,
	data_type_w pooling_1_bias2D[],
	int& weight_bias_record,
	int nn_channel_size_pooling[],
	int nn_in_number_pooling[],
	int in_number_pooling) {

    //std::cout << "Loading POOL layer bias ..." << std::endl;

	std::ifstream ifs(weight_src);
	std::string str;
	if (!ifs) {
		std::cout << "weight file not found !" << std::endl;
	}
	//vec_t bias2D;
	int layer_bias_num = nn_in_number_pooling[in_number_pooling];
	int weight_bias_count = 0;
	while (ifs >> str&&weight_bias_count < layer_bias_num + weight_bias_record)
	{
		if (str != "weights:"&&str != "bias:") {
			if (weight_bias_count >= weight_bias_record) {
				//cout << str << " ";
				data_type_w f = atof(str.c_str());
				//bias2D.push_back(f);//
				pooling_1_bias2D[weight_bias_count - weight_bias_record] = f;
			}
			weight_bias_count++;
		}
	}

	weight_bias_record = weight_bias_count;
	//std::cout << "pooling layer bias number in total is = " << weight_bias_count << std::endl;

	ifs.close();
	//return bias2D;
}

void load_bias_fc(
	const char* weight_src,
	data_type_w fc_1_bias2D[],
	int& weight_bias_record,
	int nn_channel_size_fc[],
	int nn_in_number_fc[],
	int nn_out_number_fc[],
	int in_number_fc) {

    //std::cout << "Loading FC layer bias ..." << std::endl;

	std::ifstream ifs(weight_src);
	std::string str;
	if (!ifs) {
		std::cout << "weight file not found !" << std::endl;
	}
	//vec_t bias2D;
	int layer_bias_num = nn_out_number_fc[in_number_fc];
	int weight_bias_count = 0;
	while (ifs >> str&&weight_bias_count < layer_bias_num + weight_bias_record)
	{
		if (str != "weights:"&&str != "bias:") {
			if (weight_bias_count >= weight_bias_record) {
				//cout << str << " ";
				data_type_w f = atof(str.c_str());
				//bias2D.push_back(f);
				fc_1_bias2D[weight_bias_count - weight_bias_record] = f;
			}
			weight_bias_count++;
		}
	}

	weight_bias_record = weight_bias_count;
	//std::cout << "fc layer bias number in total is = " << weight_bias_count << std::endl;
	ifs.close();
	//return bias2D;
}

#endif 
