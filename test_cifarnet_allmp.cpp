// Amazon FPGA Hardware Development Kit
//
// Copyright 2016 Amazon.com, Inc. or its affiliates. All Rights Reserved.
//
// Licensed under the Amazon Software License (the "License"). You may not use
// this file except in compliance with the License. A copy of the License is
// located at
//
//    http://aws.amazon.com/asl/
//
// or in the "license" file accompanying this file. This file is distributed on
// an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or
// implied. See the License for the specific language governing permissions and
// limitations under the License.


#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <unistd.h>

//C++ headers
#include <cstdint>
#include <iostream>
//#include <cstring>
#include <fstream>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#include <fpga_pci.h>
#include <fpga_mgmt.h>
#include <utils/lcd.h>

#define STB_IMAGE_IMPLEMENTATION
#include "./inference_net/stb_image.h"
#include "./inference_net/weight_bias_one_dim.h"
#include "./inference_net/config.h"
#include "./inference_net/inference_func.h"
#include "./inference_net/convmpool_allmp_config.h"
#include "./inference_net/max_pool_acc_innerpp.h"
#include "./inference_net/acc_instance.h"
#include "./inference_net/softmax_one_dim.h"
#include "./inference_net/predict_one_dim.h"

using namespace std;

/* use the stdout logger for printing debug information  */
const struct logger *logger = &logger_stdout;

int main(int argc, char **argv) {
    int rc;
    int slot_id;

    /* initialize the fpga_pci library so we could have access to FPGA PCIe from this applications */
    rc = fpga_pci_init();
    fail_on(rc, out, "Unable to initialize the fpga_pci library");

    /* This demo works with single FPGA slot, we pick slot #0 as it works for both f1.2xl and f1.16xl */
    slot_id = 0;

    rc = check_afi_ready(slot_id);
    fail_on(rc, out, "AFI not ready");
    
    /* Accessing the CL registers via AppPF BAR0, which maps to sh_cl_ocl_ AXI-Lite bus between AWS FPGA Shell and the CL*/
    printf("\n");

    printf("===== AXI CDMA Example =====\n");   
    rc = peek_poke_example(slot_id, FPGA_APP_PF, APP_PF_BAR1);
    fail_on(rc, out, "peek-poke example failed");
  
    return rc; 
   
out:
    return 1;
}

/*
 * An example to attach to an arbitrary slot, pf, and bar with register access.
 */
int peek_poke_example(int slot_id, int pf_id, int bar_id) {
    int rc;
    int rc_0;
    int rc_4;
    int rc_sda;
    uint32_t value;

    int loop_var;
    int loop_var_1;
    int loop_var_2;

    data_type in_data[3*32*32];
    data_type  out_temp_full[8192];
    data_type  out_temp_1[4096];
    data_type  out_temp_2[4096];
    data_type  out_temp_3[4096];
    data_type  out_temp_4[4096];
    data_type  out_temp_5[4096];
    data_type  out_temp_6[4096];
    data_type  out_temp_7[4096];
    data_type  out_temp_8[4096];
    data_type  out_temp_1_1[4096];
    data_type  out_temp_2_1[4096];
    data_type  out_temp_3_1[4096];
    data_type  out_temp_4_1[4096];
    data_type  out_temp_5_1[4096];
    data_type  out_temp_6_1[4096];
    data_type  out_temp_7_1[4096];
    data_type  out_temp_8_1[4096];
    data_type  fc_3_out[10];
    data_type_w conv_weight[2400+25600+51200+10240];
    data_type_w conv_weight_1[1]={1};
    data_type_w conv_weight_2[2400+25600+51200+10240];
    data_type_w conv_bias[32+32+64+10];
    data_type_w conv_bias_1[32+32+64+10];
    float conv_1_weight2D[2400];
    float conv_1_bias2D[32];
    float conv_2_weight2D[25600];
    float conv_2_bias2D[32];
    float conv_3_weight2D[51200];
    float conv_3_bias2D[64];
    float fc_1_weight2D[10240];
    float fc_1_bias2D[10];
    data_type_w weight_temp_1[3][32][5][5];
    data_type_w weight_temp_2[32][32][5][5];
    data_type_w weight_temp_3[32][64][5][5];
    data_type_w weight_temp_4[64][10][4][4];
    data_type_w conv_weight_mem_port_1[11680];
    data_type_w conv_weight_mem_port_2[11680];
    data_type_w conv_weight_mem_port_3[11680];
    data_type_w conv_weight_mem_port_4[11680];
    data_type_w conv_weight_mem_port_5[11680];
    data_type_w conv_weight_mem_port_6[11680];
    data_type_w conv_weight_mem_port_7[11680];
    data_type_w conv_weight_mem_port_8[11680];
    
    int ctrl_param_1[2] = {1, 0};
    int ctrl_param_2[2] = {0, 1};
    int acc_param_conv_1[16] = {3, 5, 16, 32, 32, 32, 32, 1, 2, 1, 0, 0, 0, 0, 1, 1};
    int acc_param_conv_2[16] = {32, 5, 32, 16, 16, 16, 16, 1, 2, 1, 1024, 32, 1024, 0, 1, 1};
    int acc_param_conv_3[16] = {32, 5, 64, 8, 8, 8, 8, 1, 2, 1, 4224, 64, 0, 0, 1, 1};
    int acc_param_conv_4[16] = {64, 4, 10, 4, 4, 1, 1, 4, 0, 0, 10624, 128, 0, 0, 1, 1};
    int acc_param_pool_1[9] = {32, 32, 16, 3, 16, 16, 2, 0, 0};
    int acc_param_pool_2[9] = {16, 16, 32, 3, 8, 8, 2, 0, 0};
    int acc_param_pool_3[9] = {8, 8, 64, 3, 4, 4, 2, 0, 0};

    int w;
    int h;
    int channels;
    int size;
    const unsigned char * data ;
    const unsigned char * image_orig ;
    int in_number_conv = 0;
    int in_number_fc = 0;
    int in_number_pooling = 0;
    int conv_weight_num=0;
    int conv_bias_num=0;

    string image_dir = "./netInput/50000.png";
    const char* weight_src = "./netInput/net_weights_cifar.txt";
    ifstream ifs1("./netInput/net_mean.txt");
    float f;
    float channel_mean[3] = { 0 };
    float in_data_3D_channel_swap[3][32][32] = { 0 };
    float in_data_3D[3][32][32] = { 0 };
    string str1;
    string y1 = "[";
    string y2 = "]";
    int index = 0;
    int in_data_size=0;
    std::ofstream indata;
    std::ofstream outdata;
    std::ofstream weightdata;
    std::ofstream test_output;

    //int i,j,k;
    int count = 0;

    //time mreasurement variable define
    struct timeval start,end;
    unsigned long diff;
    XInference_net InstancePtr;
    InstancePtr.ctrl_bus_baseaddress = XINFERENCE_IP_CRTL_BUS_ADDR;
    InstancePtr.IsReady = 0x01;
    uint32_t ip_status;
    /* pci_bar_handle_t is a handler for an address space exposed by one PCI BAR on one of the PCI PFs of the FPGA */

    pci_bar_handle_t pci_bar_handle = PCI_BAR_HANDLE_INIT;
    pci_bar_handle_t pci_bar_handle_0 = PCI_BAR_HANDLE_INIT;
    pci_bar_handle_t pci_bar_handle_4 = PCI_BAR_HANDLE_INIT;

    pci_bar_handle_t pci_bar_handle_sda = PCI_BAR_HANDLE_INIT;

    cout << "test point 1" << endl;

    /* attach to the fpga, with a pci_bar_handle out param
     * To attach to multiple slots or BARs, call this function multiple times,
     * saving the pci_bar_handle to specify which address space to interact with in
     * other API calls.
     * This function accepts the slot_id, physical function, and bar number
     */
    rc = fpga_pci_attach(slot_id, pf_id, bar_id, 0, &pci_bar_handle);
    fail_on(rc, out, "Unable to attach to the AFI on slot id %d", slot_id);

    rc_0 = fpga_pci_attach(slot_id, pf_id, 0, 0, &pci_bar_handle_0);
    fail_on(rc_0, out, "Unable to attach to the AFI on slot id %d", slot_id);

    rc_4 = fpga_pci_attach(slot_id, pf_id, 4, 0, &pci_bar_handle_4);
    fail_on(rc_4, out, "Unable to attach to the AFI on slot id %d", slot_id);

    rc_sda = fpga_pci_attach(slot_id, FPGA_MGMT_PF, MGMT_PF_BAR4, 0, &pci_bar_handle_sda);
    fail_on(rc_sda, out, "Unable to attach to the AFI on slot id %d", slot_id);

//--------------------------input image data initialization----------------//
    if (!ifs1) {
        cout << "mean data file not found !" << endl;
        getchar();
    }
    while (ifs1 >> str1) {
        int p1 = str1.find(y1, 0);
        if (p1 >= 0) {
            str1.erase(p1, y1.length());
        }
        int p2 = str1.find(y2, 0);
        if (p2 >= 0) {
            str1.erase(p2, y2.length());
        }
        f = atof(str1.c_str());
        channel_mean[index] = f;
        index++;
    }
    ifs1.close();
    data = loadfile(image_dir, size);
    image_orig = stbi_load_from_memory(data, size, &w, &h, &channels, 3);
    for (loop_var = 0; loop_var < 3; loop_var++) {
        for (loop_var_1 = loop_var; loop_var_1 < w*h*3; loop_var_1 += 3) {
            in_data_3D_channel_swap[2 - loop_var][loop_var_1 / (w * 3)][(loop_var_1 % (w * 3) - loop_var) / 3] = (float)image_orig[loop_var_1]; //range:0--255
        }
    }
    for (loop_var = 0; loop_var < 3; loop_var++) {
        for (loop_var_1 = 0; loop_var_1 < 32; loop_var_1++) {
            for (loop_var_2 = 0; loop_var_2 < 32; loop_var_2++) {
                in_data_3D_channel_swap[loop_var][loop_var_1][loop_var_2] /= 255;// range:0--1
            }
        }
    }
    for (loop_var = 0; loop_var < 3; loop_var++) {
        for (loop_var_1 = 0; loop_var_1 < 32; loop_var_1++) {
            for (loop_var_2 = 0; loop_var_2 < 32; loop_var_2++) {
                in_data_3D[loop_var][loop_var_1][loop_var_2] = in_data_3D_channel_swap[loop_var][loop_var_1][loop_var_2] * 255 - channel_mean[loop_var];
            }
        }
    }
    for (loop_var = 0; loop_var < 3; loop_var++) {
        for (loop_var_1 = 0; loop_var_1 < 32; loop_var_1++) {
            for (loop_var_2 = 0; loop_var_2 < 32; loop_var_2++) {
                in_data[in_data_size] = (data_type)in_data_3D[loop_var][loop_var_1][loop_var_2];
                in_data_size++;
            }
        }
    }
    indata.open("./netOutput/in_data.txt", ios::app);
    for (loop_var = 0; loop_var < acc_param_conv_1[0]; loop_var++) {
        for (loop_var_1 = 0; loop_var_1 < acc_param_conv_1[3]; loop_var_1++) {
            for (loop_var_2 = 0; loop_var_2 < acc_param_conv_1[4]; loop_var_2++) {
                indata << in_data[loop_var*acc_param_conv_1[3]*acc_param_conv_1[4] + loop_var_1*acc_param_conv_1[4] + loop_var_2] << " ";
            }
            indata << endl;
        }
        indata << endl;
    }
    indata << endl;
    indata.close();
    cout << endl;

//----------------------input weight data initialization ------------------//
    // Prepare weights and bias for conv layer 1
    memset(conv_1_weight2D, 0, 2400 * sizeof(float));
    load_weight_conv(
        weight_src, 
        conv_1_weight2D,
        weight_bias_record,
        nn_channel_size_conv, 
        nn_in_number_conv,
        nn_out_number_conv,
        in_number_conv);
    //cout << "Loading conv weight 1 to memory space, starting at: " <<conv_weight_num << '\n';
    //for (int i = 0; i < 2400; i++) {
    //    conv_weight[conv_weight_num] = (data_type_w)conv_1_weight2D[i];
    //    conv_weight_num++;
   //}
    memset(conv_1_bias2D, 0, 32 * sizeof(float));
    load_bias_conv(
        weight_src, 
        conv_1_bias2D,
        weight_bias_record,
        nn_channel_size_conv, 
        nn_in_number_conv,
        nn_out_number_conv,
        in_number_conv);
    //cout << "Loading conv bias 1 to memory space, starting at: " <<conv_bias_num << '\n';
    for (int i = 0; i < 32; i++) {
        conv_bias[conv_bias_num] = (data_type_w)conv_1_bias2D[i];
        conv_bias_num++;
    }
    in_number_conv++;

    // Prepare weights and bias for conv layer 2
    memset(conv_2_weight2D, 0, 25600 * sizeof(float));
    load_weight_conv(
        weight_src, 
        conv_2_weight2D,
        weight_bias_record,
        nn_channel_size_conv, 
        nn_in_number_conv,
        nn_out_number_conv,
        in_number_conv);
    //cout << "Loading conv weight 2 to memory space, starting at: " <<conv_weight_num << '\n';
    //for (int i = 0; i < 25600; i++) {
    //    conv_weight[conv_weight_num] = (data_type_w)conv_2_weight2D[i];
    //    conv_weight_num++;
    //}
    memset(conv_2_bias2D, 0, 32 * sizeof(float));
    load_bias_conv(
        weight_src, 
        conv_2_bias2D,
        weight_bias_record,
        nn_channel_size_conv, 
        nn_in_number_conv,
        nn_out_number_conv,
        in_number_conv);
    //cout << "Loading conv bias 2 to memory space, starting at: " <<conv_bias_num << '\n';
    for (int i = 0; i < 32; i++) {
        conv_bias[conv_bias_num] = (data_type_w)conv_2_bias2D[i];
        conv_bias_num++;
    }
    in_number_conv++;

    // Prepare weights and bias for conv layer 3
    memset(conv_3_weight2D, 0, 51200 * sizeof(float));
    load_weight_conv(
        weight_src, 
        conv_3_weight2D,
        weight_bias_record,
        nn_channel_size_conv, 
        nn_in_number_conv,
        nn_out_number_conv,
        in_number_conv);
    //cout << "Loading conv weight 3 to memory space, starting at: " <<conv_weight_num << '\n';
    //for (int i = 0; i < 51200; i++) {
    //    conv_weight[conv_weight_num] = (data_type_w)conv_3_weight2D[i];
    //    conv_weight_num++;
    //}
    memset(conv_3_bias2D, 0, 64 * sizeof(float));
    load_bias_conv(
        weight_src, 
        conv_3_bias2D,
        weight_bias_record,
        nn_channel_size_conv, 
        nn_in_number_conv,
        nn_out_number_conv,
        in_number_conv);
    //cout << "Loading conv bias 3 to memory space, starting at: " <<conv_bias_num << '\n';
    for (int i = 0; i < 64; i++) {
        conv_bias[conv_bias_num] = (data_type_w)conv_3_bias2D[i];
        conv_bias_num++;
    }
    in_number_conv++;

    //cout<<"Finished loading conv weight into memory! Total: " <<conv_weight_num  << "... ... ..."<<endl;
    //cout<<"Finished loading conv bias into memory! Total: " <<conv_bias_num  << "... ... ..."<<endl;

    // Prepare weights and bias for fc layer 1
    memset(fc_1_weight2D, 0, 10240 * sizeof(float));
    load_weight_fc(
        weight_src, 
        fc_1_weight2D,
        weight_bias_record,
        nn_channel_size_fc, 
        nn_in_number_fc,
        nn_out_number_fc,
        in_number_fc);
    //cout << "Loading fc weight 1 to memory space, starting at: " <<conv_weight_num << '\n';
    //for (int i = 0; i < 10240; i++) {
    //    conv_weight[conv_weight_num] = (data_type_w)fc_1_weight2D[i];
    //    conv_weight_num++;
    //}
    memset(fc_1_bias2D, 0, 10 * sizeof(float));
    load_bias_fc(
        weight_src, 
        fc_1_bias2D,
        weight_bias_record,
        nn_channel_size_fc, 
        nn_in_number_fc,
        nn_out_number_fc,
        in_number_fc);
    //cout << "Loading fc bias 1 to memory space, starting at: " <<conv_bias_num << '\n';
    for (int i = 0; i < 10; i++) {
        conv_bias[conv_bias_num] = (data_type_w)fc_1_bias2D[i];
        conv_bias_num++;
    }
    in_number_fc++;
    //conv_1
    for (int i = 0; i < 32; i++) {
        for(int j = 0; j < 3; j++){
            for(int k1 = 0; k1 < 5; k1++){
                for(int k2 = 0; k2 < 5; k2++){
                    weight_temp_1[j][i][k1][k2] = (data_type_w)conv_1_weight2D[i*3*5*5 + j*5*5 + k1*5 + k2];
                }
            }
        }
    }
    for(int j=0; j<32; j++){
        for(int k1=0; k1 <5; k1++){
            for(int k2=0; k2<5; k2++){
                conv_weight_mem_port_1[j*5*5 + k1*5 + k2] = weight_temp_1[0][j][k1][k2];
                conv_weight_mem_port_2[j*5*5 + k1*5 + k2] = weight_temp_1[1][j][k1][k2];
                conv_weight_mem_port_3[j*5*5 + k1*5 + k2] = weight_temp_1[2][j][k1][k2];
            }
        }
    }
    //conv_2
    for (int i = 0; i < 32; i++) {
        for(int j = 0; j < 32; j++){
            for(int k1 = 0; k1 < 5; k1++){
                for(int k2 = 0; k2 < 5; k2++){
                    weight_temp_2[j][i][k1][k2] = (data_type_w)conv_2_weight2D[i*32*5*5 + j*5*5 + k1*5 + k2];
                }
            }
        }
    }
    for(int j=0; j<32; j++){
        for(int k=0; k <4; k++){
            for(int k1=0; k1 <5; k1++){
                for(int k2=0; k2<5; k2++){
                    conv_weight_mem_port_1[800+j*100+25*k + k1*5 + k2] = weight_temp_2[0+8*k][j][k1][k2];
                    conv_weight_mem_port_2[800+j*100+25*k + k1*5 + k2] = weight_temp_2[1+8*k][j][k1][k2];
                    conv_weight_mem_port_3[800+j*100+25*k + k1*5 + k2] = weight_temp_2[2+8*k][j][k1][k2];
                    conv_weight_mem_port_4[j*100+25*k + k1*5 + k2] = weight_temp_2[3+8*k][j][k1][k2];
                    conv_weight_mem_port_5[j*100+25*k + k1*5 + k2] = weight_temp_2[4+8*k][j][k1][k2];
                    conv_weight_mem_port_6[j*100+25*k + k1*5 + k2] = weight_temp_2[5+8*k][j][k1][k2];
                    conv_weight_mem_port_7[j*100+25*k + k1*5 + k2] = weight_temp_2[6+8*k][j][k1][k2];
                    conv_weight_mem_port_8[j*100+25*k + k1*5 + k2] = weight_temp_2[7+8*k][j][k1][k2];
                }
            }
        }
    }
    //conv_3
    for (int i = 0; i < 64; i++) {
        for(int j = 0; j < 32; j++){
            for(int k1 = 0; k1 < 5; k1++){
                for(int k2 = 0; k2 < 5; k2++){
                    weight_temp_3[j][i][k1][k2] = (data_type_w)conv_3_weight2D[i*32*5*5 + j*5*5 + k1*5 + k2];
                }
            }
        }
    }
    for(int j=0; j<64; j++){
        for(int k=0; k <4; k++){
            for(int k1=0; k1 <5; k1++){
                for(int k2=0; k2<5; k2++){
                    conv_weight_mem_port_1[4000+j*100+25*k + k1*5 + k2] = weight_temp_3[0+8*k][j][k1][k2];
                    conv_weight_mem_port_2[4000+j*100+25*k + k1*5 + k2] = weight_temp_3[1+8*k][j][k1][k2];
                    conv_weight_mem_port_3[4000+j*100+25*k + k1*5 + k2] = weight_temp_3[2+8*k][j][k1][k2];
                    conv_weight_mem_port_4[3200+j*100+25*k + k1*5 + k2] = weight_temp_3[3+8*k][j][k1][k2];
                    conv_weight_mem_port_5[3200+j*100+25*k + k1*5 + k2] = weight_temp_3[4+8*k][j][k1][k2];
                    conv_weight_mem_port_6[3200+j*100+25*k + k1*5 + k2] = weight_temp_3[5+8*k][j][k1][k2];
                    conv_weight_mem_port_7[3200+j*100+25*k + k1*5 + k2] = weight_temp_3[6+8*k][j][k1][k2];
                    conv_weight_mem_port_8[3200+j*100+25*k + k1*5 + k2] = weight_temp_3[7+8*k][j][k1][k2];
                }
            }
        }
    }
    //fc_1
    for (int i = 0; i < 10; i++) {
        for(int j = 0; j < 64; j++){
            for(int k1 = 0; k1 < 4; k1++){
                for(int k2 = 0; k2 < 4; k2++){
                    weight_temp_4[j][i][k1][k2] = (data_type_w)fc_1_weight2D[i*64*4*4 + j*4*4 + k1*4 + k2];
                }
            }
        }
    }
    for(int j=0; j<10; j++){
        for(int k=0; k <8; k++){
            for(int k1=0; k1 <4; k1++){
                for(int k2=0; k2<4; k2++){
                    conv_weight_mem_port_1[10400+j*128+16*k + k1*4 + k2] = weight_temp_4[0+8*k][j][k1][k2];
                    conv_weight_mem_port_2[10400+j*128+16*k + k1*4 + k2] = weight_temp_4[1+8*k][j][k1][k2];
                    conv_weight_mem_port_3[10400+j*128+16*k + k1*4 + k2] = weight_temp_4[2+8*k][j][k1][k2];
                    conv_weight_mem_port_4[9600+j*128+16*k + k1*4 + k2] = weight_temp_4[3+8*k][j][k1][k2];
                    conv_weight_mem_port_5[9600+j*128+16*k + k1*4 + k2] = weight_temp_4[4+8*k][j][k1][k2];
                    conv_weight_mem_port_6[9600+j*128+16*k + k1*4 + k2] = weight_temp_4[5+8*k][j][k1][k2];
                    conv_weight_mem_port_7[9600+j*128+16*k + k1*4 + k2] = weight_temp_4[6+8*k][j][k1][k2];
                    conv_weight_mem_port_8[9600+j*128+16*k + k1*4 + k2] = weight_temp_4[7+8*k][j][k1][k2];
                }
            }
        }
    }
    //
    for ( loop_var = 0; loop_var < 11680; loop_var++ ){
        conv_weight[conv_weight_num] = conv_weight_mem_port_1[loop_var];
        conv_weight_num++;
    }
    for ( loop_var = 0; loop_var < 11680; loop_var++ ){
        conv_weight[conv_weight_num] = conv_weight_mem_port_2[loop_var];
        conv_weight_num++;
    }
    for ( loop_var = 0; loop_var < 11680; loop_var++ ){
        conv_weight[conv_weight_num] = conv_weight_mem_port_3[loop_var];
        conv_weight_num++;
    }
    for ( loop_var = 0; loop_var < 10880; loop_var++ ){
        conv_weight[conv_weight_num] = conv_weight_mem_port_4[loop_var];
        conv_weight_num++;
    }
    for ( loop_var = 0; loop_var < 10880; loop_var++ ){
        conv_weight[conv_weight_num] = conv_weight_mem_port_5[loop_var];
        conv_weight_num++;
    }
    for ( loop_var = 0; loop_var < 10880; loop_var++ ){
        conv_weight[conv_weight_num] = conv_weight_mem_port_6[loop_var];
        conv_weight_num++;
    }
    for ( loop_var = 0; loop_var < 10880; loop_var++ ){
        conv_weight[conv_weight_num] = conv_weight_mem_port_7[loop_var];
        conv_weight_num++;
    }
    for ( loop_var = 0; loop_var < 10880; loop_var++ ){
        conv_weight[conv_weight_num] = conv_weight_mem_port_8[loop_var];
        conv_weight_num++;
    }
    //write data to DDR_SH_ADDR
    //Fill_Bram(pci_bar_handle_4, DDR_SH_ADDR, conv_weight, 2400+25600+51200+10240);
    Fill_Bram(pci_bar_handle_4, DDR_SH_ADDR_1, conv_weight_mem_port_1, 11680);
    Fill_Bram(pci_bar_handle_4, DDR_SH_ADDR_2, conv_weight_mem_port_2, 11680);
    Fill_Bram(pci_bar_handle_4, DDR_SH_ADDR_3, conv_weight_mem_port_3, 11680);
    Fill_Bram(pci_bar_handle_4, DDR_SH_ADDR_4, conv_weight_mem_port_4, 10880);
    Fill_Bram(pci_bar_handle_4, DDR_SH_ADDR_5, conv_weight_mem_port_5, 10880);
    Fill_Bram(pci_bar_handle_4, DDR_SH_ADDR_6, conv_weight_mem_port_6, 10880);
    Fill_Bram(pci_bar_handle_4, DDR_SH_ADDR_7, conv_weight_mem_port_7, 10880);
    Fill_Bram(pci_bar_handle_4, DDR_SH_ADDR_8, conv_weight_mem_port_8, 10880);
    Fill_Bram(pci_bar_handle_4, DDR_B_ADDR, conv_bias, 32+32+64+10);
    Fill_Bram(pci_bar_handle_4, DDR_A_ADDR, in_data, 3*32*32);
    
    printf("Finished writing to SH_DDR data\n");
    //cout<<"Finished loading fc weight into memory! Total: " <<conv_weight_num  << "... ... ..."<<endl;
    //cout<<"Finished loading fc bias into memory! Total: " <<conv_bias_num  << "... ... ..."<<endl;

//---------------------conv parameter bram transmission---------------------// 
    //cout << "Finished filling conv acc parameter into param bram!" << endl;
//---------------------conv 1 weight bram ------------------------------------//
    //nn_in_number_conv[in_number_conv]*nn_out_number_conv[in_number_conv]*nn_channel_size_conv[in_number_conv]*nn_channel_size_conv[in_number_conv]
    gettimeofday(&start,0);
    set_cdma(pci_bar_handle,0x01000000,0x0000000E,0xC4000000,0x00000000,0x00000640);
    set_cdma(pci_bar_handle,0x01100000,0x0000000E,0xC4010000,0x00000000,0x00000640);
    set_cdma(pci_bar_handle,0x01200000,0x0000000E,0xC4020000,0x00000000,0x00000640);
    
//----------------------conv 1 bias bram -------------------------------------//
    //nn_out_number_conv[in_number_conv]
    Fill_Bram(pci_bar_handle_4, CONV_B_BRAM_PCIS, conv_bias, 32);

//----------------------input data buffer load------------------------------//
    set_cdma(pci_bar_handle,0x02000000,0x0000000C,0xC2000000,0x00000000,0x00001000);
    set_cdma(pci_bar_handle,0x02000000+0x00001000,0x0000000C,0xC2010000,0x00000000,0x00001000);
    set_cdma(pci_bar_handle,0x02000000+0x00002000,0x0000000C,0xC2020000,0x00000000,0x00001000);
//----------------------conv_1 layer -----------------------//  
//----------------------inference net ip status check -----------------------//    
    //conv 1-1
    Fill_param(pci_bar_handle_4, CTRL_PARAMS, ctrl_param_1, 2); 
    Fill_param(pci_bar_handle_4, ACC_PARAMS_0, acc_param_conv_1, 16); 
    gettimeofday(&end,0);
    diff = 1000000*(end.tv_sec-start.tv_sec) + end.tv_usec-start.tv_usec;
    cout << "Convolution layer 1_1 data load time = " << diff << "  us" << endl;

    gettimeofday(&start,0);
    XInference_net_Start(pci_bar_handle, &InstancePtr);
    //load_weight
    set_cdma(pci_bar_handle,0x01000000+0x00000640,0x0000000E,0xC4000000+0x00000640,0x00000000,0x00000640);
    set_cdma(pci_bar_handle,0x01100000+0x00000640,0x0000000E,0xC4010000+0x00000640,0x00000000,0x00000640);
    set_cdma(pci_bar_handle,0x01200000+0x00000640,0x0000000E,0xC4020000+0x00000640,0x00000000,0x00000640);

    //load_data
    set_cdma(pci_bar_handle,0x02000000,0x0000000C,0xC2000000+0x00001000,0x00000000,0x00001000);
    set_cdma(pci_bar_handle,0x02000000+0x00001000,0x0000000C,0xC2010000+0x00001000,0x00000000,0x00001000);
    set_cdma(pci_bar_handle,0x02000000+0x00002000,0x0000000C,0xC2020000+0x00001000,0x00000000,0x00001000);
    while (!XInference_net_IsDone(pci_bar_handle, &InstancePtr)) {
        count++;
    }
    gettimeofday(&end,0);
    diff = 1000000*(end.tv_sec-start.tv_sec) + end.tv_usec-start.tv_usec;
    cout << "Convolution layer 1_1 processing time = " << diff << "  us" << endl;
    /*gettimeofday(&start,0);
    Read_Bram(pci_bar_handle_4, BUF_OUT_1_0, &out_temp_1[0], 2048);
    Read_Bram(pci_bar_handle_4, BUF_OUT_1_1, &out_temp_2[0], 2048);
    Read_Bram(pci_bar_handle_4, BUF_OUT_1_2, &out_temp_3[0], 2048);
    Read_Bram(pci_bar_handle_4, BUF_OUT_1_3, &out_temp_4[0], 2048);
    Read_Bram(pci_bar_handle_4, BUF_OUT_1_4, &out_temp_5[0], 2048);
    Read_Bram(pci_bar_handle_4, BUF_OUT_1_5, &out_temp_6[0], 2048);
    Read_Bram(pci_bar_handle_4, BUF_OUT_1_6, &out_temp_7[0], 2048);
    Read_Bram(pci_bar_handle_4, BUF_OUT_1_7, &out_temp_8[0], 2048);*/
    /*Read_Bram(pci_bar_handle_4, BUF_OUT_1_0, out_temp_full, 2048);
    outdata.open("./netOutput/out_temp_data.txt", ios::app);
    outdata <<"conv_output:"<< endl;
    for(loop_var = 0; loop_var < 2048; loop_var++){
        outdata << out_temp_full[loop_var] << "  ";
    }
    outdata << endl;    
    outdata.close();*/
    /*gettimeofday(&end,0);
    diff = 1000000*(end.tv_sec-start.tv_sec) + end.tv_usec-start.tv_usec;
    cout << "Convolution layer 1_1 out store time = " << diff << "  us" << endl;*/
    
    //pool 1_1
    gettimeofday(&start,0);
    Fill_param(pci_bar_handle_4, CTRL_PARAMS, ctrl_param_2, 2);
    Fill_param(pci_bar_handle_4, ACC_PARAMS_1, acc_param_pool_1, 9); 
    gettimeofday(&end,0);
    diff = 1000000*(end.tv_sec-start.tv_sec) + end.tv_usec-start.tv_usec;
    cout << "Pooling layer 1_1 data load time = " << diff << "  us" << endl;
    //----------------------inference net ip status check -----------------------//    
    //ip_status = XInference_net_ReadReg(pci_bar_handle, &InstancePtr.ctrl_bus_baseaddress, XINFERENCE_NET_CRTL_BUS_ADDR_AP_CTRL);
    //cout << "Status feedback from inference ip is : " << ip_status << endl;
    gettimeofday(&start,0);
    XInference_net_Start(pci_bar_handle, &InstancePtr);
    while (!XInference_net_IsDone(pci_bar_handle, &InstancePtr)) {
        count++;
    }
    gettimeofday(&end,0);
    diff = 1000000*(end.tv_sec-start.tv_sec) + end.tv_usec-start.tv_usec;
    cout << "Pooling layer 1_1 processing time = " << diff << "  us" << endl;
    gettimeofday(&start,0);
    set_cdma(pci_bar_handle,0xC2000000,0x00000000,0x02000000,0x0000000E,0x00000400);
    set_cdma(pci_bar_handle,0xC2010000,0x00000000,0x02000000+0x00000400,0x0000000E,0x00000400);
    set_cdma(pci_bar_handle,0xC2020000,0x00000000,0x02000000+0x00000800,0x0000000E,0x00000400);
    set_cdma(pci_bar_handle,0xC2030000,0x00000000,0x02000000+0x00000C00,0x0000000E,0x00000400);
    set_cdma(pci_bar_handle,0xC2040000,0x00000000,0x02000000+0x00001000,0x0000000E,0x00000400);
    set_cdma(pci_bar_handle,0xC2050000,0x00000000,0x02000000+0x00001400,0x0000000E,0x00000400);
    set_cdma(pci_bar_handle,0xC2060000,0x00000000,0x02000000+0x00001800,0x0000000E,0x00000400);
    set_cdma(pci_bar_handle,0xC2070000,0x00000000,0x02000000+0x00001C00,0x0000000E,0x00000400);

    set_cdma(pci_bar_handle,0xC2000000+0x00000400,0x00000000,0x02000000+0x00002000,0x0000000E,0x00000400);
    set_cdma(pci_bar_handle,0xC2010000+0x00000400,0x00000000,0x02000000+0x00002400,0x0000000E,0x00000400);
    set_cdma(pci_bar_handle,0xC2020000+0x00000400,0x00000000,0x02000000+0x00002800,0x0000000E,0x00000400);
    set_cdma(pci_bar_handle,0xC2030000+0x00000400,0x00000000,0x02000000+0x00002C00,0x0000000E,0x00000400);
    set_cdma(pci_bar_handle,0xC2040000+0x00000400,0x00000000,0x02000000+0x00003000,0x0000000E,0x00000400);
    set_cdma(pci_bar_handle,0xC2050000+0x00000400,0x00000000,0x02000000+0x00003400,0x0000000E,0x00000400);
    set_cdma(pci_bar_handle,0xC2060000+0x00000400,0x00000000,0x02000000+0x00003800,0x0000000E,0x00000400);
    set_cdma(pci_bar_handle,0xC2070000+0x00000400,0x00000000,0x02000000+0x00003C00,0x0000000E,0x00000400);
    /*outdata.open("./netOutput/pool_temp_out_data.txt", ios::app);
    outdata <<"pool_output:"<< endl;
    Read_Bram(pci_bar_handle_4, BUF_OUT_0_0, out_temp_full, 512);
    for(loop_var = 0; loop_var < 512; loop_var++){
        outdata << out_temp_full[loop_var] << "  ";
        out_temp_2[loop_var]=out_temp_full[loop_var];
    }
    outdata << endl;    
    outdata.close();*/
    gettimeofday(&end,0);
    diff = 1000000*(end.tv_sec-start.tv_sec) + end.tv_usec-start.tv_usec;
    cout << "Pooling layer 1_1 out store time = " << diff << "  us" << endl;

    //conv 1-2
    gettimeofday(&start,0);
    acc_param_conv_1[10] = 400;
    acc_param_conv_1[11] = 16;
    acc_param_conv_1[12] = 1024;
    Fill_param(pci_bar_handle_4, CTRL_PARAMS, ctrl_param_1, 2); 
    Fill_param(pci_bar_handle_4, ACC_PARAMS_0, acc_param_conv_1, 16); 
    gettimeofday(&end,0);
    diff = 1000000*(end.tv_sec-start.tv_sec) + end.tv_usec-start.tv_usec;
    cout << "Convolution layer 1_2 data load time = " << diff << "  us" << endl;
    //ip_status = XInference_net_ReadReg(pci_bar_handle, &InstancePtr.ctrl_bus_baseaddress, XINFERENCE_NET_CRTL_BUS_ADDR_AP_CTRL);
    //cout << "Status feedback from inference ip is : " << ip_status << endl;
    gettimeofday(&start,0);
    XInference_net_Start(pci_bar_handle, &InstancePtr);
    //load_weight_conv_2
    set_cdma(pci_bar_handle,0x01000000+0x00000C80,0x0000000E,0xC4000000+0x00001000,0x00000000,0x00003200);
    set_cdma(pci_bar_handle,0x01100000+0x00000C80,0x0000000E,0xC4010000+0x00001000,0x00000000,0x00003200);
    set_cdma(pci_bar_handle,0x01200000+0x00000C80,0x0000000E,0xC4020000+0x00001000,0x00000000,0x00003200);
    set_cdma(pci_bar_handle,0x01300000,0x0000000E,0xC4030000+0x00001000,0x00000000,0x00003200);
    set_cdma(pci_bar_handle,0x01400000,0x0000000E,0xC4030000+0x00001000,0x00000000,0x00003200);
    set_cdma(pci_bar_handle,0x01500000,0x0000000E,0xC4050000+0x00001000,0x00000000,0x00003200);
    set_cdma(pci_bar_handle,0x01600000,0x0000000E,0xC4060000+0x00001000,0x00000000,0x00003200);
    set_cdma(pci_bar_handle,0x01700000,0x0000000E,0xC4070000+0x00001000,0x00000000,0x00003200);
    Fill_Bram(pci_bar_handle_4, CONV_B_BRAM_PCIS+32*4, conv_2_bias2D, 32);
    
    //load_data_conv_2
    set_cdma(pci_bar_handle,0x02000000,0x0000000E,0xC2000000+0x00001000,0x00000000,0x00000400);
    set_cdma(pci_bar_handle,0x02000000+0x00000400,0x0000000E,0xC2010000+0x00001000,0x00000000,0x00000400);
    set_cdma(pci_bar_handle,0x02000000+0x00000800,0x0000000E,0xC2020000+0x00001000,0x00000000,0x00000400);
    set_cdma(pci_bar_handle,0x02000000+0x00000C00,0x0000000E,0xC2030000+0x00001000,0x00000000,0x00000400);
    set_cdma(pci_bar_handle,0x02000000+0x00001000,0x0000000E,0xC2040000+0x00001000,0x00000000,0x00000400);
    set_cdma(pci_bar_handle,0x02000000+0x00001400,0x0000000E,0xC2050000+0x00001000,0x00000000,0x00000400);
    set_cdma(pci_bar_handle,0x02000000+0x00001800,0x0000000E,0xC2060000+0x00001000,0x00000000,0x00000400);
    set_cdma(pci_bar_handle,0x02000000+0x00001C00,0x0000000E,0xC2070000+0x00001000,0x00000000,0x00000400);

    set_cdma(pci_bar_handle,0x02000000+0x00002000,0x0000000E,0xC2000000+0x00001000+0x00000400,0x00000000,0x00000400);
    set_cdma(pci_bar_handle,0x02000000+0x00002400,0x0000000E,0xC2010000+0x00001000+0x00000400,0x00000000,0x00000400);
    set_cdma(pci_bar_handle,0x02000000+0x00002800,0x0000000E,0xC2020000+0x00001000+0x00000400,0x00000000,0x00000400);
    set_cdma(pci_bar_handle,0x02000000+0x00002C00,0x0000000E,0xC2030000+0x00001000+0x00000400,0x00000000,0x00000400);
    set_cdma(pci_bar_handle,0x02000000+0x00003000,0x0000000E,0xC2040000+0x00001000+0x00000400,0x00000000,0x00000400);
    set_cdma(pci_bar_handle,0x02000000+0x00003400,0x0000000E,0xC2050000+0x00001000+0x00000400,0x00000000,0x00000400);
    set_cdma(pci_bar_handle,0x02000000+0x00003800,0x0000000E,0xC2060000+0x00001000+0x00000400,0x00000000,0x00000400);
    set_cdma(pci_bar_handle,0x02000000+0x00003C00,0x0000000E,0xC2070000+0x00001000+0x00000400,0x00000000,0x00000400);
    while (!XInference_net_IsDone(pci_bar_handle, &InstancePtr)) {
        count++;
    }
    gettimeofday(&end,0);
    diff = 1000000*(end.tv_sec-start.tv_sec) + end.tv_usec-start.tv_usec;
    cout << "Convolution layer 1_2 processing time = " << diff << "  us" << endl;
    //cout << "IP is done at " << count << " attempts" << endl; 
    /*gettimeofday(&start,0);*/
    /*Read_Bram(pci_bar_handle_4, BUF_OUT_1_7, out_temp_full, 2048);
    outdata.open("./netOutput/out_temp_data.txt", ios::app);
    outdata <<"conv_output:"<< endl;
    for(loop_var = 0; loop_var < 2048; loop_var++){
        outdata << out_temp_full[loop_var] << "  ";
    }
    outdata << endl;    
    outdata.close();*/
    /*gettimeofday(&end,0);
    diff = 1000000*(end.tv_sec-start.tv_sec) + end.tv_usec-start.tv_usec;
    cout << "Convolution layer 1_2 out store time = " << diff << "  us" << endl;*/
    
    //pool 1_2
    gettimeofday(&start,0);
    Fill_param(pci_bar_handle_4, CTRL_PARAMS, ctrl_param_2, 2);
    Fill_param(pci_bar_handle_4, ACC_PARAMS_1, acc_param_pool_1, 9); 
    gettimeofday(&end,0);
    diff = 1000000*(end.tv_sec-start.tv_sec) + end.tv_usec-start.tv_usec;
    cout << "Pooling layer 1_2 data load time = " << diff << "  us" << endl;
    //----------------------inference net ip status check -----------------------//    
    //ip_status = XInference_net_ReadReg(pci_bar_handle, &InstancePtr.ctrl_bus_baseaddress, XINFERENCE_NET_CRTL_BUS_ADDR_AP_CTRL);
    //cout << "Status feedback from inference ip is : " << ip_status << endl;
    gettimeofday(&start,0);
    XInference_net_Start(pci_bar_handle, &InstancePtr);
    while (!XInference_net_IsDone(pci_bar_handle, &InstancePtr)) {
        count++;
    }
    gettimeofday(&end,0);
    diff = 1000000*(end.tv_sec-start.tv_sec) + end.tv_usec-start.tv_usec;
    cout << "Pooling layer 1_2 processing time = " << diff << "  us" << endl;
    gettimeofday(&start,0);
    set_cdma(pci_bar_handle,0xC2000000,0x00000000,0x02004000,0x0000000E,0x00000400);
    set_cdma(pci_bar_handle,0xC2010000,0x00000000,0x02004000+0x00000400,0x0000000E,0x00000400);
    set_cdma(pci_bar_handle,0xC2020000,0x00000000,0x02004000+0x00000800,0x0000000E,0x00000400);
    set_cdma(pci_bar_handle,0xC2030000,0x00000000,0x02004000+0x00000C00,0x0000000E,0x00000400);
    set_cdma(pci_bar_handle,0xC2040000,0x00000000,0x02004000+0x00001000,0x0000000E,0x00000400);
    set_cdma(pci_bar_handle,0xC2050000,0x00000000,0x02004000+0x00001400,0x0000000E,0x00000400);
    set_cdma(pci_bar_handle,0xC2060000,0x00000000,0x02004000+0x00001800,0x0000000E,0x00000400);
    set_cdma(pci_bar_handle,0xC2070000,0x00000000,0x02004000+0x00001C00,0x0000000E,0x00000400);

    set_cdma(pci_bar_handle,0xC2000000+0x00000400,0x00000000,0x02004000+0x00002000,0x0000000E,0x00000400);
    set_cdma(pci_bar_handle,0xC2010000+0x00000400,0x00000000,0x02004000+0x00002400,0x0000000E,0x00000400);
    set_cdma(pci_bar_handle,0xC2020000+0x00000400,0x00000000,0x02004000+0x00002800,0x0000000E,0x00000400);
    set_cdma(pci_bar_handle,0xC2030000+0x00000400,0x00000000,0x02004000+0x00002C00,0x0000000E,0x00000400);
    set_cdma(pci_bar_handle,0xC2040000+0x00000400,0x00000000,0x02004000+0x00003000,0x0000000E,0x00000400);
    set_cdma(pci_bar_handle,0xC2050000+0x00000400,0x00000000,0x02004000+0x00003400,0x0000000E,0x00000400);
    set_cdma(pci_bar_handle,0xC2060000+0x00000400,0x00000000,0x02004000+0x00003800,0x0000000E,0x00000400);
    set_cdma(pci_bar_handle,0xC2070000+0x00000400,0x00000000,0x02004000+0x00003C00,0x0000000E,0x00000400);
    /*outdata.open("./netOutput/pool_temp_out_data.txt", ios::app);
    outdata <<"pool_output:"<< endl;
    Read_Bram(pci_bar_handle_4, BUF_OUT_0_7, out_temp_full, 512);
    for(loop_var = 0; loop_var < 512; loop_var++){
        outdata << out_temp_full[loop_var] << "  ";
    }
    outdata << endl;    
    outdata.close();*/
    gettimeofday(&end,0);
    diff = 1000000*(end.tv_sec-start.tv_sec) + end.tv_usec-start.tv_usec;
    cout << "Pooling layer 1_2 out store time = " << diff << "  us" << endl;

    //----------------------conv_2 layer -----------------------//  
    gettimeofday(&start,0); 
    Fill_param(pci_bar_handle_4, CTRL_PARAMS, ctrl_param_1, 2);
    Fill_param(pci_bar_handle_4, ACC_PARAMS_0, acc_param_conv_2, 16); 

    //load_data
    set_cdma(pci_bar_handle,0x02000000+0x00004000,0x0000000E,0xC2000000+0x00001000+0x00000800,0x00000000,0x00000400);
    set_cdma(pci_bar_handle,0x02000000+0x00004400,0x0000000E,0xC2010000+0x00001000+0x00000800,0x00000000,0x00000400);
    set_cdma(pci_bar_handle,0x02000000+0x00004800,0x0000000E,0xC2020000+0x00001000+0x00000800,0x00000000,0x00000400);
    set_cdma(pci_bar_handle,0x02000000+0x00004C00,0x0000000E,0xC2030000+0x00001000+0x00000800,0x00000000,0x00000400);
    set_cdma(pci_bar_handle,0x02000000+0x00005000,0x0000000E,0xC2040000+0x00001000+0x00000800,0x00000000,0x00000400);
    set_cdma(pci_bar_handle,0x02000000+0x00005400,0x0000000E,0xC2050000+0x00001000+0x00000800,0x00000000,0x00000400);
    set_cdma(pci_bar_handle,0x02000000+0x00005800,0x0000000E,0xC2060000+0x00001000+0x00000800,0x00000000,0x00000400);
    set_cdma(pci_bar_handle,0x02000000+0x00005C00,0x0000000E,0xC2070000+0x00001000+0x00000800,0x00000000,0x00000400);

    set_cdma(pci_bar_handle,0x02000000+0x00006000,0x0000000E,0xC2000000+0x00001000+0x00000C00,0x00000000,0x00000400);
    set_cdma(pci_bar_handle,0x02000000+0x00006400,0x0000000E,0xC2010000+0x00001000+0x00000C00,0x00000000,0x00000400);
    set_cdma(pci_bar_handle,0x02000000+0x00006800,0x0000000E,0xC2020000+0x00001000+0x00000C00,0x00000000,0x00000400);
    set_cdma(pci_bar_handle,0x02000000+0x00006C00,0x0000000E,0xC2030000+0x00001000+0x00000C00,0x00000000,0x00000400);
    set_cdma(pci_bar_handle,0x02000000+0x00007000,0x0000000E,0xC2040000+0x00001000+0x00000C00,0x00000000,0x00000400);
    set_cdma(pci_bar_handle,0x02000000+0x00007400,0x0000000E,0xC2050000+0x00001000+0x00000C00,0x00000000,0x00000400);
    set_cdma(pci_bar_handle,0x02000000+0x00007800,0x0000000E,0xC2060000+0x00001000+0x00000C00,0x00000000,0x00000400);
    set_cdma(pci_bar_handle,0x02000000+0x00007C00,0x0000000E,0xC2070000+0x00001000+0x00000C00,0x00000000,0x00000400);
    gettimeofday(&end,0);
    diff = 1000000*(end.tv_sec-start.tv_sec) + end.tv_usec-start.tv_usec;
    cout << "Convolution layer 2 data load time = " << diff << "  us" << endl;
    //----------------------inference net ip status check -----------------------//    
    //conv 2
    //ip_status = XInference_net_ReadReg(pci_bar_handle, InstancePtr.ctrl_bus_baseaddress, XINFERENCE_NET_CRTL_BUS_ADDR_AP_CTRL);
    //cout << "Status feedback from inference ip is : " << ip_status << endl;
    gettimeofday(&start,0);
    XInference_net_Start(pci_bar_handle, &InstancePtr);
    //load_weight_conv_3
    set_cdma(pci_bar_handle,0x01000000+0x00003E80,0x0000000E,0xC4000000+0x00004200,0x00000000,0x00006400);
    set_cdma(pci_bar_handle,0x01100000+0x00003E80,0x0000000E,0xC4010000+0x00004200,0x00000000,0x00006400);
    set_cdma(pci_bar_handle,0x01200000+0x00003E80,0x0000000E,0xC4020000+0x00004200,0x00000000,0x00006400);
    set_cdma(pci_bar_handle,0x01300000+0x00003200,0x0000000E,0xC4030000+0x00004200,0x00000000,0x00006400);
    set_cdma(pci_bar_handle,0x01400000+0x00003200,0x0000000E,0xC4030000+0x00004200,0x00000000,0x00006400);
    set_cdma(pci_bar_handle,0x01500000+0x00003200,0x0000000E,0xC4050000+0x00004200,0x00000000,0x00006400);
    set_cdma(pci_bar_handle,0x01600000+0x00003200,0x0000000E,0xC4060000+0x00004200,0x00000000,0x00006400);
    set_cdma(pci_bar_handle,0x01700000+0x00003200,0x0000000E,0xC4070000+0x00004200,0x00000000,0x00006400);
    Fill_Bram(pci_bar_handle_4, CONV_B_BRAM_PCIS+64*4, conv_3_bias2D, 64);
    while (!XInference_net_IsDone(pci_bar_handle, &InstancePtr)) {
        count++;
    }
    gettimeofday(&end,0);
    diff = 1000000*(end.tv_sec-start.tv_sec) + end.tv_usec-start.tv_usec;
    cout << "Convolution layer 2 processing time = " << diff << "  us" << endl;
    //cout << "IP is done at " << count << " attempts" << endl; 
    /*gettimeofday(&start,0);
    /*Read_Bram(pci_bar_handle_4, BUF_OUT_1_0, out_temp_full, 1024);
    outdata.open("./netOutput/out_temp_data.txt", ios::app);
    outdata <<"conv_output:"<< endl;
    for(loop_var = 0; loop_var < 1024; loop_var++){
        outdata << out_temp_full[loop_var] << "  ";
        out_temp_1[loop_var]=out_temp_full[loop_var];
    }
    outdata << endl;    
    outdata.close();*/
    /*gettimeofday(&end,0);
    diff = 1000000*(end.tv_sec-start.tv_sec) + end.tv_usec-start.tv_usec;
    cout << "Convolution layer 2_1 out store time = " << diff << "  us" << endl;*/
    
    //pool 2
    gettimeofday(&start,0); 
    Fill_param(pci_bar_handle_4, CTRL_PARAMS, ctrl_param_2, 2);
    Fill_param(pci_bar_handle_4, ACC_PARAMS_1, acc_param_pool_2, 9); 
    gettimeofday(&end,0);
    diff = 1000000*(end.tv_sec-start.tv_sec) + end.tv_usec-start.tv_usec;
    cout << "Pooling layer 2 data load time = " << diff << "  us" << endl;
    //----------------------inference net ip status check -----------------------//    
    //ip_status = XInference_net_ReadReg(pci_bar_handle, InstancePtr.ctrl_bus_baseaddress, XINFERENCE_NET_CRTL_BUS_ADDR_AP_CTRL);
    //cout << "Status feedback from inference ip is : " << ip_status << endl;
    gettimeofday(&start,0);
    XInference_net_Start(pci_bar_handle, &InstancePtr);
    while (!XInference_net_IsDone(pci_bar_handle, &InstancePtr)) {
        count++;
    }
    gettimeofday(&end,0);
    diff = 1000000*(end.tv_sec-start.tv_sec) + end.tv_usec-start.tv_usec;
    cout << "Pooling layer 2 processing time = " << diff << "  us" << endl;
    /*gettimeofday(&start,0); 
    outdata.open("./netOutput/pool_temp_out_data.txt", ios::app);
    outdata <<"pool_output:"<< endl;
    //Read_Bram(pci_bar_handle_4, BUF_OUT_0_0, out_temp_3, 2048);
    for(loop_var = 0; loop_var < 256; loop_var++){
        outdata << out_temp_1_1[loop_var] << "  ";
    }
    outdata << endl;    
    outdata.close();
    gettimeofday(&end,0);
    diff = 1000000*(end.tv_sec-start.tv_sec) + end.tv_usec-start.tv_usec;
    cout << "Pooling layer 2_1 processing time = " << diff << "  us" << endl;*/

    //----------------------conv_3 layer -----------------------//   
    gettimeofday(&start,0); 
    Fill_param(pci_bar_handle_4, CTRL_PARAMS, ctrl_param_1, 2);
    Fill_param(pci_bar_handle_4, ACC_PARAMS_0, acc_param_conv_3, 16); 
    gettimeofday(&end,0);
    diff = 1000000*(end.tv_sec-start.tv_sec) + end.tv_usec-start.tv_usec;
    cout << "Convolution layer 3 data load time = " << diff << "  us" << endl;
    //----------------------inference net ip status check -----------------------//    
    //conv 3
    //ip_status = XInference_net_ReadReg(pci_bar_handle, InstancePtr.ctrl_bus_baseaddress, XINFERENCE_NET_CRTL_BUS_ADDR_AP_CTRL);
    //cout << "Status feedback from inference ip is : " << ip_status << endl;
    gettimeofday(&start,0);
    XInference_net_Start(pci_bar_handle, &InstancePtr);
    //load_weight_fc_1
    set_cdma(pci_bar_handle,0x01000000+0x0000A280,0x0000000E,0xC4000000+0x0000A600,0x00000000,0x00001400);
    set_cdma(pci_bar_handle,0x01100000+0x0000A280,0x0000000E,0xC4010000+0x0000A600,0x00000000,0x00001400);
    set_cdma(pci_bar_handle,0x01200000+0x0000A280,0x0000000E,0xC4020000+0x0000A600,0x00000000,0x00001400);
    set_cdma(pci_bar_handle,0x01300000+0x00009600,0x0000000E,0xC4030000+0x0000A600,0x00000000,0x00001400);
    set_cdma(pci_bar_handle,0x01400000+0x00009600,0x0000000E,0xC4030000+0x0000A600,0x00000000,0x00001400);
    set_cdma(pci_bar_handle,0x01500000+0x00009600,0x0000000E,0xC4050000+0x0000A600,0x00000000,0x00001400);
    set_cdma(pci_bar_handle,0x01600000+0x00009600,0x0000000E,0xC4060000+0x0000A600,0x00000000,0x00001400);
    set_cdma(pci_bar_handle,0x01700000+0x00009600,0x0000000E,0xC4070000+0x0000A600,0x00000000,0x00001400);
    Fill_Bram(pci_bar_handle_4, CONV_B_BRAM_PCIS+128*4, fc_1_bias2D, 10);
    while (!XInference_net_IsDone(pci_bar_handle, &InstancePtr)) {
        count++;
    }
    gettimeofday(&end,0);
    diff = 1000000*(end.tv_sec-start.tv_sec) + end.tv_usec-start.tv_usec;
    cout << "Convolution layer 3 processing time = " << diff << "  us" << endl;
    //cout << "IP is done at " << count << " attempts" << endl; 
    //gettimeofday(&start,0); 
    /*Read_Bram(pci_bar_handle_4, BUF_OUT_1_0, out_temp_full, 640);
    outdata.open("./netOutput/out_temp_data.txt", ios::app);
    outdata <<"conv_output:"<< endl;
    for(loop_var = 0; loop_var < 640; loop_var++){
        outdata << out_temp_full[loop_var] << "  ";
        out_temp_1[loop_var]=out_temp_full[loop_var];
    }
    outdata << endl;    
    outdata.close();
    diff = 1000000*(end.tv_sec-start.tv_sec) + end.tv_usec-start.tv_usec;
    cout << "Convolution layer 3_1 out load time = " << diff << "  us" << endl;*/
    
    //pool 3
    gettimeofday(&start,0); 
    Fill_param(pci_bar_handle_4, CTRL_PARAMS, ctrl_param_2, 2);
    Fill_param(pci_bar_handle_4, ACC_PARAMS_1, acc_param_pool_3, 9); 
    gettimeofday(&end,0);
    diff = 1000000*(end.tv_sec-start.tv_sec) + end.tv_usec-start.tv_usec;
    cout << "Pooling layer 3 data load time = " << diff << "  us" << endl;
    //----------------------inference net ip status check -----------------------//    
    //ip_status = XInference_net_ReadReg(pci_bar_handle, InstancePtr.ctrl_bus_baseaddress, XINFERENCE_NET_CRTL_BUS_ADDR_AP_CTRL);
    //cout << "Status feedback from inference ip is : " << ip_status << endl;
    gettimeofday(&start,0);
    XInference_net_Start(pci_bar_handle, &InstancePtr);
    while (!XInference_net_IsDone(pci_bar_handle, &InstancePtr)) {
        count++;
    }
    gettimeofday(&end,0);
    diff = 1000000*(end.tv_sec-start.tv_sec) + end.tv_usec-start.tv_usec;
    cout << "Pooling layer 3 processing time = " << diff << "  us" << endl;
    /*gettimeofday(&start,0); 
    outdata.open("./netOutput/pool_temp_out_data.txt", ios::app);
    outdata <<"pool_output:"<< endl;
    //Read_Bram(pci_bar_handle_4, 0xE02000000, out_temp_full, 160);
    for(loop_var = 0; loop_var < 128; loop_var++){
        outdata << out_temp_1_1[loop_var] << "  ";
    //    out_temp_2[loop_var]=out_temp_full[loop_var];
    }
    outdata << endl;    
    outdata.close();
    gettimeofday(&end,0);
    diff = 1000000*(end.tv_sec-start.tv_sec) + end.tv_usec-start.tv_usec;
    cout << "Pooling layer 3_1 processing time = " << diff << "  us" << endl;*/

    //----------------------fc layer -----------------------//   
    gettimeofday(&start,0); 
    Fill_param(pci_bar_handle_4, CTRL_PARAMS, ctrl_param_1, 2);
    Fill_param(pci_bar_handle_4, ACC_PARAMS_0, acc_param_conv_4, 16); 
    gettimeofday(&end,0);
    diff = 1000000*(end.tv_sec-start.tv_sec) + end.tv_usec-start.tv_usec;
    cout << "fc layer 1_1 data load time = " << diff << "  us" << endl;
    //----------------------inference net ip status check -----------------------//    
    //fc 1-1
    //ip_status = XInference_net_ReadReg(pci_bar_handle, InstancePtr.ctrl_bus_baseaddress, XINFERENCE_NET_CRTL_BUS_ADDR_AP_CTRL);
    //cout << "Status feedback from inference ip is : " << ip_status << endl;
    gettimeofday(&start,0);
    XInference_net_Start(pci_bar_handle, &InstancePtr);
    while (!XInference_net_IsDone(pci_bar_handle, &InstancePtr)) {
        count++;
    }
    gettimeofday(&end,0);
    diff = 1000000*(end.tv_sec-start.tv_sec) + end.tv_usec-start.tv_usec;
    cout << "fc layer 1_1 processing time = " << diff << "  us" << endl;
    //cout << "IP is done at " << count << " attempts" << endl; 
    gettimeofday(&start,0); 
    //Read_Bram(pci_bar_handle_4, BUF_OUT_1, out_temp_full, 8);
    Read_Bram(pci_bar_handle_4, BUF_OUT_1_0, out_temp_full, 1);
    Read_Bram(pci_bar_handle_4, BUF_OUT_1_1, &out_temp_full[1], 1);
    Read_Bram(pci_bar_handle_4, BUF_OUT_1_2, &out_temp_full[2], 1);
    Read_Bram(pci_bar_handle_4, BUF_OUT_1_3, &out_temp_full[3], 1);
    Read_Bram(pci_bar_handle_4, BUF_OUT_1_4, &out_temp_full[4], 1);
    Read_Bram(pci_bar_handle_4, BUF_OUT_1_5, &out_temp_full[5], 1);
    Read_Bram(pci_bar_handle_4, BUF_OUT_1_6, &out_temp_full[6], 1);
    Read_Bram(pci_bar_handle_4, BUF_OUT_1_7, &out_temp_full[7], 1);

    Read_Bram(pci_bar_handle_4, BUF_OUT_1_0+4, &out_temp_full[8], 1);
    Read_Bram(pci_bar_handle_4, BUF_OUT_1_1+4, &out_temp_full[9], 1);
    /*outdata.open("./netOutput/out_temp_data.txt", ios::app);
    outdata <<"conv_output:"<< endl;
    for(loop_var = 0; loop_var < 10; loop_var++){
        outdata << out_temp_full[loop_var] << "  ";
        out_temp_1[loop_var]=out_temp_full[loop_var];
    }
    outdata << endl;    
    outdata.close();*/
    gettimeofday(&end,0);
    diff = 1000000*(end.tv_sec-start.tv_sec) + end.tv_usec-start.tv_usec;
    cout << "fc layer 1_1 out load time = " << diff << "  us" << endl;
    for (loop_var = 0; loop_var < 10; loop_var++ ) {
        fc_3_out[loop_var]=(float)(out_temp_full[loop_var]);
    }
    softmax(fc_3_out, 10);
    predict(fc_3_out, 10);
    //gettimeofday(&end,0);
    //diff = 1000000*(end.tv_sec-start.tv_sec) + end.tv_usec-start.tv_usec;
    //cout << "Cifarnet processing time = " << diff << "  us" << endl;

//------------------------------------------------------------------------------------------
    printf("\n");
    printf("Reading and verifying DDR_B Dst Buffer 1KB\n");

    /*for ( loop_var = 0; loop_var < 256; loop_var++ ) {
        rc_4 = fpga_pci_peek(pci_bar_handle_4, (DDR_B_ADDR + loop_var*4), &value);
       fail_on(rc_4, out, "Unable to read read from the fpga !");
       //printf("register: 0x%x\n", value);
       if (value != loop_var)
        {
          printf("Data mismatch!");
        }
    }
    printf("\n");
    printf("CDMA Transfer Successful!\n");*/


out:
    /* clean up */
    if (pci_bar_handle >= 0) {
        rc = fpga_pci_detach(pci_bar_handle);
        if (rc) {
            printf("Failure while detaching from the fpga.\n");
        }
    }

    if (pci_bar_handle_0 >= 0) {
        rc_0 = fpga_pci_detach(pci_bar_handle_0);
        if (rc_0) {
            printf("Failure while detaching from the fpga.\n");
        }
    }

    if (pci_bar_handle_4 >= 0) {
        rc_4 = fpga_pci_detach(pci_bar_handle_4);
        if (rc_4) {
            printf("Failure while detaching from the fpga.\n");
        }
    }


    if (pci_bar_handle_sda >= 0) {
        rc_sda = fpga_pci_detach(pci_bar_handle_sda);
        if (rc_sda) {
            printf("Failure while detaching from the fpga.\n");
        }
    }

    /* if there is an error code, exit with status 1 */
    return (rc != 0 ? 1 : 0);
}