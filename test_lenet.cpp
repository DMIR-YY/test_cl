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
#include "./inference_net/acc_bdport_config.h"
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

    data_type in_data[32*32];
    data_type in_data_tmp[128*32];
    data_type in_data_1[128*32];
    data_type in_data_2[32*32];
    data_type in_data_part1_3D[3][20][20];
    data_type in_data_part2_3D[3][20][20];
    data_type in_data_part3_3D[3][20][20];
    data_type in_data_part4_3D[3][20][20];
    data_type in_data_part1[3*20*20];
    data_type in_data_part2[3*20*20];
    data_type in_data_part3[3*20*20];
    data_type in_data_part4[3*20*20];
    data_type  out_data[28*28];
    data_type  out_temp_1[4704];
    data_type  out_temp_2[4704];
    data_type  out_temp_3[4704];
    data_type  out_temp_4[4704];
    data_type  out_temp_5[4704];
    data_type  out_temp_6[4704];
    data_type  out_temp_7[4704];
    data_type  out_temp_8[4704];
    data_type  out_temp_9[4704];
    data_type  out_temp_10[4704];
    data_type  out_temp_11[4704];
    data_type  out_temp_12[4704];
    data_type  out_temp_13[4704];
    data_type  out_temp_14[4704];
    data_type  out_temp_15[4704];
    data_type  out_temp_16[4704];
    data_type  out_temp_tmp_0[128*32];
    data_type  out_temp_tmp_1[128*32];
    data_type  out_temp_tmp_2[128*32];
    data_type  out_temp_tmp_3[128*32];
    data_type  out_temp_tmp_4[128*32];
    data_type  out_temp_tmp_5[128*32];
    data_type  out_temp_tmp_6[128*32];
    data_type  out_temp_tmp_7[128*32];
    data_type  out_temp_tmp_8[128*32];
    data_type  fc_3_out[10];
    data_type out_res[6*28*28];
    data_type_w conv_weight[6*5*5+2400+4000];
    data_type_w conv_weight_1[6*5*5+2400+4000];
    data_type_w conv_bias[6+16+10];
    data_type_w conv_bias_1[6+16+10];
    float conv_1_weight2D[150];
    float conv_1_bias2D[6];
    float conv_2_weight2D[2400];
    float conv_2_bias2D[16];
    float fc_1_weight2D[4000];
    float fc_1_bias2D[10];
    float conv_weight_tmp[8][32][1024];

    data_type_w weight_temp_1[1][6][5][5];
    data_type_w conv_weight_mem_port_0_0[500];
    data_type_w conv_weight_mem_port_0_1[500];
    data_type_w conv_weight_mem_port_0_2[500];
    data_type_w conv_weight_mem_port_0_3[500];
    data_type_w conv_weight_mem_port_0_4[500];
    data_type_w conv_weight_mem_port_0_5[500];
    
    int ctrl_param_1[2] = {1, 0};
    int ctrl_param_2[2] = {0, 1};

    int conv_param_1[16] = {1, 5, 6, 28, 28, 28, 28, 1, 2, 1, 0, 0, 0, 0, 1, 1};
    int pool_param_1[16] = {28, 28, 6, 2, 14, 14, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0};
    int conv_param_2[16] = {6, 5, 16, 14, 14, 10, 10, 1, 0, 1, 0, 0, 0, 0, 1, 1};
    int pool_param_2[16] = {10, 10, 16, 2, 5, 5, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0};
    int conv_param_3[16] = {16, 5, 10, 5, 5, 1, 1, 5, 0, 1, 0, 0, 0, 0, 1, 1};
    int pool_param_3[16] = {10, 10, 10, 2, 1, 1, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0};

    int acc_param_conv_1[16] = {1/*S*/, 0/*n*/, 0/*r*/, 0/*c*/, 5/*K*/, 0, 0, 1/*N*/, 1, 0, 0, 0, 0, 0, 1, 0};
    int acc_param_test[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    int acc_param_conv_2[16] = {1/*S*/, 0/*n*/, 0/*r*/, 0/*c*/, 5/*K*/, 0, 0, 6/*N*/, 1, 0, 5, 0, 0, 0, 1, 0};
    int acc_param_conv_3[16] = {5/*S*/, 0/*n*/, 0/*r*/, 0/*c*/, 5/*K*/, 0, 0, 16/*N*/, 1, 0, 10, 0, 0, 0, 1, 0};
                             //C_in,R_in,N,K,C_out,R_out,stride,pad,act
    int acc_param_pool_1[16] = {2/*S*/, 0/*n*/, 0/*r*/, 0/*c*/, 2/*K*/, 28/*in_size*/, 28/*in_size*/, 0/*P*/, 16, 0, 0, 0, 0, 0, 0, 0};
    int acc_param_pool_2[16] = {2/*S*/, 0/*n*/, 0/*r*/, 0/*c*/, 2/*K*/, 10/*in_size*/, 10/*in_size*/, 0/*P*/, 16, 0, 0, 0, 0, 0, 0, 0};
    int acc_param_pool_3[16] = {0/*S*/, 0/*n*/, 0/*r*/, 0/*c*/, 0/*K*/, 1/*in_size*/, 1/*in_size*/, 0/*P*/, 0, 0, 0, 0, 0, 0, 0, 0};

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

    string image_dir = "./netInput/3.bmp";
    const char* weight_src = "./netInput/net_weights.txt";
    std::ofstream indata;
    std::ofstream outdata;
    std::ofstream weightdata;
    std::ofstream test_output;
    float in_data_3D_padding[3][32][32] = { 0 };
    int in_data_size=0;
    int w_r_offset = 0;
    int w_c_offset = 0;
    std::ofstream w_buf_t;

    //int i,j,k;
    int count = 0;

    cout << "test point 0" << endl;
    //time mreasurement variable define
    struct timeval start,end;
    struct timeval start_net,end_net;
    unsigned long diff;
    XInference_net InstancePtr;
    InstancePtr.ctrl_bus_baseaddress = XINFERENCE_IP_CRTL_BUS_ADDR_1;
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

    //Fill_param(pci_bar_handle_4, CTRL_PARAMS, ctrl_param_1, 2); 
//--------------------------input image data initialization----------------//
    data = loadfile(image_dir, size);
    image_orig = stbi_load_from_memory(data, size, &w, &h, &channels, 1);
    for (loop_var = 0; loop_var < 28*28; loop_var++) {
        in_data[loop_var] = (data_type)image_orig[loop_var];
    }
    //add padding for input
    for (loop_var = 0; loop_var < 1; loop_var++) {
        for (loop_var_1 = 0; loop_var_1 < 32; loop_var_1++) {
            for (loop_var_2 = 0; loop_var_2 < 32; loop_var_2++) {
                in_data_3D_padding[loop_var][loop_var_1][loop_var_2] = 0;
            }
        }
    }
    for (loop_var = 0; loop_var < 1; loop_var++) {
        for (loop_var_1 = 0; loop_var_1 < 28; loop_var_1++) {
            for (loop_var_2 = 0; loop_var_2 < 28; loop_var_2++) {
                in_data_3D_padding[loop_var][loop_var_1+2][loop_var_2+2] = (data_type)in_data[loop_var*28*28+loop_var_1*28+loop_var_2];
            }
        }
    }
    in_data_size=0;
    for (loop_var = 0; loop_var < 1; loop_var++) {
        for (loop_var_1 = 0; loop_var_1 < 32; loop_var_1++) {
            for (loop_var_2 = 0; loop_var_2 < 32; loop_var_2++) {
                in_data[in_data_size] = (data_type)in_data_3D_padding[loop_var][loop_var_1][loop_var_2];
                in_data_size++;
            }
        }
    }
    
    for (loop_var = 0; loop_var < 32; loop_var++) {
        for (loop_var_1 = 0; loop_var_1 < 32; loop_var_1++) {
            in_data_tmp[loop_var*128+loop_var_1] = in_data[loop_var*32+loop_var_1];
        }
    }
    /*indata.open("./netOutput/in_data.txt", ios::app);
    for (loop_var = 0; loop_var < 1; loop_var++) {
        indata <<"indata:"<< endl;
        for (loop_var_1 = 0; loop_var_1 < 32; loop_var_1++) {
            for (loop_var_2 = 0; loop_var_2 < 32; loop_var_2++) {
                indata << in_data[loop_var_1*32+loop_var_2] << " ";
            }
            indata << endl;
        }
        indata << endl;
    }
    indata << endl;
    indata.close();
    cout << endl;*/
//----------------------input weight data initialization ------------------//
    // Prepare weights and bias for conv layer 1
    memset(conv_1_weight2D, 0, 150 * sizeof(float));
    load_weight_conv(
        weight_src, 
        conv_1_weight2D,
        weight_bias_record,
        nn_channel_size_conv, 
        nn_in_number_conv,
        nn_out_number_conv,
        in_number_conv);
    //cout << "Loading conv weight 1 to memory space, starting at: " <<conv_weight_num << '\n';
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 5; j++) {
            for (int k = 0; k < 5; k++) {
                conv_weight[conv_weight_num] = (data_type_w)conv_1_weight2D[i*5*5+j*5+k];
                conv_weight_num++;
            }
        }
    }
    memset(conv_1_bias2D, 0, 6 * sizeof(float));
    load_bias_conv(
        weight_src, 
        conv_1_bias2D,
        weight_bias_record,
        nn_channel_size_conv, 
        nn_in_number_conv,
        nn_out_number_conv,
        in_number_conv);
    //cout << "Loading conv bias 1 to memory space, starting at: " <<conv_bias_num << '\n';
    for (int i = 0; i < 6; i++) {
        conv_bias[conv_bias_num] = (data_type_w)conv_1_bias2D[i];
        conv_bias_num++;
    }
    in_number_conv++;

    // Prepare weights and bias for conv layer 2
    memset(conv_2_weight2D, 0, 2400 * sizeof(float));
    load_weight_conv(
        weight_src, 
        conv_2_weight2D,
        weight_bias_record,
        nn_channel_size_conv, 
        nn_in_number_conv,
        nn_out_number_conv,
        in_number_conv);
    //cout << "Loading conv weight 2 to memory space, starting at: " <<conv_weight_num << '\n';
    for (int i = 0; i < 2400; i++) {
        conv_weight[conv_weight_num] = (data_type_w)conv_2_weight2D[i];
        conv_weight_num++;
    }
    memset(conv_2_bias2D, 0, 16 * sizeof(float));
    load_bias_conv(
        weight_src, 
        conv_2_bias2D,
        weight_bias_record,
        nn_channel_size_conv, 
        nn_in_number_conv,
        nn_out_number_conv,
        in_number_conv);
    //cout << "Loading conv bias 2 to memory space, starting at: " <<conv_bias_num << '\n';
    for (int i = 0; i < 16; i++) {
        conv_bias[conv_bias_num] = (data_type_w)conv_2_bias2D[i];
        conv_bias_num++;
    }
    in_number_conv++;

    //cout<<"Finished loading conv weight into memory! Total: " <<conv_weight_num  << "... ... ..."<<endl;
    //cout<<"Finished loading conv bias into memory! Total: " <<conv_bias_num  << "... ... ..."<<endl;

    // Prepare weights and bias for fc layer 1
    memset(fc_1_weight2D, 0, 4000 * sizeof(float));
    load_weight_fc(
        weight_src, 
        fc_1_weight2D,
        weight_bias_record,
        nn_channel_size_fc, 
        nn_in_number_fc,
        nn_out_number_fc,
        in_number_fc);
    //cout << "Loading fc weight 1 to memory space, starting at: " <<conv_weight_num << '\n';
    for (int i = 0; i < 4000; i++) {
        conv_weight[conv_weight_num] = (data_type_w)fc_1_weight2D[i];
        conv_weight_num++;
    }
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
    for (int i = 0; i < 6; i++) {
        for(int j = 0; j < 1; j++){
            for(int k1 = 0; k1 < 5; k1++){
                for(int k2 = 0; k2 < 5; k2++){
                    weight_temp_1[j][i][k1][k2] = (data_type_w)conv_1_weight2D[i*3*5*5 + j*5*5 + k1*5 + k2];
                }
            }
        }
    }
    for(int k1=0; k1 <5; k1++){
        for(int k2=0; k2<5; k2++){
            conv_weight_mem_port_0_0[k1*5 + k2] = weight_temp_1[0][0][k1][k2];
            conv_weight_mem_port_0_1[k1*5 + k2] = weight_temp_1[0][1][k1][k2];
            conv_weight_mem_port_0_2[k1*5 + k2] = weight_temp_1[0][2][k1][k2];
            conv_weight_mem_port_0_3[k1*5 + k2] = weight_temp_1[0][3][k1][k2];
            conv_weight_mem_port_0_4[k1*5 + k2] = weight_temp_1[0][4][k1][k2];
            conv_weight_mem_port_0_5[k1*5 + k2] = weight_temp_1[0][5][k1][k2];
        }
    }

//---------------------conv weight bram ------------------------------------//
    //conv_1_w_load
    w_buf_t_load(conv_weight_tmp, conv_1_weight2D, conv_param_1[10], conv_param_1[1], conv_param_1[0], conv_param_1[2], w_r_offset, w_c_offset);
    //conv_2_w_load
    w_c_offset = 5;
    acc_param_conv_2[10] = w_c_offset;
    w_buf_t_load(conv_weight_tmp, conv_2_weight2D, conv_param_2[10], conv_param_2[1], conv_param_2[0], conv_param_2[2], w_r_offset, w_c_offset);
    //fc_w_load
    w_c_offset = 10;
    acc_param_conv_3[10] = w_c_offset;
    w_buf_t_load(conv_weight_tmp, fc_1_weight2D, conv_param_3[10], conv_param_3[1], conv_param_3[0], conv_param_3[2], w_r_offset, w_c_offset);
    w_buf_t.open("netOutput/weight.txt", ios::app);
    for (int i = 0; i < 8; i++) {
      for (int j = 0; j < 32; j++) {
        w_buf_t <<"w_buf_data: "<<i<<"_"<<j<< endl;
        for(int k = 0; k < 32; k++){
          for(int l = 0; l < 20; l++){
            w_buf_t << conv_weight_tmp[i][j][k*32+l] << " ";
          }
          w_buf_t << endl;
        }
        w_buf_t << endl;
      }
      w_buf_t << endl;
    }
    w_buf_t.close();

    //write data to DDR_SH_ADDR
    //weight
    for(loop_var = 0; loop_var < 8; loop_var++){
        for(loop_var_1 = 0; loop_var_1 < 32; loop_var_1++){
            Fill_Bram(pci_bar_handle_4, 0xE01000000+0x00100000*loop_var+0x00001000*loop_var_1, conv_weight_tmp[loop_var][loop_var_1], 32*5);
        }  
    }
    //in_data
    Fill_Bram(pci_bar_handle_4, DDR_A_ADDR_1, in_data_tmp, 128*32);
    //Fill_Bram(pci_bar_handle_4, DDR_SH_ADDR, conv_weight, 6*5*5+2400+4000);
    Fill_Bram(pci_bar_handle_4, DDR_B_ADDR, conv_bias, 6+16+10);
    //Fill_Bram(pci_bar_handle_4, DDR_A_ADDR, in_data, 32*32);
    
    printf("Finished writing to SH_DDR data\n");
    //nn_in_number_conv[in_number_conv]*nn_out_number_conv[in_number_conv]*nn_channel_size_conv[in_number_conv]*nn_channel_size_conv[in_number_conv]
//----------------------conv bias bram -------------------------------------//
    //nn_out_number_conv[in_number_conv]
    //Fill_Bram(pci_bar_handle_4, CONV_B_BRAM_PCIS, conv_bias, 6+16+10);

//----------------------input data buffer load------------------------------//
    gettimeofday(&start_net,0);
    gettimeofday(&start,0);
    //weight
    //load all weight
    for(loop_var = 0; loop_var < 8; loop_var++){
        for(loop_var_1 = 0; loop_var_1 < 16; loop_var_1++){
            set_cdma(pci_bar_handle,0x01000000+loop_var*0x00100000+loop_var_1*0x00001000,0x0000000E,0x02000000+loop_var*0x00020000+loop_var_1*0x00001000,0x00000000,0x00000280);
        }  
    }
    //load conv1 weight
    /*for(loop_var = 0; loop_var < 1; loop_var++){
        for(loop_var_1 = 0; loop_var_1 < 6; loop_var_1++){
            set_cdma(pci_bar_handle,0x01000000+loop_var*0x00100000+loop_var_1*0x00001000,0x0000000E,0x02000000+loop_var*0x00020000+loop_var_1*0x00001000,0x00000000,0x00000280);
        }  
    }

    /*Read_Bram(pci_bar_handle_4, CONV_W_BRAM_PCIS_6_0, out_temp_1, 32*5);
    outdata.open("./netOutput/out_temp_data.txt", ios::app);
    outdata <<"conv_weight:"<< endl;
    for(loop_var = 0; loop_var < 5; loop_var++){
        for(loop_var_1 = 0; loop_var_1 < 32; loop_var_1++){
            outdata << out_temp_1[loop_var*32+loop_var_1] << "  ";
        }
        outdata << endl;   
    }
    outdata << endl;    
    outdata.close();*/
    //bias
    //Fill_Bram(pci_bar_handle_4, CONV_B_BRAM_PCIS, conv_bias, 6+16+10);
    /*Read_Bram(pci_bar_handle_4, CONV_B_BRAM_PCIS, conv_bias_1, 6+16+10);
    outdata.open("./netOutput/bias.txt", ios::app);
    outdata <<"conv_bias:"<< endl;
    for(loop_var = 0; loop_var < 32; loop_var++){
        outdata << conv_bias_1[loop_var] << "  ";  
    }
    outdata << endl;    
    outdata.close();*/
    //input
    set_cdma(pci_bar_handle,0x02000000,0x0000000C,0x04000000,0x00000000,0x00004000);
    //Fill_Bram(pci_bar_handle_4, BUF_OUT_0_0, in_data_tmp, 128*32);
    /*Read_Bram(pci_bar_handle_4, BUF_OUT_0_0, in_data_1, 128*32);
    outdata.open("./netOutput/in_data.txt", ios::app);
    outdata <<"in_data:"<< endl;
    for(loop_var = 0; loop_var < 32; loop_var++){
        for(loop_var_1 = 0; loop_var_1 < 128; loop_var_1++){
            outdata << in_data_1[loop_var*128+loop_var_1] << "  ";
        }
        outdata << endl;   
    }
    outdata << endl;    
    outdata.close();*/
    
//----------------------inference net ip status check -----------------------//    
    //1_1
    Fill_param(pci_bar_handle_4, CONV_CORE_PARAM, acc_param_conv_1, 16);
    Fill_param(pci_bar_handle_4, POOL_CORE_PARAM, acc_param_pool_1, 16); 
    /*Read_param(pci_bar_handle_4, POOL_CORE_PARAM, acc_param_test, 16);
    outdata.open("./netOutput/in_data.txt", ios::app);
    outdata <<"param:"<< endl;
    for(loop_var = 0; loop_var < 16; loop_var++){
        outdata << acc_param_test[loop_var] << "  ";  
    }
    outdata << endl;    
    outdata.close();*/
    gettimeofday(&end,0);
    diff = 1000000*(end.tv_sec-start.tv_sec) + end.tv_usec-start.tv_usec;
    cout << "conv1_1 data loading time = " << diff << "  us" << endl;

    gettimeofday(&start,0);
    XInference_net_Start(pci_bar_handle, &InstancePtr);
    //load conv2 weight
    /*for(loop_var_1 = 0; loop_var_1 < 10; loop_var_1++){
        set_cdma(pci_bar_handle,0x01006000+loop_var_1*0x00001000,0x0000000E,0x02006000+loop_var_1*0x00001000,0x00000000,0x00000280);
    }
    for(loop_var = 0; loop_var < 5; loop_var++){
        for(loop_var_1 = 0; loop_var_1 < 16; loop_var_1++){
            set_cdma(pci_bar_handle,0x01100000+loop_var*0x00100000+loop_var_1*0x00001000,0x0000000E,0x02020000+loop_var*0x00020000+loop_var_1*0x00001000,0x00000000,0x00000280);
        }  
    }*/
    while (!XInference_net_IsDone(pci_bar_handle, &InstancePtr)) {
        count++;
    }
    gettimeofday(&end,0);
    diff = 1000000*(end.tv_sec-start.tv_sec) + end.tv_usec-start.tv_usec;
    cout << "Convolution layer_1_1 processing time = " << diff << "  us" << endl;
    
    //1_2
    gettimeofday(&start,0);
    acc_param_conv_1[6]=16;
    Fill_param(pci_bar_handle_4, CONV_CORE_PARAM, acc_param_conv_1, 16);
    XInference_net_Start(pci_bar_handle, &InstancePtr);
    //load fc weight
    /*for(loop_var = 0; loop_var < 2; loop_var++){
        for(loop_var_1 = 0; loop_var_1 < 16; loop_var_1++){
            set_cdma(pci_bar_handle,0x01600000+loop_var*0x00100000+loop_var_1*0x00001000,0x0000000E,0x020C0000+loop_var*0x00020000+loop_var_1*0x00001000,0x00000000,0x00000280);
        }  
    }*/
    //load conv1_1 out_data
    for(loop_var = 0; loop_var < 6; loop_var++){
        for (loop_var_1 = 0; loop_var_1 < 8; loop_var_1++) {
            set_cdma(pci_bar_handle,0x00040000+loop_var*0x00001000+loop_var_1*0x00000080,0x00000000,0x04000000+loop_var*0x00010000+loop_var_1*0x00000200,0x00000000,0x00000020);
        }
    }
    while (!XInference_net_IsDone(pci_bar_handle, &InstancePtr)) {
        count++;
    }
    gettimeofday(&end,0);
    diff = 1000000*(end.tv_sec-start.tv_sec) + end.tv_usec-start.tv_usec;
    cout << "Convolution layer_1_2 processing time = " << diff << "  us" << endl;
    
    //1_3
    gettimeofday(&start,0);
    acc_param_conv_1[5]=16;
    acc_param_conv_1[6]=0;
    Fill_param(pci_bar_handle_4, CONV_CORE_PARAM, acc_param_conv_1, 16);
    XInference_net_Start(pci_bar_handle, &InstancePtr);
    //load conv1_2 out_data
    for(loop_var = 0; loop_var < 6; loop_var++){
        for (loop_var_1 = 0; loop_var_1 < 8; loop_var_1++) {
            set_cdma(pci_bar_handle,0x00040020+loop_var*0x00001000+loop_var_1*0x00000080,0x00000000,0x04000020+loop_var*0x00010000+loop_var_1*0x00000200,0x00000000,0x00000018);
        }
    }
    while (!XInference_net_IsDone(pci_bar_handle, &InstancePtr)) {
        count++;
    }
    gettimeofday(&end,0);
    diff = 1000000*(end.tv_sec-start.tv_sec) + end.tv_usec-start.tv_usec;
    cout << "Convolution layer_1_3 processing time = " << diff << "  us" << endl;
    
    //1_4
    gettimeofday(&start,0);
    acc_param_conv_1[5]=16;
    acc_param_conv_1[6]=16;
    Fill_param(pci_bar_handle_4, CONV_CORE_PARAM, acc_param_conv_1, 16); 
    XInference_net_Start(pci_bar_handle, &InstancePtr);
    //load conv1_3 out_data
    for(loop_var = 0; loop_var < 6; loop_var++){
        for (loop_var_1 = 0; loop_var_1 < 6; loop_var_1++) {
            set_cdma(pci_bar_handle,0x00040400+loop_var*0x00001000+loop_var_1*0x00000080,0x00000000,0x04001000+loop_var*0x00010000+loop_var_1*0x00000200,0x00000000,0x00000020);
        }
    }
    while (!XInference_net_IsDone(pci_bar_handle, &InstancePtr)) {
        count++;
    }
    gettimeofday(&end,0);
    diff = 1000000*(end.tv_sec-start.tv_sec) + end.tv_usec-start.tv_usec;
    cout << "Convolution layer_1_4 processing time = " << diff << "  us" << endl;

    //----------------------conv_2 layer -----------------------// 
    //loading data
    gettimeofday(&start,0);
    /*for(loop_var = 0; loop_var < 6; loop_var++){
        for (loop_var_1 = 0; loop_var_1 < 14; loop_var_1++) {
            set_cdma(pci_bar_handle,0x00040000+loop_var*0x00001000+loop_var_1*0x00000080,0x00000000,0x04000000+loop_var*0x00010000+loop_var_1*0x00000200,0x00000000,0x00000038);
        }
    }*/
    //load conv1_4 out_data
    for(loop_var = 0; loop_var < 6; loop_var++){
        for (loop_var_1 = 0; loop_var_1 < 6; loop_var_1++) {
            set_cdma(pci_bar_handle,0x00040420+loop_var*0x00001000+loop_var_1*0x00000080,0x00000000,0x04001020+loop_var*0x00010000+loop_var_1*0x00000200,0x00000000,0x00000018);
        }
    }    
    /*Read_Bram(pci_bar_handle_4, BUF_OUT_0_0, out_temp_7, 128*32);
    outdata.open("./netOutput/out_temp_data.txt", ios::app);
    outdata <<"conv_input:"<< endl;
    for(loop_var = 0; loop_var < 32; loop_var++){
        for(loop_var_1 = 0; loop_var_1 < 128; loop_var_1++){
            outdata << out_temp_7[loop_var*128+loop_var_1] << " ";
        }
        outdata << endl;   
    }
    outdata << endl;    
    outdata.close();*/
    Fill_param(pci_bar_handle_4, CONV_CORE_PARAM, acc_param_conv_2, 16);
    Fill_param(pci_bar_handle_4, POOL_CORE_PARAM, acc_param_pool_2, 16); 
    gettimeofday(&end,0);
    diff = 1000000*(end.tv_sec-start.tv_sec) + end.tv_usec-start.tv_usec;
    cout << "conv2 data loading time = " << diff << "  us" << endl;

    //set_cdma(pci_bar_handle,0xE02000000,0x0000000C,0xC4000000,0x00000000,0x00004980);
    //----------------------inference net ip status check -----------------------//    
    gettimeofday(&start,0);
    XInference_net_Start(pci_bar_handle, &InstancePtr);

    while (!XInference_net_IsDone(pci_bar_handle, &InstancePtr)) {
        count++;
    }
    gettimeofday(&end,0);
    diff = 1000000*(end.tv_sec-start.tv_sec) + end.tv_usec-start.tv_usec;
    cout << "Convolution layer2 processing time = " << diff << "  us" << endl;

    /*Read_Bram(pci_bar_handle_4, BUF_OUT_1_0, out_temp_1, 4704);
    outdata.open("./netOutput/out_temp_data.txt", ios::app);
    outdata <<"conv_output:"<< endl;
    for(int i = 0;i < 16;i++){
        for(int j = 0;j < 5;j++){
            for(int k = 0;k < 5;k++){
                outdata << out_temp_1[i*5*5+j*5+k] << " ";
            }
            outdata << endl;
        }
        outdata << endl;
    }
    outdata << endl;    
    outdata.close();*/

    //----------------------fc layer -----------------------//  
    //fc1_1
    //loading fc1_1 in_data
    gettimeofday(&start,0);
    for(loop_var = 0; loop_var < 8; loop_var++){
        for (loop_var_1 = 0; loop_var_1 < 5; loop_var_1++) {
            set_cdma(pci_bar_handle,0x00040000+loop_var*0x00001000+loop_var_1*0x00000080,0x00000000,0x04000000+loop_var*0x00010000+loop_var_1*0x00000200,0x00000000,0x00000014);
        }
    }
    Fill_param(pci_bar_handle_4, CONV_CORE_PARAM, acc_param_conv_3, 16);
    Fill_param(pci_bar_handle_4, POOL_CORE_PARAM, acc_param_pool_3, 16); 
    gettimeofday(&end,0);
    diff = 1000000*(end.tv_sec-start.tv_sec) + end.tv_usec-start.tv_usec;
    cout << "fc_1_1 data loading time = " << diff << "  us" << endl;
    
    /*Read_Bram(pci_bar_handle_4, BUF_OUT_0_0, out_temp_7, 128*32);
    outdata.open("./netOutput/out_temp_data.txt", ios::app);
    outdata <<"conv_input:"<< endl;
    for(loop_var = 0; loop_var < 32; loop_var++){
        for(loop_var_1 = 0; loop_var_1 < 128; loop_var_1++){
            outdata << out_temp_7[loop_var*128+loop_var_1] << " ";
        }
        outdata << endl;   
    }
    outdata << endl;    
    outdata.close();*/
    //set_cdma(pci_bar_handle,0xE02000000,0x0000000C,0xC4000000,0x00000000,0x00004980);
    //----------------------inference net ip status check -----------------------//    
    //ip_status = XInference_net_ReadReg(pci_bar_handle, InstancePtr.ctrl_bus_baseaddress, XINFERENCE_NET_CRTL_BUS_ADDR_AP_CTRL);
    gettimeofday(&start,0);
    XInference_net_Start(pci_bar_handle, &InstancePtr);
    //loading fc1_2 in_data
    for(loop_var = 0; loop_var < 8; loop_var++){
        for (loop_var_1 = 0; loop_var_1 < 5; loop_var_1++) {
            set_cdma(pci_bar_handle,0x00048000+loop_var*0x00001000+loop_var_1*0x00000080,0x00000000,0x04000100+loop_var*0x00010000+loop_var_1*0x00000200,0x00000000,0x00000014);
        }
    }
    while (!XInference_net_IsDone(pci_bar_handle, &InstancePtr)) {
        count++;
    }
    gettimeofday(&end,0);
    diff = 1000000*(end.tv_sec-start.tv_sec) + end.tv_usec-start.tv_usec;
    cout << "Fc layer1_1 processing time = " << diff << "  us" << endl;

    //fc1_2 
    //loading data
    gettimeofday(&start,0);
    /*for(loop_var = 0; loop_var < 8; loop_var++){
        for (loop_var_1 = 0; loop_var_1 < 5; loop_var_1++) {
            set_cdma(pci_bar_handle,0x00048000+loop_var*0x00001000+loop_var_1*0x00000080,0x00000000,0x04000000+loop_var*0x00010000+loop_var_1*0x00000200,0x00000000,0x00000014);
        }
    }*/
    acc_param_conv_3[1] = 8;//n
    acc_param_conv_3[10] = 15;//w_c_offset
    acc_param_conv_3[6] = 64;//c_offset
    Fill_param(pci_bar_handle_4, CONV_CORE_PARAM, acc_param_conv_3, 16);
    Fill_param(pci_bar_handle_4, POOL_CORE_PARAM, acc_param_pool_3, 16); 
    gettimeofday(&end,0);
    diff = 1000000*(end.tv_sec-start.tv_sec) + end.tv_usec-start.tv_usec;
    cout << "fc_1_2 data loading time = " << diff << "  us" << endl;

    /*Read_Bram(pci_bar_handle_4, BUF_OUT_0_0, out_temp_7, 128*5);
    outdata.open("./netOutput/out_temp_data.txt", ios::app);
    outdata <<"conv_input:"<< endl;
    for(loop_var = 0; loop_var < 5; loop_var++){
        for(loop_var_1 = 0; loop_var_1 < 128; loop_var_1++){
            outdata << out_temp_7[loop_var*128+loop_var_1] << " ";
        }
        outdata << endl;   
    }
    outdata << endl;    
    outdata.close();*/
    gettimeofday(&start,0);
    XInference_net_Start(pci_bar_handle, &InstancePtr);

    while (!XInference_net_IsDone(pci_bar_handle, &InstancePtr)) {
        count++;
    }
    gettimeofday(&end,0);
    diff = 1000000*(end.tv_sec-start.tv_sec) + end.tv_usec-start.tv_usec;
    cout << "Fc layer1_2 processing time = " << diff << "  us" << endl;
    //cout << "IP is done at " << count << " attempts" << endl; 
    Read_Bram(pci_bar_handle_4, BUF_OUT_1_0, out_temp_1, 1);
    Read_Bram(pci_bar_handle_4, BUF_OUT_1_1, out_temp_2, 1);
    Read_Bram(pci_bar_handle_4, BUF_OUT_1_2, out_temp_3, 1);
    Read_Bram(pci_bar_handle_4, BUF_OUT_1_3, out_temp_4, 1);
    Read_Bram(pci_bar_handle_4, BUF_OUT_1_4, out_temp_5, 1);
    Read_Bram(pci_bar_handle_4, BUF_OUT_1_5, out_temp_6, 1);
    Read_Bram(pci_bar_handle_4, BUF_OUT_1_6, out_temp_7, 1);
    Read_Bram(pci_bar_handle_4, BUF_OUT_1_7, out_temp_8, 1);
    Read_Bram(pci_bar_handle_4, BUF_OUT_1_8, out_temp_9, 1);
    Read_Bram(pci_bar_handle_4, BUF_OUT_1_9, out_temp_10, 1);

    fc_3_out[0]=(float)(out_temp_1[0]);
    fc_3_out[1]=(float)(out_temp_2[0]);
    fc_3_out[2]=(float)(out_temp_3[0]);
    fc_3_out[3]=(float)(out_temp_4[0]);
    fc_3_out[4]=(float)(out_temp_5[0]);
    fc_3_out[5]=(float)(out_temp_6[0]);
    fc_3_out[6]=(float)(out_temp_7[0]);
    fc_3_out[7]=(float)(out_temp_8[0]);
    fc_3_out[8]=(float)(out_temp_9[0]);
    fc_3_out[9]=(float)(out_temp_10[0]);
    for (loop_var = 0; loop_var < 10; loop_var++ ) {
        cout << fc_3_out[loop_var] << "  ";
    }
    cout << endl;
    softmax(fc_3_out, 10);
    predict(fc_3_out, 10);
    gettimeofday(&end_net,0);
    diff = 1000000*(end_net.tv_sec-start_net.tv_sec) + end_net.tv_usec-start_net.tv_usec;
    cout << "Lenet processing time = " << diff << "  us" << endl;

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
