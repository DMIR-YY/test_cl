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
//#include <cstdlib>
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
#include "./inference_net/acc_config.h"

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

    data_type in_data[28*28];
    data_type  out_data[28*28];
    uint32_t   out_temp[28*28];
    data_type out_res[6*28*28];
    data_type_w conv_weight[6*5*5];
    data_type_w conv_bias[1024];
    int acc_param[16] = {1, 5, 1, 28, 28, 28, 28, 1, 2, 1, 0, 1024, 0, 0, 1, 0};
    int acc_param_test_0[16] = {0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    int acc_param_test_1[16] = {0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};


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
    ifstream ifs("./test_data.txt");
    string str;

    //int i,j,k;
    int count = 0;
    int num = 0;

    //time mreasurement variable define
    struct timeval start,end;
    unsigned long diff;
    XInference_net *InstancePtr;
    InstancePtr->ctrl_bus_baseaddress = XINFERENCE_IP_CRTL_BUS_ADDR;
    InstancePtr->IsReady = 0x01;
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


    printf("Checking DDR4_A/B/D and DDR4_C(SH) Calibration with SDA-AXI GPIO\n");

    rc_sda = fpga_pci_peek(pci_bar_handle_sda, HELLO_WORLD_REG_ADDR, &value);
    fail_on(rc_sda, out, "Unable to read read from the fpga !");

   while(value != 0x0000000F)
    {
    rc_sda = fpga_pci_peek(pci_bar_handle_sda, HELLO_WORLD_REG_ADDR, &value);
    fail_on(rc_sda, out, "Unable to read read from the fpga !");
    printf("register: 0x%x\n", value);
    }  
    printf("DDR4_A/B/D and DDR4_C(SH) Calibrated! GPIO Input Value: 0x%x\n", value);

    printf("\n");

    printf("Writing for VLED (0xAAAA) with OCL-AXI GPIO\n");

    value = 0x0000AAAA;
    rc_0 = fpga_pci_poke(pci_bar_handle_0, HELLO_WORLD_REG_ADDR, value);
    fail_on(rc_0, out, "Unable to write to the fpga !");


    printf("\n");

    printf("Writing to SH_DDR Source Buffer 1KB\n");

    for ( loop_var = 0; loop_var < 256; loop_var++ ) {
       rc_4 = fpga_pci_poke(pci_bar_handle_4, (DDR_SH_ADDR + loop_var*4), loop_var);
       fail_on(rc_4, out, "Unable to write to the fpga !"); 
            
    }
    printf("Finished writing to SH_DDR Source Buffer\n");

    printf("\n");
    printf("Setting Up CDMA Transfers with USR-AXI CDMA AXI4 Lite Registers\n");

    value = 0x01000000;
    rc = fpga_pci_poke(pci_bar_handle, HELLO_WORLD_REG_ADDR_SRC_ADDR, value);
    fail_on(rc, out, "Unable to write to the fpga !");  

    value = 0x0000000E;
    rc = fpga_pci_poke(pci_bar_handle, HELLO_WORLD_REG_ADDR_SRC_MSB_ADDR, value);
    fail_on(rc, out, "Unable to write to the fpga !");  
    
    value = 0x02000000;
//    value = 0xC0000000;
    rc = fpga_pci_poke(pci_bar_handle, HELLO_WORLD_REG_ADDR_DST_ADDR, value);
    fail_on(rc, out, "Unable to write to the fpga !");

    value = 0x0000000D;
//    value = 0x00000000;
    rc = fpga_pci_poke(pci_bar_handle, HELLO_WORLD_REG_ADDR_DST_MSB_ADDR, value);
    fail_on(rc, out, "Unable to write to the fpga !");  
    
    value = 0x00000400;
    rc = fpga_pci_poke(pci_bar_handle, HELLO_WORLD_REG_ADDR_BYTES, value);
    fail_on(rc, out, "Unable to write to the fpga !");
    printf("Executing CDMA Transfers on DDR4_SH and polling status register\n");
    printf("\n");

    rc = fpga_pci_peek(pci_bar_handle, HELLO_WORLD_REG_ADDR_STATUS, &value);
    fail_on(rc, out, "Unable to read read from the fpga !");
    
    while(value != 0x00001002)
    {
    rc = fpga_pci_peek(pci_bar_handle, HELLO_WORLD_REG_ADDR_STATUS, &value);
    fail_on(rc, out, "Unable to read read from the fpga !");
    }
    printf("CDMA Transfer Complete!\n");
    printf("AXI CDMA Status Register Value: 0x%x\n", value);


//--------------------------input image data initialization----------------//
    data = loadfile(image_dir, size);
    image_orig = stbi_load_from_memory(data, size, &w, &h, &channels, 1);
    for (int j = 0; j < 28*28; j++) {
        in_data[j] = (data_type)image_orig[j];
    }
    cout << endl;
    indata.open("./netOutput/in_data.txt", ios::app);
    for (int i = 0; i < acc_param[0]; i++) {
        for (int j = 0; j < acc_param[3]; j++) {
            for (int k = 0; k < acc_param[4]; k++) {
                indata << in_data[i*acc_param[3]*acc_param[4] + j*acc_param[4] + k] << " ";
            }
            indata << endl;
        }
        indata << endl;
    }
    indata << endl;
    indata.close();
    cout << endl;
//----------------------test bram -----------------------------------------//
/*
    cout << "Test bram data write and read" << endl;
    for ( loop_var = 0; loop_var < 28*28; loop_var++ ) {
//        rc_4 = fpga_pci_poke(pci_bar_handle_4, (TEST_BRAM_0_ADDR+loop_var*4), (uint32_t)in_data[loop_var]);
        rc_4 = fpga_pci_poke(pci_bar_handle_4, (TEST_BRAM_0_ADDR+loop_var*4), *(uint32_t*)&in_data[loop_var]);
        fail_on(rc_4, out, "Unable to write to BRAM !");  
    }    
    printf("finished writing to test BRAM!!! \n");
    for ( loop_var = 0; loop_var < 28*28; loop_var++ ) {
        rc_4 = fpga_pci_peek(pci_bar_handle_4, (TEST_BRAM_0_ADDR + loop_var*4), (uint32_t*)&out_data[loop_var]);
        fail_on(rc_4, out, "Unable to read from the BRAM !");
        if(*((float*)&out_data[loop_var]) != in_data[loop_var])
        {
          printf("Data mismatch! in_data[%d] = %f,  out_data[%d] = %f\n", loop_var,in_data[loop_var], loop_var, *(float*)&out_data[loop_var]);
        }
    }
    cout << "Finished test bram read and write check!!!" << endl;
//TODO: weight and bias data initialization
*/
//----------------------input weight data initialization ------------------//
    // Prepare weights and bias for conv layer 1
    //float *conv_1_weight2D = (float*)malloc(150 * sizeof(float));
    //memset(conv_1_weight2D, 0, 150 * sizeof(float));
    load_weight_conv(
        weight_src, 
        conv_weight,
        weight_bias_record,
        nn_channel_size_conv, 
        nn_in_number_conv,
        nn_out_number_conv,
        in_number_conv);
    cout << "Loading conv weight 1 to memory space, starting at: " <<conv_weight_num << '\n';
    weightdata.open("./netOutput/weight_data.txt", ios::app);
    weightdata <<"weight:"<< endl;
    for(int i = 0;i < 6;i++){
        for(int j = 0;j < 5;j++){
            for(int k = 0;k < 5;k++){
                weightdata << conv_weight[i*5*5+j*5+k] << " ";
            }
            weightdata << endl;
        }
        weightdata << endl;
    }
    weightdata << endl;

    //for (int i = 0; i < 150; i++) {
    //   conv_weight_mem_port[conv_weight_num] = (data_type_w)conv_weight[i];
    //   conv_weight_num++;
    //}
    //free(conv_1_weight2D);
    //float *conv_1_bias2D = (float*)malloc(6 * sizeof(float));
    //memset(conv_1_bias2D, 0, 6 * sizeof(float));

//----------------------input bias data initialization---------------------//
    /*load_bias_conv(
        weight_src, 
        conv_bias,
        weight_bias_record,
        nn_channel_size_conv, 
        nn_in_number_conv,
        nn_out_number_conv,
        in_number_conv);
    cout << "Loading conv bias 1 to memory space, starting at: " <<conv_bias_num << '\n';
    weightdata <<"bias:"<< endl;
    for(int i = 0;i < 6;i++){
        weightdata << conv_bias[i] << " ";
    }
    weightdata << endl;
    weightdata << endl;*/
    weightdata.close();
    //for (int i = 0; i < 6; i++) {
    //   conv_bias_mem_port[conv_bias_num] = (data_type_w)conv_bias[i];
    //   conv_bias_num++;
    //}
    //free(conv_1_bias2D);
    //in_number_conv++;


//---------------------conv parameter bram transmission---------------------// 
//    for(int i = 0; i < 16; i++){cout << acc_param[i] << "  ";}
//    cout << endl;
    Fill_param(pci_bar_handle_4, ACC_PARAMS_0, acc_param, 16); 
    Read_param(pci_bar_handle_4, ACC_PARAMS_0, acc_param_test_0, 16);
/*
    weightdata.open("params.txt", ios::app);
    for(int i = 0; i < 16; i++){weightdata << acc_param_test_0[i] << "  ";}
    weightdata << endl;
    weightdata.close();
*/ 
    Fill_param(pci_bar_handle_4, ACC_PARAMS_1, acc_param, 16); 
    Read_param(pci_bar_handle_4, ACC_PARAMS_1, acc_param_test_1, 16);
/*
    weightdata.open("params.txt", ios::app);
    for(int i = 0; i < 16; i++)
    {
        weightdata << acc_param_test_0[i] << "  ";
        weightdata << acc_param_test_1[i] << "  ";
    }
    weightdata << endl;
    weightdata.close();
*/
    cout << "Finished filling conv acc parameter into param bram!" << endl;

//---------------------conv weight bram ------------------------------------//
    //nn_in_number_conv[in_number_conv]*nn_out_number_conv[in_number_conv]*nn_channel_size_conv[in_number_conv]*nn_channel_size_conv[in_number_conv]
    Fill_Bram(pci_bar_handle_4, CONV_W_BRAM_PCIS, conv_weight, 6*5*5);
//    Read_Bram(pci_bar_handle_4, CONV_W_BRAM_PCIS, conv_weight, 6*5*5);
//    cout << "Finished conv weight bram read and write check!!!" << endl;
//----------------------conv bias bram -------------------------------------//
    //nn_out_number_conv[in_number_conv]
    while (ifs >> str) {
        float f = atof(str.c_str());
        conv_bias[num] = f;
        num++;
    }
    Fill_Bram(pci_bar_handle_4, CONV_B_BRAM_PCIS, conv_bias, 1024);
//    Fill_Bram(pci_bar_handle_4, CONV_B_BRAM_PCIS_0, conv_bias, 6);

    Read_Bram(pci_bar_handle_4, CONV_B_BRAM_PCIS, conv_bias, 1024);

    weightdata.open("./netOutput/params.txt", ios::app);
    for (int j=0; j< 1024; j++){
        weightdata << conv_bias[j] << " ";
    }
    weightdata << endl;
    weightdata.close();

//    cout << "Finished conv bias bram read and write check!!!" << endl;
//    Fill_Bram(pci_bar_handle_4, CONV_B_BRAM_PCIS_1, conv_bias, 6);
//    Read_Bram(pci_bar_handle_4, CONV_B_BRAM_PCIS_1, conv_bias, 6);
//    cout << "Finished conv bias bram read and write check!!!" << endl;
/*
//-----------------------fc weight bram -----------------------------------//
    //Fill_Bram(pci_bar_handle_4, FC_W_BRAM_PCIS, in_data, 28*28);
//    Read_Bram(pci_bar_handle_4, FC_W_BRAM_PCIS, out_data, 28*28);
    //cout << "Finished fc weight bram read and write check!!!" << endl;
//----------------------fc bias bram ---------------------------------------//
    //Fill_Bram(pci_bar_handle_4, FC_B_BRAM_PCIS, in_data, 28*28);
    //Read_Bram(pci_bar_handle_4, FC_B_BRAM_PCIS, out_data, 28*28);
    //cout << "Finished fc bias bram read and write check!!!" << endl;
*/
//----------------------input data buffer load------------------------------//

    Fill_Bram(pci_bar_handle_4, BUF_OUT_0, in_data, 28*28);

//    Read_Bram(pci_bar_handle_4, FC_B_BRAM_PCIS, out_res, 28*28);
//    cout << "Finished input data buffer load ......" << endl;
/*
    cout << "Test bram data write and read for OUT_BUF_1 " << endl;
    for ( loop_var = 0; loop_var < 8096; loop_var++ ) {
        rc_4 = fpga_pci_poke(pci_bar_handle_4, (BUF_OUT_1 + loop_var*4), (uint32_t)loop_var);
        fail_on(rc_4, out, "Unable to write to BRAM !");  
    }    
    cout << "Read test for OUT_BUF_1" << endl;
    for ( loop_var = 0; loop_var < 8096; loop_var++ ) {
        rc_4 = fpga_pci_peek(pci_bar_handle_4, (BUF_OUT_1 + loop_var*4), &value);
        fail_on(rc_4, out, "Unable to read from the BRAM !");
        if(value == loop_var)
        {
          printf("Data mismatch! value = %d,  loop_var = %d\n", value, loop_var);
        }
    }
    cout << "Finished OUT_BUF_1 read and write check!!!" << endl;
*/
 
//----------------------inference net ip status check -----------------------//    

    ip_status = XInference_net_ReadReg(pci_bar_handle, InstancePtr->ctrl_bus_baseaddress, XINFERENCE_NET_CRTL_BUS_ADDR_AP_CTRL);
    cout << "Status feedback from inference ip is : " << ip_status << endl;

    gettimeofday(&start,0);
    XInference_net_Start(pci_bar_handle, InstancePtr);

    while (!XInference_net_IsDone(pci_bar_handle, InstancePtr)) {
        count++;
    }
    gettimeofday(&end,0);
    diff = 1000000*(end.tv_sec-start.tv_sec) + end.tv_usec-start.tv_usec;
    cout << "Convolution layer processing time = " << diff << "  us" << endl;
    cout << "IP is done at " << count << " attempts" << endl; 

//---------------Read convolution results out from output_buffer_1------------//

//TODO: read the results data out for comparison -- single layer convolution    
    cout << "Read out convolutional results" << endl;
    Read_Bram(pci_bar_handle_4, BUF_OUT_1, out_data, 6*28*28);
    outdata.open("./netOutput/out_data.txt", ios::app);
    for(int i = 0;i < acc_param[2];i++){
        for(int j = 0;j < acc_param[5];j++){
            for(int k = 0;k < acc_param[6];k++){
                outdata << out_data[i*acc_param[5]*acc_param[6]+j*acc_param[6]+k] << "  ";
            }
            outdata << endl;
        }
        outdata << endl;
    }
    outdata << endl;    
    outdata.close();

//------------------------------------------------------------------------------------------
    printf("\n");
    printf("Reading and verifying DDR_B Dst Buffer 1KB\n");

    for ( loop_var = 0; loop_var < 256; loop_var++ ) {
        rc_4 = fpga_pci_peek(pci_bar_handle_4, (DDR_B_ADDR + loop_var*4), &value);
       fail_on(rc_4, out, "Unable to read read from the fpga !");
       //printf("register: 0x%x\n", value);
       if (value != loop_var)
    	{
          printf("Data mismatch!");
    	}
    }
    printf("\n");
    printf("CDMA Transfer Successful!\n");


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
