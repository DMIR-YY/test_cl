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

using namespace std;

typedef struct {
    uint32_t ctrl_bus_baseaddress;
    uint32_t IsReady;
} XInference_net;

/*
 * pci_vendor_id and pci_device_id values below are Amazon's and avaliable to use for a given FPGA slot. 
 * Users may replace these with their own if allocated to them by PCI SIG
 */
static uint16_t pci_vendor_id = 0x1D0F; /* Amazon PCI Vendor ID */
static uint16_t pci_device_id = 0xF000; /* PCI Device ID preassigned by Amazon for F1 applications */

#define HELLO_WORLD_REG_ADDR UINT64_C(0x0)

#define DDR_SH_ADDR UINT64_C(0xE01000000)
#define DDR_B_ADDR UINT64_C(0xD02000000)
#define DDR_A_ADDR UINT64_C(0xC02000000)

#define HELLO_WORLD_REG_ADDR_CONTROL UINT64_C(0x00)
#define HELLO_WORLD_REG_ADDR_STATUS UINT64_C(0x04)

#define HELLO_WORLD_REG_ADDR_SRC_MSB_ADDR UINT64_C(0x1C)
#define HELLO_WORLD_REG_ADDR_DST_MSB_ADDR UINT64_C(0x24)

#define HELLO_WORLD_REG_ADDR_SRC_ADDR UINT64_C(0x18)
#define HELLO_WORLD_REG_ADDR_DST_ADDR UINT64_C(0x20)
#define HELLO_WORLD_REG_ADDR_BYTES UINT64_C(0x28)

 // M_AXI_BAR1 connected to inference control port
#define XINFERENCE_IP_CRTL_BUS_ADDR_1 UINT64_C(0x010000)
#define XINFERENCE_IP_CRTL_BUS_ADDR_2 UINT64_C(0x020000)

#define XINFERENCE_NET_CRTL_BUS_ADDR_AP_CTRL UINT64_C(0x0)
#define XINFERENCE_NET_CRTL_BUS_ADDR_GIE UINT64_C(0x4)
#define XINFERENCE_NET_CRTL_BUS_ADDR_IER UINT64_C(0x8)
#define XINFERENCE_NET_CRTL_BUS_ADDR_ISR UINT64_C(0xc)

/* Declaring the local functions */
int peek_poke_example(int slot, int pf_id, int bar_id);
int vled_example(int slot);

/* Declating auxilary house keeping functions */
int initialize_log(char* log_name);
int check_afi_ready(int slot);

void XInference_net_WriteReg(pci_bar_handle_t pci_bar, uint64_t BaseAddress, uint64_t RegOffset, uint32_t Data);
uint32_t XInference_net_ReadReg(pci_bar_handle_t pci_bar, uint64_t BaseAddress, uint64_t RegOffset);
int XInference_net_Initialize(pci_bar_handle_t pci_bar, XInference_net *InstancePtr, const char* InstanceName);
int XInference_net_Release(pci_bar_handle_t pci_bar, XInference_net *InstancePtr);

void XInference_net_Start(pci_bar_handle_t pci_bar, XInference_net *InstancePtr);
uint32_t XInference_net_IsDone(pci_bar_handle_t pci_bar, XInference_net *InstancePtr);
uint32_t XInference_net_IsIdle(pci_bar_handle_t pci_bar, XInference_net *InstancePtr);
uint32_t XInference_net_IsReady(pci_bar_handle_t pci_bar, XInference_net *InstancePtr);

void Fill_Bram(pci_bar_handle_t pci_bar, uint64_t BRAM_ADDRSS, float *data, int length);
void Fill_param(pci_bar_handle_t pci_bar, uint64_t BRAM_ADDRESS, int *data, int length);
void Read_Bram(pci_bar_handle_t pci_bar, uint64_t BRAM_ADDRSS, float *data, int length);
void Read_param(pci_bar_handle_t pci_bar, uint64_t BRAM_ADDRSS, int *data, int length);

/*
 * check if the corresponding AFI for hello_world is loaded
 */
int check_afi_ready(int slot_id) {
    struct fpga_mgmt_image_info info = {0}; 
    int rc;

    /* get local image description, contains status, vendor id, and device id. */
    rc = fpga_mgmt_describe_local_image(slot_id, &info,0);
    fail_on(rc, out, "Unable to get AFI information from slot %d. Are you running as root?",slot_id);

    /* check to see if the slot is ready */
    if (info.status != FPGA_STATUS_LOADED) {
        rc = 1;
        fail_on(rc, out, "AFI in Slot %d is not in READY state !", slot_id);
    }

    printf("AFI PCI  Vendor ID: 0x%x, Device ID 0x%x\n",
        info.spec.map[FPGA_APP_PF].vendor_id,
        info.spec.map[FPGA_APP_PF].device_id);

    /* confirm that the AFI that we expect is in fact loaded */
    if (info.spec.map[FPGA_APP_PF].vendor_id != pci_vendor_id ||
        info.spec.map[FPGA_APP_PF].device_id != pci_device_id) {
        printf("AFI does not show expected PCI vendor id and device ID. If the AFI "
               "was just loaded, it might need a rescan. Rescanning now.\n");

        rc = fpga_pci_rescan_slot_app_pfs(slot_id);
        fail_on(rc, out, "Unable to update PF for slot %d",slot_id);
        /* get local image description, contains status, vendor id, and device id. */
        rc = fpga_mgmt_describe_local_image(slot_id, &info,0);
        fail_on(rc, out, "Unable to get AFI information from slot %d",slot_id);

        printf("AFI PCI  Vendor ID: 0x%x, Device ID 0x%x\n",
            info.spec.map[FPGA_APP_PF].vendor_id,
            info.spec.map[FPGA_APP_PF].device_id);

        /* confirm that the AFI that we expect is in fact loaded after rescan */
        if (info.spec.map[FPGA_APP_PF].vendor_id != pci_vendor_id ||
             info.spec.map[FPGA_APP_PF].device_id != pci_device_id) {
            rc = 1;
            fail_on(rc, out, "The PCI vendor id and device of the loaded AFI are not "
                             "the expected values.");
        }
    }
    
    return rc;

out:
    return 1;
}

void XInference_net_WriteReg(pci_bar_handle_t pci_bar, uint64_t BaseAddress, uint64_t RegOffset, uint32_t Data) {
    int rc;
    rc = fpga_pci_poke(pci_bar, BaseAddress + RegOffset, Data);
    fail_on(rc, out, "Unable to read from IP !");  
out:
    //cout << "Function fail!!" << endl;
;
}

uint32_t XInference_net_ReadReg(pci_bar_handle_t pci_bar, uint64_t BaseAddress, uint64_t RegOffset) {
    uint32_t data;
    int rc;
    rc = fpga_pci_peek(pci_bar, (BaseAddress + RegOffset), &data);
    fail_on(rc, out, "Unable to read from the BRAM !");
    return data;
out:
    return 0;
}

// int XInference_net_Initialize(XInference_net *InstancePtr, const char* InstanceName);
// int XInference_net_Release(XInference_net *InstancePtr);
void XInference_net_Start(pci_bar_handle_t pci_bar, XInference_net *InstancePtr) {
    uint32_t data;
    data = XInference_net_ReadReg(pci_bar, InstancePtr->ctrl_bus_baseaddress, XINFERENCE_NET_CRTL_BUS_ADDR_AP_CTRL) & 0x80;
    XInference_net_WriteReg(pci_bar, InstancePtr->ctrl_bus_baseaddress, XINFERENCE_NET_CRTL_BUS_ADDR_AP_CTRL, data | 0x01);
}

uint32_t XInference_net_IsDone(pci_bar_handle_t pci_bar, XInference_net *InstancePtr){
    uint32_t data;
    data = XInference_net_ReadReg(pci_bar, InstancePtr->ctrl_bus_baseaddress, XINFERENCE_NET_CRTL_BUS_ADDR_AP_CTRL);
    return (data >> 1) & 0x01;
}

uint32_t XInference_net_IsIdle(pci_bar_handle_t pci_bar, XInference_net *InstancePtr) {
    uint32_t data;
    data = XInference_net_ReadReg(pci_bar, InstancePtr->ctrl_bus_baseaddress, XINFERENCE_NET_CRTL_BUS_ADDR_AP_CTRL);
    return (data >> 2) & 0x01;
}

uint32_t XInference_net_IsReady(pci_bar_handle_t pci_bar, XInference_net *InstancePtr) {
    uint32_t data;
    data = XInference_net_ReadReg(pci_bar, InstancePtr->ctrl_bus_baseaddress, XINFERENCE_NET_CRTL_BUS_ADDR_AP_CTRL);
    return !(data & 0x1);
}

void Fill_param(pci_bar_handle_t pci_bar, uint64_t BRAM_ADDRESS, int *data, int length) {
    int rc_4, loop_var;
    // cout << "Loading data to BRAM, location = " << pci_bar << "  BRAM_ADDRSS = " << BRAM_ADDRESS << endl;
    //printf("Loading data to BRAM, location = 0x%d, BRAM_ADDRESS = 0x%x \n", pci_bar, BRAM_ADDRESS);
    for ( loop_var = 0; loop_var < length; loop_var++ ) {
       rc_4 = fpga_pci_poke(pci_bar, (BRAM_ADDRESS + loop_var*4), *(uint32_t*)&data[loop_var]);
    //    fail_on(rc_4, out_fill, "Unable to write to BRAM !");  
    }    
    //cout << "Loaded data to BRAM !!!" << endl;
// out_fill:
        // cout << "failed writing" << endl;
}

void Fill_Bram(pci_bar_handle_t pci_bar, uint64_t BRAM_ADDRESS, float *data, int length) {
    int rc_4, loop_var;
    // cout << "Loading data to BRAM, location = " << pci_bar << "  BRAM_ADDRSS = " << BRAM_ADDRESS << endl;
    //printf("Loading data to BRAM, location = 0x%d, BRAM_ADDRESS = 0x%x \n", pci_bar, BRAM_ADDRESS);
    for ( loop_var = 0; loop_var < length; loop_var++ ) {
       rc_4 = fpga_pci_poke(pci_bar, (BRAM_ADDRESS + loop_var*4), *(uint32_t*)&data[loop_var]);
    //    fail_on(rc_4, out_fill, "Unable to write to BRAM !");  
    }    
    //cout << "Loaded data to BRAM !!!" << endl;
// out_fill:
        // cout << "failed writing" << endl;
}

void Read_Bram(pci_bar_handle_t pci_bar, uint64_t BRAM_ADDRESS, float *data, int length) {
    int rc_4, loop_var;
    // cout << "Reading BRAM data, location = " << pci_bar << "  BRAM_ADDRESS = " << BRAM_ADDRESS << endl;
    //printf("Reading BRAM data, location = 0x%d, BRAM_ADDRESS = 0x%x \n", pci_bar, BRAM_ADDRESS);
    for ( loop_var = 0; loop_var < length; loop_var++ ) {
        rc_4 = fpga_pci_peek(pci_bar, (BRAM_ADDRESS + loop_var*4), (uint32_t*)&data[loop_var]);
        // fail_on(rc_4, out_read, "Unable to read from the BRAM !");
    } 
    //cout << "Finished reading BRAM data!!!" << endl;
// out_read:
        // cout << "failed reading" << endl;
}

void Read_param(pci_bar_handle_t pci_bar, uint64_t BRAM_ADDRESS, int *data, int length) {
    int rc_4, loop_var;
    // cout << "Reading BRAM data, location = " << pci_bar << "  BRAM_ADDRESS = " << BRAM_ADDRESS << endl;
    //printf("Reading BRAM data, location = 0x%d, BRAM_ADDRESS = 0x%x \n", pci_bar, BRAM_ADDRESS);
    for ( loop_var = 0; loop_var < length; loop_var++ ) {
        rc_4 = fpga_pci_peek(pci_bar, (BRAM_ADDRESS + loop_var*4), (uint32_t*)&data[loop_var]);
        // fail_on(rc_4, out_read, "Unable to read from the BRAM !");
    } 
    //cout << "Finished reading BRAM data!!!" << endl;
// out_read:
        // cout << "failed reading" << endl;
}

void set_cdma(pci_bar_handle_t pci_bar_handle,uint32_t src_value_1,uint32_t src_value_2,uint32_t dst_value_1,uint32_t dst_value_2,uint32_t bytes_value){
    int rc;
    //printf("Setting Up CDMA Transfers with USR-AXI CDMA AXI4 Lite Registers\n");
    
    rc = fpga_pci_poke(pci_bar_handle, HELLO_WORLD_REG_ADDR_SRC_ADDR, src_value_1);//shujulaiyuan di 32
    //fail_on(rc, out, "Unable to write to the fpga !");  

    rc = fpga_pci_poke(pci_bar_handle, HELLO_WORLD_REG_ADDR_SRC_MSB_ADDR, src_value_2);//gao 32
    //fail_on(rc, out, "Unable to write to the fpga !");  
    
    rc = fpga_pci_poke(pci_bar_handle, HELLO_WORLD_REG_ADDR_DST_ADDR, dst_value_1);//di 32
    //fail_on(rc, out, "Unable to write to the fpga !");

    rc = fpga_pci_poke(pci_bar_handle, HELLO_WORLD_REG_ADDR_DST_MSB_ADDR, dst_value_2);//gao 32
    //fail_on(rc, out, "Unable to write to the fpga !");  
    
    //value = 0x00007168;//400 mingling
    rc = fpga_pci_poke(pci_bar_handle, HELLO_WORLD_REG_ADDR_BYTES, bytes_value);//shujuchangdu
    //fail_on(rc, out, "Unable to write to the fpga !");
    //printf("Executing CDMA Transfers on DDR4_SH and polling status register\n");
    //printf("\n");

    rc = fpga_pci_peek(pci_bar_handle, HELLO_WORLD_REG_ADDR_STATUS, &bytes_value);
    //fail_on(rc, out, "Unable to read read from the fpga !");
    
    while(bytes_value != 0x00001002)//1002 zhixingwanle
    {
    rc = fpga_pci_peek(pci_bar_handle, HELLO_WORLD_REG_ADDR_STATUS, &bytes_value);
    //fail_on(rc, out, "Unable to read read from the fpga !");
    }
    //printf("CDMA Transfer Complete!\n");
    //printf("AXI CDMA Status Register Value: 0x%x\n", bytes_value);
}

const unsigned char * loadfile(const std::string &file, int &size) {
   std::ifstream fs(file.c_str(), std::ios::binary);
   fs.seekg(0, std::ios::end);
   size = fs.tellg();
   char * data = new char[size];
   fs.seekg(0);
   fs.read(data, sizeof(char) * size);
   fs.close();
   return (unsigned char *)data;
}

void w_buf_t_load(float buf[][Tm][WBUF_t*WBUF_t], float *layer_weights, int weight_offset, int K, int N, int M, int w_r_offset, int w_c_offset){
        for (int m = 0; m < M; m += Tm) {
            for (int n = 0; n < N; n += Tn) {
                if(w_c_offset + 2*K > WBUF_t){
                    w_c_offset = 0;
                    w_r_offset += K;
                }else{
                    w_c_offset += K*(n/Tn);
                    w_r_offset += K*(m/Tm);
                }
                for(int k1 = 0; k1 < K; k1++){
                    for(int k2 = 0; k2 < K; k2++){
                        for(int j = 0; j < Tn && j < N; j++){
                            for(int i = 0; i < Tm && i < M; i++){
                                buf[j][i][WBUF_t*(k1+w_r_offset) + (k2+w_c_offset)] = *(layer_weights + weight_offset + (i+m)*N*K*K + (j+n)*K*K + k1*K + k2);
                           }
                        }
                    }
                }
            }
        }
    }

void acc_call_1(pci_bar_handle_t pci_bar_handle,XInference_net InstancePtr_1,XInference_net InstancePtr_2,
    int Tm_times,int Tn_times,int Tr_times,int Tc_times,
    uint32_t bytes_value_in,uint32_t bytes_value_weight,uint32_t bytes_value_out){
    //3 ports
    for(int r=0;r<Tr_times;r++){
        for(int c=0;c<Tc_times;c++){
            for(int m=0;m<Tm_times;m++){
                for(int n=0;n<Tn_times;n++){
                    //load weight_data
                    for (int loop_var_1 = 0; loop_var_1 < 3; loop_var_1++) {
                        for (int loop_var_2 = 0; loop_var_2 < 32; loop_var_2++) {
                            set_cdma(pci_bar_handle,0x01000000+loop_var_1*100000+loop_var_2*1000+bytes_value_weight*n*(Tm_times>1?m:1),0x0000000E,0x02000000+loop_var_1*20000+loop_var_2*1000,0x00000000,0x00000800);
                        }
                    }
                    //set_cdma(pci_bar_handle,0x01000000+bytes_value_weight*n*(Tm_times>1?m:1),0x0000000E,0x02000000,0x00000000,bytes_value_weight);
                    //set_cdma(pci_bar_handle,0x01100000+bytes_value_weight*n*(Tm_times>1?m:1),0x0000000E,0xC4010000,0x00000000,bytes_value_weight);
                    //set_cdma(pci_bar_handle,0x01200000+bytes_value_weight*n*(Tm_times>1?m:1),0x0000000E,0xC4020000,0x00000000,bytes_value_weight);
                    //load input_data
                    set_cdma(pci_bar_handle,0x02000000,0x0000000C,0x00000000,0x00000000,bytes_value_in);
                    set_cdma(pci_bar_handle,0x02000000+0x00000640,0x0000000C,0x00010000,0x00000000,bytes_value_in);
                    set_cdma(pci_bar_handle,0x02000000+0x00000C80,0x0000000C,0x00020000,0x00000000,bytes_value_in);
                    //start conv acc
                    XInference_net_Start(pci_bar_handle, &InstancePtr_1);
                    while (!XInference_net_IsDone(pci_bar_handle, &InstancePtr_1)) {
                        
                    }
                }
                //start pool acc
                XInference_net_Start(pci_bar_handle, &InstancePtr_2);
                while (!XInference_net_IsDone(pci_bar_handle, &InstancePtr_2)) {
                        
                }
                //trans output
                for (int loop_var_1 = 0; loop_var_1 < 32; loop_var_1++) {
                    set_cdma(pci_bar_handle,0x00040000+loop_var_1*1000,0x00000000,0xD0200000+loop_var_1*4000+bytes_value_out*(Tm_times>1?m:1)*(c+Tc_times*r),0x00000000,bytes_value_out);
                }
                /*set_cdma(pci_bar_handle,0xC6000000,0x00000000,0xD0200000+bytes_value_out*m,0x00000000,bytes_value_out);
                set_cdma(pci_bar_handle,0xC6040000,0x00000000,0xD0210000+bytes_value_out*m,0x00000000,bytes_value_out);
                set_cdma(pci_bar_handle,0xC6080000,0x00000000,0xD0220000+bytes_value_out*m,0x00000000,bytes_value_out);
                set_cdma(pci_bar_handle,0xC60C0000,0x00000000,0xD0230000+bytes_value_out*m,0x00000000,bytes_value_out);
                set_cdma(pci_bar_handle,0xC6100000,0x00000000,0xD0240000+bytes_value_out*m,0x00000000,bytes_value_out);
                set_cdma(pci_bar_handle,0xC6140000,0x00000000,0xD0250000+bytes_value_out*m,0x00000000,bytes_value_out);
                set_cdma(pci_bar_handle,0xC6180000,0x00000000,0xD0260000+bytes_value_out*m,0x00000000,bytes_value_out);
                set_cdma(pci_bar_handle,0xC61C0000,0x00000000,0xD0270000+bytes_value_out*m,0x00000000,bytes_value_out);*/
            }
        }
    }
}

void acc_call_2(pci_bar_handle_t pci_bar_handle,XInference_net InstancePtr,
    int Tm_times,int Tn_times,int Tr_times,int Tc_times,
    uint32_t bytes_value_in_offset,uint32_t bytes_value_in,uint32_t bytes_value_weight,uint32_t bytes_value_out){
    int flag=0;
    //8 ports
    for(int r=0;r<Tr_times;r++){
        for(int c=0;c<Tc_times;c++){
            for(int m=0;m<Tm_times;m++){
                for(int n=0;n<Tn_times;n++){
                    //load weight_data
                    for (int loop_var_1 = 0; loop_var_1 < 8; loop_var_1++) {
                        for (int loop_var_2 = 0; loop_var_2 < 32; loop_var_2++) {
                            set_cdma(pci_bar_handle,0x01000000+loop_var_1*100000+loop_var_2*1000+bytes_value_weight*n*(Tm_times>1?m:1),0x0000000E,0x02000000+loop_var_1*20000+loop_var_2*1000,0x00000000,0x00000800);
                        }
                    }
                    //set_cdma(pci_bar_handle,0x01000000+bytes_value_weight*n*(Tm_times>1?m:1),0x0000000E,0xC4000000,0x00000000,bytes_value_weight);
                    //set_cdma(pci_bar_handle,0x01100000+bytes_value_weight*n*(Tm_times>1?m:1),0x0000000E,0xC4010000,0x00000000,bytes_value_weight);
                    //set_cdma(pci_bar_handle,0x01200000+bytes_value_weight*n*(Tm_times>1?m:1),0x0000000E,0xC4020000,0x00000000,bytes_value_weight);
                    //set_cdma(pci_bar_handle,0x01300000+bytes_value_weight*n*(Tm_times>1?m:1),0x0000000E,0xC4030000,0x00000000,bytes_value_weight);
                    //set_cdma(pci_bar_handle,0x01400000+bytes_value_weight*n*(Tm_times>1?m:1),0x0000000E,0xC4040000,0x00000000,bytes_value_weight);
                    //set_cdma(pci_bar_handle,0x01500000+bytes_value_weight*n*(Tm_times>1?m:1),0x0000000E,0xC4050000,0x00000000,bytes_value_weight);
                    //set_cdma(pci_bar_handle,0x01600000+bytes_value_weight*n*(Tm_times>1?m:1),0x0000000E,0xC4060000,0x00000000,bytes_value_weight);
                    //set_cdma(pci_bar_handle,0x01700000+bytes_value_weight*n*(Tm_times>1?m:1),0x0000000E,0xC4070000,0x00000000,bytes_value_weight);
                    //load input_data
                    if(flag==0){
                        set_cdma(pci_bar_handle,0xD0200000+bytes_value_in_offset*c+bytes_value_in_offset*Tc_times*r+bytes_value_in_offset*Tc_times*Tr_times*n,0x0000000C,0x02000000,0x00000000,bytes_value_in);
                        set_cdma(pci_bar_handle,0xD0210000+bytes_value_in_offset*c+bytes_value_in_offset*Tc_times*r+bytes_value_in_offset*Tc_times*Tr_times*n,0x0000000C,0x02010000,0x00000000,bytes_value_in);
                        set_cdma(pci_bar_handle,0xD0220000+bytes_value_in_offset*c+bytes_value_in_offset*Tc_times*r+bytes_value_in_offset*Tc_times*Tr_times*n,0x0000000C,0x02020000,0x00000000,bytes_value_in);
                        set_cdma(pci_bar_handle,0xD0230000+bytes_value_in_offset*c+bytes_value_in_offset*Tc_times*r+bytes_value_in_offset*Tc_times*Tr_times*n,0x0000000C,0x02030000,0x00000000,bytes_value_in);
                        set_cdma(pci_bar_handle,0xD0240000+bytes_value_in_offset*c+bytes_value_in_offset*Tc_times*r+bytes_value_in_offset*Tc_times*Tr_times*n,0x0000000C,0x02040000,0x00000000,bytes_value_in);
                        set_cdma(pci_bar_handle,0xD0250000+bytes_value_in_offset*c+bytes_value_in_offset*Tc_times*r+bytes_value_in_offset*Tc_times*Tr_times*n,0x0000000C,0x02050000,0x00000000,bytes_value_in);
                        set_cdma(pci_bar_handle,0xD0260000+bytes_value_in_offset*c+bytes_value_in_offset*Tc_times*r+bytes_value_in_offset*Tc_times*Tr_times*n,0x0000000C,0x02060000,0x00000000,bytes_value_in);
                        set_cdma(pci_bar_handle,0xD0270000+bytes_value_in_offset*c+bytes_value_in_offset*Tc_times*r+bytes_value_in_offset*Tc_times*Tr_times*n,0x0000000C,0x02070000,0x00000000,bytes_value_in);
                        flag=1;
                    }else{
                        set_cdma(pci_bar_handle,0xC2000000+bytes_value_in_offset*c+bytes_value_in_offset*Tc_times*r+bytes_value_in_offset*Tc_times*Tr_times*n,0x0000000C,0x02000000,0x00000000,bytes_value_in);
                        set_cdma(pci_bar_handle,0xC2100000+bytes_value_in_offset*c+bytes_value_in_offset*Tc_times*r+bytes_value_in_offset*Tc_times*Tr_times*n,0x0000000C,0x02010000,0x00000000,bytes_value_in);
                        set_cdma(pci_bar_handle,0xC2200000+bytes_value_in_offset*c+bytes_value_in_offset*Tc_times*r+bytes_value_in_offset*Tc_times*Tr_times*n,0x0000000C,0x02020000,0x00000000,bytes_value_in);
                        set_cdma(pci_bar_handle,0xC2300000+bytes_value_in_offset*c+bytes_value_in_offset*Tc_times*r+bytes_value_in_offset*Tc_times*Tr_times*n,0x0000000C,0x02030000,0x00000000,bytes_value_in);
                        set_cdma(pci_bar_handle,0xC2400000+bytes_value_in_offset*c+bytes_value_in_offset*Tc_times*r+bytes_value_in_offset*Tc_times*Tr_times*n,0x0000000C,0x02040000,0x00000000,bytes_value_in);
                        set_cdma(pci_bar_handle,0xC2500000+bytes_value_in_offset*c+bytes_value_in_offset*Tc_times*r+bytes_value_in_offset*Tc_times*Tr_times*n,0x0000000C,0x02050000,0x00000000,bytes_value_in);
                        set_cdma(pci_bar_handle,0xC2600000+bytes_value_in_offset*c+bytes_value_in_offset*Tc_times*r+bytes_value_in_offset*Tc_times*Tr_times*n,0x0000000C,0x02060000,0x00000000,bytes_value_in);
                        set_cdma(pci_bar_handle,0xC2700000+bytes_value_in_offset*c+bytes_value_in_offset*Tc_times*r+bytes_value_in_offset*Tc_times*Tr_times*n,0x0000000C,0x02070000,0x00000000,bytes_value_in);
                        flag=0;
                    }
                    //start acc
                    XInference_net_Start(pci_bar_handle, &InstancePtr);
                    while (!XInference_net_IsDone(pci_bar_handle, &InstancePtr)) {
                        
                    }
                }
                //trans output
                if(flag==0){
                    set_cdma(pci_bar_handle,0xC6000000,0x00000000,0xC2000000+bytes_value_out*m,0x00000000,bytes_value_out);
                    set_cdma(pci_bar_handle,0xC6040000,0x00000000,0xC2100000+bytes_value_out*m,0x00000000,bytes_value_out);
                    set_cdma(pci_bar_handle,0xC6080000,0x00000000,0xC2200000+bytes_value_out*m,0x00000000,bytes_value_out);
                    set_cdma(pci_bar_handle,0xC60C0000,0x00000000,0xC2300000+bytes_value_out*m,0x00000000,bytes_value_out);
                    set_cdma(pci_bar_handle,0xC6100000,0x00000000,0xC2400000+bytes_value_out*m,0x00000000,bytes_value_out);
                    set_cdma(pci_bar_handle,0xC6140000,0x00000000,0xC2500000+bytes_value_out*m,0x00000000,bytes_value_out);
                    set_cdma(pci_bar_handle,0xC6180000,0x00000000,0xC2600000+bytes_value_out*m,0x00000000,bytes_value_out);
                    set_cdma(pci_bar_handle,0xC61C0000,0x00000000,0xC2700000+bytes_value_out*m,0x00000000,bytes_value_out);
                    flag=1;
                }else{
                    set_cdma(pci_bar_handle,0xC6000000,0x00000000,0xD0200000+bytes_value_out*m,0x00000000,bytes_value_out);
                    set_cdma(pci_bar_handle,0xC6040000,0x00000000,0xD0210000+bytes_value_out*m,0x00000000,bytes_value_out);
                    set_cdma(pci_bar_handle,0xC6080000,0x00000000,0xD0220000+bytes_value_out*m,0x00000000,bytes_value_out);
                    set_cdma(pci_bar_handle,0xC60C0000,0x00000000,0xD0230000+bytes_value_out*m,0x00000000,bytes_value_out);
                    set_cdma(pci_bar_handle,0xC6100000,0x00000000,0xD0240000+bytes_value_out*m,0x00000000,bytes_value_out);
                    set_cdma(pci_bar_handle,0xC6140000,0x00000000,0xD0250000+bytes_value_out*m,0x00000000,bytes_value_out);
                    set_cdma(pci_bar_handle,0xC6180000,0x00000000,0xD0260000+bytes_value_out*m,0x00000000,bytes_value_out);
                    set_cdma(pci_bar_handle,0xC61C0000,0x00000000,0xD0270000+bytes_value_out*m,0x00000000,bytes_value_out);
                    flag=0;
                }
            }
        }
    }
}

void acc_call_2_pool(pci_bar_handle_t pci_bar_handle,pci_bar_handle_t pci_bar_handle_4,XInference_net InstancePtr,
    uint64_t CTRL_PARAMS,uint64_t ACC_PARAMS,int *ctrl_param,int *acc_param_pool,
    int Tm_times,int Tn_times,int Tr_times,int Tc_times,
    uint32_t bytes_value_in_offset,uint32_t bytes_value_in,uint32_t bytes_value_weight,uint32_t bytes_value_out){
    int flag=0;
    //8 ports
    for(int r=0;r<Tr_times;r++){
        for(int c=0;c<Tc_times;c++){
            for(int m=0;m<Tm_times;m++){
                for(int n=0;n<Tn_times;n++){
                    //load input_data
                    if(flag==0){
                        set_cdma(pci_bar_handle,0xD0200000+bytes_value_in_offset*c+bytes_value_in_offset*Tc_times*r+bytes_value_in_offset*Tc_times*Tr_times*n,0x0000000C,0x02000000,0x00000000,bytes_value_in);
                        set_cdma(pci_bar_handle,0xD0210000+bytes_value_in_offset*c+bytes_value_in_offset*Tc_times*r+bytes_value_in_offset*Tc_times*Tr_times*n,0x0000000C,0x02010000,0x00000000,bytes_value_in);
                        set_cdma(pci_bar_handle,0xD0220000+bytes_value_in_offset*c+bytes_value_in_offset*Tc_times*r+bytes_value_in_offset*Tc_times*Tr_times*n,0x0000000C,0x02020000,0x00000000,bytes_value_in);
                        set_cdma(pci_bar_handle,0xD0230000+bytes_value_in_offset*c+bytes_value_in_offset*Tc_times*r+bytes_value_in_offset*Tc_times*Tr_times*n,0x0000000C,0x02030000,0x00000000,bytes_value_in);
                        set_cdma(pci_bar_handle,0xD0240000+bytes_value_in_offset*c+bytes_value_in_offset*Tc_times*r+bytes_value_in_offset*Tc_times*Tr_times*n,0x0000000C,0x02040000,0x00000000,bytes_value_in);
                        set_cdma(pci_bar_handle,0xD0250000+bytes_value_in_offset*c+bytes_value_in_offset*Tc_times*r+bytes_value_in_offset*Tc_times*Tr_times*n,0x0000000C,0x02050000,0x00000000,bytes_value_in);
                        set_cdma(pci_bar_handle,0xD0260000+bytes_value_in_offset*c+bytes_value_in_offset*Tc_times*r+bytes_value_in_offset*Tc_times*Tr_times*n,0x0000000C,0x02060000,0x00000000,bytes_value_in);
                        set_cdma(pci_bar_handle,0xD0270000+bytes_value_in_offset*c+bytes_value_in_offset*Tc_times*r+bytes_value_in_offset*Tc_times*Tr_times*n,0x0000000C,0x02070000,0x00000000,bytes_value_in);
                        flag=1;
                    }else{
                        set_cdma(pci_bar_handle,0xC2000000+bytes_value_in_offset*c+bytes_value_in_offset*Tc_times*r+bytes_value_in_offset*Tc_times*Tr_times*n,0x0000000C,0x02000000,0x00000000,bytes_value_in);
                        set_cdma(pci_bar_handle,0xC2100000+bytes_value_in_offset*c+bytes_value_in_offset*Tc_times*r+bytes_value_in_offset*Tc_times*Tr_times*n,0x0000000C,0x02010000,0x00000000,bytes_value_in);
                        set_cdma(pci_bar_handle,0xC2200000+bytes_value_in_offset*c+bytes_value_in_offset*Tc_times*r+bytes_value_in_offset*Tc_times*Tr_times*n,0x0000000C,0x02020000,0x00000000,bytes_value_in);
                        set_cdma(pci_bar_handle,0xC2300000+bytes_value_in_offset*c+bytes_value_in_offset*Tc_times*r+bytes_value_in_offset*Tc_times*Tr_times*n,0x0000000C,0x02030000,0x00000000,bytes_value_in);
                        set_cdma(pci_bar_handle,0xC2400000+bytes_value_in_offset*c+bytes_value_in_offset*Tc_times*r+bytes_value_in_offset*Tc_times*Tr_times*n,0x0000000C,0x02040000,0x00000000,bytes_value_in);
                        set_cdma(pci_bar_handle,0xC2500000+bytes_value_in_offset*c+bytes_value_in_offset*Tc_times*r+bytes_value_in_offset*Tc_times*Tr_times*n,0x0000000C,0x02050000,0x00000000,bytes_value_in);
                        set_cdma(pci_bar_handle,0xC2600000+bytes_value_in_offset*c+bytes_value_in_offset*Tc_times*r+bytes_value_in_offset*Tc_times*Tr_times*n,0x0000000C,0x02060000,0x00000000,bytes_value_in);
                        set_cdma(pci_bar_handle,0xC2700000+bytes_value_in_offset*c+bytes_value_in_offset*Tc_times*r+bytes_value_in_offset*Tc_times*Tr_times*n,0x0000000C,0x02070000,0x00000000,bytes_value_in);
                        flag=0;
                    }
                    //start conv acc
                    XInference_net_Start(pci_bar_handle, &InstancePtr);
                    while (!XInference_net_IsDone(pci_bar_handle, &InstancePtr)) {
                        
                    }
                }
                //load pool params
                Fill_param(pci_bar_handle_4, CTRL_PARAMS, ctrl_param, 2);
                Fill_param(pci_bar_handle_4, ACC_PARAMS, acc_param_pool, 9); 
                //start pool acc
                XInference_net_Start(pci_bar_handle, &InstancePtr);
                while (!XInference_net_IsDone(pci_bar_handle, &InstancePtr)) {
                    
                }
                //trans output
                if(flag==0){
                    set_cdma(pci_bar_handle,0xC6000000,0x00000000,0xC2000000+bytes_value_out*m,0x00000000,bytes_value_out);
                    set_cdma(pci_bar_handle,0xC6040000,0x00000000,0xC2100000+bytes_value_out*m,0x00000000,bytes_value_out);
                    set_cdma(pci_bar_handle,0xC6080000,0x00000000,0xC2200000+bytes_value_out*m,0x00000000,bytes_value_out);
                    set_cdma(pci_bar_handle,0xC60C0000,0x00000000,0xC2300000+bytes_value_out*m,0x00000000,bytes_value_out);
                    set_cdma(pci_bar_handle,0xC6100000,0x00000000,0xC2400000+bytes_value_out*m,0x00000000,bytes_value_out);
                    set_cdma(pci_bar_handle,0xC6140000,0x00000000,0xC2500000+bytes_value_out*m,0x00000000,bytes_value_out);
                    set_cdma(pci_bar_handle,0xC6180000,0x00000000,0xC2600000+bytes_value_out*m,0x00000000,bytes_value_out);
                    set_cdma(pci_bar_handle,0xC61C0000,0x00000000,0xC2700000+bytes_value_out*m,0x00000000,bytes_value_out);
                    flag=1;
                }else{
                    set_cdma(pci_bar_handle,0xC6000000,0x00000000,0xD0200000+bytes_value_out*m,0x00000000,bytes_value_out);
                    set_cdma(pci_bar_handle,0xC6040000,0x00000000,0xD0210000+bytes_value_out*m,0x00000000,bytes_value_out);
                    set_cdma(pci_bar_handle,0xC6080000,0x00000000,0xD0220000+bytes_value_out*m,0x00000000,bytes_value_out);
                    set_cdma(pci_bar_handle,0xC60C0000,0x00000000,0xD0230000+bytes_value_out*m,0x00000000,bytes_value_out);
                    set_cdma(pci_bar_handle,0xC6100000,0x00000000,0xD0240000+bytes_value_out*m,0x00000000,bytes_value_out);
                    set_cdma(pci_bar_handle,0xC6140000,0x00000000,0xD0250000+bytes_value_out*m,0x00000000,bytes_value_out);
                    set_cdma(pci_bar_handle,0xC6180000,0x00000000,0xD0260000+bytes_value_out*m,0x00000000,bytes_value_out);
                    set_cdma(pci_bar_handle,0xC61C0000,0x00000000,0xD0270000+bytes_value_out*m,0x00000000,bytes_value_out);
                    flag=0;
                }
            }
        }
    }
}