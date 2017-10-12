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

#include <fpga_pci.h>
#include <fpga_mgmt.h>
#include <utils/lcd.h>

/* Constants determined by the CL */
/* a set of register offsets; this CL has only one */
/* these register addresses should match the addresses in */
/* /aws-fpga/hdk/cl/examples/common/cl_common_defines.vh */

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


//BRAM PCIS address offset
// test bram
#define BRAM_BASE_ADDR UINT64_C(0xC0000000)
#define BRAM_MSB_ADDR UINT64_C(0xC0000FFF)
//bram_ctrl_9
#define CONV_W_BRAM_PCIS UINT64_C(0x00C0002000)
//bram_ctrl_10
#define CONV_B_BRAM_PCIS UINT64_C(0x00C0000000)
//bram_ctrl_11
#define FC_W_BRAM_PCIS UINT64_C(0x00C2000000)
//bram_ctrl_12
#define FC_B_BRAM_PCIS UINT64_C(0x00C4000000)
//bram_ctrl_13
#define FC_OUT UINT64_C(0x00C6000000)
//bram_ctrl_14
#define BUF_OUT_0 UINT64_C(0x00C8000000)
//bram_ctrl_15
#define BUF_OUT_1 UINT64_C(0x00CA000000)

// M_AXI_BAR1 connected to inference control port
#define XINFERENCE_IP_CRTL_BUS_ADDR UINT64_C(0x010000)

#define XINFERENCE_NET_CRTL_BUS_ADDR_AP_CTRL UINT64_C(0x0)
#define XINFERENCE_NET_CRTL_BUS_ADDR_GIE UINT64_C(0x4)
#define XINFERENCE_NET_CRTL_BUS_ADDR_IER UINT64_C(0x8)
#define XINFERENCE_NET_CRTL_BUS_ADDR_ISR UINT64_C(0xc)

typedef struct {
    uint32_t ctrl_bus_baseaddress;
    uint32_t IsReady;
} XInference_net;

//BRAM INFERENCE_IP address offset

using namespace std;

/*
 * pci_vendor_id and pci_device_id values below are Amazon's and avaliable to use for a given FPGA slot. 
 * Users may replace these with their own if allocated to them by PCI SIG
 */
static uint16_t pci_vendor_id = 0x1D0F; /* Amazon PCI Vendor ID */
static uint16_t pci_device_id = 0xF000; /* PCI Device ID preassigned by Amazon for F1 applications */


/* use the stdout logger for printing debug information  */
const struct logger *logger = &logger_stdout;

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

    float in_data[28*28];
    uint32_t out_data[28*28];

    ifstream ifs("input_3_28.txt");
    string str;

    int index = 0;
    int i, j;
    XInference_net *InstancePtr;
    InstancePtr->ctrl_bus_baseaddress = XINFERENCE_IP_CRTL_BUS_ADDR;
    InstancePtr->IsReady = 0x01;
    uint32_t ip_status;
    /* pci_bar_handle_t is a handler for an address space exposed by one PCI BAR on one of the PCI PFs of the FPGA */

    pci_bar_handle_t pci_bar_handle = PCI_BAR_HANDLE_INIT;
    pci_bar_handle_t pci_bar_handle_0 = PCI_BAR_HANDLE_INIT;
    pci_bar_handle_t pci_bar_handle_4 = PCI_BAR_HANDLE_INIT;

    pci_bar_handle_t pci_bar_handle_sda = PCI_BAR_HANDLE_INIT;

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


//--------------------------BRAM data initialization---------------------------------------
    if(!ifs) {
       printf("input data not found!!\n");
    }
    while (ifs >> str) {
        in_data[index] = uint32_t(atof(str.c_str()));
        index++;
    }
    ifs.close();

    for (i = 0; i < 28; i++) {
        for ( j = 0; j< 28; j++) {
            cout << in_data[i*28 + j] << "  ";
        }
        cout << endl;
    }

//----------------------test bram -----------------------------------------//
    cout << "Test bram data write and read" << endl;
    for ( loop_var = 0; loop_var < 28*28; loop_var++ ) {
       rc_4 = fpga_pci_poke(pci_bar_handle_4, (BRAM_BASE_ADDR+loop_var*4), in_data[loop_var]);
       fail_on(rc_4, out, "Unable to write to BRAM !");  
    }    
    printf("finished writing to test BRAM!!! \n");
    for ( loop_var = 0; loop_var < 28*28; loop_var++ ) {
        rc_4 = fpga_pci_peek(pci_bar_handle_4, (BRAM_BASE_ADDR + loop_var*4), &out_data[loop_var]);
        fail_on(rc_4, out, "Unable to read from the BRAM !");
        if(out_data[loop_var] != in_data[loop_var])
       {
          printf("Data mismatch! in_data[%d] = %d,  out_data[%d] = %d\n", loop_var, in_data[loop_var], loop_var, out_data[loop_var]);
        }
    }
    cout << "Finished test bram read and write check!!!" << endl;
    for (i = 0; i < 28; i++) {
        for ( j = 0; j< 28; j++) {
            out_data[i*28 + j] = 0;
        }
    }
//---------------------conv weight bram ------------------------------------//
    cout << endl;
    cout << "conv weight bram data write and read" << endl;
    for ( loop_var = 0; loop_var < 28*28; loop_var++ ) {
       rc_4 = fpga_pci_poke(pci_bar_handle_4, (CONV_W_BRAM_PCIS+loop_var*4), in_data[loop_var]);
       fail_on(rc_4, out, "Unable to write to BRAM !");  
    }    
    printf("finished writing to conv weight BRAM!!! \n");
    for ( loop_var = 0; loop_var < 28*28; loop_var++ ) {
        rc_4 = fpga_pci_peek(pci_bar_handle_4, (CONV_W_BRAM_PCIS + loop_var*4), &out_data[loop_var]);
        fail_on(rc_4, out, "Unable to read from the BRAM !");
        if(out_data[loop_var] != in_data[loop_var])
       {
          printf("Data mismatch! in_data[%d] = %d,  out_data[%d] = %d\n", loop_var, in_data[loop_var], loop_var, out_data[loop_var]);
        }
    }
    for (i = 0; i < 28; i++) {
        for ( j = 0; j< 28; j++) {
            out_data[i*28 + j] = 0;
        }
    }    
    cout << "Finished conv weight bram read and write check!!!" << endl;
//----------------------conv bias bram -------------------------------------//    
    cout << endl;
    cout << "conv bias bram data write and read" << endl;
    for ( loop_var = 0; loop_var < 28*28; loop_var++ ) {
       rc_4 = fpga_pci_poke(pci_bar_handle_4, (CONV_B_BRAM_PCIS+loop_var*4), in_data[loop_var]);
       fail_on(rc_4, out, "Unable to write to BRAM !");  
    }    
    printf("finished writing to test BRAM!!! \n");
    for ( loop_var = 0; loop_var < 28*28; loop_var++ ) {
        rc_4 = fpga_pci_peek(pci_bar_handle_4, (BRAM_BASE_ADDR + loop_var*4), &out_data[loop_var]);
        fail_on(rc_4, out, "Unable to read from the BRAM !");
        if(out_data[loop_var] != in_data[loop_var])
       {
          printf("Data mismatch! in_data[%d] = %d,  out_data[%d] = %d\n", loop_var, in_data[loop_var], loop_var, out_data[loop_var]);
        }
    }
    for (i = 0; i < 28; i++) {
        for ( j = 0; j< 28; j++) {
            out_data[i*28 + j] = 0;
        }
    }    
    cout << "Finished conv bias bram read and write check!!!" << endl;
//-----------------------fc weight bram -----------------------------------//
    cout << endl;
    cout << "fc weight bram data write and read" << endl;
    for ( loop_var = 0; loop_var < 28*28; loop_var++ ) {
       rc_4 = fpga_pci_poke(pci_bar_handle_4, (FC_W_BRAM_PCIS+loop_var*4), in_data[loop_var]);
       fail_on(rc_4, out, "Unable to write to BRAM !");  
    }    
    printf("finished writing to fc weight BRAM!!! \n");
    for ( loop_var = 0; loop_var < 28*28; loop_var++ ) {
        rc_4 = fpga_pci_peek(pci_bar_handle_4, (FC_W_BRAM_PCIS + loop_var*4), &out_data[loop_var]);
        fail_on(rc_4, out, "Unable to read from the BRAM !");
        if(out_data[loop_var] != in_data[loop_var])
       {
          printf("Data mismatch! in_data[%d] = %d,  out_data[%d] = %d\n", loop_var, in_data[loop_var], loop_var, out_data[loop_var]);
        }
    }
    for (i = 0; i < 28; i++) {
        for ( j = 0; j< 28; j++) {
            out_data[i*28 + j] = 0;
        }
    }    
    cout << "Finished fc weight bram read and write check!!!" << endl;
//----------------------fc bias bram ---------------------------------------//
    cout << endl;
    cout << "fc bias bram data write and read" << endl;
    for ( loop_var = 0; loop_var < 28*28; loop_var++ ) {
       rc_4 = fpga_pci_poke(pci_bar_handle_4, (FC_B_BRAM_PCIS+loop_var*4), in_data[loop_var]);
       fail_on(rc_4, out, "Unable to write to BRAM !");  
    }    
    printf("finished writing to fc bias BRAM!!! \n");
    for ( loop_var = 0; loop_var < 28*28; loop_var++ ) {
        rc_4 = fpga_pci_peek(pci_bar_handle_4, (FC_B_BRAM_PCIS + loop_var*4), &out_data[loop_var]);
        fail_on(rc_4, out, "Unable to read from the BRAM !");
        if(out_data[loop_var] != in_data[loop_var])
       {
          printf("Data mismatch! in_data[%d] = %d,  out_data[%d] = %d\n", loop_var, in_data[loop_var], loop_var, out_data[loop_var]);
        }
    }
    for (i = 0; i < 28; i++) {
        for ( j = 0; j< 28; j++) {
            cout << out_data[i*28 + j] << "  ";
        }
        cout << endl;
    }    
    cout << "Finished fc bias bram read and write check!!!" << endl;

//----------------------inference net ip status check -----------------------//    
    ip_status = XInference_net_ReadReg(pci_bar_handle, InstancePtr->ctrl_bus_baseaddress, XINFERENCE_NET_CRTL_BUS_ADDR_AP_CTRL);
    cout << "Status feedback from inference ip is : " << ip_status << endl;

    int i = 0;    
    XInference_net_Start(pci_bar_handle, InstancePtr);

    while (!XInference_net_IsDone(pci_bar_handle, InstancePtr)) {
        i++;
    }
    cout << "IP is done at " << i << "attempts" << endl; 

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
    cout << "Function fail!!" << endl;;
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