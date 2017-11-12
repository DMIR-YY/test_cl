
#ifndef _ACC_CONFIG_H_
#define _ACC_CONFIG_H_

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
// conv weight m_axi port -- axi_bram_ctrl_0 (32k)
#define CONV_W_BRAM_PCIS UINT64_C(0x00C0000000)
// conv bias m_axi port -- axi_bram_ctrl_1 (8k)
#define CONV_B_BRAM_PCIS UINT64_C(0x00C2001000)
// temp out 0 1 portA -- axi_bram_ctrl_2 (32k)
#define BUF_OUT_0 UINT64_C(0x00C4000000)
// temp out 1 1 portA -- axi_bram_ctrl_3 (32k)
#define BUF_OUT_1 UINT64_C(0x00C6000000)
// parameter bram definition -- axi_bram_ctrl_4 (4k)
#define ACC_PARAMS_0 UINT64_C(0x00C8000000)
// parameter bram definition -- axi_bram_ctrl_5 (4k)
#define ACC_PARAMS_1 UINT64_C(0x00CA000000)


#endif
