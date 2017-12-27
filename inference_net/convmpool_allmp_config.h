
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
// conv weight port --axi_bram_ctrl_19 [0xC800_0000,0xC800_4000) (256k)
#define CONV_W_BRAM_PCIS_0 UINT64_C(0xC4000000)
#define CONV_W_BRAM_PCIS_1 UINT64_C(0xC4010000)
#define CONV_W_BRAM_PCIS_2 UINT64_C(0xC4020000)
#define CONV_W_BRAM_PCIS_3 UINT64_C(0xC4030000)
#define CONV_W_BRAM_PCIS_4 UINT64_C(0xC4040000)
#define CONV_W_BRAM_PCIS_5 UINT64_C(0xC4050000)
#define CONV_W_BRAM_PCIS_6 UINT64_C(0xC4060000)
#define CONV_W_BRAM_PCIS_7 UINT64_C(0xC4070000)
// conv bias port --axi_bram_ctrl_20{0xCA00_0000,0xCA00_1000) (4k)
#define CONV_B_BRAM_PCIS UINT64_C(0xCA000000)
// temp out 0 1 port --axi_bram_ctrl_3->10 [0xC200_0000, 0xC204_0000) 8 bram controller (32k/per)
#define BUF_OUT_0_0 UINT64_C(0xC2000000)
#define BUF_OUT_0_1 UINT64_C(0xC2010000)
#define BUF_OUT_0_2 UINT64_C(0xC2020000)
#define BUF_OUT_0_3 UINT64_C(0xC2030000)
#define BUF_OUT_0_4 UINT64_C(0xC2040000)
#define BUF_OUT_0_5 UINT64_C(0xC2050000)
#define BUF_OUT_0_6 UINT64_C(0xC2060000)
#define BUF_OUT_0_7 UINT64_C(0xC2070000)
// temp out 1 1 port --axi_bram_ctrl_11->18 [0xC600_0000, 0xC600_8000) 8 bram controller (4k/per)
#define BUF_OUT_1_0 UINT64_C(0xC6000000)
#define BUF_OUT_1_1 UINT64_C(0xC6002000)
#define BUF_OUT_1_2 UINT64_C(0xC6004000)
#define BUF_OUT_1_3 UINT64_C(0xC6006000)
#define BUF_OUT_1_4 UINT64_C(0xC6008000)
#define BUF_OUT_1_5 UINT64_C(0xC600A000)
#define BUF_OUT_1_6 UINT64_C(0xC600C000)
#define BUF_OUT_1_7 UINT64_C(0xC600E000)
// ctrl_cmd_in_port -- axi_bram_ctrl_0 (4k)
#define CTRL_PARAMS UINT64_C(0xC0000000)
// conv_param_in_port -- axi_bram_ctrl_1 (4k)
#define ACC_PARAMS_0 UINT64_C(0xC0001000)
//// pool_param_in_port -- axi_bram_ctrl_2 (4k)
#define ACC_PARAMS_1 UINT64_C(0xC0002000)


#endif

