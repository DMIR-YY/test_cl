#ifndef _ACC_INSTANCE_H_
#define _ACC_INSTANCE_H_

#include "max_pool_acc_innerpp.h"
#include "config.h"

max_pool_acc<data_type, data_type_w, data_type_o, 16, 16, 16, 2, 2> maxPoolAcc1;

void max_pool_layer_new(
   int R_in,
   int C_in,
   int N,
   int K,
   int R,
   int C,
   int S,
   int P,
   bool act,
   data_type *in_data_1,
   data_type_o *out_data_1) {

   maxPoolAcc1.max_pool_layer_acc(R_in, C_in, N, K, R, C, S, P, act, in_data_1, out_data_1);

}



#endif