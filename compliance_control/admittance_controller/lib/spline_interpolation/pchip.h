//
// Created by liaoy on 2020/11/10.
//

#ifndef PCHIP_PCHIP_H
#define PCHIP_PCHIP_H

#ifdef __cplusplus
extern "C"{
#endif

void pchip(const double *x, const double *y, int x_len, const double *new_x, int new_x_len, double *new_y);

#ifdef __cplusplus
}
#endif


#endif //PCHIP_PCHIP_H
