""" links:
https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-019-0920-y

"""


import math

INIT_BODY_TEMP = 37.2


def Noisy_MH(t, T_ambient, cf, m, max_delta_t=12):
    return Marshall_Hoare(t, T_ambient, cf, m)


# NOTE: t for time, T_ambient for ambient, k for correction factor, m for mass in kg
def Marshall_Hoare(t, T_ambient, k, m):
    B_temp = -1.2815 * (k * math.pow(m, -0.625)) + 0.0284
    if T_ambient <= 23.2:
        return (INIT_BODY_TEMP - T_ambient) * (1.25 * math.exp(B_temp * t) - 0.25 * math.exp(5 * B_temp * t)) + T_ambient        
    return (INIT_BODY_TEMP - T_ambient) * (1.11 * math.exp(B_temp * t) - 0.11 * math.exp(10 * B_temp * t)) + T_ambient