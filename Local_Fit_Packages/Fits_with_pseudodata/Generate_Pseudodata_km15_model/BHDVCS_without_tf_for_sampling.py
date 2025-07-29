import numpy as np
import math

class F1F2:
    def __init__(self):
        self.GM0 = 2.792847337

    def ffGE(self, t):
        GE = 1.0 / (1.0 + (-t / 0.710649)) ** 2
        return GE

    def ffGM(self, t):
        shape = self.ffGE(t)
        return self.GM0 * shape

    def ffF2(self, t):
        f2 = (self.ffGM(t) - self.ffGE(t)) / (1.0 - t / (4.0 * 0.938272 ** 2))
        return f2

    def ffF1(self, t):
        f1 = self.ffGM(t) - self.ffF2(t)
        return f1

    def f1_f21(self, t):
        return self.ffF1(t), self.ffF2(t)


class BHDVCStf_Pure:
    def __init__(self):
        self.M = 0.938272  # GeV
        self.M2 = self.M ** 2
        self.ALP_INV = 137.0359998
        self.PI = math.pi
        self.RAD = self.PI / 180.0
        self.GeV2nb = 0.389379 * 1e6

    def SetKinematics(self, QQ, x, t, k):
        ee = 4. * self.M2 * x * x / QQ
        y = math.sqrt(QQ) / (math.sqrt(ee) * k)
        xi = x * (1. + t / (2. * QQ)) / (2. - x + x * t / QQ)
        Gamma = x * y ** 2 / (self.ALP_INV ** 3 * self.PI * 8. * QQ ** 2 * math.sqrt(1. + ee))
        tmin = -QQ * (2. * (1. - x) * (1. - math.sqrt(1. + ee)) + ee) / (4. * x * (1. - x) + ee)
        Ktilde_10 = math.sqrt(tmin - t) * math.sqrt((1. - x) * math.sqrt(1. + ee) +
                          ((t - tmin) * (ee + 4. * x * (1. - x)) / (4. * QQ))) \
                          * math.sqrt(1. - y - y ** 2 * ee / 4.) / math.sqrt(1. - y + y ** 2 * ee / 4.)
        K = math.sqrt(1. - y + y ** 2 * ee / 4.) * Ktilde_10 / math.sqrt(QQ)
        return ee, y, xi, Gamma, tmin, Ktilde_10, K

    def BHLeptonPropagators(self, phi, QQ, x, t, ee, y, K):
        phi_rad = phi * self.RAD
        KD = -QQ / (2. * y * (1. + ee)) * (1. + 2. * K * math.cos(self.PI - phi_rad)
                                           - t / QQ * (1. - x * (2. - y) + y * ee / 2.) + y * ee / 2.)
        P1 = 1. + 2. * KD / QQ
        P2 = t / QQ - 2. * KD / QQ
        return P1, P2

    def BHUU(self, phi, F1, F2, P1, P2, QQ, x, t, ee, y, Gamma, K):
        phi_rad = phi * self.RAD

        c0 = 8. * K ** 2 * ((2. + 3. * ee) * (QQ / t) * (F1 ** 2 - F2 ** 2 * t / (4. * self.M2)) +
              2. * x ** 2 * (F1 + F2) ** 2) + (2. - y) ** 2 * ((2. + ee) *
              ((4. * x ** 2 * self.M2 / t) * (1. + t / QQ) ** 2 +
              4. * (1. - x) * (1. + x * t / QQ)) * (F1 ** 2 - F2 ** 2 * t / (4. * self.M2)) +
              4. * x ** 2 * (x + (1. - x + ee / 2.) * (1. - t / QQ) ** 2 -
              x * (1. - 2. * x) * t ** 2 / QQ ** 2) * (F1 + F2) ** 2) + 8. * (1. + ee) * \
              (1. - y - ee * y ** 2 / 4.) * (2. * ee * (1. - t / (4. * self.M2)) * (F1 ** 2 -
              F2 ** 2 * t / (4. * self.M2)) - x ** 2 * (1. - t / QQ) ** 2 * (F1 + F2) ** 2)

        c1 = 8. * K * (2. - y) * ((4. * x ** 2 * self.M2 / t - 2. * x - ee) *
              (F1 ** 2 - F2 ** 2 * t / (4. * self.M2)) + 2. * x ** 2 * (1. - (1. - 2. * x) * t / QQ)
              * (F1 + F2) ** 2)

        c2 = 8. * x ** 2 * K ** 2 * ((4. * self.M2 / t) * (F1 ** 2 - F2 ** 2 * t / (4. * self.M2)) +
              2. * (F1 + F2) ** 2)

        amp2 = 1. / (x ** 2 * y ** 2 * (1. + ee) ** 2 * t * P1 * P2) * \
               (c0 + c1 * math.cos(self.PI - phi_rad) + c2 * math.cos(2. * (self.PI - phi_rad)))

        return Gamma * self.GeV2nb * amp2

    def IUU(self, phi, F1, F2, P1, P2, QQ, x, t, ee, y, K, Gamma, ReH, ReE, ReHtilde, twist, tmin, xi, Ktilde_10):
        # Placeholder for A_U_I, B_U_I, C_U_I values
        A_U_I, B_U_I, C_U_I = 0.0, 0.0, 0.0  # You can implement these if needed
        interference = 1. / (x * y ** 3 * t * P1 * P2) * (
            A_U_I * (F1 * ReH - t / 4. / self.M2 * F2 * ReE) +
            B_U_I * (F1 + F2) * (ReH + ReE) +
            C_U_I * (F1 + F2) * ReHtilde
        )
        return Gamma * self.GeV2nb * interference


class F_calc:
    def __init__(self):
        self.module = BHDVCStf_Pure()

    def fn_1(self, kins, cffs):
        phi, QQ, x, t, k, F1, F2 = kins
        ReH, ReE, ReHtilde, c0fit = cffs
        ee, y, xi, Gamma, tmin, Ktilde_10, K = self.module.SetKinematics(QQ, x, t, k)
        P1, P2 = self.module.BHLeptonPropagators(phi, QQ, x, t, ee, y, K)
        xsbhuu = self.module.BHUU(phi, F1, F2, P1, P2, QQ, x, t, ee, y, Gamma, K)
        xsiuu = self.module.IUU(phi, F1, F2, P1, P2, QQ, x, t, ee, y, K, Gamma, ReH, ReE, ReHtilde, "t2", tmin, xi, Ktilde_10)
        return xsbhuu + xsiuu + c0fit
